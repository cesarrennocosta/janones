"""GraphEngine: Multiplex 5-layer interpersonal network and social media topologies in CSR format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import scipy.sparse as sp

from src.core.agent_state import AgentStateEngine


LAYER_NAMES = (
    "family",
    "work_school",
    "church_community",
    "friends",
    "weak_ties",
)


@dataclass
class GraphEngine:
    """Manages the 5-layer interpersonal multiplex network as CSR sparse matrices,
    plus the scale-free directed feed network and closed WhatsApp group structures.
    """

    n_agents: int
    layers: Dict[str, sp.csr_matrix] = field(default_factory=dict)
    layer_weights: Dict[str, float] = field(default_factory=dict)
    composite_csr: sp.csr_matrix | None = None

    # Open algorithmic social media directed graph (rows: follower i, cols: creator/influencer j)
    social_feed_csr: sp.csr_matrix | None = None
    influencer_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    eigenvector_centrality: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    # Closed messaging (WhatsApp) group structures (CSR-like flat arrays for Numba)
    wa_group_indptr: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    wa_group_members: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    wa_agent_group_indptr: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    wa_agent_groups: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    n_wa_groups: int = 0

    @classmethod
    def build_multiplex(
        cls,
        agent_state: AgentStateEngine,
        config: dict | None = None,
        seed: int | np.random.SeedSequence | np.random.Generator = 42,
    ) -> "GraphEngine":
        """Generate all 5 interpersonal CSR layers, open scale-free graph, and closed WhatsApp groups."""
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        cfg = config or {}
        n = agent_state.n_agents

        engine = cls(n_agents=n)

        # Default layer contact frequencies / mutual trust weights
        default_weights = {
            "family": 0.38,
            "work_school": 0.22,
            "church_community": 0.18,
            "friends": 0.15,
            "weak_ties": 0.07,
        }
        user_weights = cfg.get("layer_weights", {})
        for k, v in default_weights.items():
            engine.layer_weights[k] = float(user_weights.get(k, v))

        # 1. Layer 1: Family (small dense cliques, daily contact, maximum mutual trust)
        mean_family_size = int(cfg.get("family_clique_size", 4))
        engine.layers["family"] = cls._build_family_layer(n, mean_family_size, rng)

        # 2. Layer 2: School / Work (modular network structured by age & socioeconomic homophily)
        mean_work_deg = int(cfg.get("work_school_degree", 8))
        engine.layers["work_school"] = cls._build_work_school_layer(
            n, agent_state.age, agent_state.ses, mean_work_deg, rng
        )

        # 3. Layer 3: Church / Communities (high-density clusters and strong attitudinal conformity)
        mean_church_deg = int(cfg.get("church_degree", 10))
        engine.layers["church_community"] = cls._build_church_layer(
            n, agent_state.innate_beliefs, mean_church_deg, rng
        )

        # 4. Layer 4: Social Circle / Friends (Watts-Strogatz small-world oriented by attitudinal homophily)
        ws_k = int(cfg.get("friends_k", 6))
        ws_p = float(cfg.get("friends_rewire_p", 0.15))
        engine.layers["friends"] = cls._build_friends_small_world_layer(
            n, agent_state.innate_beliefs, ws_k, ws_p, rng
        )

        # 5. Layer 5: Weak / Casual Ties (sparse random bridges generating global information shortcuts)
        weak_deg = int(cfg.get("weak_ties_degree", 3))
        engine.layers["weak_ties"] = cls._build_weak_ties_layer(n, weak_deg, rng)

        # Build row-normalized weighted composite CSR matrix for fast Numba kernel iteration
        engine.build_composite_csr()

        # Build Open Algorithmic Social Media scale-free directed network & identify influencers
        influencer_frac = float(cfg.get("influencer_fraction", 0.02))
        engine._build_scale_free_feed_network(agent_state, influencer_frac, rng)

        # Build Closed Messaging (WhatsApp) groups (K <= 256 participants)
        max_group_size = min(int(cfg.get("whatsapp_max_group_size", 256)), 256)
        mean_group_size = int(cfg.get("whatsapp_mean_group_size", 32))
        groups_per_agent = int(cfg.get("whatsapp_groups_per_agent", 3))
        engine._build_whatsapp_groups(
            agent_state, max_group_size, mean_group_size, groups_per_agent, rng
        )

        return engine

    @staticmethod
    def _build_family_layer(n: int, mean_clique_size: int, rng: np.random.Generator) -> sp.csr_matrix:
        """Partition agents into small dense cliques (size 2 to 2*mean_clique_size - 1)."""
        perm = rng.permutation(n).astype(np.int32)
        rows = []
        cols = []
        idx = 0
        while idx < n:
            c_size = int(rng.integers(2, max(3, 2 * mean_clique_size)))
            end = min(idx + c_size, n)
            clique = perm[idx:end]
            if len(clique) > 1:
                for u in clique:
                    for v in clique:
                        if u != v:
                            rows.append(u)
                            cols.append(v)
            elif n > 1:
                # Connect leftover singleton to a random neighbor
                partner = int(perm[idx - 1])
                u = int(clique[0])
                rows.extend([u, partner])
                cols.extend([partner, u])
            idx = end

        data = np.ones(len(rows), dtype=np.float64)
        mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        mat.sum_duplicates()
        mat.data[:] = 1.0
        return mat

    @staticmethod
    def _build_work_school_layer(
        n: int,
        age: np.ndarray,
        ses: np.ndarray,
        mean_deg: int,
        rng: np.random.Generator,
    ) -> sp.csr_matrix:
        """Modular network structured by age cohort and socioeconomic homophily."""
        # Sort agents by composite key of normalized age tier and SES
        age_tier = np.floor((age - 16.0) / 15.0)
        sort_key = age_tier * 2.0 + ses + rng.normal(0.0, 0.08, size=n)
        order = np.argsort(sort_key).astype(np.int32)

        half_k = max(1, mean_deg // 2)
        src_list = []
        dst_list = []
        for offset in range(1, half_k + 1):
            u = order
            v = np.roll(order, -offset)
            src_list.append(u)
            dst_list.append(v)
            src_list.append(v)
            dst_list.append(u)

        rows = np.concatenate(src_list)
        cols = np.concatenate(dst_list)
        data = np.ones(len(rows), dtype=np.float64)
        mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        mat.sum_duplicates()
        mat.data[:] = 1.0
        return mat

    @staticmethod
    def _build_church_layer(
        n: int,
        innate_beliefs: np.ndarray,
        mean_deg: int,
        rng: np.random.Generator,
    ) -> sp.csr_matrix:
        """High-density community clusters with strong attitudinal conformity on moral dimension (dim 1)."""
        moral_dim = innate_beliefs[:, min(1, innate_beliefs.shape[1] - 1)]
        n_clusters = max(2, n // max(15, mean_deg * 2))
        # Assign cluster based on quantiles of moral_dim + slight noise
        noisy_moral = moral_dim + rng.normal(0.0, 0.15, size=n)
        order = np.argsort(noisy_moral).astype(np.int32)
        clusters = np.array_split(order, n_clusters)

        rows = []
        cols = []
        for cluster in clusters:
            c_len = len(cluster)
            if c_len <= 1:
                continue
            k_local = min(c_len - 1, max(2, mean_deg))
            for idx_in_c in range(c_len):
                u = cluster[idx_in_c]
                # Connect to k_local neighbors within the same community cluster
                for step in range(1, (k_local // 2) + 1):
                    v = cluster[(idx_in_c + step) % c_len]
                    rows.extend([u, v])
                    cols.extend([v, u])

        data = np.ones(len(rows), dtype=np.float64)
        mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        mat.sum_duplicates()
        mat.data[:] = 1.0
        return mat

    @staticmethod
    def _build_friends_small_world_layer(
        n: int,
        innate_beliefs: np.ndarray,
        k: int,
        rewire_p: float,
        rng: np.random.Generator,
    ) -> sp.csr_matrix:
        """Watts-Strogatz small-world graph ordered by multidimensional attitudinal projection."""
        # Project beliefs onto first principal direction for ring ordering
        projection = innate_beliefs @ np.linspace(1.0, 0.5, innate_beliefs.shape[1])
        order = np.argsort(projection).astype(np.int32)

        half_k = max(1, min(k // 2, (n - 1) // 2))
        src_list = []
        dst_list = []

        for offset in range(1, half_k + 1):
            u = order
            v = np.roll(order, -offset)
            # Rewire fraction rewire_p of edges to homophilous / random targets
            rewire_mask = rng.random(n) < rewire_p
            if np.any(rewire_mask):
                # Pick candidate targets within moderate rank distance (homophilous shortcuts) or global
                rand_shifts = rng.integers(1, max(2, n), size=int(np.sum(rewire_mask)))
                rewired_indices = (np.where(rewire_mask)[0] + rand_shifts) % n
                v = v.copy()
                v[rewire_mask] = order[rewired_indices]

            src_list.extend([u, v])
            dst_list.extend([v, u])

        rows = np.concatenate(src_list)
        cols = np.concatenate(dst_list)
        # Remove self-loops
        valid = rows != cols
        rows = rows[valid]
        cols = cols[valid]

        data = np.ones(len(rows), dtype=np.float64)
        mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        mat.sum_duplicates()
        mat.data[:] = 1.0
        return mat

    @staticmethod
    def _build_weak_ties_layer(n: int, mean_deg: int, rng: np.random.Generator) -> sp.csr_matrix:
        """Erdos-Renyi / sparse random graph creating long-range global bridges."""
        n_edges = max(n, (n * max(1, mean_deg)) // 2)
        u = rng.integers(0, n, size=n_edges, dtype=np.int32)
        v = rng.integers(0, n, size=n_edges, dtype=np.int32)
        valid = u != v
        u = u[valid]
        v = v[valid]
        rows = np.concatenate([u, v])
        cols = np.concatenate([v, u])
        data = np.ones(len(rows), dtype=np.float64)
        mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        mat.sum_duplicates()
        mat.data[:] = 1.0
        return mat

    def build_composite_csr(self) -> sp.csr_matrix:
        """Combine the 5 multiplex layers weighted by layer_weights and row-normalize."""
        combined = sp.csr_matrix((self.n_agents, self.n_agents), dtype=np.float64)
        for name in LAYER_NAMES:
            if name in self.layers:
                w = self.layer_weights.get(name, 0.2)
                combined = combined + w * self.layers[name]

        combined = combined.tocsr()
        combined.sort_indices()

        # Row-normalize so weights sum to 1 per agent (when degree > 0)
        row_sums = np.asarray(combined.sum(axis=1)).ravel()
        inv_sums = np.where(row_sums > 0.0, 1.0 / row_sums, 0.0)
        diag_inv = sp.diags(inv_sums, format="csr")
        self.composite_csr = (diag_inv @ combined).tocsr()
        self.composite_csr.indptr = np.ascontiguousarray(self.composite_csr.indptr, dtype=np.int32)
        self.composite_csr.indices = np.ascontiguousarray(self.composite_csr.indices, dtype=np.int32)
        self.composite_csr.data = np.ascontiguousarray(self.composite_csr.data, dtype=np.float64)
        return self.composite_csr

    def _build_scale_free_feed_network(
        self,
        agent_state: AgentStateEngine,
        influencer_frac: float,
        rng: np.random.Generator,
    ) -> None:
        """Construct a directed scale-free follower->creator graph and compute eigenvector centrality."""
        n = self.n_agents
        n_influencers = max(3, min(n, int(round(n * influencer_frac))))

        # Generate Pareto / Zipf popularity weights so hubs emerge naturally
        pareto_weights = (rng.pareto(a=1.5, size=n) + 1.0) * (0.5 + agent_state.engagement_rate)
        probs = pareto_weights / np.sum(pareto_weights)

        out_deg = max(4, min(25, n - 1))
        total_links = n * out_deg
        followers = np.repeat(np.arange(n, dtype=np.int32), out_deg)
        creators = rng.choice(np.arange(n, dtype=np.int32), size=total_links, replace=True, p=probs)

        valid = followers != creators
        followers = followers[valid]
        creators = creators[valid]
        data = np.ones(len(followers), dtype=np.float64)

        feed_mat = sp.coo_matrix((data, (followers, creators)), shape=(n, n)).tocsr()
        feed_mat.sum_duplicates()
        feed_mat.data[:] = 1.0
        feed_mat.indptr = np.ascontiguousarray(feed_mat.indptr, dtype=np.int32)
        feed_mat.indices = np.ascontiguousarray(feed_mat.indices, dtype=np.int32)
        feed_mat.data = np.ascontiguousarray(feed_mat.data, dtype=np.float64)
        self.social_feed_csr = feed_mat

        # Compute in-degree / eigenvector centrality via power iteration on A^T + damping
        centrality = np.full(n, 1.0 / n, dtype=np.float64)
        adj_t = feed_mat.T.tocsr()
        for _ in range(25):
            next_c = adj_t.dot(centrality) + 0.15 / n
            norm = np.linalg.norm(next_c, ord=2)
            if norm > 0:
                centrality = next_c / norm
        self.eigenvector_centrality = np.ascontiguousarray(centrality, dtype=np.float64)

        # Select top hubs as primary content influencers
        top_ids = np.argsort(centrality)[-n_influencers:][::-1].astype(np.int32)
        self.influencer_ids = np.ascontiguousarray(top_ids, dtype=np.int32)
        agent_state.is_influencer[:] = 0
        agent_state.is_influencer[self.influencer_ids] = 1

    def _build_whatsapp_groups(
        self,
        agent_state: AgentStateEngine,
        max_group_size: int,
        mean_group_size: int,
        groups_per_agent: int,
        rng: np.random.Generator,
    ) -> None:
        """Create closed WhatsApp groups (each K <= 256 participants) with homophilous membership."""
        n = self.n_agents
        max_k = min(256, max_group_size)
        avg_k = min(max_k, max(4, mean_group_size))
        n_groups = max(2, (n * max(1, groups_per_agent)) // avg_k)

        # Order agents by first attitude dimension + district so groups have local/ideological affinity
        score = agent_state.innate_beliefs[:, 0] + 0.2 * (agent_state.district / max(1, agent_state.n_districts))
        sorted_agents = np.argsort(score).astype(np.int32)

        group_members_list = []
        group_indptr = [0]
        agent_to_groups: list[list[int]] = [[] for _ in range(n)]

        for g_id in range(n_groups):
            k_size = int(rng.integers(max(3, avg_k // 2), max_k + 1))
            k_size = min(k_size, n, 256)
            center = int(rng.integers(0, n))
            window = min(n, k_size * 3)
            start = max(0, min(n - window, center - window // 2))
            pool = sorted_agents[start : start + window]
            members = rng.choice(pool, size=k_size, replace=False).astype(np.int32)
            group_members_list.append(members)
            group_indptr.append(group_indptr[-1] + len(members))
            for m in members:
                agent_to_groups[int(m)].append(g_id)

        # Ensure every agent belongs to at least 1 group
        for a_id in range(n):
            if not agent_to_groups[a_id]:
                g_id = int(rng.integers(0, n_groups))
                if len(group_members_list[g_id]) < 256:
                    group_members_list[g_id] = np.append(
                        group_members_list[g_id], np.int32(a_id)
                    )
                    agent_to_groups[a_id].append(g_id)

        # Rebuild flat CSR arrays after ensuring coverage
        flat_members = []
        group_indptr = [0]
        for g_id in range(n_groups):
            mem = group_members_list[g_id]
            flat_members.append(mem)
            group_indptr.append(group_indptr[-1] + len(mem))

        self.n_wa_groups = n_groups
        self.wa_group_indptr = np.ascontiguousarray(group_indptr, dtype=np.int32)
        self.wa_group_members = np.ascontiguousarray(np.concatenate(flat_members), dtype=np.int32)

        agent_indptr = [0]
        flat_agent_groups = []
        for a_id in range(n):
            g_list = agent_to_groups[a_id]
            flat_agent_groups.extend(g_list)
            agent_indptr.append(len(flat_agent_groups))

        self.wa_agent_group_indptr = np.ascontiguousarray(agent_indptr, dtype=np.int32)
        self.wa_agent_groups = np.ascontiguousarray(flat_agent_groups, dtype=np.int32)

    def get_neighbors(self, layer_name: str, agent_id: int) -> np.ndarray:
        """Return the neighbor indices of agent_id in the specified CSR layer."""
        mat = self.layers[layer_name]
        start = mat.indptr[agent_id]
        end = mat.indptr[agent_id + 1]
        return mat.indices[start:end]
