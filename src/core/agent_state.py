"""AgentStateEngine: Contiguous memory allocation for demographic and attitudinal properties."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AgentStateEngine:
    """Manages contiguous 1D/2D NumPy arrays for N agents in a d-dimensional attitude space.

    All arrays are guaranteed to be C-contiguous (`np.ascontiguousarray`) to allow
    zero-copy pointer passing into Numba `@njit(parallel=True, fastmath=True)` kernels.
    """

    n_agents: int
    n_dims: int
    n_candidates: int
    n_districts: int

    # Fixed demographic & psychological attributes
    age: np.ndarray               # int32 or float64, shape (N,), in [16, 90]
    gender: np.ndarray            # int8, shape (N,), in {0, 1, 2}
    ses: np.ndarray               # float64, shape (N,), in [0, 1]
    district: np.ndarray          # int32, shape (N,), in {0, ..., n_districts - 1}
    innate_beliefs: np.ndarray    # float64, shape (N, d), s_i in [-1, 1]^d (Friedkin-Johnsen anchor)
    susceptibility: np.ndarray    # float64, shape (N,), alpha_i in [0, 1] calibrated by age
    epsilon: np.ndarray           # float64, shape (N,), epsilon_i > 0 bounded confidence threshold
    assimilation_bias: np.ndarray # float64, shape (N,), b_i >= 0 biased assimilation parameter
    is_influencer: np.ndarray     # uint8/bool, shape (N,), marks high-centrality content generators

    # Cognitive coupling matrix (issue constraint)
    coupling_matrix: np.ndarray   # float64, shape (d, d), C in R^{d x d}

    # Mutable dynamical attributes
    opinions: np.ndarray          # float64, shape (N, d), x_i(t) in [-1, 1]^d
    uncertainty: np.ndarray       # float64, shape (N,), u_i(t) in (0, 1] (Relative Agreement)
    exposure_count: np.ndarray    # int64, shape (N,), total recent informational exposures
    engagement_rate: np.ndarray   # float64, shape (N,), platform engagement rate in [0, 1]
    vote_probs: np.ndarray        # float64, shape (N, K + 1), probabilities for K candidates + abstention (index K)
    vote_choice: np.ndarray       # int32, shape (N,), categorical choice in {0, ..., K-1} or -1 for abstention

    @classmethod
    def initialize(
        cls,
        n_agents: int,
        n_dims: int = 4,
        n_candidates: int = 3,
        n_districts: int = 10,
        seed: int | np.random.SeedSequence | np.random.Generator = 42,
        coupling_matrix: np.ndarray | None = None,
        base_epsilon: float = 0.45,
        base_assimilation_bias: float = 1.2,
    ) -> "AgentStateEngine":
        """Allocate and initialize all contiguous agent arrays from a reproducible RNG stream."""
        if n_dims < 3:
            raise ValueError(f"Multidimensional space requires d >= 3, got {n_dims}.")

        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        # 1. Demographics: Age in [16, 90], Gender in {0, 1, 2}, SES in [0, 1]
        age = np.ascontiguousarray(rng.uniform(16.0, 90.0, size=n_agents), dtype=np.float64)
        gender = np.ascontiguousarray(rng.choice(np.array([0, 1, 2], dtype=np.int8), size=n_agents, p=[0.48, 0.48, 0.04]))
        ses = np.ascontiguousarray(rng.beta(2.0, 3.0, size=n_agents), dtype=np.float64)

        # Assign electoral districts with realistic population sizes
        district_weights = rng.dirichlet(np.ones(n_districts) * 3.0)
        district = np.ascontiguousarray(
            rng.choice(np.arange(n_districts, dtype=np.int32), size=n_agents, p=district_weights),
            dtype=np.int32,
        )

        # 2. Cognitive coupling matrix C in R^{d x d}
        if coupling_matrix is None:
            coupling_matrix = cls.default_coupling_matrix(n_dims)
        else:
            coupling_matrix = np.ascontiguousarray(coupling_matrix, dtype=np.float64)
            if coupling_matrix.shape != (n_dims, n_dims):
                raise ValueError(f"coupling_matrix must have shape ({n_dims}, {n_dims}).")

        # 3. Innate beliefs s_i in [-1, 1]^d correlated with SES, age, and latent ideological factor
        latent_ideology = (ses - 0.45) * 0.8 + ((age - 53.0) / 37.0) * 0.35 + rng.normal(0.0, 0.55, size=n_agents)
        latent_ideology = np.clip(latent_ideology, -1.0, 1.0)

        raw_beliefs = np.empty((n_agents, n_dims), dtype=np.float64)
        for dim in range(n_dims):
            sign = 1.0 if dim % 2 == 0 else -0.7
            raw_beliefs[:, dim] = sign * latent_ideology + rng.normal(0.0, 0.45, size=n_agents)

        # Apply cognitive coupling C so initial beliefs respect issue constraint
        correlated_beliefs = raw_beliefs @ coupling_matrix.T
        innate_beliefs = np.ascontiguousarray(np.clip(correlated_beliefs, -1.0, 1.0), dtype=np.float64)

        # 4. Susceptibility alpha_i in [0, 1] calibrated inversely by age (older agents have higher cognitive inertia)
        age_norm = (age - 16.0) / (90.0 - 16.0)  # 0 at age 16, 1 at age 90
        susceptibility = np.clip(0.85 - 0.65 * age_norm + rng.normal(0.0, 0.06, size=n_agents), 0.05, 0.98)
        susceptibility = np.ascontiguousarray(susceptibility, dtype=np.float64)

        # 5. Bounded confidence epsilon_i > 0 and biased assimilation b_i >= 0
        epsilon = np.clip(
            base_epsilon * (1.0 - 0.25 * age_norm) + rng.normal(0.0, 0.05, size=n_agents),
            0.10,
            1.50,
        )
        epsilon = np.ascontiguousarray(epsilon, dtype=np.float64)

        assimilation_bias = np.clip(
            base_assimilation_bias * (0.7 + 0.6 * np.abs(latent_ideology)) + rng.exponential(0.2, size=n_agents),
            0.0,
            5.0,
        )
        assimilation_bias = np.ascontiguousarray(assimilation_bias, dtype=np.float64)

        is_influencer = np.zeros(n_agents, dtype=np.uint8)

        # 6. Mutable dynamical state
        opinions = np.ascontiguousarray(innate_beliefs.copy(), dtype=np.float64)
        uncertainty = np.ascontiguousarray(
            np.clip(0.55 * (1.0 - 0.4 * age_norm) + rng.uniform(-0.1, 0.1, size=n_agents), 0.05, 1.0),
            dtype=np.float64,
        )
        exposure_count = np.zeros(n_agents, dtype=np.int64)
        engagement_rate = np.ascontiguousarray(
            np.clip(0.25 + 0.35 * (1.0 - age_norm) + rng.uniform(-0.05, 0.15, size=n_agents), 0.01, 1.0),
            dtype=np.float64,
        )
        vote_probs = np.zeros((n_agents, n_candidates + 1), dtype=np.float64)
        vote_choice = np.full(n_agents, -1, dtype=np.int32)

        return cls(
            n_agents=n_agents,
            n_dims=n_dims,
            n_candidates=n_candidates,
            n_districts=n_districts,
            age=age,
            gender=gender,
            ses=ses,
            district=district,
            innate_beliefs=innate_beliefs,
            susceptibility=susceptibility,
            epsilon=epsilon,
            assimilation_bias=assimilation_bias,
            is_influencer=is_influencer,
            coupling_matrix=coupling_matrix,
            opinions=opinions,
            uncertainty=uncertainty,
            exposure_count=exposure_count,
            engagement_rate=engagement_rate,
            vote_probs=vote_probs,
            vote_choice=vote_choice,
        )

    @staticmethod
    def default_coupling_matrix(n_dims: int) -> np.ndarray:
        """Construct a row-normalized cognitive coupling matrix C in R^{d x d}.

        Models interdependence across themes (e.g., Economy, Moral/Customs,
        Environment/Climate, Public Security). Diagonal is dominant (self-coherence)
        with structured off-diagonal couplings.
        """
        c_mat = np.eye(n_dims, dtype=np.float64)
        for i in range(n_dims):
            for j in range(n_dims):
                if i != j:
                    dist = abs(i - j)
                    sign = 1.0 if (i + j) % 2 == 0 else -1.0
                    c_mat[i, j] = sign * (0.18 / dist)
        # Normalize rows by L1 norm so coupled linear transforms remain bounded in [-1, 1]^d
        row_norms = np.sum(np.abs(c_mat), axis=1, keepdims=True)
        c_mat = c_mat / np.maximum(row_norms, 1.0)
        return np.ascontiguousarray(c_mat, dtype=np.float64)

    def verify_contiguity(self) -> bool:
        """Verify that all numerical arrays are C-contiguous in memory for JIT compilation."""
        arrays = [
            self.age,
            self.gender,
            self.ses,
            self.district,
            self.innate_beliefs,
            self.susceptibility,
            self.epsilon,
            self.assimilation_bias,
            self.is_influencer,
            self.coupling_matrix,
            self.opinions,
            self.uncertainty,
            self.exposure_count,
            self.engagement_rate,
            self.vote_probs,
            self.vote_choice,
        ]
        return all(arr.flags["C_CONTIGUOUS"] for arr in arrays)
