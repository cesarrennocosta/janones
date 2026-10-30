"""NewsEntityPool & TraditionalMediaEcosystem: Fact-driven traditional media (TV, Radio, Press)
and social media content pools.

Empirical Brazilian Calibration & Published Theoretical References:
-------------------------------------------------------------------
1. Empirical Brazilian Media Access by Age and Socioeconomic Status (SES):
   - Television (TV Aberta / Paga):
     * IBGE, PNAD Continua TIC 2022/2023 (Acesso a Internet e a Televisao, 2023): TV present in 94.4%
       of Brazilian households; primary news medium for lower/middle SES (Classes C/D/E) and older cohorts.
     * Kantar IBOPE Media, Inside Video 2023/2024: Linear TV reaches 78.9% of the Brazilian population
       weekly (~56% daily), averaging 5h17m/day, with a steep positive age gradient (60+ vs. 16-24).
   - Radio (Radio AM/FM):
     * Kantar IBOPE Media, Inside Audio / Inside Radio 2023 & ABERT (2023): Reaches 80%-83% of residents
       in major Brazilian metropolitan areas (~3h55m/day), peaking among working-age commuters (30-59)
       and older adults across middle and working socioeconomic strata.
   - Written Press (Jornais Impressos, Revistas e Portais da Imprensa Tradicional):
     * Reuters Institute for the Study of Journalism (Oxford), Digital News Report 2023/2024 (Brazil chapter,
       R. Carro), IVC Brasil (2023), & SECOM Pesquisa Brasileira de Midia (PBM): Print circulation (~7-11%)
       and traditional press portals (~28-34%) are strongly concentrated in higher socioeconomic strata
       (Classes A/B, university-educated, SES >= 0.65) and adults aged 30+.

2. Published Theoretical Models for Fact-Driven News, Editorial Slant, Lag, Cadence, and Decay:
   - Mullainathan, S., & Shleifer, A. (2005). "The Market for News." American Economic Review, 95(4), 1031-1053.
     (Exogenous truth/fact arrivals slanted toward a media outlet's ideological position).
   - Gentzkow, M., & Shapiro, J. M. (2006). "Media Bias and Reputation." Journal of Political Economy, 114(2), 280-316;
     and Gentzkow, M., & Shapiro, J. M. (2010). "What Drives Media Slant?" Econometrica, 78(1), 35-71.
   - DellaVigna, S., & Kaplan, E. (2007). "The Fox News Effect: Media Bias and Voting." QJE, 122(3), 1187-1234.
   - Wu, F., & Huberman, B. A. (2007). "Novelty and collective attention." PNAS, 104(45), 17599-17601.
     (Exponential temporal decay of news novelty and public attention).
   - Leskovec, J., Backstrom, L., & Kleinberg, J. (2009). "Meme-tracking and the dynamics of the news cycle."
     ACM SIGKDD, 497-506. (Publication lag and cadence differences across broadcast, radio, and print press).
   - McCombs, M. E., & Shaw, D. L. (1972). "The Agenda-Setting Function of Mass Media." Public Opinion Quarterly, 36(2), 176-187.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


# Modality constants within each Media Vehicle Group (Conglomerate)
MODALITY_TV = 0
MODALITY_RADIO = 1
MODALITY_PRESS = 2
MODALITY_NAMES = ("tv", "radio", "press")


def default_vehicle_group_centers(n_groups: int, n_dims: int) -> np.ndarray:
    """Return canonical ideological centers of mass m_g in [-1, 1]^d for M in {0, 1, 2, 3, 4} media conglomerates.

    Designed so that:
    - M = 1: Single hegemonic conglomerate at (+0.55, +0.55, ..., +0.55), testing unidirectional center-of-mass pull.
    - M = 2: Two competing conglomerates with asymmetric positions (+0.60 and -0.25), net CM = +0.175, testing duopoly pull vs cleavage.
    - M = 3: Tripolar ecosystem (Conservative +0.65, Centrist-Institutional +0.15, Progressive -0.50), net CM = +0.10.
    - M = 4: Four-quadrant pluralistic press ecosystem spanning economic and moral axes.
    """
    if n_groups <= 0:
        return np.zeros((0, n_dims), dtype=np.float64)

    centers = np.zeros((n_groups, n_dims), dtype=np.float64)
    if n_groups == 1:
        # 1 Vehicle Group: Hegemonic conglomerate with clear center of mass
        centers[0, :] = 0.55
    elif n_groups == 2:
        # 2 Vehicle Groups: Dominant center-right conglomerate (+0.60) vs center-left challenger (-0.25)
        centers[0, :] = 0.60
        centers[1, :] = -0.25
    elif n_groups == 3:
        # 3 Vehicle Groups: Right (+0.65), Centrist-Establishment (+0.15), Left (-0.50)
        centers[0, :] = 0.65
        centers[1, :] = 0.15
        centers[2, :] = -0.50
    elif n_groups == 4:
        # 4 Vehicle Groups: Multi-axis ideological quadrants with net positive center-of-mass shift (+0.08)
        base_4 = np.array(
            [
                [0.70, 0.65, 0.55, 0.65],    # Group 1: Conservative / Market-liberal
                [0.50, -0.40, 0.35, 0.20],   # Group 2: Liberal-Centrist / Reformist
                [-0.35, 0.50, -0.25, 0.45],  # Group 3: Popular-Traditional / Statist
                [-0.55, -0.50, -0.45, -0.55],# Group 4: Progressive / Left
            ],
            dtype=np.float64,
        )
        for g in range(4):
            for d in range(n_dims):
                centers[g, d] = base_4[g, d % 4]
    else:
        anchors = np.linspace(-0.65, 0.70, n_groups)
        for g in range(n_groups):
            centers[g, :] = anchors[g]

    return np.ascontiguousarray(centers, dtype=np.float64)


@dataclass
class TraditionalMediaEcosystem:
    """Models M traditional media vehicle groups (conglomerates), each operating 3 vehicles:
    1. Broadcast TV (`tv`, modality 0): lag = 1 step, high credibility (0.75), medium decay (0.30).
    2. Radio (`radio`, modality 1): lag = 0 steps (live bulletin), medium credibility (0.62), fast decay (0.45).
    3. Written Press (`press`, modality 2): lag = 2 steps (print/editorial cycle), period = 2 steps,
       high credibility (0.82), strong center-of-mass slant (0.75), slow decay (0.18).

    Dynamics of Fact-Driven News Generation (Mullainathan & Shleifer 2005; Wu & Huberman 2007):
    - Exogenous facts arrive at rate `fact_rate` (mean facts per step) with latent valence f_k in [-1, 1]^d.
    - When a fact is favorable (closer to a group's center of mass m_g), the group covers it with higher
      probability and higher salience S_{k,g,v}, framing the notice closer to m_g:
          y_{k,g,v} = (1 - eta_v) * f_k + eta_v * m_g + xi
    - Each vehicle publishes the notice after its modality lag tau_v and at its reporting frequency nu_v.
    - Published notices decay exponentially over time at rate exp(-delta_v * dt).
    """

    n_groups: int
    n_dims: int
    group_centers: np.ndarray            # float64, shape (M, d)
    ecosystem_center: np.ndarray         # float64, shape (d,)

    # Flattened vehicle arrays of length V = 3 * M
    n_vehicles: int
    vehicle_group_id: np.ndarray         # int32, shape (V,)
    vehicle_modality: np.ndarray         # int32, shape (V,) in {0=TV, 1=Radio, 2=Press}
    vehicle_centers: np.ndarray          # float64, shape (V, d)
    vehicle_lag: np.ndarray              # int32, shape (V,)
    vehicle_period: np.ndarray           # int32, shape (V,)
    vehicle_pub_prob: np.ndarray         # float64, shape (V,)
    vehicle_slant_pull: np.ndarray       # float64, shape (V,)
    vehicle_decay_rate: np.ndarray       # float64, shape (V,)
    vehicle_credibility: np.ndarray      # float64, shape (V,)

    # Dynamic state of active (decaying) notices per vehicle
    active_weighted_sum: np.ndarray      # float64, shape (V, d): sum_k w_k(t) * y_k
    active_intensity: np.ndarray         # float64, shape (V,): W_v(t) = sum_k w_k(t)
    active_framing: np.ndarray           # float64, shape (V, d): B_v(t) = active_weighted_sum / active_intensity

    # Process hyperparameters
    fact_rate: float = 1.5               # Mean number of exogenous facts occurring per step
    gatekeeping_sigma: float = 0.85      # Selectivity bandwidth for favorable facts
    framing_noise_std: float = 0.05      # Editorial framing noise std

    # Queue of delayed notices waiting for publication step: list of (pub_step, vehicle_idx, salience, framing_vec)
    pending_notices: list[tuple[int, int, float, np.ndarray]] = field(default_factory=list)
    last_step_facts_count: int = 0
    last_step_published_count: int = 0

    @classmethod
    def initialize(
        cls,
        n_groups: int,
        n_dims: int,
        group_centers: np.ndarray | list | None = None,
        fact_rate: float = 1.5,
        gatekeeping_sigma: float = 0.85,
        framing_noise_std: float = 0.05,
        seed: int | np.random.Generator | None = None,
    ) -> "TraditionalMediaEcosystem":
        """Build M vehicle groups (each with TV, Radio, and Press) and seed initial step-0 notices."""
        n_groups = max(0, int(n_groups))
        if group_centers is not None:
            centers = np.ascontiguousarray(np.asarray(group_centers, dtype=np.float64))
            if centers.ndim == 1:
                centers = centers.reshape(1, -1)
        else:
            centers = default_vehicle_group_centers(n_groups, n_dims)

        if n_groups > 0:
            ecosystem_center = np.ascontiguousarray(np.mean(centers, axis=0), dtype=np.float64)
        else:
            ecosystem_center = np.zeros(n_dims, dtype=np.float64)

        n_vehicles = 3 * n_groups
        vehicle_group_id = np.zeros(n_vehicles, dtype=np.int32)
        vehicle_modality = np.zeros(n_vehicles, dtype=np.int32)
        vehicle_centers = np.zeros((n_vehicles, n_dims), dtype=np.float64)
        vehicle_lag = np.zeros(n_vehicles, dtype=np.int32)
        vehicle_period = np.ones(n_vehicles, dtype=np.int32)
        vehicle_pub_prob = np.zeros(n_vehicles, dtype=np.float64)
        vehicle_slant_pull = np.zeros(n_vehicles, dtype=np.float64)
        vehicle_decay_rate = np.zeros(n_vehicles, dtype=np.float64)
        vehicle_credibility = np.zeros(n_vehicles, dtype=np.float64)

        # Modality specifications (TV=0, Radio=1, Press=2) based on Leskovec et al. (2009) & Wu & Huberman (2007)
        # Radio: lag=0, period=1, pub_prob=0.92, slant_pull=0.58, decay=0.45, credibility=0.62
        # TV:    lag=1, period=1, pub_prob=0.85, slant_pull=0.68, decay=0.30, credibility=0.75
        # Press: lag=2, period=2, pub_prob=0.75, slant_pull=0.78, decay=0.18, credibility=0.82
        mod_specs = (
            (MODALITY_TV, 1, 1, 0.85, 0.68, 0.30, 0.75),
            (MODALITY_RADIO, 0, 1, 0.92, 0.58, 0.45, 0.62),
            (MODALITY_PRESS, 2, 2, 0.75, 0.78, 0.18, 0.82),
        )

        for g in range(n_groups):
            for mod_id, lag, period, pub_p, slant, decay, cred in mod_specs:
                v_idx = g * 3 + mod_id
                vehicle_group_id[v_idx] = g
                vehicle_modality[v_idx] = mod_id
                vehicle_centers[v_idx, :] = centers[g, :]
                vehicle_lag[v_idx] = lag
                vehicle_period[v_idx] = period
                vehicle_pub_prob[v_idx] = pub_p
                vehicle_slant_pull[v_idx] = slant
                vehicle_decay_rate[v_idx] = decay
                vehicle_credibility[v_idx] = cred

        # Initialize active notice state at editorial center of mass with baseline intensity
        active_intensity = np.full(n_vehicles, 0.50, dtype=np.float64)
        active_framing = np.ascontiguousarray(vehicle_centers.copy(), dtype=np.float64)
        active_weighted_sum = np.ascontiguousarray(
            active_framing * active_intensity[:, None], dtype=np.float64
        )

        eco = cls(
            n_groups=n_groups,
            n_dims=n_dims,
            group_centers=centers,
            ecosystem_center=ecosystem_center,
            n_vehicles=n_vehicles,
            vehicle_group_id=vehicle_group_id,
            vehicle_modality=vehicle_modality,
            vehicle_centers=np.ascontiguousarray(vehicle_centers, dtype=np.float64),
            vehicle_lag=vehicle_lag,
            vehicle_period=vehicle_period,
            vehicle_pub_prob=vehicle_pub_prob,
            vehicle_slant_pull=vehicle_slant_pull,
            vehicle_decay_rate=vehicle_decay_rate,
            vehicle_credibility=vehicle_credibility,
            active_weighted_sum=active_weighted_sum,
            active_intensity=active_intensity,
            active_framing=active_framing,
            fact_rate=float(fact_rate),
            gatekeeping_sigma=float(gatekeeping_sigma),
            framing_noise_std=float(framing_noise_std),
        )
        return eco

    def step_facts_and_news_cycle(
        self,
        step_idx: int,
        rng: np.random.Generator,
    ) -> dict[str, int]:
        """Advance the news cycle by one timestep:
        1. Exponentially decay all active notices for each vehicle v: exp(-delta_v).
        2. Generate new exogenous facts (Poisson arrival with mean `fact_rate`).
        3. For each fact, each vehicle evaluates gatekeeping favorability toward its center of mass m_g,
           produces a slanted notice closer to m_g, and schedules it with publication lag tau_v.
        4. Publish all notices whose scheduled publication step `t_pub <= step_idx` and aligned with `vehicle_period`.
        """
        if self.n_vehicles == 0:
            self.last_step_facts_count = 0
            self.last_step_published_count = 0
            return {"facts_count": 0, "published_notices": 0}

        # 1. Exponential temporal decay of existing active notices (Wu & Huberman, 2007)
        decay_factors = np.exp(-self.vehicle_decay_rate)
        self.active_weighted_sum *= decay_factors[:, None]
        self.active_intensity *= decay_factors

        # 2. Exogenous Fact Arrival Process
        n_facts = int(rng.poisson(self.fact_rate)) if self.fact_rate > 0.0 else 0
        self.last_step_facts_count = n_facts
        inv_sqrt_d = 1.0 / math.sqrt(float(self.n_dims))

        for _ in range(n_facts):
            # Objective fact valence vector f_k in [-1, 1]^d and intrinsic salience I_k in [0.5, 1.0]
            fact_vec = rng.uniform(-1.0, 1.0, size=self.n_dims)
            fact_salience = float(rng.uniform(0.55, 1.0))

            for v_idx in range(self.n_vehicles):
                center_v = self.vehicle_centers[v_idx]
                dist_f = float(np.linalg.norm(fact_vec - center_v)) * inv_sqrt_d

                # Gatekeeping favorability: facts favorable to the vehicle's center of mass have higher coverage & salience
                favorability = math.exp(-dist_f / max(self.gatekeeping_sigma, 1e-6))
                cov_prob = self.vehicle_pub_prob[v_idx] * (0.35 + 0.65 * favorability)

                if rng.random() < cov_prob:
                    eta = self.vehicle_slant_pull[v_idx]
                    noise = rng.normal(0.0, self.framing_noise_std, size=self.n_dims)
                    # Notice framed closer to the vehicle's center of mass (Mullainathan & Shleifer, 2005)
                    notice_vec = np.clip((1.0 - eta) * fact_vec + eta * center_v + noise, -1.0, 1.0)
                    notice_salience = fact_salience * (0.40 + 0.60 * favorability)

                    # Schedule publication after modality lag tau_v and aligned to publication cadence period
                    raw_pub_step = step_idx + int(self.vehicle_lag[v_idx])
                    period = int(self.vehicle_period[v_idx])
                    if period > 1 and (raw_pub_step % period != 0):
                        raw_pub_step += period - (raw_pub_step % period)

                    self.pending_notices.append((raw_pub_step, v_idx, notice_salience, notice_vec))

        # 3. Publish all scheduled notices ready at `step_idx`
        remaining_queue: list[tuple[int, int, float, np.ndarray]] = []
        published_now = 0
        for pub_step, v_idx, sal, n_vec in self.pending_notices:
            if pub_step <= step_idx:
                self.active_weighted_sum[v_idx] += sal * n_vec
                self.active_intensity[v_idx] += sal
                published_now += 1
            else:
                remaining_queue.append((pub_step, v_idx, sal, n_vec))
        self.pending_notices = remaining_queue
        self.last_step_published_count = published_now

        # 4. Update normalized active framing B_v(t)
        for v_idx in range(self.n_vehicles):
            w = self.active_intensity[v_idx]
            if w > 1e-8:
                self.active_framing[v_idx, :] = np.clip(
                    self.active_weighted_sum[v_idx, :] / w, -1.0, 1.0
                )
            else:
                self.active_framing[v_idx, :] = self.vehicle_centers[v_idx, :]

        return {
            "facts_count": n_facts,
            "published_notices": published_now,
        }


@dataclass
class NewsEntityPool:
    """Contiguous pool of active news/content items, traditional media vehicle groups, and broadcast channels.

    Attributes of each news item n in {0, ..., M-1}:
    - framing (y_n in [-1, 1]^d): Thematic framing vector across d latent issues.
    - sensationalism (e_n in [0, 1]): Emotional appeal / clickbait index.
    - veracity (v_n in [0, 1]): Factual veracity level (low values represent deliberate misinformation).
    - creator_id (int32): Influencer agent ID (or -1 for institutional media).
    - popularity (float64): Bidirectional feedback counter incremented by reads/likes/shares.
    - viral_cascades (int64): Number of WhatsApp group forwards triggered by this item.
    """

    n_items: int
    n_dims: int
    framing: np.ndarray          # float64, shape (M, d), y_n in [-1, 1]^d
    sensationalism: np.ndarray   # float64, shape (M,), e_n in [0, 1]
    veracity: np.ndarray         # float64, shape (M,), v_n in [0, 1]
    creator_id: np.ndarray       # int32, shape (M,)
    popularity: np.ndarray       # float64, shape (M,)
    viral_cascades: np.ndarray   # int64, shape (M,)

    # Legacy broadcast channels & new fact-driven TraditionalMediaEcosystem (TV, Radio, Press)
    n_broadcast: int
    broadcast_orientations: np.ndarray  # float64, shape (B, d), m_k in [-1, 1]^d
    broadcast_credibility: np.ndarray   # float64, shape (B,), gamma_k in [0, 1]
    traditional_ecosystem: TraditionalMediaEcosystem | None = None

    @classmethod
    def initialize(
        cls,
        n_items: int,
        n_dims: int,
        influencer_ids: np.ndarray,
        influencer_opinions: np.ndarray,
        n_broadcast: int = 3,
        misinfo_fraction: float = 0.25,
        seed: int | np.random.SeedSequence | np.random.Generator = 42,
        n_vehicle_groups: int | None = None,
        vehicle_group_centers: np.ndarray | list | None = None,
        fact_rate: float = 1.5,
        gatekeeping_sigma: float = 0.85,
    ) -> "NewsEntityPool":
        """Create an initial pool of news items and initialize the TraditionalMediaEcosystem."""
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        n_inf = len(influencer_ids)
        if n_inf > 0 and len(influencer_opinions) > 0:
            creator_idx = rng.integers(0, n_inf, size=n_items)
            creator_id = np.ascontiguousarray(influencer_ids[creator_idx], dtype=np.int32)
            base_framing = influencer_opinions[creator_idx] * 1.15 + rng.normal(
                0.0, 0.20, size=(n_items, n_dims)
            )
        else:
            creator_id = np.full(n_items, -1, dtype=np.int32)
            base_framing = rng.uniform(-1.0, 1.0, size=(n_items, n_dims))

        framing = np.ascontiguousarray(np.clip(base_framing, -1.0, 1.0), dtype=np.float64)

        is_misinfo = rng.random(n_items) < misinfo_fraction
        veracity = np.where(
            is_misinfo,
            rng.beta(1.5, 5.0, size=n_items),
            rng.beta(5.0, 1.5, size=n_items),
        )
        veracity = np.ascontiguousarray(np.clip(veracity, 0.0, 1.0), dtype=np.float64)

        sensationalism = np.where(
            is_misinfo,
            rng.beta(4.5, 1.8, size=n_items),
            rng.beta(2.0, 3.5, size=n_items),
        )
        sensationalism = np.ascontiguousarray(np.clip(sensationalism, 0.0, 1.0), dtype=np.float64)

        popularity = np.ascontiguousarray(np.ones(n_items, dtype=np.float64))
        viral_cascades = np.zeros(n_items, dtype=np.int64)

        eff_groups = n_vehicle_groups if n_vehicle_groups is not None else max(1, n_broadcast)
        trad_eco = TraditionalMediaEcosystem.initialize(
            n_groups=eff_groups,
            n_dims=n_dims,
            group_centers=vehicle_group_centers,
            fact_rate=fact_rate,
            gatekeeping_sigma=gatekeeping_sigma,
            seed=rng,
        )

        # Keep broadcast_orientations synchronized with vehicle centers for backward compatibility
        if trad_eco.n_vehicles > 0:
            broadcast_orientations = trad_eco.active_framing
            broadcast_credibility = trad_eco.vehicle_credibility
            n_b = trad_eco.n_vehicles
        else:
            n_b = max(1, n_broadcast)
            broadcast_orientations = np.zeros((n_b, n_dims), dtype=np.float64)
            broadcast_credibility = np.full(n_b, 0.70, dtype=np.float64)

        return cls(
            n_items=n_items,
            n_dims=n_dims,
            framing=framing,
            sensationalism=sensationalism,
            veracity=veracity,
            creator_id=creator_id,
            popularity=popularity,
            viral_cascades=viral_cascades,
            n_broadcast=n_b,
            broadcast_orientations=broadcast_orientations,
            broadcast_credibility=broadcast_credibility,
            traditional_ecosystem=trad_eco,
        )

    def refresh_influencer_content(
        self,
        influencer_opinions: np.ndarray,
        replace_fraction: float = 0.20,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Coevolution step: influencers adapt framing toward high-engagement/polarized strategies."""
        if rng is None:
            rng = np.random.default_rng()
        if len(influencer_opinions) == 0:
            return

        n_replace = max(1, int(self.n_items * replace_fraction))
        lowest_idx = np.argsort(self.popularity)[:n_replace]
        n_inf = len(influencer_opinions)
        chosen_inf = rng.integers(0, max(1, n_inf), size=n_replace)

        raw = influencer_opinions[chosen_inf] * 1.25 + rng.normal(0.0, 0.15, size=(n_replace, self.n_dims))
        self.framing[lowest_idx] = np.clip(raw, -1.0, 1.0)
        self.sensationalism[lowest_idx] = np.clip(
            rng.beta(3.0, 2.0, size=n_replace), 0.0, 1.0
        )
        self.veracity[lowest_idx] = np.clip(
            1.0 - 0.65 * self.sensationalism[lowest_idx] + rng.normal(0.0, 0.1, size=n_replace),
            0.0,
            1.0,
        )
        self.popularity[lowest_idx] = np.mean(self.popularity) * 0.5 + 1.0

