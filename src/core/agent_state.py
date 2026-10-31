"""AgentStateEngine: Contiguous memory allocation for demographic and attitudinal properties."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# Empirical Brazilian Age Pyramid for Population >= 16 years (IBGE Demographic Census 2022)
# Brackets: (min_age, max_age, relative_population_weight)
# Median age of total Brazilian population is 35.0 years; >= 65 represents 13.6% of adult (>=16) population.
IBGE_2022_ADULT_AGE_PYRAMID = (
    (16.0, 24.0, 0.168),  # Youth / University & High-School age
    (25.0, 34.0, 0.198),  # Young Adults
    (35.0, 44.0, 0.196),  # Prime Working Adults
    (45.0, 54.0, 0.164),  # Mature Working Adults
    (55.0, 64.0, 0.138),  # Pre-Retirement Cohort
    (65.0, 90.0, 0.136),  # Seniors (65+ is 10.9% of total pop = 13.6% of 16+ pop)
)


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
    age: np.ndarray               # float64, shape (N,), in [16, 90]
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

    @staticmethod
    def sample_ibge_2022_ages(n_agents: int, rng: np.random.Generator) -> np.ndarray:
        """Sample continuous ages in [16, 90] following the IBGE 2022 Census empirical age pyramid."""
        probs = np.array([b[2] for b in IBGE_2022_ADULT_AGE_PYRAMID], dtype=np.float64)
        probs /= np.sum(probs)
        bracket_indices = rng.choice(len(IBGE_2022_ADULT_AGE_PYRAMID), size=n_agents, p=probs)
        lows = np.array([IBGE_2022_ADULT_AGE_PYRAMID[idx][0] for idx in bracket_indices], dtype=np.float64)
        highs = np.array([IBGE_2022_ADULT_AGE_PYRAMID[idx][1] for idx in bracket_indices], dtype=np.float64)
        ages = rng.uniform(lows, highs)
        return np.ascontiguousarray(ages, dtype=np.float64)

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
        init_mode: str = "correlated",
        age_distribution: str = "ibge_2022",
    ) -> "AgentStateEngine":
        """Allocate and initialize all contiguous agent arrays from a reproducible RNG stream.

        Parameters
        ----------
        init_mode : str
            - 'uniform_random': Completely independent uniform random initialization of opinions,
              innate beliefs, epsilon, susceptibility, and assimilation bias (null baseline to measure
              emergence of polarization, clustering, and issue alignment from randomness).
            - 'correlated': Demographic-correlated initial beliefs with cognitive coupling applied.
        age_distribution : str
            - 'ibge_2022': Empirical Brazilian age pyramid (IBGE Censo 2022).
            - 'uniform': Uniform distribution in [16, 90].
        """
        if n_dims < 3:
            raise ValueError(f"Multidimensional space requires d >= 3, got {n_dims}.")

        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        # 1. Demographics: Age in [16, 90] (IBGE 2022 Census pyramid or uniform), Gender, SES
        if age_distribution == "ibge_2022":
            age = cls.sample_ibge_2022_ages(n_agents, rng)
        else:
            age = np.ascontiguousarray(rng.uniform(16.0, 90.0, size=n_agents), dtype=np.float64)

        # IBGE 2022 gender split (~51.5% female, ~48.5% male/other)
        gender = np.ascontiguousarray(
            rng.choice(np.array([0, 1, 2], dtype=np.int8), size=n_agents, p=[0.48, 0.50, 0.02])
        )
        ses = np.ascontiguousarray(rng.beta(2.0, 3.0, size=n_agents), dtype=np.float64)

        # Assign electoral districts / urban neighborhoods
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

        age_norm = (age - 16.0) / (90.0 - 16.0)  # 0 at age 16, 1 at age 90

        # 3. Initialize opinions, innate beliefs, susceptibility, epsilon, and assimilation bias
        if init_mode == "uniform_random":
            # Pure random uniform baseline across [-1, 1]^d (unconstrained initial attitudes)
            opinions = np.ascontiguousarray(
                rng.uniform(-1.0, 1.0, size=(n_agents, n_dims)), dtype=np.float64
            )
            innate_beliefs = np.ascontiguousarray(opinions.copy(), dtype=np.float64)
            susceptibility = np.clip(
                0.85 - 0.65 * age_norm + rng.uniform(-0.10, 0.10, size=n_agents), 0.05, 0.98
            )
            susceptibility = np.ascontiguousarray(susceptibility, dtype=np.float64)
            epsilon = np.ascontiguousarray(
                rng.uniform(0.15, max(0.25, 2.0 * base_epsilon), size=n_agents), dtype=np.float64
            )
            assimilation_bias = np.ascontiguousarray(
                rng.uniform(0.0, max(0.5, 2.0 * base_assimilation_bias), size=n_agents),
                dtype=np.float64,
            )
            uncertainty = np.ascontiguousarray(
                rng.uniform(0.10, 0.90, size=n_agents), dtype=np.float64
            )
        elif init_mode in ("centrist_convictions", "extremist_convictions"):
            if init_mode == "centrist_convictions":
                # Unimodal centrist convictions concentrated near 0 (mostly moderate/centrist citizens)
                innate_beliefs = np.ascontiguousarray(
                    np.clip(rng.normal(0.0, 0.25, size=(n_agents, n_dims)), -1.0, 1.0),
                    dtype=np.float64,
                )
            else:
                # Bimodal polarized convictions anchored near +/- 0.72 across dimensions
                pole_signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_agents, 1))
                dim_pattern = np.array([1.0 if d % 2 == 0 else -1.0 for d in range(n_dims)], dtype=np.float64)
                base_poles = pole_signs * dim_pattern[None, :] * 0.72
                is_mod = rng.random(size=n_agents) < 0.18
                base_poles[is_mod, :] = 0.0
                innate_beliefs = np.ascontiguousarray(
                    np.clip(base_poles + rng.normal(0.0, 0.16, size=(n_agents, n_dims)), -1.0, 1.0),
                    dtype=np.float64,
                )
            opinions = np.ascontiguousarray(innate_beliefs.copy(), dtype=np.float64)
            conv_extremity = np.linalg.norm(innate_beliefs, axis=1) / np.sqrt(float(n_dims))
            susceptibility = np.clip(
                0.85 - 0.55 * age_norm - 0.22 * conv_extremity + rng.uniform(-0.08, 0.08, size=n_agents),
                0.05,
                0.98,
            )
            susceptibility = np.ascontiguousarray(susceptibility, dtype=np.float64)
            epsilon = np.ascontiguousarray(
                np.clip(base_epsilon * (1.15 - 0.45 * conv_extremity) + rng.uniform(-0.08, 0.08, size=n_agents), 0.12, 1.20),
                dtype=np.float64,
            )
            assimilation_bias = np.ascontiguousarray(
                np.clip(base_assimilation_bias * (0.65 + 0.75 * conv_extremity) + rng.uniform(0.0, 0.4, size=n_agents), 0.1, 4.0),
                dtype=np.float64,
            )
            uncertainty = np.ascontiguousarray(
                np.clip(0.65 - 0.35 * conv_extremity - 0.20 * age_norm + rng.uniform(-0.08, 0.08, size=n_agents), 0.05, 0.95),
                dtype=np.float64,
            )
        else:
            latent_ideology = (
                (ses - 0.45) * 0.8
                + ((age - 53.0) / 37.0) * 0.35
                + rng.normal(0.0, 0.55, size=n_agents)
            )
            latent_ideology = np.clip(latent_ideology, -1.0, 1.0)

            raw_beliefs = np.empty((n_agents, n_dims), dtype=np.float64)
            for dim in range(n_dims):
                sign = 1.0 if dim % 2 == 0 else -0.7
                raw_beliefs[:, dim] = sign * latent_ideology + rng.normal(0.0, 0.45, size=n_agents)

            correlated_beliefs = raw_beliefs @ coupling_matrix.T
            innate_beliefs = np.ascontiguousarray(
                np.clip(correlated_beliefs, -1.0, 1.0), dtype=np.float64
            )
            opinions = np.ascontiguousarray(innate_beliefs.copy(), dtype=np.float64)

            susceptibility = np.clip(
                0.85 - 0.65 * age_norm + rng.normal(0.0, 0.06, size=n_agents), 0.05, 0.98
            )
            susceptibility = np.ascontiguousarray(susceptibility, dtype=np.float64)

            epsilon = np.clip(
                base_epsilon * (1.0 - 0.25 * age_norm) + rng.normal(0.0, 0.05, size=n_agents),
                0.10,
                1.50,
            )
            epsilon = np.ascontiguousarray(epsilon, dtype=np.float64)

            assimilation_bias = np.clip(
                base_assimilation_bias * (0.7 + 0.6 * np.abs(latent_ideology))
                + rng.exponential(0.2, size=n_agents),
                0.0,
                5.0,
            )
            assimilation_bias = np.ascontiguousarray(assimilation_bias, dtype=np.float64)

            uncertainty = np.ascontiguousarray(
                np.clip(
                    0.55 * (1.0 - 0.4 * age_norm) + rng.uniform(-0.1, 0.1, size=n_agents),
                    0.05,
                    1.0,
                ),
                dtype=np.float64,
            )

        is_influencer = np.zeros(n_agents, dtype=np.uint8)
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
