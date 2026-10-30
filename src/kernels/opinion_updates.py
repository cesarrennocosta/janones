"""DynamicsKernel: Numba @njit(parallel=True, fastmath=True) kernels for multidimensional opinion dynamics."""

from __future__ import annotations

import math
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def step_interpersonal_dynamics_jit(
    opinions: np.ndarray,
    uncertainty: np.ndarray,
    innate_beliefs: np.ndarray,
    susceptibility: np.ndarray,
    epsilon: np.ndarray,
    assimilation_bias: np.ndarray,
    coupling_matrix: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_weights: np.ndarray,
    dt: float,
    noise_Matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform one synchronous multidimensional interpersonal update step.

    Combines:
    1. Friedkin-Johnsen anchoring to innate beliefs s_i weighted by (1 - alpha_i).
    2. Bounded confidence threshold epsilon_i and Relative Agreement uncertainty overlap h_{ij}.
    3. Biased assimilation weight (alignment boosting via b_i >= 0).
    4. Cognitive coupling matrix C in R^{d x d} imposing thematic issue constraint.
    """
    n_agents, n_dims = opinions.shape
    new_opinions = np.empty((n_agents, n_dims), dtype=np.float64)
    new_uncertainty = np.empty(n_agents, dtype=np.float64)
    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))

    for i in prange(n_agents):
        start = indptr[i]
        end = indptr[i + 1]
        alpha_i = susceptibility[i]
        eps_i = epsilon[i]
        u_i = uncertainty[i]
        b_i = assimilation_bias[i]

        # Compute norm of x_i for cosine alignment in biased assimilation
        norm_i_sq = 0.0
        for d in range(n_dims):
            norm_i_sq += opinions[i, d] * opinions[i, d]
        norm_i = math.sqrt(norm_i_sq) + 1e-12

        social_shift = np.zeros(n_dims, dtype=np.float64)
        total_weight = 0.0
        uncertainty_delta = 0.0

        for ptr in range(start, end):
            j = indices[ptr]
            w_ij = edge_weights[ptr]

            # Normalized Euclidean distance in [-1, 1]^d
            dist_sq = 0.0
            dot_ij = 0.0
            norm_j_sq = 0.0
            for d in range(n_dims):
                diff = opinions[j, d] - opinions[i, d]
                dist_sq += diff * diff
                dot_ij += opinions[i, d] * opinions[j, d]
                norm_j_sq += opinions[j, d] * opinions[j, d]

            dist = math.sqrt(dist_sq) * inv_sqrt_d

            # Bounded confidence & Relative Agreement check
            eff_threshold = eps_i + 0.5 * u_i
            if dist <= eff_threshold:
                u_j = uncertainty[j]
                # Relative agreement overlap h_ij
                overlap = min(eff_threshold, u_j) - max(-eff_threshold, dist - u_j)
                ra_factor = max(0.0, (overlap / (2.0 * u_j + 1e-9)))

                # Biased assimilation: stronger assimilation when attitudes align
                norm_j = math.sqrt(norm_j_sq) + 1e-12
                cos_sim = dot_ij / (norm_i * norm_j)
                bias_factor = math.exp(b_i * 0.35 * cos_sim) / (1.0 + b_i * 0.35)

                eff_w = w_ij * (0.4 + 0.6 * ra_factor) * bias_factor
                total_weight += eff_w

                for d in range(n_dims):
                    social_shift[d] += eff_w * (opinions[j, d] - opinions[i, d])

                uncertainty_delta += eff_w * (u_j - u_i) * 0.25

        if total_weight > 1e-12:
            inv_w = 1.0 / total_weight
            for d in range(n_dims):
                social_shift[d] *= inv_w
            uncertainty_delta *= inv_w

        # Apply cognitive coupling matrix C to social shift: coupled_shift = C @ social_shift
        coupled_shift = np.zeros(n_dims, dtype=np.float64)
        for d1 in range(n_dims):
            acc = 0.0
            for d2 in range(n_dims):
                acc += coupling_matrix[d1, d2] * social_shift[d2]
            coupled_shift[d1] = acc

        # Friedkin-Johnsen update combining innate anchor s_i and coupled social shift
        for d in range(n_dims):
            anchor_pull = (1.0 - alpha_i) * (innate_beliefs[i, d] - opinions[i, d])
            social_pull = alpha_i * coupled_shift[d]
            val = opinions[i, d] + dt * (anchor_pull + social_pull) + noise_Matrix[i, d]
            if val > 1.0:
                val = 1.0
            elif val < -1.0:
                val = -1.0
            new_opinions[i, d] = val

        # Update attitudinal uncertainty u_i(t) in (0, 1]
        u_next = u_i + dt * uncertainty_delta
        if u_next > 1.0:
            u_next = 1.0
        elif u_next < 0.02:
            u_next = 0.02
        new_uncertainty[i] = u_next

    return new_opinions, new_uncertainty


@njit(parallel=True, fastmath=True)
def apply_media_assimilation_jit(
    opinions: np.ndarray,
    uncertainty: np.ndarray,
    susceptibility: np.ndarray,
    epsilon: np.ndarray,
    assimilation_bias: np.ndarray,
    coupling_matrix: np.ndarray,
    exposure_count: np.ndarray,
    engagement_rate: np.ndarray,
    selected_news_idx: np.ndarray,
    news_framing: np.ndarray,
    news_sensationalism: np.ndarray,
    news_veracity: np.ndarray,
    media_strength: float,
) -> np.ndarray:
    """Apply bidirectional biased assimilation from consumed news items to agent opinions.

    Returns an array of interaction flags (1 if agent engaged/liked/shared, 0 otherwise)
    used to increment popularity counters on the platform.
    """
    n_agents, n_dims = opinions.shape
    engaged_flags = np.zeros(n_agents, dtype=np.int32)
    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))

    for i in prange(n_agents):
        n_idx = selected_news_idx[i]
        if n_idx < 0:
            continue

        # Compute attitudinal distance ||x_i - y_n|| / sqrt(d)
        dist_sq = 0.0
        dot_in = 0.0
        norm_i_sq = 0.0
        norm_n_sq = 0.0
        for d in range(n_dims):
            diff = news_framing[n_idx, d] - opinions[i, d]
            dist_sq += diff * diff
            dot_in += opinions[i, d] * news_framing[n_idx, d]
            norm_i_sq += opinions[i, d] * opinions[i, d]
            norm_n_sq += news_framing[n_idx, d] * news_framing[n_idx, d]

        dist = math.sqrt(dist_sq) * inv_sqrt_d
        e_n = news_sensationalism[n_idx]
        v_n = news_veracity[n_idx]
        eps_i = epsilon[i] * (1.0 + 0.35 * e_n)

        if dist <= eps_i:
            exposure_count[i] += 1
            engaged_flags[i] = 1

            b_i = assimilation_bias[i]
            norm_i = math.sqrt(norm_i_sq) + 1e-12
            norm_n = math.sqrt(norm_n_sq) + 1e-12
            cos_sim = dot_in / (norm_i * norm_n)

            # Biased assimilation amplifies confirmatory framing and sensational appeal
            assim_weight = (
                media_strength
                * susceptibility[i]
                * (1.0 + 0.45 * e_n)
                * math.exp(0.30 * b_i * max(0.0, cos_sim))
            )

            raw_delta = np.empty(n_dims, dtype=np.float64)
            for d in range(n_dims):
                raw_delta[d] = assim_weight * (news_framing[n_idx, d] - opinions[i, d])

            # Coupled thematic shift via C
            for d1 in range(n_dims):
                coupled = 0.0
                for d2 in range(n_dims):
                    coupled += coupling_matrix[d1, d2] * raw_delta[d2]
                val = opinions[i, d1] + coupled
                if val > 1.0:
                    val = 1.0
                elif val < -1.0:
                    val = -1.0
                opinions[i, d1] = val

            # High sensationalism / low veracity entrenches certainty (reduces u_i)
            u_new = uncertainty[i] * (1.0 - 0.03 * (1.0 + e_n - 0.5 * v_n))
            if u_new < 0.02:
                u_new = 0.02
            uncertainty[i] = u_new

            # Update agent's platform engagement rate
            eng_new = 0.92 * engagement_rate[i] + 0.08 * min(1.0, 0.5 + 0.5 * e_n)
            engagement_rate[i] = min(1.0, max(0.01, eng_new))

    return engaged_flags


class DynamicsKernel:
    """High-level wrapper around Numba JIT opinion dynamics routines."""

    def __init__(self, dt: float = 0.15, noise_std: float = 0.01, media_strength: float = 0.08) -> None:
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.media_strength = float(media_strength)

    def step_interpersonal(
        self,
        agent_state,
        graph_engine,
        rng: np.random.Generator,
    ) -> None:
        """Advance interpersonal multiplex dynamics by one timestep dt."""
        csr = graph_engine.composite_csr
        if csr is None:
            csr = graph_engine.build_composite_csr()

        noise_matrix = np.ascontiguousarray(
            rng.normal(0.0, self.noise_std, size=(agent_state.n_agents, agent_state.n_dims)),
            dtype=np.float64,
        )

        new_op, new_unc = step_interpersonal_dynamics_jit(
            agent_state.opinions,
            agent_state.uncertainty,
            agent_state.innate_beliefs,
            agent_state.susceptibility,
            agent_state.epsilon,
            agent_state.assimilation_bias,
            agent_state.coupling_matrix,
            csr.indptr,
            csr.indices,
            csr.data,
            self.dt,
            noise_matrix,
        )
        agent_state.opinions[:] = new_op
        agent_state.uncertainty[:] = new_unc
