"""DynamicsKernel: Numba @njit(parallel=True, fastmath=True) kernels for multidimensional opinion dynamics.

Author: César Rennó-Costa (Brain Institute, UFRN, Natal, RN, Brazil)
Paper: "Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections"
Target: Nature Human Behaviour (2026)

Governing Equations:
- Friedkin-Johnsen + Deffuant Relative Agreement: Eqs. 1-4, Eqs. S14-S18
- Direct Exogenous Fact Pull: \\gamma_i(t) (Eq. 6, Eqs. S19-S20)
- Two-Timescale Opinion-Conviction Elastic Tether:
  - Opinion deviation: g_i(t) = ||\\tilde{x}_i(t) - s_i(t)||_2 / \\sqrt{D} (Eq. 7, Eq. S23)
  - Elastic tanh restraint: g_i^{eff}(t) = \\Delta_{max,i} tanh(g_i / \\Delta_{max,i}) (Eq. 8, Eq. S24)
  - Conviction crystallization: \\eta_{cryst,i}(t) (Eq. 10, Eq. S25)
  - Sustained tension rate: \\eta_{sust,i}(t) (Eq. 11, Eq. S26)
  - Coupled update: s_i(t+1) = clip(s_i + (\\eta_{cryst} + \\eta_{sust})(x_i - s_i)) (Eq. 9, Eq. S27)
"""

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


@njit(parallel=True, fastmath=True)
def step_conviction_and_opinion_tether_jit(
    opinions: np.ndarray,
    innate_beliefs: np.ndarray,
    age: np.ndarray,
    ses: np.ndarray,
    susceptibility: np.ndarray,
    fact_bias_vec: np.ndarray,
    step_idx: int,
    max_excursion_base: float = 0.34,
    conviction_drift_rate: float = 0.0024,
    crystallization_tau: float = 280.0,
    fact_direct_pull: float = 0.055,
    fact_bias_active: int = 0,
) -> tuple[float, float]:
    """Two-timescale update coupling fast current opinions x_i(t) and slow convictions s_i(t).

    Grounded in Converse (1964), Zaller (1992, RAS Model), Alwin & Krosnick (1991, Aging-Stability),
    Howe & Krosnick (2017, Attitude Extremity), and Kiley & Vaisey (2020):
    1. Current opinions x_i(t) respond rapidly to current news/fact environments, modulated by
       individual conviction extremity E_i = ||s_i||/sqrt(d), age, and SES.
    2. Elastic Conviction Tether: x_i(t) can fluctuate around conviction s_i(t) (enough to swing
       elections), but is smoothly bounded within an individual excursion envelope Delta_{max, i}
       around s_i(t).
    3. Slow Conviction Stabilization & Adaptation: s_i(t) evolves slowly toward sustained opinion
       experience and crystallizes (stabilizes) over time, changing much more slowly than x_i(t)
       even under strong shifts in exogenous fact bias.
    """
    n_agents, n_dims = opinions.shape
    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))
    cryst_factor = 1.0 / (1.0 + math.pow(float(max(0, step_idx)) / max(10.0, crystallization_tau), 1.15))

    total_op_delta = 0.0
    total_conv_delta = 0.0

    for i in prange(n_agents):
        age_norm = (age[i] - 16.0) / 74.0
        if age_norm < 0.0:
            age_norm = 0.0
        elif age_norm > 1.0:
            age_norm = 1.0
        ses_i = ses[i]
        alpha_i = susceptibility[i]

        # Compute individual conviction extremity E_i = ||s_i||_2 / sqrt(d) in [0, 1]
        s_norm_sq = 0.0
        for d in range(n_dims):
            s_norm_sq += innate_beliefs[i, d] * innate_beliefs[i, d]
        ext_i = math.sqrt(s_norm_sq) * inv_sqrt_d
        if ext_i > 1.0:
            ext_i = 1.0

        # 1. Fast current-opinion sensitivity to ambient exogenous fact bias (when active)
        if fact_bias_active == 1:
            # Centrists and younger adults respond more strongly to shifting real-world facts
            fast_pull = (
                fact_direct_pull
                * (1.30 - 0.72 * ext_i)
                * (1.18 - 0.48 * age_norm)
                * (0.85 + 0.30 * ses_i)
            )
            for d in range(n_dims):
                opinions[i, d] += fast_pull * (fact_bias_vec[d] - opinions[i, d])

        # 2. Elastic conviction tether: keep x_i(t) within bounded excursion Delta_{max, i} of s_i(t)
        gap_sq = 0.0
        for d in range(n_dims):
            diff = opinions[i, d] - innate_beliefs[i, d]
            gap_sq += diff * diff
        gap_dist = math.sqrt(gap_sq) * inv_sqrt_d

        # Centrists (low ext_i) and younger citizens have wider opinion excursion around conviction;
        # extremists (high ext_i) are tightly anchored to their conviction (Howe & Krosnick, 2017)
        max_exc_i = (
            max_excursion_base
            * (0.65 + 0.55 * alpha_i)
            * math.pow(max(0.18, 1.15 - 0.95 * ext_i), 1.35)
            * (1.16 - 0.32 * age_norm)
        )
        if max_exc_i < 0.05:
            max_exc_i = 0.05

        if gap_dist > 1e-9:
            eff_gap = max_exc_i * math.tanh(gap_dist / max_exc_i)
            scale = eff_gap / gap_dist
        else:
            eff_gap = 0.0
            scale = 1.0

        # 3. Slow conviction update s_i(t+1): crystallizes over time, plus slow adaptation under sustained pressure
        eta_cryst = (
            conviction_drift_rate
            * cryst_factor
            * (1.25 - 0.75 * age_norm)
            * (1.22 - 0.68 * ext_i)
            * (1.10 - 0.25 * ses_i)
        )
        # Sustained opinion-conviction tension induces slow plastic drift even after initial crystallization
        tension_ratio = eff_gap / max(max_exc_i, 1e-6)
        eta_sustained = (
            conviction_drift_rate
            * 0.28
            * (tension_ratio * tension_ratio)
            * (1.18 - 0.58 * age_norm)
            * (1.20 - 0.62 * ext_i)
        )
        eta_conv = eta_cryst + eta_sustained

        conv_step_sq = 0.0
        for d in range(n_dims):
            s_old = innate_beliefs[i, d]
            x_tethered = s_old + scale * (opinions[i, d] - s_old)
            if x_tethered > 1.0:
                x_tethered = 1.0
            elif x_tethered < -1.0:
                x_tethered = -1.0
            opinions[i, d] = x_tethered

            ds = eta_conv * (x_tethered - s_old)
            s_new = s_old + ds
            if s_new > 1.0:
                s_new = 1.0
            elif s_new < -1.0:
                s_new = -1.0
            innate_beliefs[i, d] = s_new
            conv_step_sq += (s_new - s_old) * (s_new - s_old)

        total_op_delta += eff_gap
        total_conv_delta += math.sqrt(conv_step_sq) * inv_sqrt_d

    return total_op_delta / float(n_agents), total_conv_delta / float(n_agents)


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
