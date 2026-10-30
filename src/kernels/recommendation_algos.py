"""Numba JIT kernels for closed-loop algorithmic social media feeds and broadcast reach."""

from __future__ import annotations

import math
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def compute_closed_loop_recommendations_jit(
    opinions: np.ndarray,
    news_framing: np.ndarray,
    news_sensationalism: np.ndarray,
    news_popularity: np.ndarray,
    beta: float,
    emotional_boost: float,
    uniform_draws: np.ndarray,
) -> np.ndarray:
    """Select one recommended news item per agent via closed-loop algorithmic curation.

    The probability of displaying item j to agent i follows:
        q_{ij} \\propto (||x_i - c_j|| + delta)^{-beta} * (1 + emotional_boost * e_j) * log(1 + popularity_j)
    """
    n_agents, n_dims = opinions.shape
    n_items = news_framing.shape[0]
    selected_items = np.full(n_agents, -1, dtype=np.int32)
    delta = 0.08  # Regularization to prevent singularity at zero distance

    for i in prange(n_agents):
        weights = np.empty(n_items, dtype=np.float64)
        total_w = 0.0

        for j in range(n_items):
            dist_sq = 0.0
            for d in range(n_dims):
                diff = opinions[i, d] - news_framing[j, d]
                dist_sq += diff * diff
            dist = math.sqrt(dist_sq) + delta

            # Closed-loop attitudinal proximity weight: ||x_i - c_j||^{-beta}
            prox_weight = math.pow(dist, -beta)

            # Algorithmic amplification of high emotional appeal / sensationalism and popularity
            amp = (1.0 + emotional_boost * news_sensationalism[j]) * math.log1p(news_popularity[j])
            w = prox_weight * amp
            weights[j] = w
            total_w += w

        if total_w > 0.0:
            target = uniform_draws[i] * total_w
            cum = 0.0
            chosen = n_items - 1
            for j in range(n_items):
                cum += weights[j]
                if cum >= target:
                    chosen = j
                    break
            selected_items[i] = chosen

    return selected_items


@njit(parallel=True, fastmath=True)
def broadcast_exposure_kernel_jit(
    opinions: np.ndarray,
    age: np.ndarray,
    ses: np.ndarray,
    susceptibility: np.ndarray,
    epsilon: np.ndarray,
    coupling_matrix: np.ndarray,
    exposure_count: np.ndarray,
    broadcast_orientations: np.ndarray,
    broadcast_credibility: np.ndarray,
    uniform_draws: np.ndarray,
    broadcast_strength: float,
) -> int:
    """Simulate 1 -> N traditional mass media broadcast reach and opinion pull.

    Reach probability is modulated by agent age cohort (older demographics have higher
    traditional TV/radio/newspaper reach) and socioeconomic profile.
    """
    n_agents, n_dims = opinions.shape
    n_channels = broadcast_orientations.shape[0]
    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))
    total_reached = 0

    for i in prange(n_agents):
        # Age and SES modulate probability of tuning into traditional broadcast media
        age_factor = (age[i] - 16.0) / (90.0 - 16.0)
        reach_prob = 0.15 + 0.50 * age_factor + 0.15 * ses[i]

        if uniform_draws[i] < reach_prob:
            # Pick channel with closest editorial orientation or cyclical assignment
            best_c = 0
            best_dist = 1e9
            for c in range(n_channels):
                d_sq = 0.0
                for d in range(n_dims):
                    diff = opinions[i, d] - broadcast_orientations[c, d]
                    d_sq += diff * diff
                dist_c = math.sqrt(d_sq) * inv_sqrt_d
                if dist_c < best_dist:
                    best_dist = dist_c
                    best_c = c

            # Check bounded confidence scaled by institutional credibility gamma_k
            gamma_k = broadcast_credibility[best_c]
            if best_dist <= epsilon[i] * (1.0 + 0.5 * gamma_k):
                exposure_count[i] += 1
                total_reached += 1
                pull = broadcast_strength * gamma_k * susceptibility[i]

                raw_shift = np.empty(n_dims, dtype=np.float64)
                for d in range(n_dims):
                    raw_shift[d] = pull * (broadcast_orientations[best_c, d] - opinions[i, d])

                for d1 in range(n_dims):
                    coupled = 0.0
                    for d2 in range(n_dims):
                        coupled += coupling_matrix[d1, d2] * raw_shift[d2]
                    val = opinions[i, d1] + coupled
                    if val > 1.0:
                        val = 1.0
                    elif val < -1.0:
                        val = -1.0
                    opinions[i, d1] = val

    return total_reached


@njit(parallel=True, fastmath=True)
def traditional_media_exposure_kernel_jit(
    opinions: np.ndarray,
    uncertainty: np.ndarray,
    age: np.ndarray,
    ses: np.ndarray,
    susceptibility: np.ndarray,
    epsilon: np.ndarray,
    assimilation_bias: np.ndarray,
    coupling_matrix: np.ndarray,
    exposure_count: np.ndarray,
    vehicle_modality: np.ndarray,
    active_framing: np.ndarray,
    active_intensity: np.ndarray,
    vehicle_credibility: np.ndarray,
    uniform_draws: np.ndarray,
    broadcast_strength: float,
    selectivity_beta: float,
) -> tuple[int, int, int]:
    """Simulate Brazilian demographic access and opinion assimilation across M traditional media
    vehicle groups, each operating TV (modality=0), Radio (modality=1), and Press (modality=2).

    Empirical Brazilian Demographic Reach Calibration:
    1. Broadcast TV (modality=0):
       - IBGE PNAD Continua TIC (2022/2023) & Kantar IBOPE Inside Video (2023):
         94.4% household presence, ~78.9% weekly reach, strong positive age gradient (older adults
         consume significantly more linear TV news) and slightly higher reliance in lower/middle SES.
         P_TV(age, ses) = clip(0.24 + 0.48 * ((age - 16)/74)^0.9 - 0.10 * ses, 0.05, 0.85)
    2. Radio AM/FM (modality=1):
       - Kantar IBOPE Inside Audio (2023) & ABERT (2023):
         80-83% metropolitan reach (~3h55m/day), peaking among working-age commuters (30-59) and
         older adults, with strong middle/working-class SES penetration.
         P_Radio(age, ses) = clip(0.18 + 0.34 * ((age - 16)/74) + 0.14 * sin(pi * ses), 0.05, 0.75)
    3. Written Press / Newspapers & Portals (modality=2):
       - Reuters Institute Digital News Report Brazil (2023/2024), IVC Brasil, & SECOM PBM:
         Strongly concentrated in high socioeconomic status (Classes A/B, college-educated: ses^1.8)
         and adults aged 30+.
         P_Press(age, ses) = clip(0.02 + 0.42 * (ses^1.8) + 0.12 * ((age - 16)/74), 0.01, 0.65)
    """
    n_agents, n_dims = opinions.shape
    n_vehicles = vehicle_modality.shape[0]
    if n_vehicles == 0:
        return 0, 0, 0

    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))
    tv_reached = 0
    radio_reached = 0
    press_reached = 0
    delta_reg = 0.12

    for i in prange(n_agents):
        age_norm = (age[i] - 16.0) / 74.0
        if age_norm < 0.0:
            age_norm = 0.0
        elif age_norm > 1.0:
            age_norm = 1.0
        ses_i = ses[i]

        # Empirical Brazilian access probabilities per modality
        p_tv = 0.24 + 0.48 * math.pow(age_norm + 1e-6, 0.9) - 0.10 * ses_i
        if p_tv < 0.05:
            p_tv = 0.05
        elif p_tv > 0.85:
            p_tv = 0.85

        p_radio = 0.18 + 0.34 * age_norm + 0.14 * math.sin(3.141592653589793 * ses_i)
        if p_radio < 0.05:
            p_radio = 0.05
        elif p_radio > 0.75:
            p_radio = 0.75

        p_press = 0.02 + 0.42 * math.pow(ses_i + 1e-6, 1.8) + 0.12 * age_norm
        if p_press < 0.01:
            p_press = 0.01
        elif p_press > 0.65:
            p_press = 0.65

        total_shift = np.zeros(n_dims, dtype=np.float64)
        agent_exposed = 0

        # Evaluate each of the 3 modalities (0=TV, 1=Radio, 2=Press)
        for mod_id in range(3):
            if mod_id == 0:
                reach_p = p_tv
            elif mod_id == 1:
                reach_p = p_radio
            else:
                reach_p = p_press

            # uniform_draws has shape (n_agents, 6): cols 0..2 for access, cols 3..5 for group selection
            if uniform_draws[i, mod_id] < reach_p:
                # Select among the M vehicle groups for this modality weighted by active intensity & proximity
                total_w = 0.0
                chosen_v = -1
                for v in range(n_vehicles):
                    if vehicle_modality[v] != mod_id:
                        continue
                    inten = active_intensity[v]
                    if inten <= 1e-5:
                        continue
                    d_sq = 0.0
                    for d in range(n_dims):
                        diff = opinions[i, d] - active_framing[v, d]
                        d_sq += diff * diff
                    dist_v = math.sqrt(d_sq) * inv_sqrt_d
                    w_v = inten * math.pow(dist_v + delta_reg, -selectivity_beta)
                    total_w += w_v

                if total_w > 0.0:
                    target = uniform_draws[i, mod_id + 3] * total_w
                    cum = 0.0
                    for v in range(n_vehicles):
                        if vehicle_modality[v] != mod_id:
                            continue
                        inten = active_intensity[v]
                        if inten <= 1e-5:
                            continue
                        d_sq = 0.0
                        for d in range(n_dims):
                            diff = opinions[i, d] - active_framing[v, d]
                            d_sq += diff * diff
                        dist_v = math.sqrt(d_sq) * inv_sqrt_d
                        w_v = inten * math.pow(dist_v + delta_reg, -selectivity_beta)
                        cum += w_v
                        if cum >= target:
                            chosen_v = v
                            break

                if chosen_v >= 0:
                    d_sq = 0.0
                    for d in range(n_dims):
                        diff = opinions[i, d] - active_framing[chosen_v, d]
                        d_sq += diff * diff
                    dist_chosen = math.sqrt(d_sq) * inv_sqrt_d

                    gamma_v = vehicle_credibility[chosen_v]
                    # Institutional credibility widens effective confidence bound (McCombs & Shaw, 1972)
                    eff_eps = epsilon[i] * (1.0 + 0.70 * gamma_v)

                    if dist_chosen <= eff_eps:
                        overlap = 1.0 - (dist_chosen / (eff_eps + 1e-12))
                        # Decayed salience factor saturates smoothly at 1.0: 1 - exp(-intensity)
                        salience_mod = 1.0 - math.exp(-active_intensity[chosen_v])
                        # Confirmatory assimilation boost
                        conf_boost = math.exp(
                            0.25 * assimilation_bias[i] * (0.5 - dist_chosen)
                        )
                        pull = (
                            broadcast_strength
                            * gamma_v
                            * susceptibility[i]
                            * overlap
                            * salience_mod
                            * conf_boost
                        )
                        for d in range(n_dims):
                            total_shift[d] += pull * (active_framing[chosen_v, d] - opinions[i, d])

                        agent_exposed += 1
                        if mod_id == 0:
                            tv_reached += 1
                        elif mod_id == 1:
                            radio_reached += 1
                        else:
                            press_reached += 1

        if agent_exposed > 0:
            exposure_count[i] += agent_exposed
            for d1 in range(n_dims):
                coupled = 0.0
                for d2 in range(n_dims):
                    coupled += coupling_matrix[d1, d2] * total_shift[d2]
                val = opinions[i, d1] + coupled
                if val > 1.0:
                    val = 1.0
                elif val < -1.0:
                    val = -1.0
                opinions[i, d1] = val

            # Repeated institutional media exposure slightly reduces attitudinal uncertainty
            u_new = uncertainty[i] * (1.0 - 0.008 * float(agent_exposed))
            if u_new < 0.05:
                u_new = 0.05
            uncertainty[i] = u_new

    return tv_reached, radio_reached, press_reached

