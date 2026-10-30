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
