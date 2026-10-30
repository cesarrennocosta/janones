"""Multidimensional Esteban-Ray polarization index (P_ER) and Sarle's bimodality coefficient."""

from __future__ import annotations

import math
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _esteban_ray_binned_jit(
    bin_centers: np.ndarray,
    bin_masses: np.ndarray,
    alpha: float,
    kappa: float,
) -> float:
    """Compute multidimensional Esteban-Ray polarization index across K populated bins:
        P_ER = kappa * sum_{i=1}^K sum_{j=1}^K pi_i^{1 + alpha} * pi_j * ||y_i - y_j||_2
    """
    n_bins, n_dims = bin_centers.shape
    total_pol = 0.0

    for i in prange(n_bins):
        pi_i = bin_masses[i]
        if pi_i <= 0.0:
            continue
        pi_i_alpha = math.pow(pi_i, 1.0 + alpha)
        row_acc = 0.0
        for j in range(n_bins):
            pi_j = bin_masses[j]
            if pi_j <= 0.0 or i == j:
                continue
            dist_sq = 0.0
            for d in range(n_dims):
                diff = bin_centers[i, d] - bin_centers[j, d]
                dist_sq += diff * diff
            row_acc += pi_j * math.sqrt(dist_sq)
        total_pol += pi_i_alpha * row_acc

    return kappa * total_pol


def compute_esteban_ray_index(
    opinions: np.ndarray,
    alpha: float = 1.3,
    n_bins_per_dim: int = 4,
    kappa: float = 1.0,
) -> float:
    """Compute the multidimensional Esteban-Ray polarization index P_{ER}.

    Discretizes [-1, 1]^d into non-empty multidimensional histogram cells (groups),
    computing group masses pi_i and centroids y_i, then evaluates P_ER via JIT kernel.
    """
    n_agents, n_dims = opinions.shape
    if n_agents <= 1:
        return 0.0

    # Quantize each coordinate into {0, ..., n_bins_per_dim - 1}
    scaled = np.floor(((np.clip(opinions, -1.0, 1.0) + 1.0) * 0.5) * n_bins_per_dim).astype(np.int64)
    scaled = np.clip(scaled, 0, n_bins_per_dim - 1)

    # Encode multi-index into a single integer key
    multipliers = (n_bins_per_dim ** np.arange(n_dims, dtype=np.int64))
    keys = scaled @ multipliers

    unique_keys, inverse_idx, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n_groups = len(unique_keys)
    if n_groups <= 1:
        return 0.0

    masses = np.ascontiguousarray(counts.astype(np.float64) / float(n_agents))
    centers = np.zeros((n_groups, n_dims), dtype=np.float64)
    for d in range(n_dims):
        centers[:, d] = np.bincount(inverse_idx, weights=opinions[:, d], minlength=n_groups) / counts

    centers = np.ascontiguousarray(centers, dtype=np.float64)
    return float(_esteban_ray_binned_jit(centers, masses, float(alpha), float(kappa)))


def compute_sarle_bimodality(opinions: np.ndarray) -> float:
    """Compute mean Sarle's bimodality coefficient across all attitude dimensions:
        BC = (gamma^2 + 1) / (kappa_kurt + 3 * (n-1)^2 / ((n-2)*(n-3)))
    where gamma is skewness and kappa_kurt is excess kurtosis.
    Values > 5/9 (~0.555) indicate bimodal or multimodal polarization.
    """
    n_agents, n_dims = opinions.shape
    if n_agents < 4:
        return 0.0

    bcs = np.zeros(n_dims, dtype=np.float64)
    n_f = float(n_agents)
    sample_correction = (3.0 * ((n_f - 1.0) ** 2)) / ((n_f - 2.0) * (n_f - 3.0))

    for d in range(n_dims):
        x = opinions[:, d]
        mean = np.mean(x)
        centered = x - mean
        m2 = np.mean(centered ** 2)
        if m2 < 1e-12:
            bcs[d] = 0.0
            continue
        m3 = np.mean(centered ** 3)
        m4 = np.mean(centered ** 4)
        skew = m3 / (m2 ** 1.5)
        excess_kurt = (m4 / (m2 ** 2)) - 3.0
        denom = excess_kurt + sample_correction
        if denom <= 1e-12:
            bcs[d] = 0.0
        else:
            bcs[d] = (skew ** 2 + 1.0) / denom

    return float(np.mean(bcs))


class AnalyticsEngine:
    """Unified analytics engine extracting multidimensional polarization, network segregation, and vote metrics."""

    def __init__(self, er_alpha: float = 1.3, er_bins: int = 4) -> None:
        self.er_alpha = float(er_alpha)
        self.er_bins = int(er_bins)

    def compute_step_metrics(
        self,
        step: int,
        agent_state,
        graph_engine,
        electoral_result: dict,
        media_stats: dict | None = None,
    ) -> dict:
        """Compute all aggregate scalar and per-candidate metrics for a simulation step."""
        from src.metrics.network_metrics import compute_ideological_segregation

        p_er = compute_esteban_ray_index(
            agent_state.opinions,
            alpha=self.er_alpha,
            n_bins_per_dim=self.er_bins,
        )
        sarle_bc = compute_sarle_bimodality(agent_state.opinions)
        segregation = compute_ideological_segregation(
            agent_state.opinions, graph_engine.composite_csr
        )

        record = {
            "step": int(step),
            "esteban_ray": float(p_er),
            "sarle_bimodality": float(sarle_bc),
            "ideological_segregation": float(segregation),
            "mean_uncertainty": float(np.mean(agent_state.uncertainty)),
            "mean_opinion_norm": float(np.mean(np.linalg.norm(agent_state.opinions, axis=1))),
            "abstention_rate": float(electoral_result["abstention_rate"]),
            "abstentions": int(electoral_result["abstentions"]),
            "popular_winner": int(electoral_result["popular_winner"]),
            "district_winner": int(electoral_result["district_winner"]),
            "electoral_discrepancy": float(electoral_result["electoral_discrepancy"]),
            "broadcast_reached": float(media_stats["broadcast_reached"]) if media_stats else 0.0,
            "feed_engaged": float(media_stats["feed_engaged"]) if media_stats else 0.0,
            "whatsapp_cascades": float(media_stats["whatsapp_cascades"]) if media_stats else 0.0,
        }

        pop_counts = electoral_result["popular_counts"]
        pop_shares = electoral_result["popular_shares"]
        dist_seats = electoral_result["district_seats"]
        for k in range(len(pop_counts)):
            record[f"votes_cand_{k}"] = int(pop_counts[k])
            record[f"share_cand_{k}"] = float(pop_shares[k])
            record[f"seats_cand_{k}"] = int(dist_seats[k])

        return record
