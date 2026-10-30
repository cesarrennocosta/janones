"""Network metrics: Ideological segregation (echo-chamber index) and multiplex structural properties."""

from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
import scipy.sparse as sp


@njit(parallel=True, fastmath=True)
def _compute_local_homophily_jit(
    opinions: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute average weighted neighbor distance across the CSR graph."""
    n_agents, n_dims = opinions.shape
    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))
    agent_dists = np.zeros(n_agents, dtype=np.float64)

    for i in prange(n_agents):
        start = indptr[i]
        end = indptr[i + 1]
        if start == end:
            continue
        w_sum = 0.0
        d_sum = 0.0
        for ptr in range(start, end):
            j = indices[ptr]
            w = weights[ptr]
            dist_sq = 0.0
            for d in range(n_dims):
                diff = opinions[i, d] - opinions[j, d]
                dist_sq += diff * diff
            d_sum += w * math.sqrt(dist_sq) * inv_sqrt_d
            w_sum += w
        if w_sum > 0.0:
            agent_dists[i] = d_sum / w_sum

    return float(np.mean(agent_dists))


def compute_ideological_segregation(opinions: np.ndarray, csr_mat: sp.csr_matrix) -> float:
    """Compute ideological segregation / echo-chamber index in [0, 1].

    Defined as 1 - (mean_neighbor_distance / expected_random_pair_distance),
    clipped to [0, 1], where higher values indicate stronger ideological clustering.
    """
    n_agents, n_dims = opinions.shape
    if n_agents <= 1 or csr_mat.nnz == 0:
        return 0.0

    local_dist = _compute_local_homophily_jit(
        opinions, csr_mat.indptr, csr_mat.indices, csr_mat.data
    )

    # Estimate global pairwise distance from sample variance across dimensions
    var_sum = float(np.sum(np.var(opinions, axis=0)))
    global_dist = math.sqrt(2.0 * var_sum / float(n_dims))
    if global_dist < 1e-12:
        return 0.0

    seg = 1.0 - (local_dist / global_dist)
    return float(np.clip(seg, 0.0, 1.0))


def compute_multiplex_structural_stats(graph_engine) -> dict[str, float]:
    """Extract mean degrees and structural statistics across the 5 multiplex layers."""
    stats: dict[str, float] = {}
    for name, mat in graph_engine.layers.items():
        degrees = np.diff(mat.indptr)
        stats[f"{name}_mean_degree"] = float(np.mean(degrees))
        stats[f"{name}_max_degree"] = float(np.max(degrees)) if len(degrees) > 0 else 0.0
    stats["whatsapp_groups_count"] = float(graph_engine.n_wa_groups)
    stats["influencers_count"] = float(len(graph_engine.influencer_ids))
    return stats
