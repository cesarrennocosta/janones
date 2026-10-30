"""Multidimensional polarization, clustering, and issue alignment reference metrics.

Reference Literature Implemented:
1. Esteban, J., & Ray, D. (1994). "On the Measurement of Polarization." Econometrica, 62(4), 819-851.
   - Multidimensional Esteban-Ray Index (`esteban_ray`).
2. Sarle, W. S. (1983) / SAS Institute; Pfister et al. (2013). "Good things peak in pairs: a note on the bimodality coefficient."
   - Sarle's Bimodality Coefficient (`sarle_bimodality`), where BC > 5/9 (~0.555) indicates bimodality.
3. Bramson, A., et al. (2016). "Disambiguation of social polarization concepts and measures." The Journal of Mathematical Sociology, 40(2), 80-111.
   - Normalized Dispersion Index (`dispersion_from_random`): ratio of empirical multidimensional variance to the uniform random baseline variance in [-1, 1]^d (Var_unif = 1/3 per dimension).
4. Laakso, M., & Taagepera, R. (1979) / Herfindahl-Hirschman Inverse Participation Ratio (IPR) & Jost (2006) True Diversity:
   - Effective Number of Opinion Clusters (`effective_cluster_count`, N_eff = 1 / sum(p_c^2)) and Shannon Entropy Cluster Count (`entropy_cluster_count`, exp(H)).
5. Baldassarri, D., & Gelman, A. (2008). "Partisans without Constraint: Political Polarization and Trends in American Public Opinion." American Journal of Sociology, 114(2), 408-446.
   - Issue Alignment / Cross-Dimensional Correlation (`mean_issue_correlation`, `max_issue_correlation`, and pairwise `corr_d{i}_d{j}`) plus Roy's (1953) Participation Ratio (`effective_dimensionality`, PR = (Tr Sigma)^2 / Tr(Sigma^2)) measuring collapse of multidimensional space into a 1D ideological axis.
"""

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


def compute_clustering_and_esteban_ray(
    opinions: np.ndarray,
    alpha: float = 1.3,
    n_bins_per_dim: int = 4,
    kappa: float = 1.0,
) -> dict[str, float]:
    """Simultaneously compute Esteban-Ray polarization and cluster count metrics in O(N d) time.

    Returns
    -------
    dict with:
    - esteban_ray: Multidimensional Esteban-Ray polarization P_ER (Esteban & Ray, 1994).
    - effective_clusters_ipr: Inverse Participation Ratio N_eff = 1 / sum(pi_c^2) (Laakso & Taagepera, 1979).
      Equals 1.0 for total consensus, 2.0 for symmetric bipolarization, and ~n_bins^d for uniform random.
    - effective_clusters_entropy: Exponential of Shannon entropy exp(-sum pi_c ln pi_c) (Jost, 2006).
    - occupied_clusters: Count of populated ideological cells exceeding 1% population threshold.
    """
    n_agents, n_dims = opinions.shape
    if n_agents <= 1:
        return {
            "esteban_ray": 0.0,
            "effective_clusters_ipr": 1.0,
            "effective_clusters_entropy": 1.0,
            "occupied_clusters": 1,
        }

    # Quantize each coordinate into {0, ..., n_bins_per_dim - 1}
    scaled = np.floor(((np.clip(opinions, -1.0, 1.0) + 1.0) * 0.5) * n_bins_per_dim).astype(np.int64)
    scaled = np.clip(scaled, 0, n_bins_per_dim - 1)

    multipliers = n_bins_per_dim ** np.arange(n_dims, dtype=np.int64)
    keys = scaled @ multipliers

    unique_keys, inverse_idx, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n_groups = len(unique_keys)

    masses = np.ascontiguousarray(counts.astype(np.float64) / float(n_agents))

    # Inverse Participation Ratio (Herfindahl-Hirschman effective number of clusters)
    sum_sq = float(np.sum(masses * masses))
    n_eff_ipr = 1.0 / max(sum_sq, 1e-12)

    # Exponential Shannon Entropy cluster count
    valid_m = masses[masses > 0.0]
    shannon_h = -float(np.sum(valid_m * np.log(valid_m)))
    n_eff_entropy = math.exp(shannon_h)

    # Occupied major clusters (>= 1% of population)
    occupied_major = int(np.sum(masses >= 0.01))

    if n_groups <= 1:
        p_er = 0.0
    else:
        centers = np.zeros((n_groups, n_dims), dtype=np.float64)
        for d in range(n_dims):
            centers[:, d] = np.bincount(inverse_idx, weights=opinions[:, d], minlength=n_groups) / counts
        centers = np.ascontiguousarray(centers, dtype=np.float64)
        p_er = float(_esteban_ray_binned_jit(centers, masses, float(alpha), float(kappa)))

    return {
        "esteban_ray": p_er,
        "effective_clusters_ipr": float(n_eff_ipr),
        "effective_clusters_entropy": float(n_eff_entropy),
        "occupied_clusters": max(1, occupied_major),
    }


def compute_esteban_ray_index(
    opinions: np.ndarray,
    alpha: float = 1.3,
    n_bins_per_dim: int = 4,
    kappa: float = 1.0,
) -> float:
    """Compute the multidimensional Esteban-Ray polarization index P_{ER}."""
    return compute_clustering_and_esteban_ray(
        opinions, alpha=alpha, n_bins_per_dim=n_bins_per_dim, kappa=kappa
    )["esteban_ray"]


def compute_sarle_bimodality(opinions: np.ndarray) -> float:
    """Compute mean Sarle's bimodality coefficient across all attitude dimensions:
        BC = (gamma^2 + 1) / (kappa_kurt + 3 * (n-1)^2 / ((n-2)*(n-3)))
    Values > 5/9 (~0.555) indicate bimodal or multimodal polarization (uniform random has BC = 5/9).
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


def compute_issue_alignment_and_dimensionality(opinions: np.ndarray) -> dict[str, float]:
    """Compute Baldassarri & Gelman (2008) issue alignment correlations, Roy (1953) effective
    dimensionality (Participation Ratio), and Bramson et al. (2016) dispersion relative to uniform random.
    """
    n_agents, n_dims = opinions.shape
    result: dict[str, float] = {}

    # 1. Bramson et al. (2016) Dispersion relative to Uniform(-1, 1) where Var_unif = 1/3
    dim_vars = np.var(opinions, axis=0)
    mean_var = float(np.mean(dim_vars))
    # Ratio: 1.0 = uniform random spread, 0.0 = total consensus, >1.0 = hyper-polarized towards +/-1
    result["dispersion_ratio_vs_random"] = float(mean_var / (1.0 / 3.0))

    # 2. Covariance & Correlation matrix across the d dimensions
    if n_agents < 3 or np.all(dim_vars < 1e-12):
        result["mean_issue_correlation"] = 0.0
        result["max_issue_correlation"] = 0.0
        result["effective_dimensionality"] = 1.0
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                result[f"corr_d{i}_d{j}"] = 0.0
        return result

    cov_mat = np.cov(opinions, rowvar=False)
    trace_cov = float(np.trace(cov_mat))
    trace_cov_sq = float(np.sum(cov_mat * cov_mat))
    # Participation Ratio of eigenvalues: (sum lambda_k)^2 / sum(lambda_k^2) in [1, d]
    if trace_cov_sq > 1e-15:
        eff_dim = (trace_cov * trace_cov) / trace_cov_sq
    else:
        eff_dim = 1.0
    result["effective_dimensionality"] = float(np.clip(eff_dim, 1.0, float(n_dims)))

    # Standardize covariance into Pearson correlation matrix safely
    std_devs = np.sqrt(np.maximum(np.diag(cov_mat), 1e-15))
    corr_mat = cov_mat / np.outer(std_devs, std_devs)
    corr_mat = np.clip(corr_mat, -1.0, 1.0)

    off_diag_abs = []
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            c_ij = float(corr_mat[i, j])
            result[f"corr_d{i}_d{j}"] = c_ij
            off_diag_abs.append(abs(c_ij))

    result["mean_issue_correlation"] = float(np.mean(off_diag_abs)) if off_diag_abs else 0.0
    result["max_issue_correlation"] = float(np.max(off_diag_abs)) if off_diag_abs else 0.0
    return result


class AnalyticsEngine:
    """Unified analytics engine extracting multidimensional polarization, cluster counts,
    issue alignment correlations, network segregation, and vote metrics.
    """

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
        """Compute all aggregate scalar, clustering, correlation, and candidate metrics for a step."""
        from src.metrics.network_metrics import compute_ideological_segregation

        cluster_pol = compute_clustering_and_esteban_ray(
            agent_state.opinions,
            alpha=self.er_alpha,
            n_bins_per_dim=self.er_bins,
        )
        sarle_bc = compute_sarle_bimodality(agent_state.opinions)
        alignment = compute_issue_alignment_and_dimensionality(agent_state.opinions)
        segregation = compute_ideological_segregation(
            agent_state.opinions, graph_engine.composite_csr
        )

        record = {
            "step": int(step),
            "esteban_ray": float(cluster_pol["esteban_ray"]),
            "sarle_bimodality": float(sarle_bc),
            "dispersion_ratio_vs_random": float(alignment["dispersion_ratio_vs_random"]),
            "effective_clusters_ipr": float(cluster_pol["effective_clusters_ipr"]),
            "effective_clusters_entropy": float(cluster_pol["effective_clusters_entropy"]),
            "occupied_clusters": int(cluster_pol["occupied_clusters"]),
            "mean_issue_correlation": float(alignment["mean_issue_correlation"]),
            "max_issue_correlation": float(alignment["max_issue_correlation"]),
            "effective_dimensionality": float(alignment["effective_dimensionality"]),
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

        # Add pairwise dimension correlations (corr_d0_d1, corr_d0_d2, ...)
        for k, v in alignment.items():
            if k.startswith("corr_d"):
                record[k] = float(v)

        pop_counts = electoral_result["popular_counts"]
        pop_shares = electoral_result["popular_shares"]
        dist_seats = electoral_result["district_seats"]
        for k in range(len(pop_counts)):
            record[f"votes_cand_{k}"] = int(pop_counts[k])
            record[f"share_cand_{k}"] = float(pop_shares[k])
            record[f"seats_cand_{k}"] = int(dist_seats[k])

        return record
