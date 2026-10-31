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
    total_possible_bins = int(n_bins_per_dim ** n_dims)

    if total_possible_bins <= 4096:
        full_counts = np.bincount(keys, minlength=total_possible_bins)
        non_empty = full_counts > 0
        counts = full_counts[non_empty]
        n_groups = len(counts)
        masses = np.ascontiguousarray(counts.astype(np.float64) / float(n_agents))

        sum_sq = float(np.sum(masses * masses))
        n_eff_ipr = 1.0 / max(sum_sq, 1e-12)
        shannon_h = -float(np.sum(masses * np.log(masses)))
        n_eff_entropy = math.exp(shannon_h)
        occupied_major = int(np.sum(masses >= 0.01))

        if n_groups <= 1:
            p_er = 0.0
        else:
            centers = np.zeros((n_groups, n_dims), dtype=np.float64)
            for d in range(n_dims):
                bin_sums = np.bincount(keys, weights=opinions[:, d], minlength=total_possible_bins)
                centers[:, d] = bin_sums[non_empty] / counts
            centers = np.ascontiguousarray(centers, dtype=np.float64)
            p_er = float(_esteban_ray_binned_jit(centers, masses, float(alpha), float(kappa)))
    else:
        unique_keys, inverse_idx, counts = np.unique(keys, return_inverse=True, return_counts=True)
        n_groups = len(unique_keys)
        masses = np.ascontiguousarray(counts.astype(np.float64) / float(n_agents))

        sum_sq = float(np.sum(masses * masses))
        n_eff_ipr = 1.0 / max(sum_sq, 1e-12)
        valid_m = masses[masses > 0.0]
        shannon_h = -float(np.sum(valid_m * np.log(valid_m)))
        n_eff_entropy = math.exp(shannon_h)
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


def compute_media_center_of_mass_metrics(
    opinions: np.ndarray,
    group_centers: np.ndarray | None = None,
    group_tendentiousness: np.ndarray | None = None,
) -> dict[str, float]:
    """Quantify how traditional media (TV, Radio, Press) and its tendentiousness level tau_g
    shift public opinion toward the media center of mass and reshape the population opinion distribution
    (DellaVigna & Kaplan, 2007; Gentzkow & Shapiro, 2006, 2010; Mullainathan & Shleifer, 2005).

    Computes:
    - pop_cm_d{k}: Population opinion center of mass along dimension k.
    - pop_cm_norm: Euclidean displacement ||bar{x}(t)||_2 / sqrt(d) from neutral origin 0.
    - pop_skewness: Mean standardized skewness across dimensions (captures asymmetric distribution skew).
    - frac_positive_d0: Share of population with positive attitude x_{i,0} > 0.0.
    - media_cm_distance: Euclidean distance ||bar{x}(t) - bar{m}||_2 / sqrt(d) to press center of mass.
    - shift_toward_media_cm: Directional projection (bar{x}(t) . bar{m}) / (||bar{m}||_2 * sqrt(d)).
    - dist_to_g0_mean: Mean normalized distance of agents to Group 1 center of mass m_0.
    - dist_to_g1_mean: Mean normalized distance of agents to Group 2 center of mass m_1 (if M >= 2).
    - frac_closer_to_g0: Fraction of population closer to Group 1 (m_0) than to any other vehicle group.
    - mean_dist_to_closest_vehicle: Mean normalized distance of agents to their closest vehicle group CM.
    """
    n_agents, n_dims = opinions.shape
    inv_sqrt_d = 1.0 / math.sqrt(float(max(1, n_dims)))
    pop_cm = np.mean(opinions, axis=0) if n_agents > 0 else np.zeros(n_dims, dtype=np.float64)
    pop_cm_norm = float(np.linalg.norm(pop_cm)) * inv_sqrt_d

    # Population distribution skewness & positive hemisphere share
    skews = []
    for d in range(n_dims):
        centered = opinions[:, d] - pop_cm[d]
        m2 = float(np.mean(centered ** 2))
        if m2 > 1e-12:
            m3 = float(np.mean(centered ** 3))
            skews.append(m3 / (m2 ** 1.5))
    pop_skewness = float(np.mean(skews)) if skews else 0.0
    frac_positive_d0 = float(np.mean(opinions[:, 0] > 0.0)) if n_agents > 0 else 0.5

    res: dict[str, float] = {
        "pop_cm_norm": pop_cm_norm,
        "pop_skewness": pop_skewness,
        "frac_positive_d0": frac_positive_d0,
        "mean_tendentiousness": 0.0,
        "tendentiousness_g0": 0.0,
        "tendentiousness_g1": 0.0,
        "tendentiousness_asymmetry": 0.0,
    }
    for d in range(n_dims):
        res[f"pop_cm_d{d}"] = float(pop_cm[d])

    if group_centers is None or len(group_centers) == 0:
        res["media_cm_distance"] = pop_cm_norm
        res["shift_toward_media_cm"] = 0.0
        res["dist_to_g0_mean"] = float(np.mean(np.linalg.norm(opinions, axis=1)) * inv_sqrt_d)
        res["dist_to_g1_mean"] = res["dist_to_g0_mean"]
        res["frac_closer_to_g0"] = frac_positive_d0
        res["mean_dist_to_closest_vehicle"] = res["dist_to_g0_mean"]
        return res

    centers = np.asarray(group_centers, dtype=np.float64)
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    n_groups = centers.shape[0]

    if group_tendentiousness is not None and len(group_tendentiousness) > 0:
        t_arr = np.asarray(group_tendentiousness, dtype=np.float64).ravel()
        res["mean_tendentiousness"] = float(np.mean(t_arr))
        res["tendentiousness_g0"] = float(t_arr[0])
        res["tendentiousness_g1"] = float(t_arr[1]) if len(t_arr) >= 2 else float(t_arr[0])
        res["tendentiousness_asymmetry"] = (
            float(abs(t_arr[0] - t_arr[1])) if len(t_arr) >= 2 else 0.0
        )
        t_sum = float(np.sum(t_arr[:n_groups]))
        if t_sum > 1e-8:
            media_cm = np.sum(centers * t_arr[:n_groups, None], axis=0) / t_sum
        else:
            media_cm = np.mean(centers, axis=0)
    else:
        media_cm = np.mean(centers, axis=0)

    media_cm_norm = float(np.linalg.norm(media_cm))
    res["media_cm_distance"] = float(np.linalg.norm(pop_cm - media_cm)) * inv_sqrt_d

    if media_cm_norm > 1e-8:
        res["shift_toward_media_cm"] = (
            float(np.dot(pop_cm, media_cm) / media_cm_norm) * inv_sqrt_d
        )
    else:
        # Fallback to Group 1 center direction if unweighted/weighted centers cancel out
        g0_norm = float(np.linalg.norm(centers[0]))
        if g0_norm > 1e-8:
            res["shift_toward_media_cm"] = (
                float(np.dot(pop_cm, centers[0]) / g0_norm) * inv_sqrt_d
            )
        else:
            res["shift_toward_media_cm"] = 0.0

    # Compute agent distances to each vehicle group center of mass
    all_dists = np.empty((n_agents, n_groups), dtype=np.float64)
    for g in range(n_groups):
        all_dists[:, g] = np.linalg.norm(opinions - centers[g, None, :], axis=1) * inv_sqrt_d

    res["dist_to_g0_mean"] = float(np.mean(all_dists[:, 0]))
    res["dist_to_g1_mean"] = (
        float(np.mean(all_dists[:, 1])) if n_groups >= 2 else res["dist_to_g0_mean"]
    )
    closest_group = np.argmin(all_dists, axis=1)
    res["frac_closer_to_g0"] = float(np.mean(closest_group == 0)) if n_groups >= 2 else frac_positive_d0
    res["mean_dist_to_closest_vehicle"] = float(np.mean(np.min(all_dists, axis=1)))
    return res


@njit(fastmath=True)
def _conviction_cohort_kernel_jit(
    opinions: np.ndarray,
    innate_beliefs: np.ndarray,
    prev_opinions: np.ndarray,
    prev_convictions: np.ndarray,
    age: np.ndarray,
    ses: np.ndarray,
    candidate_profiles: np.ndarray,
) -> tuple:
    """Single-pass JIT kernel computing conviction vs. opinion gaps, step volatilities,
    counterfactual conviction vote shares, and demographic/extremity cohort breakdowns.
    """
    n_agents, n_dims = opinions.shape
    n_cands = candidate_profiles.shape[0]
    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))

    sum_gap = 0.0
    sum_op_vol = 0.0
    sum_conv_vol = 0.0
    sum_ext = 0.0
    swing_total = 0

    # Age cohorts: young (<30), mid (30..55), senior (>=55)
    cnt_young = 0
    cnt_mid = 0
    cnt_senior = 0
    gap_young = 0.0
    gap_mid = 0.0
    gap_senior = 0.0
    op_vol_young = 0.0
    op_vol_mid = 0.0
    op_vol_senior = 0.0
    conv_vol_young = 0.0
    conv_vol_mid = 0.0
    conv_vol_senior = 0.0
    swing_young = 0
    swing_mid = 0
    swing_senior = 0

    # SES cohorts: low (<0.35), high (>=0.55)
    cnt_low_ses = 0
    cnt_high_ses = 0
    gap_low_ses = 0.0
    gap_high_ses = 0.0
    op_vol_low_ses = 0.0
    op_vol_high_ses = 0.0
    swing_low_ses = 0
    swing_high_ses = 0

    # Conviction extremity cohorts: centrist (<0.35), moderate (0.35..0.60), extremist (>0.60)
    cnt_cent = 0
    cnt_mod = 0
    cnt_ext = 0
    gap_cent = 0.0
    gap_mod = 0.0
    gap_ext = 0.0
    op_vol_cent = 0.0
    op_vol_mod = 0.0
    op_vol_ext = 0.0
    conv_vol_cent = 0.0
    conv_vol_ext = 0.0
    swing_cent = 0
    swing_mod = 0
    swing_ext = 0

    conv_cand_counts = np.zeros(n_cands, dtype=np.int64)

    for i in range(n_agents):
        gap_sq = 0.0
        op_step_sq = 0.0
        conv_step_sq = 0.0
        s_norm_sq = 0.0

        for d in range(n_dims):
            x_d = opinions[i, d]
            s_d = innate_beliefs[i, d]
            dx_s = x_d - s_d
            gap_sq += dx_s * dx_s
            s_norm_sq += s_d * s_d

            dx_t = x_d - prev_opinions[i, d]
            op_step_sq += dx_t * dx_t

            ds_t = s_d - prev_convictions[i, d]
            conv_step_sq += ds_t * ds_t

        gap_i = math.sqrt(gap_sq) * inv_sqrt_d
        op_vol_i = math.sqrt(op_step_sq) * inv_sqrt_d
        conv_vol_i = math.sqrt(conv_step_sq) * inv_sqrt_d
        ext_i = math.sqrt(s_norm_sq) * inv_sqrt_d

        sum_gap += gap_i
        sum_op_vol += op_vol_i
        sum_conv_vol += conv_vol_i
        sum_ext += ext_i

        # Closest candidate under current opinion vs. under conviction + ideological runoff midline flip
        best_op_k = 0
        best_op_dist = 1e12
        best_conv_k = 0
        best_conv_dist = 1e12
        for k in range(n_cands):
            d_op = 0.0
            d_cv = 0.0
            for d in range(n_dims):
                diff_op = opinions[i, d] - candidate_profiles[k, d]
                d_op += diff_op * diff_op
                diff_cv = innate_beliefs[i, d] - candidate_profiles[k, d]
                d_cv += diff_cv * diff_cv
            if d_op < best_op_dist:
                best_op_dist = d_op
                best_op_k = k
            if d_cv < best_conv_dist:
                best_conv_dist = d_cv
                best_conv_k = k

        conv_cand_counts[best_conv_k] += 1
        # Check whether opinion shift flipped either multi-candidate preference or primary issue side (d=0)
        runoff_flipped = (opinions[i, 0] * innate_beliefs[i, 0]) < 0.0
        is_swing = 1 if (best_op_k != best_conv_k or runoff_flipped) else 0
        swing_total += is_swing

        a_i = age[i]
        if a_i < 30.0:
            cnt_young += 1
            gap_young += gap_i
            op_vol_young += op_vol_i
            conv_vol_young += conv_vol_i
            swing_young += is_swing
        elif a_i < 55.0:
            cnt_mid += 1
            gap_mid += gap_i
            op_vol_mid += op_vol_i
            conv_vol_mid += conv_vol_i
            swing_mid += is_swing
        else:
            cnt_senior += 1
            gap_senior += gap_i
            op_vol_senior += op_vol_i
            conv_vol_senior += conv_vol_i
            swing_senior += is_swing

        ses_i = ses[i]
        if ses_i < 0.35:
            cnt_low_ses += 1
            gap_low_ses += gap_i
            op_vol_low_ses += op_vol_i
            swing_low_ses += is_swing
        elif ses_i >= 0.55:
            cnt_high_ses += 1
            gap_high_ses += gap_i
            op_vol_high_ses += op_vol_i
            swing_high_ses += is_swing

        if ext_i < 0.35:
            cnt_cent += 1
            gap_cent += gap_i
            op_vol_cent += op_vol_i
            conv_vol_cent += conv_vol_i
            swing_cent += is_swing
        elif ext_i <= 0.60:
            cnt_mod += 1
            gap_mod += gap_i
            op_vol_mod += op_vol_i
            swing_mod += is_swing
        else:
            cnt_ext += 1
            gap_ext += gap_i
            op_vol_ext += op_vol_i
            conv_vol_ext += conv_vol_i
            swing_ext += is_swing

    return (
        sum_gap,
        sum_op_vol,
        sum_conv_vol,
        sum_ext,
        swing_total,
        cnt_young,
        cnt_mid,
        cnt_senior,
        gap_young,
        gap_mid,
        gap_senior,
        op_vol_young,
        op_vol_mid,
        op_vol_senior,
        conv_vol_young,
        conv_vol_mid,
        conv_vol_senior,
        swing_young,
        swing_mid,
        swing_senior,
        cnt_low_ses,
        cnt_high_ses,
        gap_low_ses,
        gap_high_ses,
        op_vol_low_ses,
        op_vol_high_ses,
        swing_low_ses,
        swing_high_ses,
        cnt_cent,
        cnt_mod,
        cnt_ext,
        gap_cent,
        gap_mod,
        gap_ext,
        op_vol_cent,
        op_vol_mod,
        op_vol_ext,
        conv_vol_cent,
        conv_vol_ext,
        swing_cent,
        swing_mod,
        swing_ext,
        conv_cand_counts,
    )


def compute_conviction_and_cohort_metrics(
    agent_state,
    candidate_profiles: np.ndarray,
    popular_winner: int,
    prev_opinions: np.ndarray | None = None,
    prev_convictions: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute two-timescale conviction vs. opinion metrics, counterfactual conviction-only
    electoral outcomes, and demographic/extremity cohort breakdowns.
    """
    n_agents, n_dims = agent_state.opinions.shape
    inv_sqrt_d = 1.0 / math.sqrt(float(max(1, n_dims)))
    prev_op = prev_opinions if prev_opinions is not None else agent_state.opinions
    prev_cv = prev_convictions if prev_convictions is not None else agent_state.innate_beliefs

    res = _conviction_cohort_kernel_jit(
        agent_state.opinions,
        agent_state.innate_beliefs,
        prev_op,
        prev_cv,
        agent_state.age,
        agent_state.ses,
        candidate_profiles,
    )
    (
        sum_gap,
        sum_op_vol,
        sum_conv_vol,
        sum_ext,
        swing_total,
        cnt_young,
        cnt_mid,
        cnt_senior,
        gap_young,
        gap_mid,
        gap_senior,
        op_vol_young,
        op_vol_mid,
        op_vol_senior,
        conv_vol_young,
        conv_vol_mid,
        conv_vol_senior,
        swing_young,
        swing_mid,
        swing_senior,
        cnt_low_ses,
        cnt_high_ses,
        gap_low_ses,
        gap_high_ses,
        op_vol_low_ses,
        op_vol_high_ses,
        swing_low_ses,
        swing_high_ses,
        cnt_cent,
        cnt_mod,
        cnt_ext,
        gap_cent,
        gap_mod,
        gap_ext,
        op_vol_cent,
        op_vol_mod,
        op_vol_ext,
        conv_vol_cent,
        conv_vol_ext,
        swing_cent,
        swing_mod,
        swing_ext,
        conv_cand_counts,
    ) = res

    n_f = float(max(1, n_agents))
    conv_cm = np.mean(agent_state.innate_beliefs, axis=0)
    conv_cm_norm = float(np.linalg.norm(conv_cm)) * inv_sqrt_d
    conv_vars = np.var(agent_state.innate_beliefs, axis=0)
    conv_disp_ratio = float(np.mean(conv_vars) / (1.0 / 3.0))

    conv_winner = int(np.argmax(conv_cand_counts)) if len(conv_cand_counts) > 0 else -1
    conv_shares = conv_cand_counts / n_f

    out: dict[str, float | int] = {
        "conv_cm_d0": float(conv_cm[0]) if n_dims > 0 else 0.0,
        "conv_cm_norm": conv_cm_norm,
        "conv_dispersion_ratio": conv_disp_ratio,
        "mean_conviction_extremity": float(sum_ext / n_f),
        "mean_op_conv_gap": float(sum_gap / n_f),
        "opinion_step_volatility": float(sum_op_vol / n_f),
        "conviction_step_volatility": float(sum_conv_vol / n_f),
        "conv_popular_winner": conv_winner,
        "election_flipped_by_opinion": 1.0 if (popular_winner >= 0 and popular_winner != conv_winner) else 0.0,
        "swing_voter_fraction": float(swing_total / n_f),
        "op_conv_gap_young": float(gap_young / max(1, cnt_young)),
        "op_conv_gap_mid": float(gap_mid / max(1, cnt_mid)),
        "op_conv_gap_senior": float(gap_senior / max(1, cnt_senior)),
        "op_vol_young": float(op_vol_young / max(1, cnt_young)),
        "op_vol_mid": float(op_vol_mid / max(1, cnt_mid)),
        "op_vol_senior": float(op_vol_senior / max(1, cnt_senior)),
        "conv_vol_young": float(conv_vol_young / max(1, cnt_young)),
        "conv_vol_mid": float(conv_vol_mid / max(1, cnt_mid)),
        "conv_vol_senior": float(conv_vol_senior / max(1, cnt_senior)),
        "swing_vote_young": float(swing_young / max(1, cnt_young)),
        "swing_vote_mid": float(swing_mid / max(1, cnt_mid)),
        "swing_vote_senior": float(swing_senior / max(1, cnt_senior)),
        "op_conv_gap_low_ses": float(gap_low_ses / max(1, cnt_low_ses)),
        "op_conv_gap_high_ses": float(gap_high_ses / max(1, cnt_high_ses)),
        "op_vol_low_ses": float(op_vol_low_ses / max(1, cnt_low_ses)),
        "op_vol_high_ses": float(op_vol_high_ses / max(1, cnt_high_ses)),
        "swing_vote_low_ses": float(swing_low_ses / max(1, cnt_low_ses)),
        "swing_vote_high_ses": float(swing_high_ses / max(1, cnt_high_ses)),
        "frac_centrist_convictions": float(cnt_cent / n_f),
        "frac_extremist_convictions": float(cnt_ext / n_f),
        "op_conv_gap_centrist": float(gap_cent / max(1, cnt_cent)),
        "op_conv_gap_moderate": float(gap_mod / max(1, cnt_mod)),
        "op_conv_gap_extremist": float(gap_ext / max(1, cnt_ext)),
        "op_vol_centrist": float(op_vol_cent / max(1, cnt_cent)),
        "op_vol_moderate": float(op_vol_mod / max(1, cnt_mod)),
        "op_vol_extremist": float(op_vol_ext / max(1, cnt_ext)),
        "conv_vol_centrist": float(conv_vol_cent / max(1, cnt_cent)),
        "conv_vol_extremist": float(conv_vol_ext / max(1, cnt_ext)),
        "swing_vote_centrist": float(swing_cent / max(1, cnt_cent)),
        "swing_vote_moderate": float(swing_mod / max(1, cnt_mod)),
        "swing_vote_extremist": float(swing_ext / max(1, cnt_ext)),
    }
    for k in range(len(conv_shares)):
        out[f"conv_share_cand_{k}"] = float(conv_shares[k])
    return out


class AnalyticsEngine:
    """Unified analytics engine extracting multidimensional polarization, cluster counts,
    issue alignment correlations, press center-of-mass pull, network segregation, and vote metrics.
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
        news_pool=None,
        candidate_profiles: np.ndarray | None = None,
        prev_opinions: np.ndarray | None = None,
        prev_convictions: np.ndarray | None = None,
        fact_bias_d0: float = 0.0,
    ) -> dict:
        """Compute all aggregate scalar, clustering, correlation, press CM, and candidate metrics for a step."""
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

        group_centers = None
        group_tendentiousness = None
        n_vehicle_groups = 0
        ext_target = np.full(agent_state.n_dims, 0.85, dtype=np.float64)
        if news_pool is not None:
            if getattr(news_pool, "extreme_imposed_target", None) is not None:
                ext_target = news_pool.extreme_imposed_target
            if getattr(news_pool, "traditional_ecosystem", None) is not None:
                trad_eco = news_pool.traditional_ecosystem
                group_centers = trad_eco.group_centers
                group_tendentiousness = trad_eco.group_tendentiousness
                n_vehicle_groups = int(trad_eco.n_groups)

        # Quantify population tendency to follow the Extreme Imposed ideals (e_imp = [+0.85, ..., +0.85])
        inv_sqrt_d = 1.0 / math.sqrt(float(max(1, agent_state.n_dims)))
        dists_to_ext = (
            np.linalg.norm(agent_state.opinions - ext_target[None, :], axis=1) * inv_sqrt_d
        )
        dist_to_extreme_imposed_mean = float(np.mean(dists_to_ext))
        frac_following_extreme_imposed = float(np.mean(dists_to_ext <= 0.45))
        ext_norm = float(np.linalg.norm(ext_target))
        if ext_norm > 1e-9:
            pop_cm = np.mean(agent_state.opinions, axis=0)
            extreme_imposed_alignment = (
                float(np.dot(pop_cm, ext_target) / ext_norm) * inv_sqrt_d
            )
        else:
            extreme_imposed_alignment = 0.0

        cm_metrics = compute_media_center_of_mass_metrics(
            agent_state.opinions,
            group_centers=group_centers,
            group_tendentiousness=group_tendentiousness,
        )

        record = {
            "step": int(step),
            "n_vehicle_groups": int(n_vehicle_groups),
            "mean_tendentiousness": float(cm_metrics["mean_tendentiousness"]),
            "tendentiousness_g0": float(cm_metrics["tendentiousness_g0"]),
            "tendentiousness_g1": float(cm_metrics["tendentiousness_g1"]),
            "tendentiousness_asymmetry": float(cm_metrics["tendentiousness_asymmetry"]),
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
            "pop_cm_norm": float(cm_metrics["pop_cm_norm"]),
            "pop_skewness": float(cm_metrics["pop_skewness"]),
            "frac_positive_d0": float(cm_metrics["frac_positive_d0"]),
            "media_cm_distance": float(cm_metrics["media_cm_distance"]),
            "shift_toward_media_cm": float(cm_metrics["shift_toward_media_cm"]),
            "dist_to_g0_mean": float(cm_metrics["dist_to_g0_mean"]),
            "dist_to_g1_mean": float(cm_metrics["dist_to_g1_mean"]),
            "frac_closer_to_g0": float(cm_metrics["frac_closer_to_g0"]),
            "mean_dist_to_closest_vehicle": float(cm_metrics["mean_dist_to_closest_vehicle"]),
            "mean_uncertainty": float(np.mean(agent_state.uncertainty)),
            "mean_opinion_norm": float(np.mean(np.linalg.norm(agent_state.opinions, axis=1))),
            "abstention_rate": float(electoral_result["abstention_rate"]),
            "abstentions": int(electoral_result["abstentions"]),
            "popular_winner": int(electoral_result["popular_winner"]),
            "district_winner": int(electoral_result["district_winner"]),
            "electoral_discrepancy": float(electoral_result["electoral_discrepancy"]),
            "broadcast_reached": float(media_stats.get("broadcast_reached", 0.0)) if media_stats else 0.0,
            "tv_reached": float(media_stats.get("tv_reached", 0.0)) if media_stats else 0.0,
            "radio_reached": float(media_stats.get("radio_reached", 0.0)) if media_stats else 0.0,
            "press_reached": float(media_stats.get("press_reached", 0.0)) if media_stats else 0.0,
            "facts_count": float(media_stats.get("facts_count", 0.0)) if media_stats else 0.0,
            "published_notices": float(media_stats.get("published_notices", 0.0)) if media_stats else 0.0,
            "dictated_trad_share": float(media_stats.get("dictated_trad_share", 0.0)) if media_stats else 0.0,
            "mean_fact_longevity": float(media_stats.get("mean_fact_longevity", 1.0)) if media_stats else 1.0,
            "mean_fact_relevance": float(media_stats.get("mean_fact_relevance", 0.65)) if media_stats else 0.65,
            "sync_network_interest_push": float(media_stats.get("sync_network_interest_push", 0.0)) if media_stats else 0.0,
            "synchronized_fact_share": float(media_stats.get("synchronized_fact_share", 0.0)) if media_stats else 0.0,
            "feed_engaged": float(media_stats.get("feed_engaged", 0.0)) if media_stats else 0.0,
            "likes_count": float(media_stats.get("likes_count", 0.0)) if media_stats else 0.0,
            "dislikes_count": float(media_stats.get("dislikes_count", 0.0)) if media_stats else 0.0,
            "dislike_share": float(media_stats.get("dislike_share", 0.0)) if media_stats else 0.0,
            "ig_tiktok_reached": float(media_stats.get("ig_tiktok_reached", 0.0)) if media_stats else 0.0,
            "youtube_x_reached": float(media_stats.get("youtube_x_reached", 0.0)) if media_stats else 0.0,
            "facebook_reached": float(media_stats.get("facebook_reached", 0.0)) if media_stats else 0.0,
            "influencer_direct_impacts": float(media_stats.get("influencer_direct_impacts", 0.0)) if media_stats else 0.0,
            "cowork_cross_exposures": float(media_stats.get("cowork_cross_exposures", 0.0)) if media_stats else 0.0,
            "influencer_unfollows": float(media_stats.get("influencer_unfollows", 0.0)) if media_stats else 0.0,
            "followers_gained": float(media_stats.get("followers_gained", 0.0)) if media_stats else 0.0,
            "followers_lost": float(media_stats.get("followers_lost", 0.0)) if media_stats else 0.0,
            "net_follower_change": float(media_stats.get("net_follower_change", 0.0)) if media_stats else 0.0,
            "mean_inf_followers": float(media_stats.get("mean_inf_followers", 0.0)) if media_stats else 0.0,
            "max_inf_followers": float(media_stats.get("max_inf_followers", 0.0)) if media_stats else 0.0,
            "influencer_follower_gini": float(media_stats.get("influencer_follower_gini", 0.0)) if media_stats else 0.0,
            "mean_inf_tendentiousness": float(media_stats.get("mean_inf_tendentiousness", 0.0)) if media_stats else 0.0,
            "tendentious_follower_share": float(media_stats.get("tendentious_follower_share", 0.5)) if media_stats else 0.5,
            "frac_following_extreme_imposed": frac_following_extreme_imposed,
            "dist_to_extreme_imposed_mean": dist_to_extreme_imposed_mean,
            "extreme_imposed_alignment": extreme_imposed_alignment,
            "extreme_imposed_inf_followers_mean": float(media_stats.get("extreme_imposed_inf_followers_mean", 0.0)) if media_stats else 0.0,
            "other_inf_followers_mean": float(media_stats.get("other_inf_followers_mean", 0.0)) if media_stats else 0.0,
            "extreme_imposed_follower_ratio": float(media_stats.get("extreme_imposed_follower_ratio", 1.0)) if media_stats else 1.0,
            "extreme_imposed_follower_share": float(media_stats.get("extreme_imposed_follower_share", 0.0)) if media_stats else 0.0,
            "extreme_imposed_popularity_ratio": float(media_stats.get("extreme_imposed_popularity_ratio", 1.0)) if media_stats else 1.0,
            "extreme_imposed_popularity_share": float(media_stats.get("extreme_imposed_popularity_share", 0.0)) if media_stats else 0.0,
            "whatsapp_cascades": float(media_stats.get("whatsapp_cascades", 0.0)) if media_stats else 0.0,
        }

        for k, v in cm_metrics.items():
            if k.startswith("pop_cm_d"):
                record[k] = float(v)

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

        if candidate_profiles is not None:
            conv_metrics = compute_conviction_and_cohort_metrics(
                agent_state=agent_state,
                candidate_profiles=candidate_profiles,
                popular_winner=int(electoral_result["popular_winner"]),
                prev_opinions=prev_opinions,
                prev_convictions=prev_convictions,
            )
            record.update(conv_metrics)
            record["fact_bias_d0"] = float(fact_bias_d0)

        return record


