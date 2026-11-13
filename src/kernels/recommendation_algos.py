"""Numba JIT kernels for closed-loop algorithmic social media feeds and broadcast reach.

Author: César Rennó-Costa (Brain Institute, UFRN, Natal, RN, Brazil)
Paper: "Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections"
Target: Nature Human Behaviour (2026)

Governing Equations:
- Algorithmic feed homophily scoring: Eq. S41
- Dual-valence like assimilation & dislike outrage repulsion: Eqs. S43, S45
- Algorithmic relevance metric on controversy diagonal: R_m = Likes_m + \\omega Dislikes_m (Eq. S46)
"""

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
    innate_beliefs: np.ndarray,
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
        return 0, 0, 0, 0, 0

    max_mod = 0
    for v in range(n_vehicles):
        if vehicle_modality[v] > max_mod:
            max_mod = vehicle_modality[v]
    n_mods = max_mod + 1

    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))
    tv_reached = 0
    radio_reached = 0
    press_reached = 0
    outdoor_reached = 0
    rally_reached = 0
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

        # 4. Outdoor Marketing (Billboards / OOH): Ambient urban transit exposure
        p_outdoor = 0.35 + 0.25 * (1.0 - math.fabs(age_norm - 0.45) / 0.45) + 0.05 * ses_i
        if p_outdoor < 0.10:
            p_outdoor = 0.10
        elif p_outdoor > 0.70:
            p_outdoor = 0.70

        # 5. Public Rallies & Municipal Gatherings: Episodic civic mobilization
        p_rally = 0.04 + 0.14 * (1.0 - age_norm) + 0.08 * (1.0 - ses_i)
        if p_rally < 0.02:
            p_rally = 0.02
        elif p_rally > 0.30:
            p_rally = 0.30

        total_shift = np.zeros(n_dims, dtype=np.float64)
        agent_exposed = 0

        # Evaluate modalities present in ecosystem
        for mod_id in range(n_mods):
            if mod_id == 0:
                reach_p = p_tv
                mod_selectivity = selectivity_beta
            elif mod_id == 1:
                reach_p = p_radio
                mod_selectivity = selectivity_beta
            elif mod_id == 2:
                reach_p = p_press
                mod_selectivity = selectivity_beta
            elif mod_id == 3:
                reach_p = p_outdoor
                mod_selectivity = 0.15
            else:
                reach_p = p_rally
                mod_selectivity = selectivity_beta * 1.8

            # uniform_draws has shape (n_agents, 2 * n_mods): cols 0..n_mods-1 for access, cols n_mods..2*n_mods-1 for group selection
            if uniform_draws[i, mod_id] < reach_p:
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
                    w_v = inten * math.pow(dist_v + delta_reg, -mod_selectivity)
                    total_w += w_v

                if total_w > 0.0:
                    target = uniform_draws[i, mod_id + n_mods] * total_w
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
                        w_v = inten * math.pow(dist_v + delta_reg, -mod_selectivity)
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
                    eff_eps = epsilon[i] * (1.0 + 1.10 * gamma_v)

                    if dist_chosen <= eff_eps:
                        overlap = 1.0 - (dist_chosen / (eff_eps + 1e-12))
                        salience_mod = 1.0 - math.exp(-active_intensity[chosen_v])
                        conf_boost = math.exp(
                            0.25 * assimilation_bias[i] * (0.5 - dist_chosen)
                        )
                        eff_susc = 0.35 + 0.65 * susceptibility[i]
                        pull = (
                            broadcast_strength
                            * gamma_v
                            * eff_susc
                            * (0.35 + 0.65 * overlap)
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
                        elif mod_id == 2:
                            press_reached += 1
                        elif mod_id == 3:
                            outdoor_reached += 1
                        else:
                            rally_reached += 1

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

                # Agenda-setting & cultivation (McCombs & Shaw, 1972; DellaVigna & Kaplan, 2007):
                # Repeated traditional media exposure gradually internalizes into reference beliefs s_i
                s_val = innate_beliefs[i, d1] + 0.35 * coupled
                if s_val > 1.0:
                    s_val = 1.0
                elif s_val < -1.0:
                    s_val = -1.0
                innate_beliefs[i, d1] = s_val

            # Repeated institutional media exposure slightly reduces attitudinal uncertainty
            u_new = uncertainty[i] * (1.0 - 0.008 * float(agent_exposed))
            if u_new < 0.05:
                u_new = 0.05
            uncertainty[i] = u_new

    return tv_reached, radio_reached, press_reached, outdoor_reached, rally_reached


@njit(parallel=True, fastmath=True)
def social_media_ecosystem_kernel_jit(
    opinions: np.ndarray,
    innate_beliefs: np.ndarray,
    uncertainty: np.ndarray,
    age: np.ndarray,
    ses: np.ndarray,
    susceptibility: np.ndarray,
    epsilon: np.ndarray,
    assimilation_bias: np.ndarray,
    coupling_matrix: np.ndarray,
    exposure_count: np.ndarray,
    engagement_rate: np.ndarray,
    news_framing: np.ndarray,
    news_sensationalism: np.ndarray,
    news_veracity: np.ndarray,
    news_popularity: np.ndarray,
    news_platform: np.ndarray,
    news_creator_inf_idx: np.ndarray,
    influencer_followers: np.ndarray,
    influencer_is_extreme_imposed: np.ndarray,
    uniform_draws: np.ndarray,
    beta: float,
    emotional_boost: float,
    feed_strength: float,
    dislike_reaction_strength: float,
    backfire_repulsion_strength: float,
    extreme_imposed_reach_boost: float,
    sync_tactics_enabled: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int]:
    """Simulate multi-platform Brazilian social media usage (Instagram/TikTok, YouTube/X, Facebook),
    influencer reach weighting, and dual-valence reactions (Like / Assimilation vs. Dislike / Backfire).

    Empirical Brazilian Usage Calibration (DataReportal Digital Brazil 2024; CETIC.br TIC Domicilios 2023;
    Reuters Institute Digital News Report Brazil 2023/2024):
    1. Instagram & TikTok (Platform 0 - Visual Influencer & Short-Form Video):
       - 81.4% Instagram & 63.3% TikTok penetration among Brazilian internet users; 50% use Instagram for news.
       - Steep youth & young-adult concentration (16-39 yrs):
         P_IG_TT(age, ses) = clip(0.84 - 0.62 * ((age - 16)/74)^0.85 + 0.10 * ses, 0.08, 0.94)
    2. YouTube & X / Twitter (Platform 1 - Video Channels, Political Commentary & Public Microblogging):
       - YouTube reaches 86.2% (41% for news), X reaches 22.1% with higher SES/urban political engagement:
         P_YT_X(age, ses) = clip(0.56 - 0.28 * ((age - 16)/74) + 0.24 * (ses^1.2), 0.12, 0.88)
    3. Facebook (Platform 2 - Social Graph & Legacy Algorithmic Feed):
       - 72.5% reach (39% for news), shifting strongly toward mature and older Brazilian cohorts (35-75 yrs):
         P_FB(age, ses) = clip(0.22 + 0.44 * sin(pi * ((age - 16)/74)^0.75) - 0.08 * ses, 0.10, 0.78)

    Dual-Valence Affective Reaction (Like vs. Dislike / Moral Outrage & Backfire):
    - References: Bail et al. (2018) PNAS; Rathje, Van Bavel, & van der Linden (2021) PNAS;
      Brady et al. (2017) PNAS; Vosoughi, Roy, & Aral (2018) Science; Jager & Amblard (2005) CMS.
    - Positive Reaction (Like / Share, reaction_type == 1):
      Occurs when content falls within bounded confidence dist <= eps_like. Assimilates opinion toward y_j.
    - Negative Reaction (Dislike / Moral Outrage / Hate-Engagement, reaction_type == -1):
      Occurs when content is strongly discordant (dist >= tau_dislike) and emotionally/morally charged.
      Generates out-group animosity reaction (which still feeds algorithmic virality) and induces repulsive
      differentiation (backfire shift away from y_j).
    """
    n_agents, n_dims = opinions.shape
    n_items = news_framing.shape[0]
    selected_items = np.full(n_agents, -1, dtype=np.int32)
    reaction_types = np.zeros(n_agents, dtype=np.int8)  # +1 = Like, -1 = Dislike, 0 = None

    if n_items == 0:
        return selected_items, reaction_types, 0, 0, 0, 0, 0

    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))
    delta = 0.08
    ig_tt_reached = 0
    yt_x_reached = 0
    fb_reached = 0
    likes_total = 0
    dislikes_total = 0

    for i in prange(n_agents):
        age_norm = (age[i] - 16.0) / 74.0
        if age_norm < 0.0:
            age_norm = 0.0
        elif age_norm > 1.0:
            age_norm = 1.0
        ses_i = ses[i]

        # Platform usage probabilities calibrated to Brazilian demographic surveys
        p_ig_tt = 0.84 - 0.62 * math.pow(age_norm + 1e-6, 0.85) + 0.10 * ses_i
        if p_ig_tt < 0.08:
            p_ig_tt = 0.08
        elif p_ig_tt > 0.94:
            p_ig_tt = 0.94

        p_yt_x = 0.56 - 0.28 * age_norm + 0.24 * math.pow(ses_i + 1e-6, 1.2)
        if p_yt_x < 0.12:
            p_yt_x = 0.12
        elif p_yt_x > 0.88:
            p_yt_x = 0.88

        p_fb = 0.22 + 0.44 * math.sin(3.141592653589793 * math.pow(age_norm + 1e-6, 0.75)) - 0.08 * ses_i
        if p_fb < 0.10:
            p_fb = 0.10
        elif p_fb > 0.78:
            p_fb = 0.78

        active_ig = uniform_draws[i, 0] < p_ig_tt
        active_yt = uniform_draws[i, 1] < p_yt_x
        active_fb = uniform_draws[i, 2] < p_fb

        if active_ig:
            ig_tt_reached += 1
        if active_yt:
            yt_x_reached += 1
        if active_fb:
            fb_reached += 1

        if not (active_ig or active_yt or active_fb):
            continue

        # Select content item across active platforms via algorithmic curation + influencer follower reach
        total_w = 0.0
        weights = np.empty(n_items, dtype=np.float64)
        for j in range(n_items):
            plat_j = news_platform[j]
            if plat_j == 0 and not active_ig:
                weights[j] = 0.0
                continue
            elif plat_j == 1 and not active_yt:
                weights[j] = 0.0
                continue
            elif plat_j == 2 and not active_fb:
                weights[j] = 0.0
                continue

            dist_sq = 0.0
            for d in range(n_dims):
                diff = opinions[i, d] - news_framing[j, d]
                dist_sq += diff * diff
            dist = math.sqrt(dist_sq) * inv_sqrt_d

            # Algorithmic recommendation combines homophilous match and high-conflict outrage salience
            homophily_w = math.pow(dist + delta, -beta)
            outrage_salience = (
                dislike_reaction_strength
                * 0.35
                * news_sensationalism[j]
                * math.pow(dist, 1.4)
            )
            inf_idx = news_creator_inf_idx[j]
            reach_mult = 1.0
            if inf_idx >= 0 and inf_idx < len(influencer_followers):
                inf_reach = math.log1p(influencer_followers[inf_idx])
                if influencer_is_extreme_imposed[inf_idx] == 1 and extreme_imposed_reach_boost > 1.0:
                    reach_mult = extreme_imposed_reach_boost
                    # Artificial algorithmic push flattens attitudinal distance penalty so content reaches broad users
                    homophily_w = math.pow(dist * 0.55 + delta, -beta)
            else:
                inf_reach = 2.0

            amp = (
                (1.0 + emotional_boost * news_sensationalism[j])
                * math.log1p(news_popularity[j])
                * (0.5 + 0.5 * inf_reach)
                * reach_mult
            )
            w_j = (homophily_w + outrage_salience) * amp
            weights[j] = w_j
            total_w += w_j

        if total_w <= 0.0:
            continue

        target = uniform_draws[i, 3] * total_w
        cum = 0.0
        chosen = n_items - 1
        for j in range(n_items):
            cum += weights[j]
            if cum >= target:
                chosen = j
                break
        selected_items[i] = chosen

        # Evaluate agent's reaction to chosen item: Like (concordant) vs. Dislike (strongly discordant)
        chosen_inf = news_creator_inf_idx[chosen]
        is_ext_item = (
            chosen_inf >= 0
            and chosen_inf < len(influencer_is_extreme_imposed)
            and influencer_is_extreme_imposed[chosen_inf] == 1
            and extreme_imposed_reach_boost > 1.0
        )
        dist_sq = 0.0
        dot_in = 0.0
        norm_i_sq = 0.0
        norm_n_sq = 0.0
        for d in range(n_dims):
            diff = news_framing[chosen, d] - opinions[i, d]
            dist_sq += diff * diff
            dot_in += opinions[i, d] * news_framing[chosen, d]
            norm_i_sq += opinions[i, d] * opinions[i, d]
            norm_n_sq += news_framing[chosen, d] * news_framing[chosen, d]

        dist = math.sqrt(dist_sq) * inv_sqrt_d
        e_n = news_sensationalism[chosen]
        v_n = news_veracity[chosen]
        b_i = assimilation_bias[i]
        #Artificial popularity & social proof (Muchnik et al., 2013, Science) + coordinated synchronization
        # (Pacheco et al., 2021; Pennycook et al., 2018) widen acceptance bound for extreme_imposed posts
        sync_bonus = 1.25 if (is_ext_item and sync_tactics_enabled) else 1.0
        social_proof_widening = (
            (1.0 + 0.35 * math.log1p(extreme_imposed_reach_boost)) * sync_bonus
            if is_ext_item
            else 1.0
        )
        eps_like = epsilon[i] * (1.0 + 0.30 * e_n) * social_proof_widening
        # Dislike / outrage threshold: triggered when distance exceeds repulsion bound
        tau_dislike = max(
            eps_like + (0.28 if is_ext_item else 0.08),
            1.45 * epsilon[i] * (1.0 - 0.18 * e_n),
        )

        if dist <= eps_like:
            # 1. POSITIVE REACTION (LIKE / CONCORDANT ENGAGEMENT)
            exposure_count[i] += 1
            reaction_types[i] = 1
            likes_total += 1

            norm_i = math.sqrt(norm_i_sq) + 1e-12
            norm_n = math.sqrt(norm_n_sq) + 1e-12
            cos_sim = dot_in / (norm_i * norm_n)

            assim_weight = (
                feed_strength
                * susceptibility[i]
                * (1.0 + 0.45 * e_n)
                * math.exp(0.30 * b_i * max(0.0, cos_sim))
                * (1.35 if is_ext_item else 1.0)
                * (1.22 if (is_ext_item and sync_tactics_enabled) else 1.0)
            )

            raw_delta = np.empty(n_dims, dtype=np.float64)
            for d in range(n_dims):
                raw_delta[d] = assim_weight * (news_framing[chosen, d] - opinions[i, d])

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

                # Gradual cultivation of reference beliefs under repeated digital confirmation
                # Synchronized multi-source repetition accelerates long-term belief internalization
                cult_rate = (0.26 if sync_tactics_enabled else 0.16) if is_ext_item else 0.08
                s_val = innate_beliefs[i, d1] + cult_rate * coupled
                if s_val > 1.0:
                    s_val = 1.0
                elif s_val < -1.0:
                    s_val = -1.0
                innate_beliefs[i, d1] = s_val

            u_new = uncertainty[i] * (1.0 - 0.025 * (1.0 + e_n - 0.4 * v_n))
            if u_new < 0.02:
                u_new = 0.02
            uncertainty[i] = u_new

            eng_new = 0.92 * engagement_rate[i] + 0.08 * min(1.0, 0.5 + 0.5 * e_n)
            engagement_rate[i] = min(1.0, max(0.01, eng_new))

        elif dist >= tau_dislike and dislike_reaction_strength > 0.0:
            # 2. NEGATIVE REACTION (DISLIKE / OUT-GROUP ANIMOSITY & BACKFIRE REPULSION)
            # Probability of reacting due to strong dislike increases with discordance, sensationalism, and bias
            excess_dist = min(1.5, dist - tau_dislike)
            p_dislike = min(
                0.85,
                dislike_reaction_strength
                * (0.28 + 0.52 * e_n)
                * (0.45 + 0.35 * min(2.0, b_i))
                * (0.40 + 0.60 * excess_dist),
            )
            if uniform_draws[i, 4] < p_dislike:
                exposure_count[i] += 1
                reaction_types[i] = -1
                dislikes_total += 1

                # Repulsive backfire shift away from strongly disliked content (Bail et al., 2018; Jager & Amblard, 2005)
                # Social proof dampens repulsive backfire when content has massive artificial popularity
                backfire_damp = 0.25 if is_ext_item else 1.0
                repel_weight = (
                    feed_strength
                    * backfire_repulsion_strength
                    * backfire_damp
                    * susceptibility[i]
                    * (0.35 + 0.65 * e_n)
                    * min(1.8, 0.5 + 0.35 * b_i)
                )
                raw_repel = np.empty(n_dims, dtype=np.float64)
                for d in range(n_dims):
                    # Shift in the opposite direction: (opinions[i, d] - news_framing[chosen, d])
                    raw_repel[d] = repel_weight * (opinions[i, d] - news_framing[chosen, d])

                for d1 in range(n_dims):
                    coupled = 0.0
                    for d2 in range(n_dims):
                        coupled += coupling_matrix[d1, d2] * raw_repel[d2]
                    val = opinions[i, d1] + coupled
                    if val > 1.0:
                        val = 1.0
                    elif val < -1.0:
                        val = -1.0
                    opinions[i, d1] = val

                # Outrage reaction also entrenches certainty and boosts platform engagement
                u_new = uncertainty[i] * (1.0 - 0.02 * (1.0 + e_n))
                if u_new < 0.02:
                    u_new = 0.02
                uncertainty[i] = u_new

                eng_new = 0.92 * engagement_rate[i] + 0.08 * min(1.0, 0.6 + 0.4 * e_n)
                engagement_rate[i] = min(1.0, max(0.01, eng_new))

    return (
        selected_items,
        reaction_types,
        ig_tt_reached,
        yt_x_reached,
        fb_reached,
        likes_total,
        dislikes_total,
    )


@njit(fastmath=True)
def influencer_network_broadcast_and_follower_dynamics_jit(
    opinions: np.ndarray,
    susceptibility: np.ndarray,
    epsilon: np.ndarray,
    assimilation_bias: np.ndarray,
    coupling_matrix: np.ndarray,
    exposure_count: np.ndarray,
    influencer_ids: np.ndarray,
    influencer_tendentiousness: np.ndarray,
    influencer_is_extreme_imposed: np.ndarray,
    inf_follower_indptr: np.ndarray,
    inf_follower_members: np.ndarray,
    influencer_followers: np.ndarray,
    inf_step_likes: np.ndarray,
    inf_step_dislikes: np.ndarray,
    uniform_draws_inf: np.ndarray,
    influencer_impact_strength: float,
    extreme_imposed_reach_boost: float,
    extreme_imposed_popularity_boost: float,
    cowork_enabled: bool = False,
    cowork_prob: float = 0.65,
    cowork_reach_fraction: float = 0.50,
    sync_tactics_enabled: bool = False,
) -> tuple[int, int, float, float, int]:
    """Simulate direct influencer-to-follower network impact on Instagram/TikTok, influencer co-working
    (accessing collaborator networks & cross-recruiting followers), and literature-grounded
    influencer follower gain/loss dynamics.

    Literature Grounding:
    1. Preferential Attachment with Fitness / Emotional Virality (Bianconi & Barabasi, 2001, Europhys. Lett.;
       Barabasi & Albert, 1999, Science; Vosoughi, Roy, & Aral, 2018, Science):
       New follower acquisition scales with current audience size (preferential attachment) multiplied by
       viral engagement (likes + outrage-driven visibility) and influencer tendentiousness fitness.
    2. Influencer Co-Working & Cross-Network Funneling (Lewis, 2018, "Alternative Influence", Data & Society;
       Ribeiro et al., 2020, ACM FAT*; Centola, 2010, Science):
       Collaborating influencers cross-promote and access each other's follower networks with parasocial
       endorsement transfer, expanding their network reach and recruiting followers from partner audiences.
    3. Homophilous Tie Decay / Cognitive Dissonance Unfollowing (Sasahara et al., 2021, Nature Human Behaviour,
       "Social influence and unfollowing accelerate the emergence of echo chambers"; Myers & Leskovec, 2014, WWW,
       "The bursty dynamics of the twitter information network"):
       Followers unfollow influencers when attitudinal dissonance ||x_i - x_inf|| exceeds tolerance or when
       repeatedly exposed to disliked/alienating framing, accelerating ideological sorting into echo chambers.
    """
    n_inf = len(influencer_ids)
    n_agents, n_dims = opinions.shape
    if n_inf == 0 or len(inf_follower_indptr) < n_inf + 1:
        return 0, 0, 0.0, 0.0, 0

    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))
    direct_follower_impacts = 0
    unfollow_events = 0
    total_gained = 0.0
    total_lost = 0.0
    cowork_cross_exposures = 0

    # 1. Influencer Network of Impact: direct push to connected followers in CSR graph
    for k in range(n_inf):
        inf_agent_id = influencer_ids[k]
        tau_k = influencer_tendentiousness[k]
        inf_op = opinions[inf_agent_id]
        # Tendentious influencers project a sharper, more extreme persona to their follower network
        proj_mult = 0.90 + 0.35 * tau_k

        start = inf_follower_indptr[k]
        end = inf_follower_indptr[k + 1]
        local_unfollows = 0
        local_organic_likes = 0
        cowork_likes_k = 0
        cowork_partner_followers = 0.0

        for ptr in range(start, end):
            follower = inf_follower_members[ptr]
            if follower < 0 or follower >= n_agents:
                continue

            dist_sq = 0.0
            for d in range(n_dims):
                target_d = inf_op[d] * proj_mult
                if target_d > 1.0:
                    target_d = 1.0
                elif target_d < -1.0:
                    target_d = -1.0
                diff = target_d - opinions[follower, d]
                dist_sq += diff * diff
            dist = math.sqrt(dist_sq) * inv_sqrt_d

            # Parasocial trust in followed Instagram/TikTok creator widens confidence bound
            eff_eps = epsilon[follower] * (1.15 + 0.25 * tau_k)
            if dist <= eff_eps:
                concordance = 1.0 - (dist / (eff_eps + 1e-9))
                pull = (
                    influencer_impact_strength
                    * susceptibility[follower]
                    * (0.45 + 0.55 * concordance)
                    * (0.85 + 0.45 * tau_k)
                )
                for d in range(n_dims):
                    target_d = inf_op[d] * proj_mult
                    if target_d > 1.0:
                        target_d = 1.0
                    elif target_d < -1.0:
                        target_d = -1.0
                    val = opinions[follower, d] + pull * (target_d - opinions[follower, d])
                    if val > 1.0:
                        val = 1.0
                    elif val < -1.0:
                        val = -1.0
                    opinions[follower, d] = val
                exposure_count[follower] += 1
                direct_follower_impacts += 1
                local_organic_likes += 1
            elif dist > eff_eps * 1.35:
                # Sasahara et al. (2021) / Myers & Leskovec (2014): Dissonance-induced unfollowing
                draw = uniform_draws_inf[k, (ptr - start) % uniform_draws_inf.shape[1]]
                unfollow_prob = 0.04 * (dist - eff_eps) * (0.70 + 0.60 * tau_k)
                if draw < unfollow_prob:
                    local_unfollows += 1
                    unfollow_events += 1

        # 1B. Influencer Co-Working (Lewis, 2018; Ribeiro et al., 2020):
        # Extreme Imposed influencers collaborate with other influencers, accessing their follower networks
        # with co-host trust endorsement and recruiting followers across networks.
        is_ext_k = (
            k < len(influencer_is_extreme_imposed)
            and influencer_is_extreme_imposed[k] == 1
            and extreme_imposed_reach_boost > 1.0
        )
        if cowork_enabled and is_ext_k and n_inf > 1:
            cw_draw = uniform_draws_inf[k, 0]
            if cw_draw < cowork_prob:
                # Choose collaborator partner_k (either another extreme_imposed creator or a bridged creator)
                partner_seed = int(uniform_draws_inf[k, 1] * float(n_inf)) % n_inf
                partner_k = partner_seed if partner_seed != k else (k + 1) % n_inf
                p_start = inf_follower_indptr[partner_k]
                p_end = inf_follower_indptr[partner_k + 1]
                p_deg = p_end - p_start
                cowork_partner_followers = max(10.0, influencer_followers[partner_k])
                if p_deg > 0:
                    stride = max(1, int(round(1.0 / max(0.10, min(1.0, cowork_reach_fraction)))))
                    offset = int(uniform_draws_inf[k, 2] * float(stride)) % stride
                    for p_ptr in range(p_start + offset, p_end, stride):
                        co_follower = inf_follower_members[p_ptr]
                        if co_follower < 0 or co_follower >= n_agents:
                            continue
                        dist_sq_c = 0.0
                        for d in range(n_dims):
                            target_d = inf_op[d] * proj_mult
                            if target_d > 1.0:
                                target_d = 1.0
                            elif target_d < -1.0:
                                target_d = -1.0
                            diff_c = target_d - opinions[co_follower, d]
                            dist_sq_c += diff_c * diff_c
                        dist_c = math.sqrt(dist_sq_c) * inv_sqrt_d
                        # Co-host parasocial endorsement + synchronized campaign widens confidence threshold
                        co_eps = epsilon[co_follower] * (1.35 + 0.35 * tau_k) * (1.20 if sync_tactics_enabled else 1.0)
                        if dist_c <= co_eps:
                            concord_c = 1.0 - (dist_c / (co_eps + 1e-9))
                            co_pull = (
                                influencer_impact_strength
                                * susceptibility[co_follower]
                                * (0.55 + 0.45 * concord_c)
                                * (1.0 + 0.40 * tau_k)
                                * (1.25 if sync_tactics_enabled else 1.10)
                            )
                            for d in range(n_dims):
                                target_d = inf_op[d] * proj_mult
                                if target_d > 1.0:
                                    target_d = 1.0
                                elif target_d < -1.0:
                                    target_d = -1.0
                                val_c = opinions[co_follower, d] + co_pull * (target_d - opinions[co_follower, d])
                                if val_c > 1.0:
                                    val_c = 1.0
                                elif val_c < -1.0:
                                    val_c = -1.0
                                opinions[co_follower, d] = val_c
                            exposure_count[co_follower] += 1
                            direct_follower_impacts += 1
                            cowork_cross_exposures += 1
                            cowork_likes_k += 1

        # 2. Follower Gain / Loss Dynamics (Bianconi-Barabasi fitness + Co-Work Cross-Pollination + Sasahara unfollowing)
        cur_f = max(1.0, influencer_followers[k])
        base_niche_f = max(25.0, float(end - start) * 0.45)
        pop_mult = extreme_imposed_popularity_boost if is_ext_k else 1.0
        reach_mult = extreme_imposed_reach_boost if is_ext_k else 1.0
        likes_k = (
            float(inf_step_likes[k])
            + 0.35 * float(local_organic_likes)
            + 0.65 * float(cowork_likes_k)
        ) * pop_mult
        dislikes_k = float(inf_step_dislikes[k]) * (0.65 * pop_mult if is_ext_k else 1.0)

        # Algorithmic discovery gain + Co-Work network expansion:
        # Co-working expands addressable audience carrying capacity by bridging multiple creator networks
        cap_mult = (1.35 if (cowork_enabled and is_ext_k) else 0.85) if is_ext_k else 0.35
        carrying_cap = float(n_agents) * cap_mult
        saturation = max(0.02, 1.0 - (cur_f / max(100.0, carrying_cap)))
        fitness_k = (0.65 + 0.70 * tau_k) * (0.75 + 0.25 * reach_mult) * (1.20 if (sync_tactics_enabled and is_ext_k) else 1.0)
        viral_signal = likes_k + 0.55 * dislikes_k
        cowork_recruit = (
            0.045 * math.log1p(cowork_partner_followers) * (1.0 + 0.25 * float(cowork_likes_k))
            if (cowork_enabled and is_ext_k and cowork_partner_followers > 0.0)
            else 0.0
        )
        gained_k = saturation * (
            0.032 * math.log1p(cur_f) * fitness_k * math.log1p(viral_signal)
            + 0.055 * likes_k
            + cowork_recruit
        )

        # Follower loss: dissonance unfollows + backlash from high dislike-to-like ratio + natural churn
        backlash_loss = 0.035 * float(inf_step_dislikes[k]) * (1.15 - 0.50 * tau_k) * (0.45 if is_ext_k else 1.0)
        # Dissonance unfollowing scales with excess followers above loyal homophilous core
        active_excess_ratio = max(0.05, min(1.5, (cur_f - base_niche_f) / max(50.0, base_niche_f)))
        churn_loss = 0.0018 * max(0.0, cur_f - base_niche_f) + float(local_unfollows) * 0.22 * active_excess_ratio
        lost_k = backlash_loss + churn_loss

        next_f = cur_f + gained_k - lost_k
        if next_f < base_niche_f * 0.50:
            next_f = base_niche_f * 0.50
        influencer_followers[k] = next_f
        total_gained += gained_k
        total_lost += lost_k

    return direct_follower_impacts, unfollow_events, total_gained, total_lost, cowork_cross_exposures


