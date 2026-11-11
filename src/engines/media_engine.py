"""MediaEngine: Concurrent simulation of fact-driven traditional media (TV, Radio, Press),
closed-loop social feeds, and WhatsApp cascades."""

from __future__ import annotations

import numpy as np

from src.core.agent_state import AgentStateEngine
from src.core.multiplex_network import GraphEngine
from src.core.news_entity import NewsEntityPool
from src.kernels.opinion_updates import apply_media_assimilation_jit
from src.kernels.recommendation_algos import (
    broadcast_exposure_kernel_jit,
    compute_closed_loop_recommendations_jit,
    influencer_network_broadcast_and_follower_dynamics_jit,
    social_media_ecosystem_kernel_jit,
    traditional_media_exposure_kernel_jit,
)
from src.kernels.whatsapp_cascades import propagate_whatsapp_cascades_jit


class MediaEngine:
    """Orchestrates competitive media vectors across three distinct ecosystems:
    1. Traditional Mass Media Conglomerates (M vehicle groups, each with TV, Radio, and Press
       producing slanted notices from exogenous facts with modality-specific lag, frequency, and decay).
    2. Multi-Platform Algorithmic Social Media Ecosystem:
       - Platform 0: Instagram & TikTok (Visual Influencer & Short-Form Video, youth-skewed, creator network of impact).
       - Platform 1: YouTube & X / Twitter (Video Commentary & Public Microblogging, adult/SES-skewed).
       - Platform 2: Facebook (Social Graph & Legacy Algorithmic Feed, middle/older-age-skewed).
       Includes dual-valence reactions (Like / Assimilation vs. Dislike / Outrage & Backfire Repulsion),
       factual vs. tendentious influencer behavior, and Bianconi-Barabasi + Sasahara follower gain/loss dynamics.
    3. Closed Private Messaging (WhatsApp / Telegram groups K <= 256 with viral cascade forwarding).
    """

    def __init__(
        self,
        beta: float = 2.0,
        emotional_boost: float = 1.5,
        broadcast_strength: float = 0.05,
        feed_strength: float = 0.07,
        whatsapp_trust: float = 0.85,
        whatsapp_forward_prob: float = 0.25,
        enable_traditional_media: bool = True,
        enable_social_media: bool = True,
        traditional_selectivity_beta: float = 1.2,
        dislike_reaction_strength: float = 0.45,
        backfire_repulsion_strength: float = 0.35,
        influencer_impact_strength: float = 0.04,
    ) -> None:
        self.beta = float(beta)
        self.emotional_boost = float(emotional_boost)
        self.broadcast_strength = float(broadcast_strength)
        self.feed_strength = float(feed_strength)
        self.whatsapp_trust = float(whatsapp_trust)
        self.whatsapp_forward_prob = float(whatsapp_forward_prob)
        self.enable_traditional_media = bool(enable_traditional_media)
        self.enable_social_media = bool(enable_social_media)
        self.traditional_selectivity_beta = float(traditional_selectivity_beta)
        self.dislike_reaction_strength = float(dislike_reaction_strength)
        self.backfire_repulsion_strength = float(backfire_repulsion_strength)
        self.influencer_impact_strength = float(influencer_impact_strength)
        self.group_active_news: np.ndarray | None = None

    def step_all_platforms(
        self,
        agent_state: AgentStateEngine,
        graph_engine: GraphEngine,
        news_pool: NewsEntityPool,
        rng: np.random.Generator,
        step_idx: int = 0,
    ) -> dict[str, float]:
        """Run one concurrent cycle of enabled media platforms and update coevolution counters."""
        n_agents = agent_state.n_agents
        tv_reached = 0
        radio_reached = 0
        press_reached = 0
        outdoor_reached = 0
        rally_reached = 0
        broadcast_reached = 0
        facts_count = 0
        published_notices = 0

        # 1. Fact-Driven Traditional Media Conglomerates (TV, Radio, Press)
        dictated_trad_share = 0.0
        mean_fact_longevity = 1.0
        mean_fact_relevance = 0.65
        if self.enable_traditional_media:
            trad_eco = news_pool.traditional_ecosystem
            if trad_eco is not None and trad_eco.n_vehicles > 0:
                dict_vec = (
                    news_pool.sync_focal_fact_vec
                    if (news_pool.extreme_imposed_enabled and news_pool.fact_dictation_enabled)
                    else None
                )
                dict_framed = (
                    news_pool.sync_focal_fact_framed
                    if (news_pool.extreme_imposed_enabled and news_pool.fact_dictation_enabled)
                    else None
                )
                dict_push = (
                    news_pool.sync_network_interest_push
                    if (news_pool.extreme_imposed_enabled and news_pool.fact_dictation_enabled)
                    else 0.0
                )
                dict_longev = (
                    news_pool.sync_focal_fact_longevity
                    if (news_pool.extreme_imposed_enabled and news_pool.fact_dictation_enabled)
                    else 1.0
                )
                dict_pull = (
                    news_pool.fact_dictation_pull
                    if (news_pool.extreme_imposed_enabled and news_pool.fact_dictation_enabled)
                    else 0.0
                )
                cycle_info = trad_eco.step_facts_and_news_cycle(
                    step_idx=step_idx,
                    rng=rng,
                    dictated_fact_vec=dict_vec,
                    dictated_framed_vec=dict_framed,
                    dictated_interest_push=dict_push,
                    dictated_longevity_mult=dict_longev,
                    dictated_pull=dict_pull,
                )
                facts_count = int(cycle_info["facts_count"])
                published_notices = int(cycle_info["published_notices"])
                dictated_trad_share = float(cycle_info.get("dictated_trad_share", 0.0))
                mean_fact_longevity = float(cycle_info.get("mean_fact_longevity", 1.0))
                mean_fact_relevance = float(cycle_info.get("mean_fact_relevance", 0.65))

                n_mods = max(3, int(np.max(trad_eco.vehicle_modality)) + 1) if trad_eco.n_vehicles > 0 else 3
                u_draws = np.ascontiguousarray(rng.random((n_agents, 2 * n_mods), dtype=np.float64))
                reach_res = traditional_media_exposure_kernel_jit(
                    agent_state.opinions,
                    agent_state.innate_beliefs,
                    agent_state.uncertainty,
                    agent_state.age,
                    agent_state.ses,
                    agent_state.susceptibility,
                    agent_state.epsilon,
                    agent_state.assimilation_bias,
                    agent_state.coupling_matrix,
                    agent_state.exposure_count,
                    trad_eco.vehicle_modality,
                    trad_eco.active_framing,
                    trad_eco.active_intensity,
                    trad_eco.vehicle_credibility,
                    u_draws,
                    self.broadcast_strength,
                    self.traditional_selectivity_beta,
                )
                tv_reached = reach_res[0]
                radio_reached = reach_res[1]
                press_reached = reach_res[2]
                outdoor_reached = reach_res[3] if len(reach_res) > 3 else 0
                rally_reached = reach_res[4] if len(reach_res) > 4 else 0
                broadcast_reached = tv_reached + radio_reached + press_reached + outdoor_reached + rally_reached
            elif news_pool.n_broadcast > 0:
                b_draws = np.ascontiguousarray(rng.random(n_agents, dtype=np.float64))
                broadcast_reached = broadcast_exposure_kernel_jit(
                    agent_state.opinions,
                    agent_state.age,
                    agent_state.ses,
                    agent_state.susceptibility,
                    agent_state.epsilon,
                    agent_state.coupling_matrix,
                    agent_state.exposure_count,
                    news_pool.broadcast_orientations,
                    news_pool.broadcast_credibility,
                    b_draws,
                    self.broadcast_strength,
                )

        # 2 & 3. Open Algorithmic Social Feeds, Influencer Networks, & Closed WhatsApp Cascades
        feed_engaged = 0.0
        likes_count = 0
        dislikes_count = 0
        ig_tiktok_reached = 0
        youtube_x_reached = 0
        facebook_reached = 0
        influencer_direct_impacts = 0
        influencer_unfollows = 0
        followers_gained = 0.0
        followers_lost = 0.0
        cowork_cross_exposures = 0
        total_cascades = 0
        n_groups = graph_engine.n_wa_groups
        n_inf = len(graph_engine.influencer_ids)

        if self.enable_social_media and news_pool.n_items > 0:
            if n_groups > 0 and (
                self.group_active_news is None or len(self.group_active_news) != n_groups
            ):
                self.group_active_news = rng.integers(
                    0, news_pool.n_items, size=n_groups, dtype=np.int32
                )

            eff_reach_boost = (
                news_pool.extreme_imposed_reach_boost
                if news_pool.extreme_imposed_enabled
                else 1.0
            )
            eff_pop_boost = (
                news_pool.extreme_imposed_popularity_boost
                if news_pool.extreme_imposed_enabled
                else 1.0
            )

            f_draws = np.ascontiguousarray(rng.random((n_agents, 5), dtype=np.float64))
            (
                selected_news,
                reaction_types,
                ig_tiktok_reached,
                youtube_x_reached,
                facebook_reached,
                likes_count,
                dislikes_count,
            ) = social_media_ecosystem_kernel_jit(
                agent_state.opinions,
                agent_state.innate_beliefs,
                agent_state.uncertainty,
                agent_state.age,
                agent_state.ses,
                agent_state.susceptibility,
                agent_state.epsilon,
                agent_state.assimilation_bias,
                agent_state.coupling_matrix,
                agent_state.exposure_count,
                agent_state.engagement_rate,
                news_pool.framing,
                news_pool.sensationalism,
                news_pool.veracity,
                news_pool.popularity,
                news_pool.platform_type,
                news_pool.creator_inf_idx,
                graph_engine.influencer_followers,
                news_pool.influencer_is_extreme_imposed,
                f_draws,
                self.beta,
                self.emotional_boost,
                self.feed_strength,
                self.dislike_reaction_strength,
                self.backfire_repulsion_strength,
                eff_reach_boost,
                bool(news_pool.sync_tactics_enabled and news_pool.extreme_imposed_enabled),
            )

            feed_engaged = float(likes_count + dislikes_count)

            # Update per-item popularity, like_count, and dislike_count; aggregate per-influencer step reactions
            inf_step_likes = np.zeros(max(1, n_inf), dtype=np.int64)
            inf_step_dislikes = np.zeros(max(1, n_inf), dtype=np.int64)
            item_pop_mult = np.ones(news_pool.n_items, dtype=np.float64)
            if news_pool.extreme_imposed_enabled and n_inf > 0:
                valid_c_all = news_pool.creator_inf_idx >= 0
                is_ext_item_arr = np.zeros(news_pool.n_items, dtype=bool)
                is_ext_item_arr[valid_c_all] = (
                    news_pool.influencer_is_extreme_imposed[news_pool.creator_inf_idx[valid_c_all]] == 1
                )
                item_pop_mult[is_ext_item_arr] = eff_pop_boost

            liked_items = selected_news[reaction_types == 1]
            if len(liked_items) > 0:
                l_counts = np.bincount(liked_items, minlength=news_pool.n_items)
                news_pool.like_count += l_counts
                news_pool.popularity += l_counts.astype(np.float64) * item_pop_mult
                if n_inf > 0:
                    valid_c = news_pool.creator_inf_idx >= 0
                    inf_step_likes += np.bincount(
                        news_pool.creator_inf_idx[valid_c],
                        weights=l_counts[valid_c],
                        minlength=n_inf,
                    ).astype(np.int64)

            disliked_items = selected_news[reaction_types == -1]
            if len(disliked_items) > 0:
                d_counts = np.bincount(disliked_items, minlength=news_pool.n_items)
                news_pool.dislike_count += d_counts
                # Outrage / dislike comments also boost algorithmic virality (Rathje et al., 2021; Brady et al., 2017)
                news_pool.popularity += 0.75 * d_counts.astype(np.float64) * item_pop_mult
                if n_inf > 0:
                    valid_c = news_pool.creator_inf_idx >= 0
                    inf_step_dislikes += np.bincount(
                        news_pool.creator_inf_idx[valid_c],
                        weights=d_counts[valid_c],
                        minlength=n_inf,
                    ).astype(np.int64)

            # Step 2B: Influencer Network of Impact (Instagram/TikTok follower graph), Co-Working & Follower Gain/Loss
            if n_inf > 0 and len(graph_engine.inf_follower_indptr) == n_inf + 1:
                inf_draws = np.ascontiguousarray(rng.random((n_inf, 64), dtype=np.float64))
                (
                    influencer_direct_impacts,
                    influencer_unfollows,
                    followers_gained,
                    followers_lost,
                    cowork_cross_exposures,
                ) = influencer_network_broadcast_and_follower_dynamics_jit(
                    agent_state.opinions,
                    agent_state.susceptibility,
                    agent_state.epsilon,
                    agent_state.assimilation_bias,
                    agent_state.coupling_matrix,
                    agent_state.exposure_count,
                    graph_engine.influencer_ids,
                    news_pool.influencer_tendentiousness,
                    news_pool.influencer_is_extreme_imposed,
                    graph_engine.inf_follower_indptr,
                    graph_engine.inf_follower_members,
                    graph_engine.influencer_followers,
                    inf_step_likes,
                    inf_step_dislikes,
                    inf_draws,
                    self.influencer_impact_strength,
                    eff_reach_boost,
                    eff_pop_boost,
                    bool(news_pool.cowork_enabled and news_pool.extreme_imposed_enabled),
                    float(news_pool.cowork_prob),
                    float(news_pool.cowork_reach_fraction),
                    bool(news_pool.sync_tactics_enabled and news_pool.extreme_imposed_enabled),
                )

            # Advance synchronized focal fact campaign using step engagements from extreme_imposed creators
            ext_step_eng = 0.0
            if n_inf > 0 and len(news_pool.influencer_is_extreme_imposed) == n_inf:
                ext_m = news_pool.influencer_is_extreme_imposed == 1
                ext_step_eng = float(
                    np.sum(inf_step_likes[ext_m]) + np.sum(inf_step_dislikes[ext_m]) + cowork_cross_exposures
                )
            news_pool.step_synchronized_fact_campaign(
                rng=rng,
                step_ext_engagements=ext_step_eng,
                n_agents=n_agents,
            )

            # Step 3: Closed Private Messaging (WhatsApp / Telegram groups K <= 256)
            if n_groups > 0 and self.group_active_news is not None:
                rand_mat = np.ascontiguousarray(rng.random((n_groups, 256), dtype=np.float64))
                group_forwards, forwarding_members = propagate_whatsapp_cascades_jit(
                    agent_state.opinions,
                    agent_state.susceptibility,
                    agent_state.epsilon,
                    agent_state.assimilation_bias,
                    agent_state.coupling_matrix,
                    agent_state.exposure_count,
                    graph_engine.wa_group_indptr,
                    graph_engine.wa_group_members,
                    self.group_active_news,
                    news_pool.framing,
                    news_pool.sensationalism,
                    self.whatsapp_trust,
                    self.whatsapp_forward_prob,
                    rand_mat,
                )

                total_cascades = int(np.sum(group_forwards))
                for g in range(n_groups):
                    if group_forwards[g] > 0:
                        n_idx = int(self.group_active_news[g])
                        news_pool.viral_cascades[n_idx] += group_forwards[g]
                        news_pool.popularity[n_idx] += float(group_forwards[g]) * 1.5
                        fwd_agent = int(forwarding_members[g])
                        if fwd_agent >= 0:
                            g_start = graph_engine.wa_agent_group_indptr[fwd_agent]
                            g_end = graph_engine.wa_agent_group_indptr[fwd_agent + 1]
                            for ptr in range(g_start, g_end):
                                target_g = graph_engine.wa_agent_groups[ptr]
                                self.group_active_news[target_g] = n_idx
                    elif rng.random() < 0.15:
                        top_item = int(
                            np.argmax(news_pool.popularity * (0.5 + rng.random(news_pool.n_items)))
                        )
                        self.group_active_news[g] = top_item

            if step_idx > 0 and step_idx % 5 == 0 and n_inf > 0:
                inf_opinions = agent_state.opinions[graph_engine.influencer_ids]
                news_pool.refresh_influencer_content(
                    inf_opinions,
                    influencer_ids=graph_engine.influencer_ids,
                    replace_fraction=0.15,
                    rng=rng,
                    step_idx=step_idx,
                )

        # Compute aggregate influencer follower inequality (Gini), tendentious audience share,
        # and Extreme Imposed influencer popularity/follower dominance vs. general influencers
        mean_inf_followers = 0.0
        max_inf_followers = 0.0
        follower_gini = 0.0
        tendentious_follower_share = 0.5
        mean_inf_tendentiousness = 0.0
        extreme_imposed_inf_followers_mean = 0.0
        other_inf_followers_mean = 0.0
        extreme_imposed_follower_ratio = 1.0
        extreme_imposed_follower_share = 0.0
        extreme_imposed_popularity_ratio = 1.0
        extreme_imposed_popularity_share = 0.0
        synchronized_fact_share = 0.0

        if news_pool.item_is_synchronized is not None and news_pool.n_items > 0:
            synchronized_fact_share = float(np.mean(news_pool.item_is_synchronized == 1))

        if n_inf > 0 and len(graph_engine.influencer_followers) == n_inf:
            f_arr = graph_engine.influencer_followers
            mean_inf_followers = float(np.mean(f_arr))
            max_inf_followers = float(np.max(f_arr))
            f_sorted = np.sort(f_arr)
            f_sum = float(np.sum(f_sorted))
            if f_sum > 1e-9 and n_inf > 1:
                idx_arr = np.arange(1, n_inf + 1, dtype=np.float64)
                follower_gini = float(
                    (2.0 * np.sum(idx_arr * f_sorted) / (n_inf * f_sum)) - (n_inf + 1.0) / n_inf
                )
            if len(news_pool.influencer_tendentiousness) == n_inf:
                t_inf = news_pool.influencer_tendentiousness
                mean_inf_tendentiousness = float(np.mean(t_inf))
                if f_sum > 1e-9:
                    tendentious_follower_share = float(np.sum(f_arr * t_inf) / f_sum)

            if len(news_pool.influencer_is_extreme_imposed) == n_inf:
                ext_inf_mask = news_pool.influencer_is_extreme_imposed == 1
                oth_inf_mask = ~ext_inf_mask
                if np.any(ext_inf_mask):
                    extreme_imposed_inf_followers_mean = float(np.mean(f_arr[ext_inf_mask]))
                    if f_sum > 1e-9:
                        extreme_imposed_follower_share = float(np.sum(f_arr[ext_inf_mask]) / f_sum)
                if np.any(oth_inf_mask):
                    other_inf_followers_mean = float(np.mean(f_arr[oth_inf_mask]))
                else:
                    other_inf_followers_mean = extreme_imposed_inf_followers_mean
                if other_inf_followers_mean > 1e-9:
                    extreme_imposed_follower_ratio = float(
                        extreme_imposed_inf_followers_mean / other_inf_followers_mean
                    )

                # Post popularity dominance of extreme_imposed influencers vs. other influencers
                valid_c = news_pool.creator_inf_idx >= 0
                if np.any(valid_c) and news_pool.n_items > 0:
                    item_ext = np.zeros(news_pool.n_items, dtype=bool)
                    item_ext[valid_c] = ext_inf_mask[news_pool.creator_inf_idx[valid_c]]
                    pop_sum = float(np.sum(news_pool.popularity))
                    if pop_sum > 1e-9 and np.any(item_ext):
                        extreme_imposed_popularity_share = float(
                            np.sum(news_pool.popularity[item_ext]) / pop_sum
                        )
                    if np.any(item_ext) and np.any(~item_ext):
                        n_ext_inf = max(1, int(np.sum(ext_inf_mask)))
                        n_oth_inf = max(1, int(np.sum(oth_inf_mask)))
                        mean_p_ext = float(np.sum(news_pool.popularity[item_ext])) / float(n_ext_inf)
                        mean_p_oth = float(np.sum(news_pool.popularity[~item_ext])) / float(n_oth_inf)
                        if mean_p_oth > 1e-9:
                            extreme_imposed_popularity_ratio = float(mean_p_ext / mean_p_oth)

        return {
            "broadcast_reached": float(broadcast_reached),
            "tv_reached": float(tv_reached),
            "radio_reached": float(radio_reached),
            "press_reached": float(press_reached),
            "outdoor_reached": float(outdoor_reached),
            "rally_reached": float(rally_reached),
            "facts_count": float(facts_count),
            "published_notices": float(published_notices),
            "dictated_trad_share": float(dictated_trad_share),
            "mean_fact_longevity": float(
                news_pool.sync_focal_fact_longevity
                if (news_pool.extreme_imposed_enabled and news_pool.sync_tactics_enabled)
                else mean_fact_longevity
            ),
            "mean_fact_relevance": float(
                news_pool.sync_focal_fact_relevance
                if (news_pool.extreme_imposed_enabled and news_pool.sync_tactics_enabled)
                else mean_fact_relevance
            ),
            "sync_network_interest_push": float(news_pool.sync_network_interest_push),
            "synchronized_fact_share": float(synchronized_fact_share),
            "feed_engaged": float(feed_engaged),
            "likes_count": float(likes_count),
            "dislikes_count": float(dislikes_count),
            "dislike_share": float(dislikes_count / max(1, likes_count + dislikes_count)),
            "ig_tiktok_reached": float(ig_tiktok_reached),
            "youtube_x_reached": float(youtube_x_reached),
            "facebook_reached": float(facebook_reached),
            "influencer_direct_impacts": float(influencer_direct_impacts),
            "cowork_cross_exposures": float(cowork_cross_exposures),
            "influencer_unfollows": float(influencer_unfollows),
            "followers_gained": float(followers_gained),
            "followers_lost": float(followers_lost),
            "net_follower_change": float(followers_gained - followers_lost),
            "mean_inf_followers": mean_inf_followers,
            "max_inf_followers": max_inf_followers,
            "influencer_follower_gini": follower_gini,
            "mean_inf_tendentiousness": mean_inf_tendentiousness,
            "tendentious_follower_share": tendentious_follower_share,
            "extreme_imposed_inf_followers_mean": extreme_imposed_inf_followers_mean,
            "other_inf_followers_mean": other_inf_followers_mean,
            "extreme_imposed_follower_ratio": extreme_imposed_follower_ratio,
            "extreme_imposed_follower_share": extreme_imposed_follower_share,
            "extreme_imposed_popularity_ratio": extreme_imposed_popularity_ratio,
            "extreme_imposed_popularity_share": extreme_imposed_popularity_share,
            "whatsapp_cascades": float(total_cascades),
        }

