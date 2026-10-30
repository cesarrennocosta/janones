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
    traditional_media_exposure_kernel_jit,
)
from src.kernels.whatsapp_cascades import propagate_whatsapp_cascades_jit


class MediaEngine:
    """Orchestrates competitive media vectors across three distinct ecosystems:
    1. Traditional Mass Media Conglomerates (M vehicle groups, each with TV, Radio, and Press
       producing slanted notices from exogenous facts with modality-specific lag, frequency, and decay).
    2. Algorithmic Social Media Feeds (Open scale-free with beta bias & emotional amplification).
    3. Closed Private Messaging (WhatsApp groups K <= 256 with viral cascade forwarding).
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
        broadcast_reached = 0
        facts_count = 0
        published_notices = 0

        # 1. Fact-Driven Traditional Media Conglomerates (TV, Radio, Press)
        if self.enable_traditional_media:
            trad_eco = news_pool.traditional_ecosystem
            if trad_eco is not None and trad_eco.n_vehicles > 0:
                cycle_info = trad_eco.step_facts_and_news_cycle(step_idx=step_idx, rng=rng)
                facts_count = cycle_info["facts_count"]
                published_notices = cycle_info["published_notices"]

                u_draws = np.ascontiguousarray(rng.random((n_agents, 6), dtype=np.float64))
                tv_reached, radio_reached, press_reached = traditional_media_exposure_kernel_jit(
                    agent_state.opinions,
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
                broadcast_reached = tv_reached + radio_reached + press_reached
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

        # 2 & 3. Open Algorithmic Social Feeds & Closed WhatsApp Cascades (only if enable_social_media=True)
        feed_engaged = 0.0
        total_cascades = 0
        n_groups = graph_engine.n_wa_groups

        if self.enable_social_media and news_pool.n_items > 0:
            if n_groups > 0 and (
                self.group_active_news is None or len(self.group_active_news) != n_groups
            ):
                self.group_active_news = rng.integers(
                    0, news_pool.n_items, size=n_groups, dtype=np.int32
                )

            f_draws = np.ascontiguousarray(rng.random(n_agents, dtype=np.float64))
            selected_news = compute_closed_loop_recommendations_jit(
                agent_state.opinions,
                news_pool.framing,
                news_pool.sensationalism,
                news_pool.popularity,
                self.beta,
                self.emotional_boost,
                f_draws,
            )

            engaged_flags = apply_media_assimilation_jit(
                agent_state.opinions,
                agent_state.uncertainty,
                agent_state.susceptibility,
                agent_state.epsilon,
                agent_state.assimilation_bias,
                agent_state.coupling_matrix,
                agent_state.exposure_count,
                agent_state.engagement_rate,
                selected_news,
                news_pool.framing,
                news_pool.sensationalism,
                news_pool.veracity,
                self.feed_strength,
            )
            feed_engaged = float(np.sum(engaged_flags))

            engaged_items = selected_news[engaged_flags == 1]
            if len(engaged_items) > 0:
                counts = np.bincount(engaged_items, minlength=news_pool.n_items)
                news_pool.popularity += counts.astype(np.float64)

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

            if step_idx > 0 and step_idx % 5 == 0 and len(graph_engine.influencer_ids) > 0:
                inf_opinions = agent_state.opinions[graph_engine.influencer_ids]
                news_pool.refresh_influencer_content(inf_opinions, replace_fraction=0.15, rng=rng)

        return {
            "broadcast_reached": float(broadcast_reached),
            "tv_reached": float(tv_reached),
            "radio_reached": float(radio_reached),
            "press_reached": float(press_reached),
            "facts_count": float(facts_count),
            "published_notices": float(published_notices),
            "feed_engaged": float(feed_engaged),
            "whatsapp_cascades": float(total_cascades),
        }

