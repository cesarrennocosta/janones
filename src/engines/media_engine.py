"""MediaEngine: Concurrent simulation of broadcast channels, closed-loop social feeds, and WhatsApp cascades."""

from __future__ import annotations

import numpy as np

from src.core.agent_state import AgentStateEngine
from src.core.multiplex_network import GraphEngine
from src.core.news_entity import NewsEntityPool
from src.kernels.opinion_updates import apply_media_assimilation_jit
from src.kernels.recommendation_algos import (
    broadcast_exposure_kernel_jit,
    compute_closed_loop_recommendations_jit,
)
from src.kernels.whatsapp_cascades import propagate_whatsapp_cascades_jit


class MediaEngine:
    """Orchestrates competitive media vectors across three distinct platforms:
    1. Traditional Mass Media (Broadcast 1 -> N)
    2. Algorithmic Social Media Feeds (Open scale-free with beta bias & emotional amplification)
    3. Closed Private Messaging (WhatsApp groups K <= 256 with viral cascade forwarding)
    """

    def __init__(
        self,
        beta: float = 2.0,
        emotional_boost: float = 1.5,
        broadcast_strength: float = 0.05,
        feed_strength: float = 0.07,
        whatsapp_trust: float = 0.85,
        whatsapp_forward_prob: float = 0.25,
    ) -> None:
        self.beta = float(beta)
        self.emotional_boost = float(emotional_boost)
        self.broadcast_strength = float(broadcast_strength)
        self.feed_strength = float(feed_strength)
        self.whatsapp_trust = float(whatsapp_trust)
        self.whatsapp_forward_prob = float(whatsapp_forward_prob)
        self.group_active_news: np.ndarray | None = None

    def step_all_platforms(
        self,
        agent_state: AgentStateEngine,
        graph_engine: GraphEngine,
        news_pool: NewsEntityPool,
        rng: np.random.Generator,
        step_idx: int = 0,
    ) -> dict[str, float]:
        """Run one concurrent cycle of all three media platforms and update coevolution counters."""
        n_agents = agent_state.n_agents
        n_groups = graph_engine.n_wa_groups

        if self.group_active_news is None or len(self.group_active_news) != n_groups:
            self.group_active_news = rng.integers(
                0, news_pool.n_items, size=n_groups, dtype=np.int32
            )

        # 1. Traditional Broadcast Media (1 -> N)
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

        # 2. Algorithmic Open Social Media Feeds (Closed-Loop with beta bias)
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

        # Bidirectional feedback: increment popularity of engaged news items
        engaged_items = selected_news[engaged_flags == 1]
        if len(engaged_items) > 0:
            counts = np.bincount(engaged_items, minlength=news_pool.n_items)
            news_pool.popularity += counts.astype(np.float64)

        # 3. Closed Social Media (WhatsApp groups K <= 256)
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

        # Viral cross-group cascade routing: members who forwarded inject the news item into their other WhatsApp groups
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
                # Inject trending social media item into stagnant group
                top_item = int(np.argmax(news_pool.popularity * (0.5 + rng.random(news_pool.n_items))))
                self.group_active_news[g] = top_item

        # Periodic influencer strategy coevolution
        if step_idx > 0 and step_idx % 5 == 0 and len(graph_engine.influencer_ids) > 0:
            inf_opinions = agent_state.opinions[graph_engine.influencer_ids]
            news_pool.refresh_influencer_content(inf_opinions, replace_fraction=0.15, rng=rng)

        return {
            "broadcast_reached": float(broadcast_reached),
            "feed_engaged": float(np.sum(engaged_flags)),
            "whatsapp_cascades": float(total_cascades),
        }
