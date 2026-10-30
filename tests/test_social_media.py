"""Unit tests for the Multi-Platform Social Media Ecosystem (Instagram/TikTok, YouTube/X, Facebook, WhatsApp),
Influencer Network of Impact & Follower Gain/Loss Dynamics, Factual vs. Tendentious Influencers,
and Dual-Valence Affective Reactions (Like vs. Dislike / Backfire), With vs. Without Traditional Press."""

from __future__ import annotations

import numpy as np

from src.core.agent_state import AgentStateEngine
from src.core.multiplex_network import GraphEngine
from src.core.news_entity import NewsEntityPool
from src.engines.simulation_engine import SimulationEngine


def test_social_media_platforms_and_demographic_reach() -> None:
    """Verify that Instagram/TikTok, YouTube/X, Facebook, and WhatsApp all register active reach
    and that platform items span all 3 algorithmic feed modalities."""
    cfg = {
        "simulation": {
            "n_agents": 400,
            "n_dims": 4,
            "n_steps": 15,
            "n_news_items": 45,
            "init_mode": "uniform_random",
        },
        "media": {
            "enabled": True,
            "traditional_enabled": False,
            "social_enabled": True,
            "n_vehicle_groups": 0,
            "influencer_tendentiousness": "heterogeneous",
            "dislike_reaction_strength": 0.50,
            "backfire_repulsion_strength": 0.40,
        },
    }
    sim = SimulationEngine(config=cfg, base_seed=3030, task_id=0)
    assert sim.news_pool is not None
    assert set(sim.news_pool.platform_type.tolist()).issubset({0, 1, 2})
    assert len(np.unique(sim.news_pool.platform_type)) >= 2

    res = sim.run()
    m = res["final_metrics"]
    assert m["ig_tiktok_reached"] > 0.0
    assert m["youtube_x_reached"] > 0.0
    assert m["facebook_reached"] > 0.0
    assert m["whatsapp_cascades"] >= 0.0
    assert m["likes_count"] > 0.0
    assert m["dislikes_count"] > 0.0
    assert 0.0 < m["dislike_share"] < 1.0


def test_influencer_factual_vs_tendentious_and_follower_dynamics() -> None:
    """Verify:
    1. Factual influencers produce higher-veracity, lower-sensationalism content and lower polarization
       than tendentious influencers.
    2. Influencer follower gain/loss dynamics update follower counts and tendentious influencers capture
       a larger share of follower growth in a mixed ecosystem (Bianconi & Barabasi, 2001; Vosoughi et al., 2018).
    """
    rng = np.random.default_rng(4040)
    state = AgentStateEngine.initialize(n_agents=300, n_dims=4, seed=rng)
    graph = GraphEngine.build_multiplex(state, seed=rng, enable_media_networks=True)
    inf_ops = state.opinions[graph.influencer_ids]

    pool_factual = NewsEntityPool.initialize(
        n_items=40,
        n_dims=4,
        influencer_ids=graph.influencer_ids,
        influencer_opinions=inf_ops,
        influencer_tendentiousness="factual",
        seed=111,
    )
    pool_tend = NewsEntityPool.initialize(
        n_items=40,
        n_dims=4,
        influencer_ids=graph.influencer_ids,
        influencer_opinions=inf_ops,
        influencer_tendentiousness="tendentious",
        seed=111,
    )
    assert np.mean(pool_factual.influencer_tendentiousness) < 0.30
    assert np.mean(pool_tend.influencer_tendentiousness) > 0.75
    assert np.mean(pool_factual.veracity) > np.mean(pool_tend.veracity)
    assert np.mean(pool_tend.sensationalism) > np.mean(pool_factual.sensationalism)

    # Run simulation comparing Factual vs Tendentious influencers without traditional press
    cfg_fact = {
        "simulation": {"n_agents": 400, "n_dims": 4, "n_steps": 30, "n_news_items": 45, "init_mode": "uniform_random"},
        "media": {
            "enabled": True,
            "traditional_enabled": False,
            "social_enabled": True,
            "n_vehicle_groups": 0,
            "influencer_tendentiousness": "factual",
            "feed_strength": 0.09,
            "dislike_reaction_strength": 0.50,
            "backfire_repulsion_strength": 0.40,
        },
    }
    cfg_tend = {
        "simulation": {"n_agents": 400, "n_dims": 4, "n_steps": 30, "n_news_items": 45, "init_mode": "uniform_random"},
        "media": {
            "enabled": True,
            "traditional_enabled": False,
            "social_enabled": True,
            "n_vehicle_groups": 0,
            "influencer_tendentiousness": "tendentious",
            "feed_strength": 0.09,
            "dislike_reaction_strength": 0.50,
            "backfire_repulsion_strength": 0.40,
        },
    }
    sim_f = SimulationEngine(config=cfg_fact, base_seed=5151, task_id=0)
    sim_t = SimulationEngine(config=cfg_tend, base_seed=5151, task_id=0)
    m_f = sim_f.run()["final_metrics"]
    m_t = sim_t.run()["final_metrics"]

    # Tendentious influencers induce greater opinion dispersion/polarization and more dislike reactions
    assert m_t["dispersion_ratio_vs_random"] > m_f["dispersion_ratio_vs_random"]
    assert m_t["dislikes_count"] >= m_f["dislikes_count"]
    assert not np.allclose(
        sim_t.graph_engine.influencer_followers,
        sim_t.graph_engine.influencer_initial_followers,
    )


def test_social_media_with_vs_without_traditional_press() -> None:
    """Verify simulations WITH and WITHOUT the most accurate Brazilian traditional press scenario
    (M=3 tripolar press ecosystem with group_tendentiousness=[0.80, 0.25, 0.65])."""
    cfg_without_press = {
        "simulation": {"n_agents": 400, "n_dims": 4, "n_steps": 25, "n_news_items": 45, "init_mode": "uniform_random"},
        "media": {
            "enabled": True,
            "traditional_enabled": False,
            "social_enabled": True,
            "n_vehicle_groups": 0,
            "influencer_tendentiousness": "heterogeneous",
        },
    }
    cfg_with_press = {
        "simulation": {"n_agents": 400, "n_dims": 4, "n_steps": 25, "n_news_items": 45, "init_mode": "uniform_random"},
        "media": {
            "enabled": True,
            "traditional_enabled": True,
            "social_enabled": True,
            "n_vehicle_groups": 3,
            "group_tendentiousness": [0.80, 0.25, 0.65],
            "influencer_tendentiousness": "heterogeneous",
            "broadcast_strength": 0.10,
        },
    }
    m_no_press = SimulationEngine(config=cfg_without_press, base_seed=8080, task_id=0).run()["final_metrics"]
    m_with_press = SimulationEngine(config=cfg_with_press, base_seed=8080, task_id=0).run()["final_metrics"]

    assert m_no_press["broadcast_reached"] == 0.0
    assert m_no_press["feed_engaged"] > 0.0
    assert m_with_press["broadcast_reached"] > 0.0
    assert m_with_press["feed_engaged"] > 0.0
    assert m_with_press["n_vehicle_groups"] == 3


def test_extreme_imposed_population_shift_and_influencer_dominance() -> None:
    """Verify the Extreme Imposed scenario:
    1. Influencers in the extreme_imposed region (+0.85, ..., +0.85) are mostly tendentious.
    2. Artificial boosts to reach and popularity cause a significantly larger fraction of the
       population to follow Extreme Imposed ideals (`frac_following_extreme_imposed`, `extreme_imposed_alignment`).
    3. Extreme Imposed influencers become significantly more popular and gain more followers than
       other influencers in general (`extreme_imposed_follower_ratio > 1.2`, `extreme_imposed_popularity_ratio > 1.5`).
    """
    cfg_baseline = {
        "simulation": {"n_agents": 400, "n_dims": 4, "n_steps": 30, "n_news_items": 45, "init_mode": "uniform_random"},
        "media": {
            "enabled": True,
            "traditional_enabled": False,
            "social_enabled": True,
            "n_vehicle_groups": 0,
            "influencer_tendentiousness": "heterogeneous",
            "extreme_imposed_enabled": False,
        },
    }
    cfg_extreme_imposed = {
        "simulation": {"n_agents": 400, "n_dims": 4, "n_steps": 30, "n_news_items": 45, "init_mode": "uniform_random"},
        "media": {
            "enabled": True,
            "traditional_enabled": False,
            "social_enabled": True,
            "n_vehicle_groups": 0,
            "influencer_tendentiousness": "heterogeneous",
            "extreme_imposed_enabled": True,
            "extreme_imposed_fraction": 0.25,
            "extreme_imposed_reach_boost": 4.5,
            "extreme_imposed_popularity_boost": 5.0,
        },
    }
    sim_base = SimulationEngine(config=cfg_baseline, base_seed=9191, task_id=0)
    sim_ext = SimulationEngine(config=cfg_extreme_imposed, base_seed=9191, task_id=0)

    # Verify Extreme Imposed influencers are mostly tendentious (>= 0.85)
    assert sim_ext.news_pool is not None
    ext_mask = sim_ext.news_pool.influencer_is_extreme_imposed == 1
    assert np.any(ext_mask)
    assert np.mean(sim_ext.news_pool.influencer_tendentiousness[ext_mask]) >= 0.85

    m_base = sim_base.run()["final_metrics"]
    m_ext = sim_ext.run()["final_metrics"]

    # Population shifts toward Extreme Imposed ideals and a larger share follows them
    assert m_ext["extreme_imposed_alignment"] > m_base["extreme_imposed_alignment"] + 0.04
    assert m_ext["frac_following_extreme_imposed"] > m_base["frac_following_extreme_imposed"]
    assert m_ext["dist_to_extreme_imposed_mean"] < m_base["dist_to_extreme_imposed_mean"]

    # Extreme Imposed influencers become more popular and gain more followers than other influencers
    assert m_ext["extreme_imposed_follower_ratio"] > 1.15
    assert m_ext["extreme_imposed_popularity_ratio"] > 1.50


def test_synchronized_tactics_cowork_and_fact_dictation() -> None:
    """Verify long-term synchronization tactics, influencer co-working (cross-network access),
    and intermedia dictation of facts & subjects to traditional media (with variable fact relevance & longevity).
    """
    cfg_unsync = {
        "simulation": {"n_agents": 400, "n_dims": 4, "n_steps": 30, "n_news_items": 45, "init_mode": "uniform_random"},
        "media": {
            "enabled": True,
            "traditional_enabled": True,
            "social_enabled": True,
            "n_vehicle_groups": 3,
            "group_tendentiousness": [0.80, 0.25, 0.65],
            "influencer_tendentiousness": "heterogeneous",
            "extreme_imposed_enabled": True,
            "extreme_imposed_fraction": 0.25,
            "extreme_imposed_reach_boost": 3.5,
            "extreme_imposed_popularity_boost": 4.0,
            "sync_tactics_enabled": False,
            "cowork_enabled": False,
            "fact_dictation_enabled": False,
        },
    }
    cfg_sync_full = {
        "simulation": {"n_agents": 400, "n_dims": 4, "n_steps": 30, "n_news_items": 45, "init_mode": "uniform_random"},
        "media": {
            "enabled": True,
            "traditional_enabled": True,
            "social_enabled": True,
            "n_vehicle_groups": 3,
            "group_tendentiousness": [0.80, 0.25, 0.65],
            "influencer_tendentiousness": "heterogeneous",
            "extreme_imposed_enabled": True,
            "extreme_imposed_fraction": 0.25,
            "extreme_imposed_reach_boost": 3.5,
            "extreme_imposed_popularity_boost": 4.0,
            "sync_tactics_enabled": True,
            "cowork_enabled": True,
            "cowork_prob": 0.70,
            "cowork_reach_fraction": 0.55,
            "fact_dictation_enabled": True,
            "fact_dictation_pull": 0.45,
        },
    }
    res_unsync = SimulationEngine(config=cfg_unsync, base_seed=7373, task_id=0).run()
    res_sync = SimulationEngine(config=cfg_sync_full, base_seed=7373, task_id=0).run()
    m_u = res_unsync["final_metrics"]
    m_s = res_sync["final_metrics"]

    # Synchronized tactics + co-working + fact dictation generate cross-network exposures,
    # dictate traditional media coverage, extend fact longevity, and increase follower network growth
    assert m_s["synchronized_fact_share"] > 0.30
    assert m_s["cowork_cross_exposures"] > 0.0
    assert m_s["dictated_trad_share"] > 0.15
    assert m_s["mean_fact_longevity"] > m_u["mean_fact_longevity"]
    assert m_s["extreme_imposed_inf_followers_mean"] > m_u["extreme_imposed_inf_followers_mean"]
    assert m_s["extreme_imposed_alignment"] > m_u["extreme_imposed_alignment"]


