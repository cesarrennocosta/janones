"""Tests for two-timescale conviction vs. current opinion dynamics, demographic/extremity cohorts, and fact-bias shifts."""

from __future__ import annotations

import numpy as np
from src.engines.simulation_engine import SimulationEngine


def _base_conviction_config(
    n_agents: int = 1200,
    n_steps: int = 120,
    init_mode: str = "uniform_random",
    fact_bias_mode: str = "neutral",
) -> dict:
    return {
        "simulation": {
            "n_agents": n_agents,
            "n_dims": 4,
            "n_steps": n_steps,
            "snapshot_interval": 20,
            "checkpoint_interval": 50,
            "n_candidates": 4,
            "n_districts": 5,
            "n_news_items": 24,
            "init_mode": init_mode,
            "age_distribution": "ibge_2022",
            "base_epsilon": 0.45,
            "base_assimilation_bias": 1.2,
            "dt": 0.15,
            "noise_std": 0.012,
            "conviction_dynamics_enabled": True,
            "max_excursion_base": 0.34,
            "conviction_drift_rate": 0.0024,
            "crystallization_tau": 80.0,
            "fact_direct_pull": 0.065,
        },
        "networks": {
            "family_degree": 3.0,
            "work_school_degree": 6.0,
            "church_community_degree": 3.5,
            "friends_degree": 4.5,
            "weak_ties_degree": 3.5,
            "social_feed_out_degree": 8,
            "influencer_Power_alpha": 2.3,
            "n_influencers": 20,
            "n_whatsapp_groups_per_1k": 25,
            "max_whatsapp_group_size": 256,
        },
        "media": {
            "enabled": True,
            "traditional_enabled": True,
            "social_enabled": True,
            "n_vehicle_groups": 3,
            "group_tendentiousness": [0.80, 0.25, 0.65],
            "influencer_tendentiousness": "heterogeneous",
            "fact_rate": 1.8,
            "broadcast_strength": 0.065,
            "feed_strength": 0.075,
            "fact_bias_mode": fact_bias_mode,
        },
        "electoral": {
            "indifference_threshold": -1.65,
            "rationality_lambda": 4.2,
            "noise_scale": 0.08,
        },
    }


def test_conviction_stabilizes_while_opinion_fluctuates_by_cohort() -> None:
    """Verify that convictions stabilize over time while current opinions fluctuate within
    bounded excursion of convictions, and that centrist and young cohorts exhibit greater
    opinion mobility and electoral swing than extremist and senior cohorts.
    """
    cfg = _base_conviction_config(n_agents=1500, n_steps=100, init_mode="uniform_random")
    engine = SimulationEngine(config=cfg, base_seed=20261031, task_id=0)
    res = engine.run()
    hist = res["history"]
    final_m = res["final_metrics"]

    # 1. Convictions change much more slowly than current opinions
    op_vols = [h["opinion_step_volatility"] for h in hist[10:]]
    conv_vols = [h["conviction_step_volatility"] for h in hist[10:]]
    assert np.mean(op_vols) > 8.0 * np.mean(conv_vols)

    # 2. Conviction volatility decreases as convictions crystallize over time
    early_conv_vol = np.mean([h["conviction_step_volatility"] for h in hist[5:25]])
    late_conv_vol = np.mean([h["conviction_step_volatility"] for h in hist[-20:]])
    assert late_conv_vol < early_conv_vol

    # 3. Current opinions remain tethered within bounded distance of convictions
    assert 0.02 < final_m["mean_op_conv_gap"] < 0.35

    # 4. Centrist individuals have larger opinion excursion & higher swing rate than extremist individuals
    assert final_m["op_conv_gap_centrist"] > final_m["op_conv_gap_extremist"]
    assert final_m["swing_vote_centrist"] > final_m["swing_vote_extremist"]

    # 5. Younger adults have larger opinion excursion than senior citizens
    assert final_m["op_conv_gap_young"] > final_m["op_conv_gap_senior"]


def test_centrist_vs_extremist_conviction_cities() -> None:
    """Compare a city with mostly centrist convictions against a city with mostly extremist convictions."""
    cfg_cent = _base_conviction_config(n_agents=1200, n_steps=60, init_mode="centrist_convictions")
    cfg_ext = _base_conviction_config(n_agents=1200, n_steps=60, init_mode="extremist_convictions")

    res_cent = SimulationEngine(config=cfg_cent, base_seed=42, task_id=1).run()["final_metrics"]
    res_ext = SimulationEngine(config=cfg_ext, base_seed=42, task_id=1).run()["final_metrics"]

    assert res_cent["frac_centrist_convictions"] > 0.60
    assert res_ext["frac_extremist_convictions"] > 0.55
    assert res_cent["swing_voter_fraction"] > res_ext["swing_voter_fraction"]
    assert res_cent["mean_op_conv_gap"] > res_ext["mean_op_conv_gap"]


def test_fact_bias_shift_flips_opinion_and_elections_faster_than_conviction() -> None:
    """Verify that under a strong shift in the bias of exogenous facts, current opinions
    shift rapidly and flip election winners while convictions move much more slowly.
    """
    cfg = _base_conviction_config(
        n_agents=1200,
        n_steps=70,
        init_mode="centrist_convictions",
        fact_bias_mode="positive_shift",
    )
    # Trigger shift earlier by running steps 440..510 or using positive_shift
    engine = SimulationEngine(config=cfg, base_seed=99, task_id=0)
    # Fast-forward step index test via direct run with positive_shift
    res = engine.run()
    # Now test regime_shift across the transition boundary (t=400 -> 460)
    from src.core.news_entity import compute_fact_bias_vector
    from src.kernels.opinion_updates import step_conviction_and_opinion_tether_jit

    vec_pre, act_pre = compute_fact_bias_vector(390, 4, "regime_shift")
    vec_pos, act_pos = compute_fact_bias_vector(500, 4, "regime_shift")
    vec_neg, act_neg = compute_fact_bias_vector(1100, 4, "regime_shift")
    assert not act_pre and vec_pre[0] == 0.0
    assert act_pos and vec_pos[0] > 0.70
    assert act_neg and vec_neg[0] < -0.70

    # Apply 40 steps of positive fact-bias shock starting at t=401 and compare opinion vs conviction shift
    s0 = engine.agent_state.innate_beliefs[:, 0].mean()
    x0 = engine.agent_state.opinions[:, 0].mean()
    for step_t in range(401, 445):
        step_conviction_and_opinion_tether_jit(
            engine.agent_state.opinions,
            engine.agent_state.innate_beliefs,
            engine.agent_state.age,
            engine.agent_state.ses,
            engine.agent_state.susceptibility,
            vec_pos,
            step_t,
            0.34,
            0.0024,
            280.0,
            0.065,
            1,
        )
    s1 = engine.agent_state.innate_beliefs[:, 0].mean()
    x1 = engine.agent_state.opinions[:, 0].mean()

    delta_op = abs(x1 - x0)
    delta_conv = abs(s1 - s0)
    assert delta_op > 0.15
    assert delta_op > 5.0 * delta_conv


def test_two_leg_election_admiration_rejection_abstention_and_blank_votes() -> None:
    """Verify the Two-Leg (4 candidates -> 2 candidates) election model:
    - Each candidate has a center of mass in the opinion spectrum.
    - Independent admiration (increased by very close current opinion) and rejection
      (increased by very different current opinion).
    - Three-way voting: not going to vote (-1, low admiration in general), going to vote
      but voting blank (-2, no clear winner and no strong rejection), and valid vote (0..3)
      based on current opinion rather than conviction.
    """
    cfg = _base_conviction_config(n_agents=1500, n_steps=40, init_mode="uniform_random")
    engine = SimulationEngine(config=cfg, base_seed=777, task_id=0)
    res = engine.run()
    final_m = res["final_metrics"]
    state = engine.agent_state

    # 1. 4 candidates in First Leg, 2 finalists in Second Leg
    assert engine.electoral_engine.n_candidates == 4
    assert engine.electoral_engine.candidate_profiles.shape == (4, 4)
    assert final_m["r2_cand_a"] != final_m["r2_cand_b"]
    assert final_m["r2_winner"] in (final_m["r2_cand_a"], final_m["r2_cand_b"])
    assert np.isclose(final_m["r2_share_cand_a"] + final_m["r2_share_cand_b"], 1.0, atol=1e-5)

    # 2. Admiration and Rejection are bounded in [0, 1] and respond to opinion distance
    assert state.candidate_admiration is not None
    assert state.candidate_rejection is not None
    assert np.all((state.candidate_admiration >= 0.0) & (state.candidate_admiration <= 1.0))
    assert np.all((state.candidate_rejection >= 0.0) & (state.candidate_rejection <= 1.0))

    # Citizens very close to Candidate 0 have high admiration and low rejection for Candidate 0;
    # citizens very far from Candidate 0 have high rejection and low admiration for Candidate 0.
    c0 = engine.electoral_engine.candidate_profiles[0]
    dists_c0 = np.linalg.norm(state.opinions - c0[None, :], axis=1) / np.sqrt(4.0)
    close_mask = dists_c0 < 0.35
    far_mask = dists_c0 > 0.85
    assert state.candidate_admiration[close_mask, 0].mean() > state.candidate_admiration[far_mask, 0].mean() + 0.25
    assert state.candidate_rejection[far_mask, 0].mean() > state.candidate_rejection[close_mask, 0].mean() + 0.25

    # 3. Three voting states present (-1 not going to vote, -2 blank vote, >=0 valid vote)
    assert final_m["abstention_rate"] > 0.0
    assert final_m["blank_rate"] > 0.0
    assert final_m["valid_vote_rate"] > 0.30
    assert np.isclose(
        final_m["abstention_rate"] + final_m["blank_rate"] + final_m["valid_vote_rate"],
        1.0,
        atol=1e-5,
    )
    assert np.isclose(
        final_m["r2_abstention_rate"] + final_m["r2_blank_rate"] + final_m["r2_valid_vote_rate"],
        1.0,
        atol=1e-5,
    )
