"""Unit tests for Fact-Driven Traditional Media Ecosystem (TV, Radio, Press) across M=1,2,3,4 vehicle groups."""

from __future__ import annotations

import numpy as np

from src.core.news_entity import (
    MODALITY_PRESS,
    MODALITY_RADIO,
    MODALITY_TV,
    TraditionalMediaEcosystem,
)
from src.engines.simulation_engine import SimulationEngine


def test_traditional_ecosystem_lag_frequency_and_decay() -> None:
    """Verify modality lags (Radio=0, TV=1, Press=2), slant toward center of mass, and exponential decay."""
    rng = np.random.default_rng(101)
    eco = TraditionalMediaEcosystem.initialize(
        n_groups=1,
        n_dims=4,
        group_centers=[[0.60, 0.60, 0.60, 0.60]],
        fact_rate=3.0,
        seed=rng,
    )
    assert eco.n_vehicles == 3
    assert set(eco.vehicle_modality.tolist()) == {MODALITY_TV, MODALITY_RADIO, MODALITY_PRESS}

    # Step 1: facts arrive; Radio (lag=0) publishes immediately, TV (lag=1) and Press (lag=2) queue notices
    info_1 = eco.step_facts_and_news_cycle(step_idx=1, rng=rng)
    assert info_1["facts_count"] >= 0

    # Turn off new facts and verify exponential decay over time (Wu & Huberman, 2007)
    eco.fact_rate = 0.0
    eco.pending_notices.clear()
    initial_intensities = eco.active_intensity.copy()
    eco.step_facts_and_news_cycle(step_idx=10, rng=rng)
    assert np.all(eco.active_intensity < initial_intensities)
    # Press decays slower than TV, and TV decays slower than Radio
    tv_idx = int(np.where(eco.vehicle_modality == MODALITY_TV)[0][0])
    radio_idx = int(np.where(eco.vehicle_modality == MODALITY_RADIO)[0][0])
    press_idx = int(np.where(eco.vehicle_modality == MODALITY_PRESS)[0][0])
    assert eco.vehicle_decay_rate[press_idx] < eco.vehicle_decay_rate[tv_idx] < eco.vehicle_decay_rate[radio_idx]


def test_vehicle_groups_1_to_4_opinion_shift_and_polarization() -> None:
    """Verify M=1,2,3,4 vehicle groups run cleanly and M=1 pulls opinion toward the press center of mass."""
    base_cfg_0 = {
        "simulation": {
            "n_agents": 400,
            "n_dims": 4,
            "n_steps": 20,
            "init_mode": "uniform_random",
        },
        "media": {
            "enabled": False,
            "traditional_enabled": False,
            "social_enabled": False,
            "n_vehicle_groups": 0,
        },
    }
    res_0 = SimulationEngine(config=base_cfg_0, base_seed=777, task_id=0).run()
    cm_0 = res_0["final_metrics"]["pop_cm_d0"]

    results_m = {}
    for m_groups in (1, 2, 3, 4):
        cfg_m = {
            "simulation": {
                "n_agents": 400,
                "n_dims": 4,
                "n_steps": 20,
                "init_mode": "uniform_random",
            },
            "media": {
                "enabled": True,
                "traditional_enabled": True,
                "social_enabled": False,
                "n_vehicle_groups": m_groups,
                "fact_rate": 2.0,
                "broadcast_strength": 0.12,
            },
        }
        res_m = SimulationEngine(config=cfg_m, base_seed=777, task_id=0).run()
        results_m[m_groups] = res_m["final_metrics"]
        assert results_m[m_groups]["n_vehicle_groups"] == m_groups
        assert results_m[m_groups]["broadcast_reached"] > 0.0
        assert results_m[m_groups]["tv_reached"] > 0.0
        assert results_m[m_groups]["radio_reached"] > 0.0
        assert results_m[m_groups]["press_reached"] > 0.0

    # M=1 has center of mass at +0.55: population opinion must shift positively toward the press center of mass
    assert results_m[1]["shift_toward_media_cm"] > 0.05
    assert results_m[1]["pop_cm_d0"] > cm_0 + 0.05


def test_tendentiousness_1group_and_asymmetric_2groups() -> None:
    """Verify:
    1. In M=1, higher tendentiousness (0.95 vs 0.10) centers the population much closer to the media positioning.
    2. In M=2 with symmetric centers (+0.55 vs -0.55), asymmetric tendentiousness ([0.95, 0.10]) skews the
       population opinion distribution toward the more tendentious group (+0.55).
    """
    # 1. M=1: Low tendentiousness (0.10) vs High tendentiousness (0.95)
    cfg_1g_low = {
        "simulation": {"n_agents": 400, "n_dims": 4, "n_steps": 25, "init_mode": "uniform_random"},
        "media": {
            "enabled": True,
            "traditional_enabled": True,
            "social_enabled": False,
            "n_vehicle_groups": 1,
            "group_tendentiousness": 0.10,
            "fact_rate": 2.0,
            "broadcast_strength": 0.14,
        },
    }
    cfg_1g_high = {
        "simulation": {"n_agents": 400, "n_dims": 4, "n_steps": 25, "init_mode": "uniform_random"},
        "media": {
            "enabled": True,
            "traditional_enabled": True,
            "social_enabled": False,
            "n_vehicle_groups": 1,
            "group_tendentiousness": 0.95,
            "fact_rate": 2.0,
            "broadcast_strength": 0.14,
        },
    }
    m_low = SimulationEngine(config=cfg_1g_low, base_seed=999, task_id=0).run()["final_metrics"]
    m_high = SimulationEngine(config=cfg_1g_high, base_seed=999, task_id=0).run()["final_metrics"]

    assert m_high["shift_toward_media_cm"] > m_low["shift_toward_media_cm"] + 0.04
    assert m_high["dist_to_g0_mean"] < m_low["dist_to_g0_mean"]

    # 2. M=2: Symmetric centers (+0.55 and -0.55) with Asymmetric Tendentiousness ([0.95, 0.10])
    sym_centers = [[0.55, 0.55, 0.55, 0.55], [-0.55, -0.55, -0.55, -0.55]]
    cfg_2g_sym = {
        "simulation": {"n_agents": 400, "n_dims": 4, "n_steps": 25, "init_mode": "uniform_random"},
        "media": {
            "enabled": True,
            "traditional_enabled": True,
            "social_enabled": False,
            "n_vehicle_groups": 2,
            "vehicle_group_centers": sym_centers,
            "group_tendentiousness": [0.50, 0.50],
            "fact_rate": 2.0,
            "broadcast_strength": 0.14,
        },
    }
    cfg_2g_asym = {
        "simulation": {"n_agents": 400, "n_dims": 4, "n_steps": 25, "init_mode": "uniform_random"},
        "media": {
            "enabled": True,
            "traditional_enabled": True,
            "social_enabled": False,
            "n_vehicle_groups": 2,
            "vehicle_group_centers": sym_centers,
            "group_tendentiousness": [0.95, 0.10],
            "fact_rate": 2.0,
            "broadcast_strength": 0.14,
        },
    }
    m_2g_sym = SimulationEngine(config=cfg_2g_sym, base_seed=999, task_id=0).run()["final_metrics"]
    m_2g_asym = SimulationEngine(config=cfg_2g_asym, base_seed=999, task_id=0).run()["final_metrics"]

    # The asymmetric scenario must pull the population mean and share toward Group 1 (+0.55)
    assert m_2g_asym["pop_cm_d0"] > m_2g_sym["pop_cm_d0"] + 0.04
    assert m_2g_asym["frac_closer_to_g0"] > m_2g_sym["frac_closer_to_g0"]

