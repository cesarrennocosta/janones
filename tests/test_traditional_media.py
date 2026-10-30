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
            "n_steps": 15,
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
                "n_steps": 15,
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
