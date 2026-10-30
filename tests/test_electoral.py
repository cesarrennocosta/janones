"""Unit tests for ElectoralEngine, Esteban-Ray polarization, and full SimulationEngine with HDF5/Parquet I/O."""

from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import pyarrow.parquet as pq

from src.core.agent_state import AgentStateEngine
from src.engines.electoral_engine import ElectoralEngine
from src.engines.simulation_engine import SimulationEngine
from src.metrics.polarization import (
    compute_esteban_ray_index,
    compute_sarle_bimodality,
)


def test_electoral_logit_abstention_and_districts():
    rng = np.random.default_rng(606)
    state = AgentStateEngine.initialize(n_agents=400, n_dims=4, n_candidates=3, n_districts=8, seed=rng)

    # Case 1: High indifference threshold forces high abstention
    strict_elec = ElectoralEngine(
        n_candidates=3,
        n_dims=4,
        indifference_threshold=10.0,  # Impossible for negative squared distance utility to exceed
        noise_scale=0.01,
    )
    res_strict = strict_elec.evaluate_election(state, rng)
    assert res_strict["abstention_rate"] == 1.0
    assert np.all(state.vote_choice == -1)
    assert np.allclose(state.vote_probs[:, 3], 1.0)

    # Case 2: Normal indifference threshold produces valid probability distribution and district counts
    normal_elec = ElectoralEngine(
        n_candidates=3,
        n_dims=4,
        indifference_threshold=-2.2,
        rationality_lambda=5.0,
        noise_scale=0.1,
    )
    res_norm = normal_elec.evaluate_election(state, rng)
    assert 0.0 <= res_norm["abstention_rate"] < 1.0
    assert np.allclose(np.sum(state.vote_probs, axis=1), 1.0)
    assert len(res_norm["popular_counts"]) == 3
    assert len(res_norm["district_winners"]) == 8
    assert 0.0 <= res_norm["electoral_discrepancy"] <= 1.0


def test_polarization_metrics_consensus_vs_bipolar():
    # Consensus state -> near-zero Esteban-Ray polarization
    consensus = np.zeros((200, 4), dtype=np.float64)
    er_cons = compute_esteban_ray_index(consensus, alpha=1.3, n_bins_per_dim=4)
    assert er_cons == 0.0

    # Bipolar state -> strictly higher Esteban-Ray polarization and high Sarle bimodality
    bipolar = np.vstack([
        np.full((100, 4), -0.9, dtype=np.float64),
        np.full((100, 4), 0.9, dtype=np.float64),
    ])
    er_bip = compute_esteban_ray_index(bipolar, alpha=1.3, n_bins_per_dim=4)
    sarle_bip = compute_sarle_bimodality(bipolar)
    assert er_bip > 0.5
    assert sarle_bip > 0.555


def test_end_to_end_simulation_and_hpc_storage(tmp_path: Path):
    cfg = {
        "simulation": {
            "n_agents": 120,
            "n_dims": 4,
            "n_steps": 6,
            "snapshot_interval": 2,
            "n_candidates": 3,
            "n_districts": 4,
            "n_news_items": 15,
        }
    }
    sim = SimulationEngine(config=cfg, base_seed=999, task_id=7, output_dir=tmp_path)
    out = sim.run()

    h5_file = out["hdf5_path"]
    pq_file = out["parquet_path"]
    assert h5_file is not None and h5_file.exists()
    assert pq_file is not None and pq_file.exists()

    # Verify HDF5 datasets
    with h5py.File(h5_file, "r") as h5f:
        assert "static" in h5f
        assert "trajectories" in h5f
        assert "final_state" in h5f
        assert h5f["trajectories/opinions"].shape == (4, 120, 4)  # steps 0, 2, 4, 6

    # Verify Apache Parquet table
    table = pq.read_table(pq_file)
    assert table.num_rows == 7  # steps 0..6 inclusive
    assert "esteban_ray" in table.column_names
    assert "abstention_rate" in table.column_names
