#!/usr/bin/env python3
"""Run a single OmniOpinion stochastic simulation task (SLURM array compatible)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engines.simulation_engine import SimulationEngine


def load_merged_config(
    base_cfg_path: str | Path,
    net_cfg_path: str | Path | None = None,
    media_cfg_path: str | Path | None = None,
) -> dict:
    """Load and merge YAML configuration files."""
    merged: dict = {}
    for p in (base_cfg_path, net_cfg_path, media_cfg_path):
        if p is not None and Path(p).exists():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                merged.update(data)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a single OmniOpinion simulation instance (HPC / SLURM Array task)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_simulation.yaml",
        help="Path to base simulation YAML configuration.",
    )
    parser.add_argument(
        "--network-config",
        type=str,
        default="configs/networks/multiplex_default.yaml",
        help="Path to multiplex network YAML configuration.",
    )
    parser.add_argument(
        "--media-config",
        type=str,
        default="configs/media/platform_profiles.yaml",
        help="Path to media platforms YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/slurm_runs",
        help="Directory to write HDF5 trajectories and Parquet metrics.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20261029,
        help="Master entropy seed for numpy.random.SeedSequence.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Explicit task ID (overrides SLURM_ARRAY_TASK_ID environment variable).",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=None,
        help="Override number of agents N.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override number of timesteps.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override algorithmic recommendation bias parameter beta >= 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve task ID from CLI or SLURM_ARRAY_TASK_ID environment variable
    env_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    task_id = args.task_id if args.task_id is not None else int(env_task_id)

    cfg = load_merged_config(args.config, args.network_config, args.media_config)
    sim_cfg = cfg.setdefault("simulation", {})
    media_cfg = cfg.setdefault("media", {})

    if args.n_agents is not None:
        sim_cfg["n_agents"] = args.n_agents
    if args.n_steps is not None:
        sim_cfg["n_steps"] = args.n_steps
    if args.beta is not None:
        media_cfg["beta"] = args.beta
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        # Optional parameter sweep across SLURM tasks: vary beta smoothly across ensemble
        sweep_betas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        media_cfg["beta"] = sweep_betas[task_id % len(sweep_betas)]

    engine = SimulationEngine(
        config=cfg,
        base_seed=args.base_seed,
        task_id=task_id,
        output_dir=args.output_dir,
    )
    result = engine.run()

    final_m = result["final_metrics"]
    print(
        f"[OmniOpinion Task {task_id:04d}] Completed {engine.n_steps} steps | "
        f"Esteban-Ray P_ER={final_m['esteban_ray']:.4f} | "
        f"Sarle BC={final_m['sarle_bimodality']:.4f} | "
        f"Abstention={final_m['abstention_rate']:.2%} | "
        f"Popular Winner={final_m['popular_winner']} | "
        f"District Winner={final_m['district_winner']}"
    )


if __name__ == "__main__":
    main()
