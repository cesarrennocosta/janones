#!/usr/bin/env python3
"""Run a single OmniOpinion stochastic simulation task (SLURM array & multi-scale city compatible)."""

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
        description="Execute a single OmniOpinion simulation instance (HPC / SLURM task)."
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
        default="/scratch/global/crcosta/omni_opinion/slurm_runs",
        help="Directory to write HDF5 trajectories, checkpoints, and Parquet metrics.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20261030,
        help="Master entropy seed for numpy.random.SeedSequence.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Explicit seed / task ID (0..100, overrides SLURM_ARRAY_TASK_ID).",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=None,
        help="Override number of agents N (e.g. 10000, 100000, 1000000, 10000000).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override number of timesteps.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save mid-simulation checkpoint every K steps for preempt_strict recovery.",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=["uniform_random", "correlated"],
        default=None,
        help="Parameter initialization mode ('uniform_random' for pure random initial conditions).",
    )
    parser.add_argument(
        "--no-media",
        action="store_true",
        help="Disable all media platforms (run pure local interpersonal connections only).",
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
    if args.checkpoint_interval is not None:
        sim_cfg["checkpoint_interval"] = args.checkpoint_interval
    if args.init_mode is not None:
        sim_cfg["init_mode"] = args.init_mode
    if args.no_media:
        media_cfg["enabled"] = False
    if args.beta is not None:
        media_cfg["beta"] = args.beta

    engine = SimulationEngine(
        config=cfg,
        base_seed=args.base_seed,
        task_id=task_id,
        output_dir=args.output_dir,
    )
    result = engine.run()

    final_m = result["final_metrics"]
    status = "PREEMPTED (Checkpoint Saved)" if result.get("preempted") else "COMPLETED"
    print(
        f"[OmniOpinion Task {task_id:04d} | N={engine.n_agents} | {status}] "
        f"Step={final_m['step']} | "
        f"P_ER={final_m['esteban_ray']:.4f} | "
        f"DispVsRand={final_m['dispersion_ratio_vs_random']:.3f} | "
        f"Clusters(IPR)={final_m['effective_clusters_ipr']:.2f} | "
        f"IssueCorr={final_m['mean_issue_correlation']:.3f} | "
        f"EffDim={final_m['effective_dimensionality']:.2f}"
    )


if __name__ == "__main__":
    main()
