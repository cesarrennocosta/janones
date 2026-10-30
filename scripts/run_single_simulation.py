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
        "--traditional-media-only",
        action="store_true",
        help="Enable Traditional Media (TV, Radio, Press) while keeping algorithmic social media & WhatsApp disabled.",
    )
    parser.add_argument(
        "--social-media-only",
        action="store_true",
        help="Enable Social Media (Instagram/TikTok, YouTube/X, Facebook, WhatsApp) without Traditional Press.",
    )
    parser.add_argument(
        "--full-media-ecosystem",
        action="store_true",
        help="Enable BOTH Traditional Media (TV, Radio, Press) and the Multi-Platform Social Media Layer.",
    )
    parser.add_argument(
        "--n-vehicle-groups",
        type=int,
        default=None,
        help="Number of traditional media vehicle groups M in {0, 1, 2, 3, 4} (each operating TV, Radio, and Press).",
    )
    parser.add_argument(
        "--tendentiousness",
        type=str,
        default=None,
        help="Tendentiousness level tau in [0, 1] as a scalar (e.g. '0.15', '0.90') or comma-separated per-group list (e.g. '0.90,0.15').",
    )
    parser.add_argument(
        "--influencer-tendentiousness",
        type=str,
        default=None,
        help="Influencer tendentiousness mode ('factual', 'tendentious', 'heterogeneous') or scalar/comma-separated values in [0, 1].",
    )
    parser.add_argument(
        "--dislike-strength",
        type=float,
        default=None,
        help="Strength of negative/dislike out-group reaction probability on social media.",
    )
    parser.add_argument(
        "--backfire-strength",
        type=float,
        default=None,
        help="Repulsive backfire shift strength when reacting negatively (dislike) to discordant content.",
    )
    parser.add_argument(
        "--extreme-imposed",
        action="store_true",
        help="Enable the Extreme Imposed scenario: tendentious influencers in one extreme of subject space have artificially boosted reach and popularity.",
    )
    parser.add_argument(
        "--extreme-reach-boost",
        type=float,
        default=None,
        help="Artificial algorithmic reach multiplier for Extreme Imposed influencers (e.g. 3.5, 6.0).",
    )
    parser.add_argument(
        "--extreme-popularity-boost",
        type=float,
        default=None,
        help="Artificial popularity/interaction multiplier for Extreme Imposed influencer posts (e.g. 4.0, 8.0).",
    )
    parser.add_argument(
        "--extreme-fraction",
        type=float,
        default=None,
        help="Fraction of influencers anchored in the Extreme Imposed subject zone (default 0.25).",
    )
    parser.add_argument(
        "--sync-tactics",
        action="store_true",
        help="Enable synchronized focal-fact narratives among Extreme Imposed influencers.",
    )
    parser.add_argument(
        "--cowork",
        action="store_true",
        help="Enable influencer co-working (collaborations accessing partner follower networks & mutual recruitment).",
    )
    parser.add_argument(
        "--fact-dictation",
        action="store_true",
        help="Enable intermedia fact & subject dictation (synchronized network pushes focal facts into traditional media coverage & extends longevity).",
    )
    parser.add_argument(
        "--fact-dictation-pull",
        type=float,
        default=None,
        help="Strength of intermedia frame dictation pull on traditional media coverage (default 0.42).",
    )
    parser.add_argument(
        "--symmetric-duopoly",
        action="store_true",
        help="Place 2 vehicle groups at symmetric opposite centers (+0.55 and -0.55) to isolate asymmetric tendentiousness effects.",
    )
    parser.add_argument(
        "--fact-rate",
        type=float,
        default=None,
        help="Mean number of exogenous facts occurring per simulation step.",
    )
    parser.add_argument(
        "--broadcast-strength",
        type=float,
        default=None,
        help="Persuasive pull strength of traditional broadcast/press media.",
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
        media_cfg["traditional_enabled"] = False
        media_cfg["social_enabled"] = False
    if args.traditional_media_only:
        media_cfg["enabled"] = True
        media_cfg["traditional_enabled"] = True
        media_cfg["social_enabled"] = False
    if args.social_media_only:
        media_cfg["enabled"] = True
        media_cfg["traditional_enabled"] = False
        media_cfg["social_enabled"] = True
        media_cfg["n_vehicle_groups"] = 0
    if args.full_media_ecosystem:
        media_cfg["enabled"] = True
        media_cfg["traditional_enabled"] = True
        media_cfg["social_enabled"] = True
    if args.n_vehicle_groups is not None:
        media_cfg["n_vehicle_groups"] = args.n_vehicle_groups
        if args.n_vehicle_groups > 0:
            media_cfg["traditional_enabled"] = True
            media_cfg["enabled"] = True
        else:
            media_cfg["traditional_enabled"] = False
    if args.tendentiousness is not None:
        parts = [float(x.strip()) for x in args.tendentiousness.split(",") if x.strip()]
        media_cfg["group_tendentiousness"] = parts[0] if len(parts) == 1 else parts
    if args.influencer_tendentiousness is not None:
        raw_inf_t = args.influencer_tendentiousness.strip()
        if raw_inf_t.lower() in ("factual", "tendentious", "heterogeneous", "mixed", "bimodal"):
            media_cfg["influencer_tendentiousness"] = raw_inf_t.lower()
        else:
            parts = [float(x.strip()) for x in raw_inf_t.split(",") if x.strip()]
            media_cfg["influencer_tendentiousness"] = parts[0] if len(parts) == 1 else parts
    if args.dislike_strength is not None:
        media_cfg["dislike_reaction_strength"] = args.dislike_strength
    if args.backfire_strength is not None:
        media_cfg["backfire_repulsion_strength"] = args.backfire_strength
    if args.extreme_imposed:
        media_cfg["extreme_imposed_enabled"] = True
    if args.extreme_reach_boost is not None:
        media_cfg["extreme_imposed_reach_boost"] = args.extreme_reach_boost
        if args.extreme_reach_boost > 1.0:
            media_cfg["extreme_imposed_enabled"] = True
    if args.extreme_popularity_boost is not None:
        media_cfg["extreme_imposed_popularity_boost"] = args.extreme_popularity_boost
        if args.extreme_popularity_boost > 1.0:
            media_cfg["extreme_imposed_enabled"] = True
    if args.extreme_fraction is not None:
        media_cfg["extreme_imposed_fraction"] = args.extreme_fraction
    if args.sync_tactics:
        media_cfg["sync_tactics_enabled"] = True
        media_cfg["extreme_imposed_enabled"] = True
    if args.cowork:
        media_cfg["cowork_enabled"] = True
        media_cfg["extreme_imposed_enabled"] = True
    if args.fact_dictation:
        media_cfg["fact_dictation_enabled"] = True
        media_cfg["sync_tactics_enabled"] = True
        media_cfg["extreme_imposed_enabled"] = True
    if args.fact_dictation_pull is not None:
        media_cfg["fact_dictation_pull"] = args.fact_dictation_pull
    if args.symmetric_duopoly:
        n_dims = int(sim_cfg.get("n_dims", 4))
        media_cfg["vehicle_group_centers"] = [
            [0.55] * n_dims,
            [-0.55] * n_dims,
        ]
    if args.fact_rate is not None:
        media_cfg["fact_rate"] = args.fact_rate
    if args.broadcast_strength is not None:
        media_cfg["broadcast_strength"] = args.broadcast_strength
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
        f"[OmniOpinion Task {task_id:04d} | N={engine.n_agents} | M={engine.n_vehicle_groups} | "
        f"Tend=({final_m.get('tendentiousness_g0', 0.0):.2f},{final_m.get('tendentiousness_g1', 0.0):.2f}) | {status}] "
        f"Step={final_m['step']} | "
        f"P_ER={final_m['esteban_ray']:.4f} | "
        f"ShiftCM={final_m.get('shift_toward_media_cm', 0.0):+.3f} | "
        f"DistG0={final_m.get('dist_to_g0_mean', 0.0):.3f} | "
        f"FracG0={final_m.get('frac_closer_to_g0', 0.5):.3f}"
    )


if __name__ == "__main__":
    main()


