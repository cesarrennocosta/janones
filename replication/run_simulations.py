#!/usr/bin/env python3
"""Master Execution Pipeline: Run Paper Simulations and Reproduce Publication Figures.

Author: César Rennó-Costa (Brain Institute, UFRN, Natal, RN, Brazil)
Paper: "Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections"
Target: Nature Human Behaviour (2026)
Contact: rennocosta@neuro.ufrn.br

Usage:
  # Run all figures in fast verification mode (~2 minutes)
  python replication/run_simulations.py --all --mode fast

  # Run all figures in full publication mode (full Monte Carlo sweeps)
  python replication/run_simulations.py --all --mode full

  # Run specific figures
  python replication/run_simulations.py --figures 1 2 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from replication.sim_figure1_empirical_data import generate_figure1
from replication.sim_figure2_urban_network import generate_figure2
from replication.sim_figure3_traditional_media import generate_figure3
from replication.sim_figure4_social_media import generate_figure4
from replication.sim_figure5_first_round import generate_figure5
from replication.sim_figure6_second_round import generate_figure6


BANNER = """
================================================================================
  Late-Campaign Opinion Volatility and Algorithmic Counter-Mobilization
  Official Simulation & Figure Replication Pipeline
  Author: César Rennó-Costa (Brain Institute, UFRN, Natal, RN, Brazil)
  Target: Nature Human Behaviour (2026)
================================================================================
"""

FIGURE_REGISTRY = {
    1: ("Figure 1: Empirical Election & Polling Trajectories", generate_figure1),
    2: ("Figure 2: Urban Multiplex Network & Two-Timescale Dynamics", generate_figure2),
    3: ("Figure 3: Traditional Mass Media Gatekeeping & Polarization", generate_figure3),
    4: ("Figure 4: Social Media Feeds, Influencer Power-Law & Cascades", generate_figure4),
    5: ("Figure 5: First-Round Fit (RMSE=0.85%) & 5-Mechanism Sensitivity", generate_figure5),
    6: ("Figure 6: Second-Round Runoff & Janonismo Counter-Tactics (RMSE=0.56%)", generate_figure6),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run paper simulations and generate publication figures for Nature Human Behaviour.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all", action="store_true", help="Run all 6 figures sequentially")
    parser.add_argument("--figures", type=int, nargs="+", choices=[1, 2, 3, 4, 5, 6], help="Specific figure numbers to reproduce")
    parser.add_argument("--mode", type=str, choices=["fast", "full"], default="fast", help="Execution mode (fast or full)")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "figures"), help="Output directory for generated figures")
    args = parser.parse_args()

    if not args.all and not args.figures:
        parser.print_help()
        print("\nPlease specify --all or one or more figures using --figures 1 2 ...")
        sys.exit(1)

    figs_to_run = [1, 2, 3, 4, 5, 6] if args.all else sorted(args.figures)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(BANNER)
    print(f"Execution Mode:    {args.mode.upper()}")
    print(f"Output Directory:  {out_dir}")
    print(f"Target Figures:    {figs_to_run}\n")

    start_total = time.time()
    results_summary = []

    for fig_num in figs_to_run:
        label, func = FIGURE_REGISTRY[fig_num]
        print(f"--- Running {label} ---")
        t0 = time.time()
        try:
            if fig_num in (2, 5, 6):
                pdf, png = func(output_dir=out_dir, mode=args.mode)
            else:
                pdf, png = func(output_dir=out_dir)
            elapsed = time.time() - t0
            print(f"✓ Completed {label} in {elapsed:.1f}s\n")
            results_summary.append((fig_num, label, "SUCCESS", elapsed, pdf.name))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"✗ Failed {label} in {elapsed:.1f}s: {e}\n")
            results_summary.append((fig_num, label, f"FAILED ({e})", elapsed, "N/A"))

    total_elapsed = time.time() - start_total
    print("================================================================================")
    print("  Replication Summary")
    print("================================================================================")
    print(f"{'Fig':<4} {'Description':<52} {'Status':<10} {'Time':<8} {'File'}")
    print("-" * 90)
    for num, desc, status, el, fname in results_summary:
        print(f"{num:<4} {desc[:50]:<52} {status:<10} {el:>5.1f}s  {fname}")
    print("-" * 90)
    print(f"Total Elapsed Time: {total_elapsed:.1f}s ({total_elapsed/60.0:.2f} min)\n")


if __name__ == "__main__":
    main()
