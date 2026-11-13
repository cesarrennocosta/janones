#!/usr/bin/env python3
"""Figure 6 Replication: Strategic Evaluation of Algorithmic Counter-Tactics in the 2026 Runoff.

Author: César Rennó-Costa (Brain Institute, UFRN, Natal, RN, Brazil)
Paper: "Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections"
Target: Nature Human Behaviour (2026)
Contact: rennocosta@neuro.ufrn.br

Analytical Panels:
- Panel a: Runoff tracking trajectories comparing unopposed blitz (passive defense, Flávio wins 53.6%) vs full deployment of Janonismo Cultural (Lula surges to 51.79%, winning the election), calibrated to Datafolha polls and official TSE tallies (R^2 = 0.9128, RMSE = 0.56 percentage points).
- Panel b: Official 2nd-Round TSE ballot outcome and reversal dynamics.
- Panel c: Net valid vote share gain attributed to each of the 6 tactical pillars across 100 simulation seeds with error bars (mean and range).
- Panel d: Net vote share gain for Lula and corresponding share loss for Flávio across the 6 tactics.
- Panel e: Parametric sensitivity curves across physical empirical domains (Signal denial, monothematic variance, loss-aversion multiplier, WhatsApp trust, rally turnout locking, and live saturation ceiling).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import core simulation and rendering from existing pipeline
from scripts.generate_second_round_fit_and_sensibility_figure import (
    run_second_round_janones_simulations,
    render_second_round_master_figure,
)


def run_second_round_simulation(mode: str = "fast", n_agents: int = None) -> dict:
    """Run Second-Round Janonismo ABM simulation with 6-tactic sensitivity."""
    if n_agents is None:
        n_agents = 10_000 if mode == "fast" else 30_000
    print(f"[Figure 6] Running Second-Round Runoff ABM calibration (N={n_agents:,} citizens, mode={mode})...")
    sim_data = run_second_round_janones_simulations(n_agents=n_agents)
    return sim_data


def generate_figure6(
    sim_data: dict = None,
    output_dir: Path | str = None,
    mode: str = "fast",
) -> tuple[Path, Path]:
    """Render and save Figure 6 (Second-Round Janonismo Counter-Tactics & Sensitivity)."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sim_data is None:
        sim_data = run_second_round_simulation(mode=mode)

    pdf_path = output_dir / "figure5_jan_election_second.pdf"
    png_path = output_dir / "figure5_jan_election_second.png"
    out_files = [
        pdf_path,
        png_path,
        output_dir / "figure5_janones_countertactics_fit.pdf",
        output_dir / "figure5_janones_countertactics_fit.png",
    ]

    render_second_round_master_figure(sim_data, out_files)
    print(f"[Figure 6] Saved: {pdf_path}")
    print(f"[Figure 6] Saved: {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Figure 6 Second-Round simulation and generate plots.")
    parser.add_argument("--mode", type=str, choices=["fast", "full"], default="fast", help="Execution mode")
    parser.add_argument("--n-agents", type=int, default=None, help="Agent population size")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "figures"), help="Output directory")
    args = parser.parse_args()

    sim_data = run_second_round_simulation(mode=args.mode, n_agents=args.n_agents)
    generate_figure6(sim_data=sim_data, output_dir=args.output_dir, mode=args.mode)
