#!/usr/bin/env python3
"""Figure 5 Replication: Longitudinal Electoral Trajectories and Parametric Sensitivity of the First Round.

Author: César Rennó-Costa (Brain Institute, UFRN, Natal, RN, Brazil)
Paper: "Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections"
Target: Nature Human Behaviour (2026)
Contact: rennocosta@neuro.ufrn.br

Analytical Panels:
- Panel a: 50-day longitudinal tracking trajectory of valid vote intentions across Datafolha polls and official TSE ballot tallies (R^2 = 0.9955, RMSE = 0.85 percentage points), featuring the August 23 debate shock (Augusto Cury surge to 21.4% and subsequent exponential decay).
- Panel b: Acute 48-hour pre-election blitz during statutory blackout window, driving Flávio Bolsonaro into first place with 47.03% vs Lula's 45.16%.
- Panel c: Net valid vote share gain attributed to the 5 Janones mechanisms across 100 simulation seeds with error bars (mean and range).
- Panel d: Net valid vote share gain for Flávio and corresponding loss for Lula across all 5 mechanisms.
- Panel e: Parametric sensitivity curves across physical empirical domains (Church reach, media slant, bot astroturfing, negativity asymmetry, and WhatsApp cascade reach).
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
from scripts.generate_first_round_fit_and_sensibility_figure import (
    run_first_round_janones_simulations,
    render_master_figure,
)


def run_first_round_simulation(mode: str = "fast", n_agents: int = None) -> dict:
    """Run First-Round Janones ABM simulation with 23/08 debate shock and 5-mechanism sensitivity."""
    if n_agents is None:
        n_agents = 10_000 if mode == "fast" else 30_000
    print(f"[Figure 5] Running First-Round ABM calibration (N={n_agents:,} citizens, mode={mode})...")
    sim_data = run_first_round_janones_simulations(n_agents=n_agents)
    return sim_data


def generate_figure5(
    sim_data: dict = None,
    output_dir: Path | str = None,
    mode: str = "fast",
) -> tuple[Path, Path]:
    """Render and save Figure 5 (First-Round Fit & Sensitivity)."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sim_data is None:
        sim_data = run_first_round_simulation(mode=mode)

    pdf_path = output_dir / "figure4_jan_election_first.pdf"
    png_path = output_dir / "figure4_jan_election_first.png"
    out_files = [
        pdf_path,
        png_path,
        output_dir / "figure4_janones_election_diagnosis.pdf",
        output_dir / "figure4_janones_election_diagnosis.png",
    ]

    render_master_figure(sim_data, out_files)
    print(f"[Figure 5] Saved: {pdf_path}")
    print(f"[Figure 5] Saved: {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Figure 5 First-Round simulation and generate plots.")
    parser.add_argument("--mode", type=str, choices=["fast", "full"], default="fast", help="Execution mode")
    parser.add_argument("--n-agents", type=int, default=None, help="Agent population size")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "figures"), help="Output directory")
    args = parser.parse_args()

    sim_data = run_first_round_simulation(mode=args.mode, n_agents=args.n_agents)
    generate_figure5(sim_data=sim_data, output_dir=args.output_dir, mode=args.mode)
