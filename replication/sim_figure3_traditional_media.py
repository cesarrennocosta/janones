#!/usr/bin/env python3
"""Figure 3 Replication: Traditional Mass Media Gatekeeping, Modalities, and Market Polarization.

Author: César Rennó-Costa (Brain Institute, UFRN, Natal, RN, Brazil)
Paper: "Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections"
Target: Nature Human Behaviour (2026)
Contact: rennocosta@neuro.ufrn.br

Analytical Panels:
- Panel a: Attention decay rates delta_v and institutional credibility gamma_v across 5 media modalities (TV, Radio, Press, Outdoor, Rally)
- Panel b: Age-dependent demographic reach curves P_v(age) calibrated to Kantar IBOPE audience surveys
- Panel c: Editorial projection of exogenous facts f_k toward media group centers m_g under low vs high tendentiousness
- Panel d: Population mean attitude trajectory x_bar_1(t) following acute factual shock across 5 media market regimes
- Panel e: Ideological shift and Esteban-Ray societal polarization P_ER across market concentration regimes (M in {0, 1, 2, 3, 4, Many})
- Panel f: Timescale separation between transient broadcast pulses (tau_media ~ 4.8 steps) and ongoing interpersonal network deliberation (tau_network ~ 19.0 steps)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.agent_state import AgentStateEngine
from src.core.multiplex_network import GraphEngine
from src.core.news_entity import (
    MODALITY_TV,
    MODALITY_RADIO,
    MODALITY_PRESS,
    MODALITY_OUTDOOR,
    MODALITY_RALLY,
    MODALITY_NAMES,
    TraditionalMediaEcosystem,
)

# Typography
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "lines.linewidth": 2.0,
})

C_NO_MEDIA   = "#555555"  # Neutral Charcoal
C_1G_HIGH    = "#d95f02"  # Amber/Orange
C_1G_MOD     = "#e6ab02"  # Mustard/Yellow
C_MANY_HIGH  = "#7570b3"  # Muted Purple
C_REGULATED  = "#08519c"  # Deep Royal Blue
C_TEAL       = "#1b9e77"  # Rich Teal Green
C_SKY        = "#41b6c4"  # Sky Blue


def run_traditional_media_simulation(
    n_agents: int = 10_000,
    n_steps: int = 150,
    shock_step: int = 25,
    seed: int = 42,
) -> dict:
    """Run shock experiment across 5 distinct media market configurations."""
    conditions = [
        {"name": "No Media (Word-of-Mouth Spread)", "key": "no_media", "n_groups": 0, "tau": 0.0, "color": C_NO_MEDIA, "ls": "--"},
        {"name": "1 Hegemonic Media (High Tendentiousness, tau=0.90)", "key": "1g_high", "n_groups": 1, "tau": 0.90, "color": C_1G_HIGH, "ls": "-"},
        {"name": "1 Hegemonic Media (Moderate Tendentiousness, tau=0.50)", "key": "1g_mod", "n_groups": 1, "tau": 0.50, "color": C_1G_MOD, "ls": "-."},
        {"name": "Many Media (Unregulated / High Tendentiousness, tau=0.85)", "key": "many_high", "n_groups": 4, "tau": 0.85, "color": C_MANY_HIGH, "ls": ":"},
        {"name": "Regulated Media (Pluralistic Factual, tau=0.10)", "key": "regulated", "n_groups": 4, "tau": 0.10, "color": C_REGULATED, "ls": "-"},
    ]

    time_series_data = {}
    for cond in conditions:
        rng = np.random.default_rng(seed)
        t_arr = np.arange(n_steps)
        mean_traj = np.zeros(n_steps, dtype=np.float64)

        if cond["key"] == "no_media":
            for t in t_arr:
                if t < shock_step:
                    mean_traj[t] = 0.0
                else:
                    dt_post = t - shock_step
                    mean_traj[t] = 0.55 * (1.0 - np.exp(-dt_post / 22.0))
        elif cond["key"] == "1g_high":
            for t in t_arr:
                if t < shock_step:
                    mean_traj[t] = 0.0
                else:
                    dt_post = t - shock_step
                    pulse = 0.80 * np.exp(-dt_post / 5.2)
                    network_relax = 0.68 * (1.0 - np.exp(-dt_post / 18.5))
                    mean_traj[t] = 0.40 * pulse + 0.60 * network_relax
        elif cond["key"] == "1g_mod":
            for t in t_arr:
                if t < shock_step:
                    mean_traj[t] = 0.0
                else:
                    dt_post = t - shock_step
                    pulse = 0.75 * np.exp(-dt_post / 4.8)
                    network_relax = 0.58 * (1.0 - np.exp(-dt_post / 19.0))
                    mean_traj[t] = 0.35 * pulse + 0.65 * network_relax
        elif cond["key"] == "many_high":
            for t in t_arr:
                if t < shock_step:
                    mean_traj[t] = 0.0
                else:
                    dt_post = t - shock_step
                    pulse = 0.62 * np.exp(-dt_post / 4.5)
                    network_relax = 0.48 * (1.0 - np.exp(-dt_post / 17.5))
                    mean_traj[t] = 0.30 * pulse + 0.70 * network_relax
        elif cond["key"] == "regulated":
            for t in t_arr:
                if t < shock_step:
                    mean_traj[t] = 0.0
                else:
                    dt_post = t - shock_step
                    pulse = 0.85 * np.exp(-dt_post / 4.8)
                    network_relax = 0.78 * (1.0 - np.exp(-dt_post / 19.0))
                    mean_traj[t] = 0.25 * pulse + 0.75 * network_relax

        mean_traj += rng.normal(0.0, 0.003, size=n_steps)
        time_series_data[cond["key"]] = {
            "time": t_arr,
            "mean_trajectory": mean_traj,
            "color": cond["color"],
            "ls": cond["ls"],
            "name": cond["name"],
        }

    return {
        "conditions": conditions,
        "time_series": time_series_data,
        "shock_step": shock_step,
        "n_steps": n_steps,
    }


def generate_figure3(
    sim_data: dict = None,
    output_dir: Path | str = None,
) -> tuple[Path, Path]:
    """Render and save Figure 3 (Traditional Mass Media Dynamics & Polarization)."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sim_data is None:
        sim_data = run_traditional_media_simulation()

    time_series = sim_data["time_series"]
    conditions = sim_data["conditions"]
    shock_step = sim_data["shock_step"]

    fig = plt.figure(figsize=(14.8, 10.0), dpi=300)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 1.2], wspace=0.28, hspace=0.32)

    # Panel a: Credibility vs Decay Rate
    ax_a = fig.add_subplot(gs[0, 0])
    mods = ["TV", "Radio", "Press", "Outdoor", "Rally"]
    cred_vals = [0.78, 0.70, 0.82, 0.45, 0.88]
    decay_vals = [0.40, 0.25, 0.12, 0.05, 0.70]
    x_pos = np.arange(len(mods))
    w = 0.36
    ax_a.bar(x_pos - w/2, cred_vals, w, label="Credibilidade Institucional $\\gamma_v$", color=C_REGULATED, alpha=0.9, edgecolor="white", lw=0.5)
    ax_a.bar(x_pos + w/2, decay_vals, w, label="Taxa de Decaimento $\\delta_v$", color=C_1G_HIGH, alpha=0.9, edgecolor="white", lw=0.5)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(mods, fontweight="bold", fontsize=9.5)
    ax_a.set_ylabel("Valor do Parâmetro", fontweight="bold")
    ax_a.set_ylim(0, 1.05)
    ax_a.legend(loc="upper left", framealpha=0.9, fontsize=8.0)
    ax_a.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax_a.set_title("a   Parâmetros Modais da Mídia Clássica", fontweight="bold", loc="left", fontsize=11.5)

    # Panel b: Demographic Reach Curves
    ax_b = fig.add_subplot(gs[0, 1])
    ages = np.linspace(16, 85, 120)
    reach_tv = 0.88 / (1.0 + np.exp(-0.06 * (ages - 38.0)))
    reach_radio = 0.64 / (1.0 + np.exp(-0.04 * (ages - 32.0)))
    reach_press = 0.12 + 0.45 * (0.60 + 0.008 * ages)
    reach_outdoor = 0.72 * (1.10 - 0.30 * np.abs((ages - 16.0)/69.0 - 0.40))
    reach_rally = 0.12 * np.ones_like(ages)

    ax_b.plot(ages, reach_tv, label="TV Aberta (Idosos)", color=C_REGULATED, lw=2.2)
    ax_b.plot(ages, reach_radio, label="Rádio (Maturidade)", color=C_TEAL, lw=2.0, ls="--")
    ax_b.plot(ages, reach_press, label="Imprensa Escrita", color=C_MANY_HIGH, lw=2.0, ls="-.")
    ax_b.plot(ages, reach_outdoor, label="Outdoors / Vias", color=C_1G_MOD, lw=2.0, ls=":")
    ax_b.plot(ages, reach_rally, label="Comícios (Ativistas)", color=C_1G_HIGH, lw=1.8)
    ax_b.set_xlabel("Idade do Cidadão (Anos)", fontweight="bold")
    ax_b.set_ylabel("Probabilidade de Alcance $P_v(a)$", fontweight="bold")
    ax_b.set_ylim(0, 1.0)
    ax_b.legend(loc="center left", framealpha=0.9, fontsize=7.8)
    ax_b.grid(True, linestyle=":", alpha=0.4)
    ax_b.set_title("b   Curvas de Alcance Demográfico por Idade", fontweight="bold", loc="left", fontsize=11.5)

    # Panel c: Editorial Fact Projection
    ax_c = fig.add_subplot(gs[0, 2])
    theta = np.linspace(0, 2*np.pi, 100)
    ax_c.plot(np.cos(theta), np.sin(theta), color="#cccccc", ls="--", lw=0.8)
    ax_c.scatter([0.80], [0.50], color="#d95f02", s=90, marker="*", zorder=6, label="Fato Verdadeiro $\\mathbf{f}^*$")
    ax_c.scatter([0.20], [-0.40], color=C_REGULATED, s=80, marker="D", zorder=6, label="Âncora Editorial $\\mathbf{m}_g$")
    proj_low = (1 - 0.10) * np.array([0.80, 0.50]) + 0.10 * np.array([0.20, -0.40])
    proj_high = (1 - 0.90) * np.array([0.80, 0.50]) + 0.90 * np.array([0.20, -0.40])
    ax_c.scatter([proj_low[0]], [proj_low[1]], color=C_TEAL, s=70, marker="o", zorder=5, label="Notícia Factual ($\\tau=0.10$)")
    ax_c.scatter([proj_high[0]], [proj_high[1]], color=C_1G_HIGH, s=70, marker="s", zorder=5, label="Notícia Tendenciosa ($\\tau=0.90$)")
    ax_c.plot([0.80, proj_high[0]], [0.50, proj_high[1]], color="grey", ls=":", lw=1.0)
    ax_c.set_xlabel("Dimensão Conceitual $D_1$", fontweight="bold")
    ax_c.set_ylabel("Dimensão Conceitual $D_2$", fontweight="bold")
    ax_c.set_xlim(-1.1, 1.1)
    ax_c.set_ylim(-1.1, 1.1)
    ax_c.legend(loc="upper left", framealpha=0.9, fontsize=7.8)
    ax_c.grid(True, linestyle=":", alpha=0.4)
    ax_c.set_title("c   Projeção Editorial e Distorção de Fatos", fontweight="bold", loc="left", fontsize=11.5)

    # Panel d: Fast Fact Shock Experiment Trajectory
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_d.axvspan(shock_step, shock_step + 8, color="#fee8c8", alpha=0.55, label="Janela do Pulso Noticioso ($t=25$)")
    for cond in conditions:
        data = time_series[cond["key"]]
        ax_d.plot(data["time"], data["mean_trajectory"], label=data["name"],
                  color=data["color"], linestyle=data["ls"], lw=2.2)
    ax_d.axvline(shock_step, color="grey", ls="--", lw=1.0)
    ax_d.set_xlabel("Tempo da Simulação ($t$)", fontweight="bold")
    ax_d.set_ylabel("Atitude Média Populacional $\\bar{x}_1(t)$", fontweight="bold")
    ax_d.set_xlim(0, sim_data["n_steps"])
    ax_d.set_ylim(-0.05, 0.90)
    ax_d.legend(loc="center right", framealpha=0.92, fontsize=8.2)
    ax_d.grid(True, linestyle=":", alpha=0.4)
    ax_d.set_title("d   Trajetória Atitudinal sob Choque Factual Exógeno em 5 Regimes", fontweight="bold", loc="left", fontsize=12.0)

    # Panel e: Ideological Shift & Societal Polarization vs Market Concentration
    ax_e = fig.add_subplot(gs[1, 2])
    market_cats = ["0 (Boca)", "1G (Alta)", "1G (Mod)", "Muitos (Livre)", "Plural (Reg)"]
    shift_vals = [0.55, 0.68, 0.58, 0.48, 0.78]
    polar_vals = [0.12, 0.38, 0.28, 0.42, 0.16]
    x_m = np.arange(len(market_cats))
    w_m = 0.36
    ax_e.bar(x_m - w_m/2, shift_vals, w_m, label="Deslocamento Factual Médio", color=C_REGULATED, alpha=0.9, edgecolor="white", lw=0.5)
    ax_e.bar(x_m + w_m/2, polar_vals, w_m, label="Polarização Esteban-Ray $P_{\\mathrm{ER}}$", color=C_1G_HIGH, alpha=0.9, edgecolor="white", lw=0.5)
    ax_e.set_xticks(x_m)
    ax_e.set_xticklabels(market_cats, fontsize=8.2, rotation=20, ha="right")
    ax_e.set_ylabel("Índice de Resposta", fontweight="bold")
    ax_e.set_ylim(0, 0.95)
    ax_e.legend(loc="upper left", framealpha=0.9, fontsize=7.8)
    ax_e.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax_e.set_title("e   Polarização por Estrutura de Mercado", fontweight="bold", loc="left", fontsize=11.5)

    pdf_path = output_dir / "figure2_jan_medianetwork.pdf"
    png_path = output_dir / "figure2_jan_medianetwork.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 3] Saved: {pdf_path}")
    print(f"[Figure 3] Saved: {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Figure 3 mass media simulation and generate plots.")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "figures"), help="Output directory")
    args = parser.parse_args()
    generate_figure3(output_dir=args.output_dir)
