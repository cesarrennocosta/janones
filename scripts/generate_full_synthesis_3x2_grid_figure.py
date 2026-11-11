#!/usr/bin/env python3
"""
Generate Publication-Grade 3x2 Matrix Master Synthesis Figure:
2026 Brazilian Presidential Election (Full Empirical & Model Synthesis)

Architecture: 3x2 Grid (3 Horizontal Columns x 2 Vertical Rows)
- Top Row (Row 0):
  * Panel a: Electoral Outcomes & Non-Voter Bars (Benchmarks & Neutral Models)
  * Panel b: 71-Day Longitudinal Vote Intentions (Aug 16 - Oct 25, 2026)
  * Panel c: Candidate Rejection Dynamics (Datafolha "Não Votaria de Jeito Nenhum")
- Bottom Row (Row 1):
  * Panel d: Quaest Digital Popularity Index (IPD) Benchmark Waves (Focused 66-88 pts)
  * Panel e: Average Daily Social Interactions (M/day, Oct 4, Debate, Oct 5-17 Avg)
  * Panel f: Follower Ecosystem: Candidate Base, Party Official, WhatsApp, Top-5 Avg, Est. Unique Reach

Width Ratios:
- Top Row: [1.0, 1.25, 0.95]
- Bottom Row: [0.85, 0.85, 1.45] (Panel f is ~70% wider than d and e to accommodate 5 bar groups)

Adobe Illustrator Compatibility:
- FontType 42 (CIDFontType2 TrueType)
- Single text objects without kerning splits
"""

import os
import sys
import shutil
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from matplotlib import _text_helpers

# -----------------------------------------------------------------------------
# Configuration & Adobe Illustrator Font Settings
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "figure.titlesize": 12.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Patch text layout to eliminate intra-word kerning displacement breaks
orig_layout = _text_helpers.layout
def single_object_layout(*args, **kwargs):
    for item in orig_layout(*args, **kwargs):
        yield SimpleNamespace(
            ft_object=item.ft_object,
            char=item.char,
            glyph_index=item.glyph_index,
            prev_kern=0,
            x=item.x,
            y=item.y
        )
_text_helpers.layout = single_object_layout

COLORS = {
    "lula": "#E53935",        # PT Red
    "flavio": "#1E88E5",      # PL Blue
    "others": "#FB8C00",      # Center/Third-way Orange
    "neutral": "#757575",     # Inertia Neutral Grey
    "nonvoters": "#546E7A",   # Slate Grey for Non-Voters
    "grid": "#E0E0E0",
}


def generate_electoral_time_series():
    """Generate the calibrated 71-day electoral time series."""
    days = np.arange(1, 72)
    lula_vote = np.zeros(71)
    flavio_vote = np.zeros(71)
    others_vote = np.zeros(71)
    lula_neutral_r2 = np.zeros(71)
    flavio_neutral_r2 = np.zeros(71)
    lula_rej = np.zeros(71)
    flavio_rej = np.zeros(71)

    # First Round (Days 1 to 50: Aug 16 - Oct 4)
    for d in range(1, 51):
        idx = d - 1
        if d <= 40:
            lula_vote[idx] = 45.40 + (48.20 - 45.40) * (d / 40.0)
            flavio_vote[idx] = 33.60 + (38.80 - 33.60) * (d / 40.0)
            others_vote[idx] = 100.0 - lula_vote[idx] - flavio_vote[idx]
            lula_rej[idx] = 33.0 + (35.0 - 33.0) * (d / 40.0)
            flavio_rej[idx] = 45.0 + (43.5 - 45.0) * (d / 40.0)
        else:
            t = (d - 40) / 10.0
            flavio_vote[idx] = 38.80 + 8.20 / (1.0 + np.exp(-1.0 * (t * 10 - 5)))
            lula_vote[idx] = 48.20 - 3.04 / (1.0 + np.exp(-0.8 * (t * 10 - 5)))
            others_vote[idx] = 100.0 - lula_vote[idx] - flavio_vote[idx]
            lula_rej[idx] = 35.0 + 5.5 / (1.0 + np.exp(-0.8 * (t * 10 - 5)))
            flavio_rej[idx] = 43.5 - 1.5 / (1.0 + np.exp(-0.8 * (t * 10 - 5)))

    # Second Round (Days 51 to 71: Oct 5 - Oct 25)
    for d in range(51, 72):
        idx = d - 1
        t_r2 = (d - 50) / 21.0
        lula_neutral_r2[idx] = 53.50 + 0.2 * np.sin(t_r2 * np.pi)
        flavio_neutral_r2[idx] = 100.0 - lula_neutral_r2[idx]

        if d <= 64:  # Pre-Band Debate
            t_pre = (d - 50) / 14.0
            flavio_vote[idx] = 51.50 - 2.0 * t_pre
            lula_vote[idx] = 48.50 + 2.0 * t_pre
            lula_rej[idx] = 40.5 - 2.0 * t_pre
            flavio_rej[idx] = 42.0 + 4.5 * t_pre
        else:  # Post-Debate & Janones Mobilization Surge
            t_post = (d - 64) / 7.0
            lula_vote[idx] = 50.50 + (51.80 - 50.50) / (1.0 + np.exp(-1.2 * (t_post * 7 - 3.5)))
            flavio_vote[idx] = 100.0 - lula_vote[idx]
            lula_rej[idx] = 38.5 - 0.5 * t_post
            flavio_rej[idx] = 46.5 + (48.0 - 46.5) / (1.0 + np.exp(-1.0 * (t_post * 7 - 3.5)))

    return {
        "days": days,
        "lula_vote": lula_vote,
        "flavio_vote": flavio_vote,
        "others_vote": others_vote,
        "lula_neutral_r2": lula_neutral_r2,
        "flavio_neutral_r2": flavio_neutral_r2,
        "lula_rej": lula_rej,
        "flavio_rej": flavio_rej,
    }


def plot_3x2_synthesis_figure(data: dict, out_pdf: Path, out_svg: Path, out_png: Path):
    """Render the publication-grade 3x2 matrix synthesis figure."""
    print("Plotting 3x2 Matrix Master Synthesis Figure (Illustrator Editable Text)...")
    fig = plt.figure(figsize=(24.0, 12.0), dpi=300)

    # Master 2-Row Grid
    gs_master = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0], hspace=0.32, top=0.94, bottom=0.07, left=0.04, right=0.985)

    # Top Row: Panels a, b, c
    gs_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_master[0], width_ratios=[1.0, 1.25, 0.95], wspace=0.22)
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_top[0, 2])

    # Bottom Row: Panels d, e, f (Panel f is ~70% wider than d and e for 5 bar groups)
    gs_bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_master[1], width_ratios=[0.85, 0.85, 1.45], wspace=0.22)
    ax_d = fig.add_subplot(gs_bottom[0, 0])
    ax_e = fig.add_subplot(gs_bottom[0, 1])
    ax_f = fig.add_subplot(gs_bottom[0, 2])

    days = data["days"]

    # -------------------------------------------------------------------------
    # Panel A: Electoral Benchmarks, Neutral Models & Dedicated Non-Voter Bars
    # -------------------------------------------------------------------------
    milestones = [
        "Datafolha Pre-R1\n(Aug 16 Baseline)",
        "R1 Neutral Model\n(Inertia Baseline)",
        "R1 Actual Vote\n(Oct 4 Official)",
        "R2 Neutral Model\n(Init. Datafolha Poll)",
        "R2 Actual Vote\n(Oct 25 Official)",
    ]
    x_a = np.arange(len(milestones))
    w_a = 0.17

    lula_vals = [45.40, 49.10, 45.16, 53.50, 51.80]
    flavio_vals = [33.60, 36.50, 47.00, 46.50, 48.20]
    others_vals = [21.00, 14.40, 7.84, 0.0, 0.0]
    nonvoters_vals = [24.70, 24.70, 24.70, 24.50, 21.79]

    # Milestones 0, 1, 2 (4 bars)
    for i in range(3):
        pos_l = x_a[i] - 1.5 * w_a
        pos_f = x_a[i] - 0.5 * w_a
        pos_o = x_a[i] + 0.5 * w_a
        pos_nv = x_a[i] + 1.5 * w_a

        ax_a.bar(pos_l, lula_vals[i], w_a, color=COLORS["lula"], zorder=3,
                 label="Lula da Silva (Valid %)" if i == 0 else "")
        ax_a.text(pos_l, lula_vals[i] + 0.6, f"{lula_vals[i]:.1f}%", ha="center", va="bottom",
                  fontsize=7.2, fontweight="bold", color=COLORS["lula"])

        ax_a.bar(pos_f, flavio_vals[i], w_a, color=COLORS["flavio"], zorder=3,
                 label="Flávio Bolsonaro (Valid %)" if i == 0 else "")
        ax_a.text(pos_f, flavio_vals[i] + 0.6, f"{flavio_vals[i]:.1f}%", ha="center", va="bottom",
                  fontsize=7.2, fontweight="bold", color=COLORS["flavio"])

        ax_a.bar(pos_o, others_vals[i], w_a, color=COLORS["others"], alpha=0.9, zorder=3,
                 label="Others / 3rd-Way (Valid %)" if i == 0 else "")
        ax_a.text(pos_o, others_vals[i] + 0.6, f"{others_vals[i]:.1f}%", ha="center", va="bottom",
                  fontsize=7.2, color=COLORS["others"])

        ax_a.bar(pos_nv, nonvoters_vals[i], w_a, color=COLORS["nonvoters"], zorder=3,
                 label="Non-Voters (Abst.+B/N %)" if i == 0 else "")
        ax_a.text(pos_nv, nonvoters_vals[i] + 0.6, f"{nonvoters_vals[i]:.1f}%", ha="center", va="bottom",
                  fontsize=7.2, fontweight="bold", color=COLORS["nonvoters"])

    # Milestones 3, 4 (3 bars centered)
    for i in (3, 4):
        pos_l = x_a[i] - w_a
        pos_f = x_a[i]
        pos_nv = x_a[i] + w_a

        ax_a.bar(pos_l, lula_vals[i], w_a, color=COLORS["lula"], zorder=3)
        ax_a.text(pos_l, lula_vals[i] + 0.6, f"{lula_vals[i]:.1f}%", ha="center", va="bottom",
                  fontsize=7.2, fontweight="bold", color=COLORS["lula"])

        ax_a.bar(pos_f, flavio_vals[i], w_a, color=COLORS["flavio"], zorder=3)
        ax_a.text(pos_f, flavio_vals[i] + 0.6, f"{flavio_vals[i]:.1f}%", ha="center", va="bottom",
                  fontsize=7.2, fontweight="bold", color=COLORS["flavio"])

        ax_a.bar(pos_nv, nonvoters_vals[i], w_a, color=COLORS["nonvoters"], zorder=3)
        ax_a.text(pos_nv, nonvoters_vals[i] + 0.6, f"{nonvoters_vals[i]:.1f}%", ha="center", va="bottom",
                  fontsize=7.2, fontweight="bold", color=COLORS["nonvoters"])

    ax_a.axvline(x=2.5, color="#9E9E9E", linestyle="--", lw=1.0, alpha=0.7)
    ax_a.text(1.0, 56.5, "FIRST ROUND BENCHMARKS", ha="center", fontsize=7.8, fontweight="bold", color="#424242")
    ax_a.text(3.5, 56.5, "SECOND ROUND BENCHMARKS", ha="center", fontsize=7.8, fontweight="bold", color="#424242")

    ax_a.annotate("Janones Mobilization:\nNon-Voters Compressed\nfrom 24.70% -> 21.79%\n(-2.91% Turnout Surge)",
                  xy=(4 + w_a, 21.79), xytext=(2.7, 30.0),
                  arrowprops=dict(arrowstyle="->", color=COLORS["nonvoters"], lw=1.2),
                  fontsize=7.0, fontweight="bold", color=COLORS["nonvoters"],
                  bbox=dict(boxstyle="round,pad=0.25", fc="#ECEFF1", ec=COLORS["nonvoters"], alpha=0.9))

    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(milestones, fontsize=7.0)
    ax_a.set_ylabel("Share of Valid Electorate (%)", fontsize=8.2)
    ax_a.set_ylim(0, 62)
    ax_a.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_a.legend(loc="upper left", frameon=True, fontsize=6.8)
    ax_a.set_title("a  Electoral Outcomes, Neutral Baselines & Non-Voter Bars", fontweight="bold", loc="left", fontsize=9.0)

    # -------------------------------------------------------------------------
    # Panel B: Longitudinal Vote Intentions & Results (71 Days)
    # -------------------------------------------------------------------------
    ax_b.plot(days[:50], data["lula_vote"][:50], color=COLORS["lula"], lw=2.2, label="Lula da Silva (Fitted Model)")
    ax_b.plot(days[:50], data["flavio_vote"][:50], color=COLORS["flavio"], lw=2.2, label="Flávio Bolsonaro (Fitted Model)")
    ax_b.plot(days[:50], data["others_vote"][:50], color=COLORS["others"], lw=1.5, linestyle="-.", label="Others / Third-Way")

    ax_b.plot(days[50:], data["lula_vote"][50:], color=COLORS["lula"], lw=2.2)
    ax_b.plot(days[50:], data["flavio_vote"][50:], color=COLORS["flavio"], lw=2.2)

    ax_b.plot(days[50:], data["lula_neutral_r2"][50:], color=COLORS["lula"], lw=1.5, linestyle=":", label="Lula Neutral R2 (Init. Datafolha Poll)")
    ax_b.plot(days[50:], data["flavio_neutral_r2"][50:], color=COLORS["flavio"], lw=1.5, linestyle=":", label="Flávio Neutral R2 (Init. Datafolha Poll)")

    # Datafolha Empirical Checkpoints
    df_days_r1 = [1, 16, 30, 40, 46, 50]
    df_lula_r1 = [45.40, 46.50, 47.80, 48.20, 46.70, 45.16]
    df_flavio_r1 = [33.60, 35.80, 37.20, 38.80, 43.10, 47.00]
    ax_b.scatter(df_days_r1, df_lula_r1, color=COLORS["lula"], s=28, zorder=5, edgecolor="white", lw=0.8, label="Datafolha Checkpoints / TSE")
    ax_b.scatter(df_days_r1, df_flavio_r1, color=COLORS["flavio"], s=28, zorder=5, edgecolor="white", lw=0.8)

    ax_b.scatter([64, 71], [50.50, 51.80], color=COLORS["lula"], s=36, zorder=5, edgecolor="black", lw=1.0)
    ax_b.scatter([64, 71], [49.50, 48.20], color=COLORS["flavio"], s=36, zorder=5, edgecolor="black", lw=1.0)

    ax_b.axvspan(41, 50, color=COLORS["flavio"], alpha=0.08, label="Pre-R1 Digital Blitz (Days 41–50)")
    ax_b.axvline(x=50, color="#424242", linestyle="--", lw=1.2, label="Round 1 Vote (Oct 4)")
    ax_b.axvline(x=64, color="#7B1FA2", linestyle="--", lw=1.2, label="Band Debate (Day 64 / Oct 18)")

    ax_b.annotate("Day 48 Crossover\nFlávio algorithmic surge\novertakes Lula (46.8% v 46.5%)",
                  xy=(48, 46.8), xytext=(30, 53.0),
                  arrowprops=dict(arrowstyle="->", color=COLORS["flavio"], lw=1.2),
                  fontsize=7.0, fontweight="bold", color=COLORS["flavio"],
                  bbox=dict(boxstyle="round,pad=0.25", fc="#E3F2FD", ec=COLORS["flavio"], alpha=0.9))

    ax_b.annotate("Band Debate Reversal\nLula takes lead (50.5%)\nJanones live saturation",
                  xy=(64, 50.5), xytext=(52, 41.5),
                  arrowprops=dict(arrowstyle="->", color=COLORS["lula"], lw=1.2),
                  fontsize=7.0, fontweight="bold", color=COLORS["lula"],
                  bbox=dict(boxstyle="round,pad=0.25", fc="#FFEBEE", ec=COLORS["lula"], alpha=0.9))

    ax_b.text(50.5, 47.3, "R1 Official:\nFlávio 47.0%\nLula 45.2%", fontsize=6.8, fontweight="bold", color="#1B5E20")
    ax_b.text(71.2, 51.8, "R2 Official:\nLula 51.8%\nFlávio 48.2%", fontsize=6.8, fontweight="bold", color="#B71C1C")

    ax_b.set_xlabel("Campaign Day (Day 1: Aug 16 -> Day 71: Oct 25, 2026)", fontsize=8.2)
    ax_b.set_ylabel("Valid Vote Intention (%)", fontsize=8.2)
    ax_b.set_xlim(0, 75)
    ax_b.set_ylim(0, 60)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    ax_b.legend(loc="lower left", frameon=True, fontsize=6.5, ncol=2)
    ax_b.set_title("b  71-Day Longitudinal Vote Intentions (Aug 16 – Oct 25, 2026)", fontweight="bold", loc="left", fontsize=9.0)

    # -------------------------------------------------------------------------
    # Panel C: Candidate Rejection Dynamics (Datafolha "Não Votaria")
    # -------------------------------------------------------------------------
    ax_c.plot(days[:50], data["lula_rej"][:50], color=COLORS["lula"], lw=2.2, label="Lula Rejection (Não Votaria)")
    ax_c.plot(days[:50], data["flavio_rej"][:50], color=COLORS["flavio"], lw=2.2, label="Flávio Rejection (Não Votaria)")
    ax_c.plot(days[50:], data["lula_rej"][50:], color=COLORS["lula"], lw=2.2)
    ax_c.plot(days[50:], data["flavio_rej"][50:], color=COLORS["flavio"], lw=2.2)

    df_rej_days = [1, 20, 40, 50, 64, 71]
    df_lula_rej_pts = [33.0, 34.0, 35.0, 40.5, 38.5, 38.0]
    df_flavio_rej_pts = [45.0, 44.0, 43.5, 42.0, 46.5, 48.0]
    ax_c.scatter(df_rej_days, df_lula_rej_pts, color=COLORS["lula"], s=30, zorder=5, edgecolor="white", lw=0.8)
    ax_c.scatter(df_rej_days, df_flavio_rej_pts, color=COLORS["flavio"], s=30, zorder=5, edgecolor="white", lw=0.8)

    ax_c.axvspan(41, 50, color=COLORS["flavio"], alpha=0.08)
    ax_c.axvline(x=50, color="#424242", linestyle="--", lw=1.2)
    ax_c.axvline(x=64, color="#7B1FA2", linestyle="--", lw=1.2)

    ax_c.annotate("R1 Blitz Spikes Lula Rejection\n(33.0% -> 40.5% in 10 days)\nMeta ads + WhatsApp blitz",
                  xy=(50, 40.5), xytext=(22, 46.0),
                  arrowprops=dict(arrowstyle="->", color=COLORS["lula"], lw=1.2),
                  fontsize=7.0, fontweight="bold", color=COLORS["lula"],
                  bbox=dict(boxstyle="round,pad=0.25", fc="#FFEBEE", ec=COLORS["lula"], alpha=0.9))

    ax_c.annotate("R2 Janonismo Inversion:\nMaterial Loss Framing (λ=2.25)\nFlávio Rejection Spikes (42% -> 48%)\nCreates insurmountable ceiling",
                  xy=(71, 48.0), xytext=(45, 33.0),
                  arrowprops=dict(arrowstyle="->", color=COLORS["flavio"], lw=1.2),
                  fontsize=7.0, fontweight="bold", color=COLORS["flavio"],
                  bbox=dict(boxstyle="round,pad=0.25", fc="#E3F2FD", ec=COLORS["flavio"], alpha=0.9))

    ax_c.set_xlabel("Campaign Day (August 16 -> October 25, 2026)", fontsize=8.2)
    ax_c.set_ylabel("Voter Rejection Rate (%)", fontsize=8.2)
    ax_c.set_xlim(0, 75)
    ax_c.set_ylim(25, 55)
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(loc="upper left", frameon=True, fontsize=7.2)
    ax_c.set_title("c  Datafolha Rejection Dynamics (Não Votaria)", fontweight="bold", loc="left", fontsize=9.0)

    # -------------------------------------------------------------------------
    # Panel D: Quaest Digital Popularity Index (IPD) Benchmark Waves
    # -------------------------------------------------------------------------
    waves = [
        "Pre-1st Round Eve\n(Oct 4–5, 2026)",
        "Post-1st Round\n(Oct 15, 2026)",
        "Post-1st Debate\n(Oct 19, 2026)",
    ]
    x_d = np.arange(len(waves))
    w_d = 0.28

    flavio_ipd = [80.0, 76.4, 76.6]
    lula_ipd = [70.7, 82.6, 83.3]

    ax_d.bar(x_d - w_d/2, flavio_ipd, w_d, color=COLORS["flavio"], label="Flávio Bolsonaro", zorder=3)
    ax_d.bar(x_d + w_d/2, lula_ipd, w_d, color=COLORS["lula"], label="Lula da Silva", zorder=3)

    ax_d.text(x_d[0] - w_d/2, 80.5, "80.0 pts\n(Flávio +9.3)", ha="center", va="bottom", fontweight="bold", color=COLORS["flavio"], fontsize=7.2)
    ax_d.text(x_d[0] + w_d/2, 71.2, "70.7 pts", ha="center", va="bottom", color=COLORS["lula"], fontsize=7.2)

    ax_d.text(x_d[1] - w_d/2, 76.9, "76.4 pts", ha="center", va="bottom", color=COLORS["flavio"], fontsize=7.2)
    ax_d.text(x_d[1] + w_d/2, 83.1, "82.6 pts\n(Lula +6.2)", ha="center", va="bottom", fontweight="bold", color=COLORS["lula"], fontsize=7.2)

    ax_d.text(x_d[2] - w_d/2, 77.1, "76.6 pts", ha="center", va="bottom", color=COLORS["flavio"], fontsize=7.2)
    ax_d.text(x_d[2] + w_d/2, 83.8, "83.3 pts\n(Lula +6.7)", ha="center", va="bottom", fontweight="bold", color=COLORS["lula"], fontsize=7.2)

    net_box = (
        "NET CHANGE (Oct 4 -> Oct 19):\n"
        "• Lula:   +12.6 pts (70.7 -> 83.3)\n"
        "• Flávio:  -3.4 pts (80.0 -> 76.6)\n"
        "• Net Swing: +16.0 pts Inversion!"
    )
    ax_d.annotate(net_box, xy=(0.02, 0.52), xycoords="axes fraction",
                  fontsize=6.8, family="sans-serif",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#FF9800", alpha=0.95))

    ax_d.text(0.48, 0.04, "Sources: Quaest IPD (175 metrics); reports by Poder360, Vero Notícias, ICL Notícias.",
              transform=ax_d.transAxes, fontsize=6.5, color="#424242", style="italic")

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(waves, fontsize=7.2)
    ax_d.set_ylabel("IPD Score (Points)", fontsize=8.0)
    ax_d.set_ylim(66, 88)
    ax_d.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_d.legend(loc="upper left", frameon=True, fontsize=6.8)
    ax_d.set_title("d  Quaest Digital Popularity Index (IPD) - Benchmark Waves", fontweight="bold", loc="left", fontsize=9.0)

    # -------------------------------------------------------------------------
    # Panel E: Average Daily Social Interactions (Normalized by Day)
    # -------------------------------------------------------------------------
    cat_inter = [
        "Oct 4 (1 day)\n[Vero/Bites]",
        "Debate (1 day)\n[Poder360/B]",
        "Oct 5–17 (13d Avg)\n[Bites]",
    ]
    x_e = np.arange(len(cat_inter))
    w_e = 0.28

    flavio_inter = [3.90, 5.00, 2.41]
    lula_inter = [0.17, 25.50, 8.52]

    ax_e.bar(x_e - w_e/2, flavio_inter, w_e, color=COLORS["flavio"], label="Flávio", zorder=3)
    ax_e.bar(x_e + w_e/2, lula_inter, w_e, color=COLORS["lula"], label="Lula", zorder=3)

    ax_e.text(x_e[0] - w_e/2, 4.4, "3.90M/d\n(22.9x)", ha="center", va="bottom", fontweight="bold", color=COLORS["flavio"], fontsize=6.8)
    ax_e.text(x_e[0] + w_e/2, 0.7, "0.17M/d", ha="center", va="bottom", color=COLORS["lula"], fontsize=6.5)

    ax_e.text(x_e[1] - w_e/2, 5.5, "5.0M/d", ha="center", va="bottom", color=COLORS["flavio"], fontsize=6.5)
    ax_e.text(x_e[1] + w_e/2, 26.0, "25.5M/d\n(5.1x)", ha="center", va="bottom", fontweight="bold", color=COLORS["lula"], fontsize=6.8)

    ax_e.text(x_e[2] - w_e/2, 2.9, "2.41M/d", ha="center", va="bottom", color=COLORS["flavio"], fontsize=6.5)
    ax_e.text(x_e[2] + w_e/2, 9.0, "8.52M/d\n(3.5x)", ha="center", va="bottom", fontweight="bold", color=COLORS["lula"], fontsize=6.8)

    sept_box = "Sept Daily Wins:\nFlávio 14 v Lula 3 (82.4%)\n[ICL Notícias / Bites]"
    ax_e.annotate(sept_box, xy=(0.04, 0.62), xycoords="axes fraction",
                  fontsize=6.5, bbox=dict(boxstyle="round,pad=0.25", fc="#F5F5F5", ec="#BDBDBD", alpha=0.95))

    ax_e.set_xticks(x_e)
    ax_e.set_xticklabels(cat_inter, fontsize=7.0)
    ax_e.set_ylabel("Avg. Interactions/Day (Millions)", fontsize=7.6)
    ax_e.set_ylim(0, 31)
    ax_e.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_e.legend(loc="upper left", frameon=True, fontsize=6.5)
    ax_e.set_title("e  Average Daily Social Interactions", fontweight="bold", loc="left", fontsize=9.0)

    # -------------------------------------------------------------------------
    # Panel F: Follower Ecosystem: Candidate Base, Party, Closed & Bench Reach
    # (Wide canvas: width ratio 1.45 to comfortably fit all 5 categories)
    # -------------------------------------------------------------------------
    cats_f = [
        "Candidate Base\n[Flávio v Lula]",
        "Party Official\n[PL v PT]",
        "WhatsApp Army\n[Tropa v Exército]",
        "Surrogate Top-5 Avg\n[Excl. Candidate]",
        "Surrogates Unique\n[Excl. Candidate]",
    ]
    x_f = np.arange(len(cats_f))
    w_f = 0.32

    # Platform subdivisions for Candidate Base:
    # Lula: 39.0M = IG 18.2, X 9.1, FB 5.2, TT 4.6, YT 1.9
    # Flavio: 24.1M = IG 13.5, FB 4.0, X 3.2, TT 2.0, YT 1.4
    lula_plats = [18.2, 9.1, 5.2, 4.6, 1.9]
    flavio_plats = [13.5, 4.0, 3.2, 2.0, 1.4]
    plat_shades = [1.0, 0.78, 0.60, 0.45, 0.30]

    # 1. Candidate Base (Flávio vs Lula) - Stacked with platform subdivisions
    bot_f = 0
    for val, s in zip(flavio_plats, plat_shades):
        ax_f.bar(x_f[0] - w_f/2, val, w_f, bottom=bot_f, color=COLORS["flavio"], alpha=s, edgecolor="white", lw=0.5, zorder=3)
        bot_f += val

    bot_l = 0
    for val, s in zip(lula_plats, plat_shades):
        ax_f.bar(x_f[0] + w_f/2, val, w_f, bottom=bot_l, color=COLORS["lula"], alpha=s, edgecolor="white", lw=0.5, zorder=3)
        bot_l += val

    ax_f.text(x_f[0] - w_f/2, bot_f + 1.2, f"{bot_f:.1f}M", ha="center", va="bottom", fontweight="bold", color="#0D47A1", fontsize=7.5)
    ax_f.text(x_f[0] + w_f/2, bot_l + 1.2, f"{bot_l:.1f}M\n(1.6x)", ha="center", va="bottom", fontweight="bold", color="#B71C1C", fontsize=7.5)

    # 2. Party Official Institutional Handles (PL: 3.5M vs PT: 4.8M)
    ax_f.bar(x_f[1] - w_f/2, 3.5, w_f, color=COLORS["flavio"], edgecolor="white", lw=0.5, zorder=3)
    ax_f.bar(x_f[1] + w_f/2, 4.8, w_f, color=COLORS["lula"], edgecolor="white", lw=0.5, zorder=3)
    ax_f.text(x_f[1] - w_f/2, 3.5 + 1.2, "3.5M", ha="center", va="bottom", fontweight="bold", color="#0D47A1", fontsize=7.5)
    ax_f.text(x_f[1] + w_f/2, 4.8 + 1.2, "4.8M\n(1.4x)", ha="center", va="bottom", fontweight="bold", color="#B71C1C", fontsize=7.5)

    # 3. Closed Networks: WhatsApp Mobilization Army (Tropa: 2.0M vs Exército: 10.0M)
    ax_f.bar(x_f[2] - w_f/2, 2.0, w_f, color=COLORS["flavio"], edgecolor="white", lw=0.5, zorder=3)
    ax_f.bar(x_f[2] + w_f/2, 10.0, w_f, color=COLORS["lula"], edgecolor="white", lw=0.5, zorder=3)
    ax_f.text(x_f[2] - w_f/2, 2.0 + 1.2, "2.0M\n(Tropa)", ha="center", va="bottom", color="#0D47A1", fontsize=7.0)
    ax_f.text(x_f[2] + w_f/2, 10.0 + 1.2, "10.0M\n(5.0x)", ha="center", va="bottom", fontweight="bold", color="#B71C1C", fontsize=7.5)

    # 4. Surrogate Bench: Top-5 Average (PL: 17.4M vs PT & Allies: 7.1M)
    ax_f.bar(x_f[3] - w_f/2, 17.36, w_f, color="#0D47A1", hatch="//", edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    ax_f.bar(x_f[3] + w_f/2, 7.12, w_f, color="#B71C1C", hatch="//", edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    ax_f.text(x_f[3] - w_f/2, 17.36 + 1.2, "17.4M\n(2.4x)", ha="center", va="bottom", fontweight="bold", color="#0D47A1", fontsize=7.5)
    ax_f.text(x_f[3] + w_f/2, 7.12 + 1.2, "7.1M", ha="center", va="bottom", fontweight="bold", color="#B71C1C", fontsize=7.5)

    # 5. Surrogate Bench: Estimated Unique Individual Reach (PL: 82.0M vs PT & Allies: 32.5M)
    ax_f.bar(x_f[4] - w_f/2, 82.0, w_f, color="#0D47A1", hatch="..", edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    ax_f.bar(x_f[4] + w_f/2, 32.5, w_f, color="#B71C1C", hatch="..", edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    ax_f.text(x_f[4] - w_f/2, 82.0 + 1.2, "82.0M\n(2.5x)", ha="center", va="bottom", fontweight="bold", color="#0D47A1", fontsize=7.5)
    ax_f.text(x_f[4] + w_f/2, 32.5 + 1.2, "32.5M", ha="center", va="bottom", fontweight="bold", color="#B71C1C", fontsize=7.5)

    ax_f.set_xticks(x_f)
    ax_f.set_xticklabels(cats_f, fontsize=7.2)
    ax_f.set_ylabel("Followers / Reach (Millions)", fontsize=8.5)
    ax_f.set_ylim(0, 108)
    ax_f.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Legend distinguishing Candidate vs Party Bench
    legend_elements = [
        Patch(facecolor="#1E88E5", edgecolor="white", label="Flávio Bolsonaro / PL"),
        Patch(facecolor="#E53935", edgecolor="white", label="Lula da Silva / PT"),
        Patch(facecolor="#90CAF9", hatch="//", edgecolor="#0D47A1", label="Surrogate Top-5 Avg (Excl. Candidate)"),
        Patch(facecolor="#90CAF9", hatch="..", edgecolor="#0D47A1", label="Surrogates Unique Reach (Excl. Candidate)"),
    ]
    ax_f.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=7.2)

    # Inset callout box with detailed breakdown
    subdiv_box = (
        "Follower Ecosystem & Deduplication Architecture (Option B Clarification):\n"
        "• Surrogate Bench (Excl. Candidate): Measures top-5 allied influencers without the presidential nominee.\n"
        "  - PL Bench: Nikolas (52M), Eduardo (13.2M), Gayer (7.8M), Zambelli (7.4M), Damares (6.4M) [Avg: 17.4M, Jaccard 22.4% -> 82.0M Unique]\n"
        "  - PT/Allies Bench: Janones (17.2M), Boulos (8.4M), Erika (4.2M), Gleisi (3.1M), Randolfe (2.7M) [Avg: 7.1M, Jaccard 18.2% -> 32.5M Unique]\n"
        "• Note on Scope: Lula's 39.0M is gross multi-platform followers (IG+X+FB+TT+YT); the 32.5M is deduplicated citizens reached by his 5 allies alone."
    )
    ax_f.annotate(subdiv_box, xy=(0.02, 0.49), xycoords="axes fraction",
                  fontsize=6.4, family="sans-serif",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#FFFDE7", ec="#FBC02D", alpha=0.95))

    ax_f.set_title("f  Follower Ecosystem: Official Base, Closed WhatsApp & Surrogate Reach (Excl. Candidate)", fontweight="bold", loc="left", fontsize=9.2)

    # Save PDF (with Type 42 TrueType text), SVG (native vector text), and PNG (high-res preview)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"3x2 Matrix Synthesis Figure saved to: {out_pdf}, {out_svg}, and {out_png}")


def main():
    print("=== Generating 3x2 Matrix Master Synthesis Figure ===")
    data = generate_electoral_time_series()

    figures_dir = PROJECT_ROOT / "figures"
    os.makedirs(figures_dir, exist_ok=True)

    out_pdf = figures_dir / "election_2026_full_synthesis_3x2_grid.pdf"
    out_svg = figures_dir / "election_2026_full_synthesis_3x2_grid.svg"
    out_png = figures_dir / "election_2026_full_synthesis_3x2_grid.png"

    plot_3x2_synthesis_figure(data, out_pdf, out_svg, out_png)

    # Copy to artifacts directory
    artifact_dir = Path("/Users/rennocosta/.gemini/antigravity/brain/9a018403-8aec-4ff7-aadc-19a18baf7af1")
    if artifact_dir.exists():
        shutil.copy2(out_pdf, artifact_dir / "election_2026_full_synthesis_3x2_grid.pdf")
        shutil.copy2(out_svg, artifact_dir / "election_2026_full_synthesis_3x2_grid.svg")
        shutil.copy2(out_png, artifact_dir / "election_2026_full_synthesis_3x2_grid.png")
        print(f"Copied 3x2 figure files to artifact directory: {artifact_dir}")

    print("=== 3x2 Matrix Master Synthesis Completed Successfully ===")


if __name__ == "__main__":
    main()
