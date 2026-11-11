#!/usr/bin/env python3
"""
Generate Publication-Grade 3x3 Master Synthesis Figure:
2026 Brazilian Presidential Election (Full Empirical, Media, Finance & Model Synthesis)

Optimized for:
- Full-width A4 scientific paper insertion (width: 182 mm / 7.17 in; height: 232 mm / 9.13 in)
- Typography: Helvetica Neue, base font size 6.0 pt
- Adobe Illustrator Compatibility: FontType 42 (CIDFontType2 TrueType), single text objects

Architecture: 3x3 Grid (3 Rows x 3 Columns = 9 Analytical Panels)
- Row 1: Electoral & Behavioral Dynamics (Urnas & Longitudinal Polls)
  * Panel a: Official TSE Outcomes & Voter Turnout Mobilization
  * Panel b: 71-Day Longitudinal Polling Trajectory (Aug 16 – Oct 25, 2026)
  * Panel c: Candidate Rejection Dynamics (Datafolha "Não Votaria")
- Row 2: Institutional Resources & Campaign Finance (Media & War Chests)
  * Panel d: Free TV & Radio Airtime Share (HGPE 1st & 2nd Rounds)
  * Panel e: Macro Party Public Funding (FEFC) vs TSE Spending Ceilings
  * Panel f: Presidential Campaign Receipts: Public Injections vs. Private Capital
- Row 3: Digital Hegemony & Mobilization Ecosystem (Algorithms & Networks)
  * Panel g: Quaest Digital Popularity Index (IPD) Benchmark Waves
  * Panel h: Average Daily Social Interactions (M/day)
  * Panel i: Follower Ecosystem & Deduplicated Reach (Option B Clarified)
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
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

# Strict typography: Helvetica Neue, 6pt base for A4 paper insertion
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6.0,
    "axes.titlesize": 6.5,
    "axes.labelsize": 6.0,
    "xtick.labelsize": 5.2,
    "ytick.labelsize": 5.2,
    "legend.fontsize": 5.0,
    "figure.titlesize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.35,
    "lines.linewidth": 0.8,
    "lines.markersize": 3.8,
})

# Single text object patch for Adobe Illustrator compatibility
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
    "lula": "#D32F2F",         # PT Red
    "flavio": "#1976D2",       # PL Blue
    "others": "#F57C00",       # Third-Way / Center Orange
    "neutral": "#616161",      # Inertia / Baseline Grey
    "nonvoters": "#455A64",    # Slate Grey for Abstention / Blank / Null
    "public_fund": "#388E3C",  # Green for Public Funds
    "private_fund": "#7B1FA2", # Purple for Private Capital
    "grid": "#E0E0E0",
}


def build_master_3x3_figure():
    # A4 text width ~ 182 mm (7.165 in); height ~ 232 mm (9.134 in)
    fig = plt.figure(figsize=(7.16, 9.15), dpi=300)

    # Master Grid: 3 rows with optimized vertical spacing
    gs_master = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.0, 1.05],
                         hspace=0.28, top=0.965, bottom=0.045, left=0.065, right=0.985)

    # Row 0: Panels a, b, c
    gs_row0 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_master[0], width_ratios=[0.95, 1.15, 0.90], wspace=0.25)
    ax_a = fig.add_subplot(gs_row0[0, 0])
    ax_b = fig.add_subplot(gs_row0[0, 1])
    ax_c = fig.add_subplot(gs_row0[0, 2])

    # Row 1: Panels d, e, f
    gs_row1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_master[1], width_ratios=[1.0, 1.0, 1.0], wspace=0.25)
    ax_d = fig.add_subplot(gs_row1[0, 0])
    ax_e = fig.add_subplot(gs_row1[0, 1])
    ax_f = fig.add_subplot(gs_row1[0, 2])

    # Row 2: Panels g, h, i
    gs_row2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_master[2], width_ratios=[0.85, 0.85, 1.30], wspace=0.25)
    ax_g = fig.add_subplot(gs_row2[0, 0])
    ax_h = fig.add_subplot(gs_row2[0, 1])
    ax_i = fig.add_subplot(gs_row2[0, 2])

    # =========================================================================
    # ROW 1 - PANEL A: Official TSE Election Outcomes & Mobilization
    # =========================================================================
    rounds = ["TSE 1º Turno\n(04/Out)", "TSE 2º Turno\n(25/Out)", "Não-Votantes\n(Abst. + B/N)"]
    x_a = np.arange(len(rounds))
    w_a = 0.26

    # 1º Turno TSE Official: Flávio 47.03%, Lula 45.16%, Outros 7.81%
    ax_a.bar(x_a[0] - w_a, 45.16, w_a, color=COLORS["lula"], zorder=3, label="Lula (PT)")
    ax_a.bar(x_a[0], 47.03, w_a, color=COLORS["flavio"], zorder=3, label="Flávio (PL)")
    ax_a.bar(x_a[0] + w_a, 7.81, w_a, color=COLORS["others"], zorder=3, label="Outros / 3ª Via")
    ax_a.text(x_a[0] - w_a, 45.9, "45.16%", ha="center", va="bottom", fontsize=5.0, fontweight="bold", color=COLORS["lula"])
    ax_a.text(x_a[0], 47.8, "47.03%", ha="center", va="bottom", fontsize=5.0, fontweight="bold", color=COLORS["flavio"])
    ax_a.text(x_a[0] + w_a, 8.5, "7.81%", ha="center", va="bottom", fontsize=4.8, color=COLORS["others"])

    # 2º Turno TSE Official: Lula 51.79%, Flávio 48.21%
    ax_a.bar(x_a[1] - w_a/2, 51.79, w_a, color=COLORS["lula"], zorder=3)
    ax_a.bar(x_a[1] + w_a/2, 48.21, w_a, color=COLORS["flavio"], zorder=3)
    ax_a.text(x_a[1] - w_a/2, 52.6, "51.79%", ha="center", va="bottom", fontsize=5.0, fontweight="bold", color=COLORS["lula"])
    ax_a.text(x_a[1] + w_a/2, 49.0, "48.21%", ha="center", va="bottom", fontsize=5.0, fontweight="bold", color=COLORS["flavio"])

    # Turnout Dynamics (Non-Voters 1ºT: 24.70% -> 2ºT: 21.79%, -2.91% net drop)
    ax_a.bar(x_a[2] - w_a/2, 24.70, w_a, color=COLORS["nonvoters"], alpha=0.6, zorder=3, label="Não-Votantes 1ºT")
    ax_a.bar(x_a[2] + w_a/2, 21.79, w_a, color=COLORS["nonvoters"], zorder=3, label="Não-Votantes 2ºT")
    ax_a.text(x_a[2] - w_a/2, 25.4, "24.70%\n(1ºT)", ha="center", va="bottom", fontsize=4.6, color=COLORS["nonvoters"])
    ax_a.text(x_a[2] + w_a/2, 22.5, "21.79%\n(2ºT)", ha="center", va="bottom", fontsize=4.6, fontweight="bold", color=COLORS["nonvoters"])

    # Turnout surge annotation
    ax_a.annotate("Queda Não-Votantes\n-2,91 pp (Mobilização)",
                  xy=(x_a[2] + w_a/2, 21.79), xytext=(x_a[2] - 0.52, 35.0),
                  arrowprops=dict(arrowstyle="->", color=COLORS["nonvoters"], lw=0.7),
                  fontsize=4.6, fontweight="bold", color=COLORS["nonvoters"],
                  bbox=dict(boxstyle="round,pad=0.2", fc="#ECEFF1", ec=COLORS["nonvoters"], alpha=0.9))

    ax_a.axvline(x=1.5, color="#BDBDBD", linestyle="--", lw=0.6)
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(rounds, fontsize=5.0)
    ax_a.set_ylabel("Participação Eleitoral (%)", fontsize=5.8)
    ax_a.set_ylim(0, 65)
    ax_a.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_a.legend(loc="upper left", frameon=True, fontsize=4.6, ncol=2)
    ax_a.set_title("a  Resultados Oficiais TSE & Mobilização", fontweight="bold", loc="left", fontsize=6.5)

    # =========================================================================
    # ROW 1 - PANEL B: 71-Day Longitudinal Polling Trajectory (Official Datafolha & TSE)
    # =========================================================================
    # Exact Official Datafolha tracking polls from database:
    # 1º Turno (6 polls): 19/Ago, 02/Set, 10/Set, 17/Set, 23/Set, 01/Out
    # 2º Turno (4 polls): 08/Out, 15/Out, 22/Out, 24/Out (Véspera: Flávio 51% vs Lula 49%)
    days_1t = [4, 18, 26, 33, 39, 47]   # Campaign days (base: 16/Ago = 1)
    df_lula_1t = [44.3, 41.8, 43.3, 43.0, 44.9, 45.0]
    df_flav_1t = [37.5, 36.3, 37.8, 39.6, 40.4, 41.0]
    df_outr_1t = [18.2, 22.0, 18.9, 17.4, 14.7, 14.0]

    days_2t = [54, 61, 68, 70]          # 08/Out, 15/Out, 22/Out, 24/Out (Véspera)
    df_lula_2t = [48.0, 49.0, 50.0, 49.0]
    df_flav_2t = [52.0, 51.0, 50.0, 51.0]

    day_urna_1t = 50                     # 04/Out (TSE Urnas 1ºT: Flávio 47.03%, Lula 45.16%, Outros 7.81%)
    day_urna_2t = 71                     # 25/Out (TSE Urnas 2ºT: Lula 51.79%, Flávio 48.21%)

    # Dashed connection lines for 1º Turno
    ax_b.plot(days_1t, df_lula_1t, color=COLORS["lula"], linestyle="--", lw=0.9, alpha=0.85)
    ax_b.plot(days_1t, df_flav_1t, color=COLORS["flavio"], linestyle="--", lw=0.9, alpha=0.85)
    ax_b.plot(days_1t, df_outr_1t, color=COLORS["others"], linestyle="--", lw=0.8, alpha=0.7)

    # Dashed connection lines for 2º Turno
    ax_b.plot(days_2t, df_lula_2t, color=COLORS["lula"], linestyle="--", lw=0.9, alpha=0.85)
    ax_b.plot(days_2t, df_flav_2t, color=COLORS["flavio"], linestyle="--", lw=0.9, alpha=0.85)

    # Scatter points 1º Turno
    ax_b.scatter(days_1t, df_lula_1t, color=COLORS["lula"], marker="o", s=22, zorder=4, edgecolor="white", lw=0.6, label="Lula (Datafolha)")
    ax_b.scatter(days_1t, df_flav_1t, color=COLORS["flavio"], marker="^", s=24, zorder=4, edgecolor="white", lw=0.6, label="Flávio (Datafolha)")
    ax_b.scatter(days_1t, df_outr_1t, color=COLORS["others"], marker="d", s=18, zorder=4, edgecolor="white", lw=0.5, label="Outros (Datafolha)")

    # Scatter points 2º Turno (at Day 68 / 22/Out tie 50%/50%, offset slightly to show both markers)
    ax_b.scatter([54, 61, 67.6, 70], [48.0, 49.0, 50.0, 49.0], color=COLORS["lula"], marker="o", s=22, zorder=4, edgecolor="white", lw=0.6)
    ax_b.scatter([54, 61, 68.4, 70], [52.0, 51.0, 50.0, 51.0], color=COLORS["flavio"], marker="^", s=24, zorder=4, edgecolor="white", lw=0.6)

    # Official TSE Urna checkpoints (Dourado com borda da cor do candidato)
    ax_b.scatter([day_urna_1t, day_urna_2t], [45.16, 51.79], color="#FFD700", marker="o", s=32, zorder=5, edgecolor=COLORS["lula"], lw=1.0, label="TSE Urnas (Lula)")
    ax_b.scatter([day_urna_1t, day_urna_2t], [47.03, 48.21], color="#FFD700", marker="^", s=34, zorder=5, edgecolor=COLORS["flavio"], lw=1.0, label="TSE Urnas (Flávio)")

    # Annotation of the Eve poll vs Urnas turnaround
    ax_b.annotate("Véspera Datafolha:\nFlávio 51% x Lula 49%\n(Virada de +2.79% nas urnas)",
                  xy=(70, 49.0), xytext=(38, 28.0),
                  arrowprops=dict(arrowstyle="->", color=COLORS["lula"], lw=0.7),
                  fontsize=4.6, fontweight="bold", color=COLORS["lula"],
                  bbox=dict(boxstyle="round,pad=0.2", fc="#FFEBEE", ec=COLORS["lula"], alpha=0.9))

    ax_b.axvline(x=50, color="#424242", linestyle=":", lw=0.7)
    ax_b.text(50, 56.5, "1ºT (04/Out)", ha="center", fontsize=4.8, fontweight="bold", color="#424242")

    ax_b.set_xlabel("Dia da Campanha (19/Ago -> 25/Out)", fontsize=5.8)
    ax_b.set_ylabel("Intenção de Voto Válido (%)", fontsize=5.8)
    ax_b.set_xlim(0, 75)
    ax_b.set_ylim(0, 60)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    ax_b.legend(loc="lower left", frameon=True, fontsize=4.5, ncol=2)
    ax_b.set_title("b  Trajetória Longitudinal (Datafolha & TSE)", fontweight="bold", loc="left", fontsize=6.5)

    # =========================================================================
    # ROW 1 - PANEL C: Candidate Rejection Dynamics (Official Datafolha)
    # =========================================================================
    # Exact Official Datafolha Rejection dataset:
    # "Em quais desses candidatos você não votaria de jeito nenhum?"
    # 1ºT: 19/Ago (45/46), 02/Set (45/47), 10/Set (46/46), 17/Set (47/47), 23/Set (45/45), 01/Out (45/45)
    # 2ºT: 08/Out (47/43), 15/Out (46/44), 22/Out (45/46), 24/Out (46/45)
    df_rej_days_1t = [4, 18, 26, 33, 39, 47]
    df_lula_rej_1t = [45.0, 45.0, 46.0, 47.0, 45.0, 45.0]
    df_flav_rej_1t = [46.0, 47.0, 46.0, 47.0, 45.0, 45.0]

    df_rej_days_2t = [54, 61, 68, 70]
    df_lula_rej_2t = [47.0, 46.0, 45.0, 46.0]
    df_flav_rej_2t = [43.0, 44.0, 46.0, 45.0]

    # Dashed connection lines
    ax_c.plot(df_rej_days_1t, df_lula_rej_1t, color=COLORS["lula"], linestyle="--", lw=0.9, alpha=0.85)
    ax_c.plot(df_rej_days_1t, df_flav_rej_1t, color=COLORS["flavio"], linestyle="--", lw=0.9, alpha=0.85)
    ax_c.plot(df_rej_days_2t, df_lula_rej_2t, color=COLORS["lula"], linestyle="--", lw=0.9, alpha=0.85)
    ax_c.plot(df_rej_days_2t, df_flav_rej_2t, color=COLORS["flavio"], linestyle="--", lw=0.9, alpha=0.85)

    # Scatter points with slight offset at exact ties so both circle and triangle are visible:
    # 19/Ago: 45 Lula, 46 Flavio (no tie)
    # 02/Set: 45 Lula, 47 Flavio (no tie)
    # 10/Set: 46 Tie
    # 17/Set: 47 Tie
    # 23/Set: 45 Tie
    # 01/Out: 45 Tie
    x_lula_1t = [4, 18, 25.6, 32.6, 38.6, 46.6]
    x_flav_1t = [4, 18, 26.4, 33.4, 39.4, 47.4]

    ax_c.scatter(x_lula_1t, df_lula_rej_1t, color=COLORS["lula"], marker="o", s=22, zorder=4, edgecolor="white", lw=0.6, label="Rejeição Lula")
    ax_c.scatter(x_flav_1t, df_flav_rej_1t, color=COLORS["flavio"], marker="^", s=24, zorder=4, edgecolor="white", lw=0.6, label="Rejeição Flávio")

    # 2º Turno points (08/Out 47/43, 15/Out 46/44, 22/Out 45/46, 24/Out 46/45)
    ax_c.scatter(df_rej_days_2t, df_lula_rej_2t, color=COLORS["lula"], marker="o", s=22, zorder=4, edgecolor="white", lw=0.6)
    ax_c.scatter(df_rej_days_2t, df_flav_rej_2t, color=COLORS["flavio"], marker="^", s=24, zorder=4, edgecolor="white", lw=0.6)

    ax_c.axvline(x=50, color="#424242", linestyle=":", lw=0.7)

    ax_c.annotate("Janonismo no 2ºT:\nRejeição de Flávio\nreverte de 43% -> 46%\n(perda da lua de mel)",
                  xy=(68, 46.0), xytext=(44, 38.5),
                  arrowprops=dict(arrowstyle="->", color=COLORS["flavio"], lw=0.7),
                  fontsize=4.6, fontweight="bold", color=COLORS["flavio"],
                  bbox=dict(boxstyle="round,pad=0.2", fc="#E3F2FD", ec=COLORS["flavio"], alpha=0.9))

    ax_c.set_xlabel("Dia da Campanha (19/Ago -> 25/Out)", fontsize=5.8)
    ax_c.set_ylabel("Rejeição Datafolha (%)", fontsize=5.8)
    ax_c.set_xlim(0, 75)
    ax_c.set_ylim(35, 55)
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(loc="upper left", frameon=True, fontsize=4.6)
    ax_c.set_title("c  Dinâmica da Rejeição Eleitoral", fontweight="bold", loc="left", fontsize=6.5)

    # =========================================================================
    # ROW 2 - PANEL D: TV & Radio Free Airtime Share (HGPE)
    # =========================================================================
    hgpe_stages = ["1º Turno (Bloco 12m30s)\n[28/Ago - 01/Out]", "2º Turno (Bloco 10m00s)\n[10/Out - 24/Out]"]
    x_d = np.arange(len(hgpe_stages))
    w_d = 0.28

    # 1º Turno: Lula 331s (5m31s, 44.1%), Flavio 260s (4m20s, 34.7%), Others 159s (2m39s, 21.2%)
    # In Minutes: Lula 5.517m, Flavio 4.333m, Others 2.65m
    ax_d.bar(x_d[0] - w_d, 5.517, w_d, color=COLORS["lula"], zorder=3, label="Lula / Coligação")
    ax_d.bar(x_d[0], 4.333, w_d, color=COLORS["flavio"], zorder=3, label="Flávio / PL")
    ax_d.bar(x_d[0] + w_d, 2.650, w_d, color=COLORS["others"], zorder=3, label="Outros Candidatos")
    ax_d.text(x_d[0] - w_d, 5.65, "5m31s\n(44.1%)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["lula"])
    ax_d.text(x_d[0], 4.45, "4m20s\n(34.7%)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["flavio"])
    ax_d.text(x_d[0] + w_d, 2.75, "2m39s\n(21.2%)", ha="center", va="bottom", fontsize=4.6, color=COLORS["others"])

    # 2º Turno: Parity 5m00s each (50% x 50%)
    ax_d.bar(x_d[1] - w_d/2, 5.0, w_d, color=COLORS["lula"], zorder=3)
    ax_d.bar(x_d[1] + w_d/2, 5.0, w_d, color=COLORS["flavio"], zorder=3)
    ax_d.text(x_d[1] - w_d/2, 5.25, "5m00s\n(50%)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["lula"])
    ax_d.text(x_d[1] + w_d/2, 5.25, "5m00s\n(50%)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["flavio"])

    spots_box = (
        "Inserções 30s Diárias:\n"
        "• 1ºT: Lula 433 vs Flávio 339 (+27.7% Lula)\n"
        "• 2ºT: 25 inserções/dia cada (Paridade Legal)"
    )
    ax_d.annotate(spots_box, xy=(0.04, 0.18), xycoords="axes fraction",
                  fontsize=4.6, bbox=dict(boxstyle="round,pad=0.2", fc="#FFFDE7", ec="#FBC02D", alpha=0.95))

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(hgpe_stages, fontsize=5.2)
    ax_d.set_ylabel("Tempo de Antena por Bloco (Minutos)", fontsize=5.8)
    ax_d.set_ylim(0, 7.5)
    ax_d.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_d.legend(loc="upper right", frameon=True, fontsize=4.6)
    ax_d.set_title("d  Horário Gratuito de TV & Rádio (HGPE)", fontweight="bold", loc="left", fontsize=6.5)

    # =========================================================================
    # ROW 2 - PANEL E: Macro Party FEFC vs TSE Spending Limits
    # =========================================================================
    fefc_categories = ["Fundo Partidário/FEFC\n(Cota Nacional da Legenda)", "Teto de Gastos TSE\n(Chapa Presidencial)"]
    x_e = np.arange(len(fefc_categories))
    w_e = 0.28

    # Macro Party Pool: PL R$ 881.66M (17.8%), PT R$ 615.42M (12.4%), PT+Fed R$ 692.81M
    ax_e.bar(x_e[0] - w_e/2, 881.66, w_e, color=COLORS["flavio"], zorder=3, label="PL (Bancada Nacional)")
    ax_e.bar(x_e[0] + w_e/2, 692.81, w_e, color=COLORS["lula"], zorder=3, label="Federação Brasil da Esperança")
    ax_e.text(x_e[0] - w_e/2, 895, "R$ 881.7M\n(17.8% FEFC)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["flavio"])
    ax_e.text(x_e[0] + w_e/2, 705, "R$ 692.8M\n(14.0% FEFC)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["lula"])

    # TSE Spending Limits: 1ºT R$ 88.9M -> 2ºT R$ 133.4M
    ax_e.bar(x_e[1] - w_e/2, 88.94, w_e, color="#78909C", zorder=3, label="Teto TSE 1º Turno")
    ax_e.bar(x_e[1] + w_e/2, 133.42, w_e, color="#37474F", zorder=3, label="Teto TSE Total (1º+2ºT)")
    ax_e.text(x_e[1] - w_e/2, 98, "R$ 88.9M\n(Teto 1ºT)", ha="center", va="bottom", fontsize=4.8, color="#37474F")
    ax_e.text(x_e[1] + w_e/2, 142, "R$ 133.4M\n(Teto 2ºT)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color="#263238")

    fefc_total_box = (
        "FEFC Global 2026: R$ 4,96 Bilhões\n"
        "• PL detém a maior cota parlamentar\n"
        "• Teto presidencial = 2,7% do fundo global"
    )
    ax_e.annotate(fefc_total_box, xy=(0.04, 0.64), xycoords="axes fraction",
                  fontsize=4.6, bbox=dict(boxstyle="round,pad=0.2", fc="#F5F5F5", ec="#BDBDBD", alpha=0.95))

    ax_e.set_xticks(x_e)
    ax_e.set_xticklabels(fefc_categories, fontsize=5.2)
    ax_e.set_ylabel("Recursos Financeiros (R$ Milhões)", fontsize=5.8)
    ax_e.set_ylim(0, 1050)
    ax_e.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_e.legend(loc="upper right", frameon=True, fontsize=4.5)
    ax_e.set_title("e  Fundo Eleitoral Macro (FEFC) & Tetos TSE", fontweight="bold", loc="left", fontsize=6.5)

    # =========================================================================
    # ROW 2 - PANEL F: Presidential Campaign Funding Origin: Public vs Private
    # =========================================================================
    candidates_f = ["Lula da Silva (PT)\n[R$ 128.6M Total]", "Flávio Bolsonaro (PL)\n[R$ 90.2M Total]"]
    x_f = np.arange(len(candidates_f))
    w_f = 0.38

    # Lula: FEFC 123.8M (96.3%), Private 4.8M (3.7%)
    # Flavio: FEFC 47.0M (52.1%), Private 43.2M (47.9%)
    ax_f.bar(x_f[0], 123.8, w_f, color=COLORS["lula"], zorder=3, label="Fundo Público (FEFC)")
    ax_f.bar(x_f[0], 4.8, w_f, bottom=123.8, color="#FFCDD2", edgecolor=COLORS["lula"], lw=0.6, zorder=3, label="Doações Privadas / Pix")
    ax_f.text(x_f[0], 60, "R$ 123.8M\n(96.3% FEFC)", ha="center", va="center", fontsize=4.8, fontweight="bold", color="white")
    ax_f.text(x_f[0], 130, "R$ 4.8M (3.7%)", ha="center", va="bottom", fontsize=4.8, color=COLORS["lula"])

    ax_f.bar(x_f[1], 47.0, w_f, color=COLORS["flavio"], zorder=3)
    ax_f.bar(x_f[1], 43.2, w_f, bottom=47.0, color="#BBDEFB", edgecolor=COLORS["flavio"], lw=0.6, zorder=3)
    ax_f.text(x_f[1], 23.5, "R$ 47.0M\n(52.1% FEFC)", ha="center", va="center", fontsize=4.8, fontweight="bold", color="white")
    ax_f.text(x_f[1], 68.6, "R$ 43.2M\n(47.9% Privado)", ha="center", va="center", fontsize=4.8, fontweight="bold", color="#0D47A1")

    asym_box = (
        "Assimetria Estrutural:\n"
        "• Flávio arrecadou 9.0x mais capital\n"
        "  privado de pessoas físicas/Pix\n"
        "• PT centralizou 96% do teto via FEFC"
    )
    ax_f.annotate(asym_box, xy=(0.04, 0.68), xycoords="axes fraction",
                  fontsize=4.6, bbox=dict(boxstyle="round,pad=0.2", fc="#EDE7F6", ec="#7E57C2", alpha=0.95))

    ax_f.set_xticks(x_f)
    ax_f.set_xticklabels(candidates_f, fontsize=5.2)
    ax_f.set_ylabel("Receitas Declaradas TSE (R$ Milhões)", fontsize=5.8)
    ax_f.set_ylim(0, 150)
    ax_f.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_f.legend(loc="upper right", frameon=True, fontsize=4.6)
    ax_f.set_title("f  Origem do Financiamento da Chapa", fontweight="bold", loc="left", fontsize=6.5)

    # =========================================================================
    # ROW 3 - PANEL G: Quaest Digital Popularity Index (IPD) Benchmark Waves
    # =========================================================================
    ipd_waves = ["Véspera 1ºT\n(04/Out)", "Pós-1º Turno\n(15/Out)", "Pós-Debate\n(19/Out)"]
    x_g = np.arange(len(ipd_waves))
    w_g = 0.28

    # Authentic Quaest IPD Benchmark Waves (175 metrics):
    # Oct 4: Flavio 80.0 vs Lula 70.7 (+9.3 Flavio)
    # Oct 15: Lula 82.6 vs Flavio 76.4 (+6.2 Lula)
    # Oct 19: Lula 83.3 vs Flavio 76.6 (+6.7 Lula)
    flavio_ipd = [80.0, 76.4, 76.6]
    lula_ipd = [70.7, 82.6, 83.3]

    ax_g.bar(x_g - w_g/2, flavio_ipd, w_g, color=COLORS["flavio"], label="Flávio Bolsonaro", zorder=3)
    ax_g.bar(x_g + w_g/2, lula_ipd, w_g, color=COLORS["lula"], label="Lula da Silva", zorder=3)

    ax_g.text(x_g[0] - w_g/2, 80.6, "80.0\n(+9.3)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["flavio"])
    ax_g.text(x_g[0] + w_g/2, 71.3, "70.7", ha="center", va="bottom", fontsize=4.8, color=COLORS["lula"])

    ax_g.text(x_g[1] - w_g/2, 77.0, "76.4", ha="center", va="bottom", fontsize=4.8, color=COLORS["flavio"])
    ax_g.text(x_g[1] + w_g/2, 83.2, "82.6\n(+6.2)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["lula"])

    ax_g.text(x_g[2] - w_g/2, 77.2, "76.6", ha="center", va="bottom", fontsize=4.8, color=COLORS["flavio"])
    ax_g.text(x_g[2] + w_g/2, 83.9, "83.3\n(+6.7)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["lula"])

    net_box = (
        "Inversão Líquida:\n"
        "• Lula: 70.7 -> 83.3 (+12.6 pts)\n"
        "• Flávio: 80.0 -> 76.6 (-3.4 pts)\n"
        "• Saldo Líquido: +16.0 pts Inversão!"
    )
    ax_g.annotate(net_box, xy=(0.03, 0.04), xycoords="axes fraction",
                  fontsize=4.6, bbox=dict(boxstyle="round,pad=0.2", fc="#FFF3E0", ec="#FF9800", alpha=0.95))

    ax_g.set_xticks(x_g)
    ax_g.set_xticklabels(ipd_waves, fontsize=5.2)
    ax_g.set_ylabel("Índice IPD Quaest (Pontos)", fontsize=5.8)
    ax_g.set_ylim(48, 93)
    ax_g.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_g.legend(loc="upper right", frameon=True, fontsize=4.6)
    ax_g.set_title("g  Índice de Popularidade Digital (IPD)", fontweight="bold", loc="left", fontsize=6.5)

    # =========================================================================
    # ROW 3 - PANEL H: Average Daily Social Interactions (M/day)
    # =========================================================================
    cat_inter = ["04/Out (1d)\n[Noite 1ºT]", "16/Out (1d)\n[Debate Band]", "05-17/Out (13d)\n[Média 2ºT]"]
    x_h = np.arange(len(cat_inter))
    w_h = 0.28

    flavio_inter = [3.90, 5.00, 2.41]
    lula_inter = [0.17, 25.50, 8.52]

    ax_h.bar(x_h - w_h/2, flavio_inter, w_h, color=COLORS["flavio"], label="Flávio", zorder=3)
    ax_h.bar(x_h + w_h/2, lula_inter, w_h, color=COLORS["lula"], label="Lula", zorder=3)

    ax_h.text(x_h[0] - w_h/2, 4.3, "3.90M\n(22.9x)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["flavio"])
    ax_h.text(x_h[0] + w_h/2, 0.5, "0.17M", ha="center", va="bottom", fontsize=4.6, color=COLORS["lula"])

    ax_h.text(x_h[1] - w_h/2, 5.4, "5.0M", ha="center", va="bottom", fontsize=4.6, color=COLORS["flavio"])
    ax_h.text(x_h[1] + w_h/2, 26.0, "25.5M\n(5.1x)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["lula"])

    ax_h.text(x_h[2] - w_h/2, 2.8, "2.41M", ha="center", va="bottom", fontsize=4.6, color=COLORS["flavio"])
    ax_h.text(x_h[2] + w_h/2, 8.9, "8.52M\n(3.5x)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["lula"])

    ax_h.set_xticks(x_h)
    ax_h.set_xticklabels(cat_inter, fontsize=5.2)
    ax_h.set_ylabel("Interações/Dia (Milhões)", fontsize=5.8)
    ax_h.set_ylim(0, 31)
    ax_h.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_h.legend(loc="upper left", frameon=True, fontsize=4.6)
    ax_h.set_title("h  Volume Médio de Interações Sociais", fontweight="bold", loc="left", fontsize=6.5)

    # =========================================================================
    # ROW 3 - PANEL I: Follower Ecosystem & Top-5 Surrogate Bench (Empirical Mean)
    # =========================================================================
    # 4 Pure Empirical Categories (Removing non-peer-reviewed deduplicated reach estimate):
    # 1. Base Própria (Candidato)
    # 2. Contas Partido (Oficiais)
    # 3. Grupos Fechados (WhatsApp)
    # 4. Média dos 5 Maiores Aliados (Bancada Parlamentar)
    cats_i = [
        "Base Própria\n(Candidato)",
        "Contas Partido\n(Oficiais)",
        "Grupos Fechados\n(WhatsApp)",
        "Média Top-5 Aliados\n(Bancada Parlamentar)",
    ]
    x_i = np.arange(len(cats_i))
    w_i = 0.32

    # 1. Candidate Base - Platform Subdivisions
    # Lula: 39.0M = IG 18.2, X 9.1, FB 5.2, TT 4.6, YT 1.9
    # Flavio: 24.1M = IG 13.5, FB 4.0, X 3.2, TT 2.0, YT 1.4
    lula_plats = [18.2, 9.1, 5.2, 4.6, 1.9]
    flavio_plats = [13.5, 4.0, 3.2, 2.0, 1.4]
    plat_shades = [1.0, 0.78, 0.60, 0.45, 0.30]

    bot_f = 0
    for val, s in zip(flavio_plats, plat_shades):
        ax_i.bar(x_i[0] - w_i/2, val, w_i, bottom=bot_f, color=COLORS["flavio"], alpha=s, edgecolor="white", lw=0.4, zorder=3)
        bot_f += val

    bot_l = 0
    for val, s in zip(lula_plats, plat_shades):
        ax_i.bar(x_i[0] + w_i/2, val, w_i, bottom=bot_l, color=COLORS["lula"], alpha=s, edgecolor="white", lw=0.4, zorder=3)
        bot_l += val

    ax_i.text(x_i[0] - w_i/2, bot_f + 0.8, f"{bot_f:.1f}M", ha="center", va="bottom", fontsize=4.8, color=COLORS["flavio"])
    ax_i.text(x_i[0] + w_i/2, bot_l + 0.8, f"{bot_l:.1f}M\n(1.6x)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["lula"])

    # 2. Party Official (PL 4.2M vs PT 6.7M)
    ax_i.bar(x_i[1] - w_i/2, 4.2, w_i, color=COLORS["flavio"], zorder=3)
    ax_i.bar(x_i[1] + w_i/2, 6.7, w_i, color=COLORS["lula"], zorder=3)
    ax_i.text(x_i[1] - w_i/2, 4.8, "4.2M", ha="center", va="bottom", fontsize=4.8, color=COLORS["flavio"])
    ax_i.text(x_i[1] + w_i/2, 7.3, "6.7M\n(1.6x)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["lula"])

    # 3. WhatsApp Closed Ecosystem (Exército do Lula 10.0M vs Tropa do Flávio 2.0M)
    ax_i.bar(x_i[2] - w_i/2, 2.0, w_i, color=COLORS["flavio"], zorder=3)
    ax_i.bar(x_i[2] + w_i/2, 10.0, w_i, color=COLORS["lula"], zorder=3)
    ax_i.text(x_i[2] - w_i/2, 2.6, "2.0M", ha="center", va="bottom", fontsize=4.8, color=COLORS["flavio"])
    ax_i.text(x_i[2] + w_i/2, 10.6, "10.0M\n(5.0x)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["lula"])

    # 4. Surrogate Bench: Arithmetic Mean of the 5 Largest Allies
    # PL Top-5: Nikolas (52M), Eduardo (13.2M), Gayer (7.8M), Zambelli (7.4M), Damares (6.4M) -> Soma 86.8M / 5 = 17.36M
    # PT Top-5: Janones (17.2M), Boulos (8.4M), Erika (4.2M), Gleisi (3.1M), Randolfe (2.7M) -> Soma 35.6M / 5 = 7.12M
    ax_i.bar(x_i[3] - w_i/2, 17.4, w_i, color=COLORS["flavio"], zorder=3)
    ax_i.bar(x_i[3] + w_i/2, 7.1, w_i, color=COLORS["lula"], zorder=3)
    ax_i.text(x_i[3] - w_i/2, 18.0, "17.4M\n(2.4x)", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=COLORS["flavio"])
    ax_i.text(x_i[3] + w_i/2, 7.7, "7.1M", ha="center", va="bottom", fontsize=4.8, color=COLORS["lula"])

    # Methodological breakdown box for Top-5 and Candidate Base
    method_box = (
        "Base Própria:\n"
        "• IG 50%  • X/Tw 23%  • FB 13%  • TT 10%  • YT 4%\n"
        "─────────────────────────────────────────\n"
        "Média dos 5 Maiores Aliados (Top-5):\n"
        "• PL (17.4M méd / 86.8M total):\n"
        "  Nikolas (52M), Eduardo B. (13.2M),\n"
        "  Gayer (7.8M), Zambelli (7.4M), Damares (6.4M)\n"
        "• PT (7.1M méd / 35.6M total):\n"
        "  Janones (17.2M), Boulos (8.4M),\n"
        "  Erika (4.2M), Gleisi (3.1M), Randolfe (2.7M)"
    )
    ax_i.annotate(method_box, xy=(0.17, 0.47), xycoords="axes fraction",
                  fontsize=4.0, bbox=dict(boxstyle="round,pad=0.2", fc="#FAFAFA", ec="#BDBDBD", alpha=0.95))

    ax_i.set_xticks(x_i)
    ax_i.set_xticklabels(cats_i, fontsize=5.0)
    ax_i.set_ylabel("Seguidores Totais / Média (Milhões)", fontsize=5.8)
    ax_i.set_ylim(0, 52)
    ax_i.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Custom legend for panel i
    p_flav = Patch(facecolor=COLORS["flavio"], label="Rede Flávio / PL")
    p_lula = Patch(facecolor=COLORS["lula"], label="Rede Lula / Janones")
    ax_i.legend(handles=[p_flav, p_lula], loc="upper right", frameon=True, fontsize=4.6)
    ax_i.set_title("i  Ecossistema de Redes & Média dos 5 Maiores Aliados", fontweight="bold", loc="left", fontsize=6.5)

    # -------------------------------------------------------------------------
    # Save outputs (PDF, SVG, PNG)
    # -------------------------------------------------------------------------
    pdf_path = FIGURES_DIR / "election_2026_master_synthesis_3x3_grid.pdf"
    svg_path = FIGURES_DIR / "election_2026_master_synthesis_3x3_grid.svg"
    png_path = FIGURES_DIR / "election_2026_master_synthesis_3x3_grid.png"

    plt.savefig(pdf_path, format="pdf", dpi=300)
    plt.savefig(svg_path, format="svg", dpi=300)
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()

    print(f"Generated {pdf_path}")
    print(f"Generated {svg_path}")
    print(f"Generated {png_path}")

    # Copy to Brain Artifact Directory
    brain_dir = Path("/Users/rennocosta/.gemini/antigravity/brain/9a018403-8aec-4ff7-aadc-19a18baf7af1")
    if brain_dir.exists():
        for ext in [".pdf", ".svg", ".png"]:
            src = FIGURES_DIR / f"election_2026_master_synthesis_3x3_grid{ext}"
            dst = brain_dir / f"election_2026_master_synthesis_3x3_grid{ext}"
            shutil.copy2(src, dst)
            print(f"Copied to brain: {dst}")

if __name__ == "__main__":
    build_master_3x3_figure()
