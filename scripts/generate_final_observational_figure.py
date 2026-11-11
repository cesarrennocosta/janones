#!/usr/bin/env python3
"""
Generate Final Observational Figure for the 2026 Brazilian Presidential Election.

Figure Architecture:
- Top Row (Row 0 - 4 Blocks, a & b narrower):
  * (a) Official TSE Results: 1st and 2nd Rounds (Valid Votes & Non-Voters)
  * (b) Pre-Campaign Electorate Conviction Segmentation (Datafolha 1-to-5 Scale)
  * (c) Longitudinal Vote Intentions (Datafolha 6 Polls in 1ºT + 4 in 2ºT + TSE Urnas - Pure Dots)
  * (d) Candidate Rejection Dynamics (Datafolha "Não Votaria de Jeito Nenhum": Lula vs. Flávio)
- Bottom Row (Row 1 - 3 Blocks, Lower Part of 3x2 Grid Figure):
  * (e) Quaest Digital Popularity Index (IPD) Benchmark Waves (66-88 pts)
  * (f) Average Daily Social Interactions (M/day: Oct 4, Debate, Oct 5-17 Avg)
  * (g) Follower Ecosystem: Candidate Base, Party Official, WhatsApp, Top-5 Avg, Est. Unique Reach

Adobe Illustrator Compatibility:
- FontType 42 (TrueType)
- Single text object layout patch to eliminate fragmented kerning glyphs.
"""

import os
import sys
import shutil
import json
from pathlib import Path
from types import SimpleNamespace
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import _text_helpers
from scipy.interpolate import PchipInterpolator

# -----------------------------------------------------------------------------
# Configuration & Adobe Illustrator Font Settings
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['text.antialiased'] = True

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

# -----------------------------------------------------------------------------
# Color Palette
# -----------------------------------------------------------------------------
COLORS = {
    "lula": "#D32F2F",           # PT Red
    "lula_dark": "#C62828",
    "lula_light": "#EF9A9A",
    "flavio": "#1976D2",         # PL Blue
    "flavio_dark": "#1565C0",
    "flavio_light": "#90CAF9",
    "others": "#7E57C2",         # 3rd-Way Purple
    "nonvoters": "#546E7A",      # Non-voters Slate Grey
    "neutral": "#9E9E9E",        # Neutral Grey
    "amber": "#FFA000"
}

# -----------------------------------------------------------------------------
# Data Definition
# -----------------------------------------------------------------------------

# (a) TSE Actual Results (1st and 2nd Rounds)
tse_rounds = ["1º Turno TSE\n(04/Out/2026)", "2º Turno TSE\n(25/Out/2026)"]
# 1st Round: Flavio 47.03%, Lula 45.16%, Others 7.81%, Non-voters 24.70%
# 2nd Round: Lula 51.79%, Flavio 48.21%, Non-voters 21.79%

# (b) Datafolha Pre-Campaign Conviction Segmentation (1-to-5 Scale)
segment_categories = ['Lulista', 'Bolsonarista', 'Neutro', 'Nenhum']
convict_vals = [28.0, 23.0, 0.0, 0.0]
non_convict_vals = [12.0, 11.0, 0.0, 0.0]
single_vals = [0.0, 0.0, 19.0, 7.0]
total_vals = [40.0, 34.0, 19.0, 7.0]

# (c) Vote Intention Points (Datafolha 6 polls in 1ºT + 4 in 2ºT + TSE)
base_date = datetime.date(2026, 8, 19)

r1_poll_dates = [
    datetime.date(2026, 8, 19),
    datetime.date(2026, 9, 2),
    datetime.date(2026, 9, 10),
    datetime.date(2026, 9, 17),
    datetime.date(2026, 9, 23),
    datetime.date(2026, 10, 1)
]
r1_poll_lula = [44.3, 41.8, 43.3, 43.0, 44.9, 45.0]
r1_poll_flavio = [37.5, 36.3, 37.8, 39.6, 40.4, 41.0]
r1_poll_others = [18.2, 22.0, 18.9, 17.4, 14.7, 14.0]

tse1_date = datetime.date(2026, 10, 4)
tse1_lula = 45.16
tse1_flavio = 47.03
tse1_others = 7.81

r2_poll_dates = [
    datetime.date(2026, 10, 8),
    datetime.date(2026, 10, 15),
    datetime.date(2026, 10, 22),
    datetime.date(2026, 10, 24)
]
r2_poll_flavio = [52.0, 51.0, 50.0, 51.0]
r2_poll_lula = [48.0, 49.0, 50.0, 49.0]

tse2_date = datetime.date(2026, 10, 25)
tse2_lula = 51.79
tse2_flavio = 48.21

r1_poll_x = np.array([(d - base_date).days for d in r1_poll_dates], dtype=float)
tse1_x = (tse1_date - base_date).days
r2_poll_x = np.array([(d - base_date).days for d in r2_poll_dates], dtype=float)
tse2_x = (tse2_date - base_date).days

# (d) Candidate Rejection Points (Datafolha "Não Votaria de Jeito Nenhum")
rej_r1_dates = [
    datetime.date(2026, 8, 19),
    datetime.date(2026, 9, 2),
    datetime.date(2026, 9, 10),
    datetime.date(2026, 9, 17),
    datetime.date(2026, 9, 23),
    datetime.date(2026, 10, 1)
]
rej_r1_lula = [45.0, 45.0, 46.0, 47.0, 45.0, 45.0]
rej_r1_flavio = [46.0, 47.0, 46.0, 47.0, 45.0, 45.0]

rej_r2_dates = [
    datetime.date(2026, 10, 8),
    datetime.date(2026, 10, 15),
    datetime.date(2026, 10, 22),
    datetime.date(2026, 10, 24)
]
rej_r2_lula = [47.0, 46.0, 45.0, 46.0]
rej_r2_flavio = [43.0, 44.0, 46.0, 45.0]

rej_r1_x = np.array([(d - base_date).days for d in rej_r1_dates], dtype=float)
rej_r2_x = np.array([(d - base_date).days for d in rej_r2_dates], dtype=float)

rej_x_dense_r1 = np.linspace(rej_r1_x[0], rej_r1_x[-1], 200)
spline_rej_r1_lula = PchipInterpolator(rej_r1_x, rej_r1_lula)(rej_x_dense_r1)
spline_rej_r1_flavio = PchipInterpolator(rej_r1_x, rej_r1_flavio)(rej_x_dense_r1)

rej_x_dense_r2 = np.linspace(rej_r2_x[0], rej_r2_x[-1], 200)
spline_rej_r2_lula = PchipInterpolator(rej_r2_x, rej_r2_lula)(rej_x_dense_r2)
spline_rej_r2_flavio = PchipInterpolator(rej_r2_x, rej_r2_flavio)(rej_x_dense_r2)


def main():
    fig = plt.figure(figsize=(25.0, 12.8), dpi=300)
    
    # 2 Main Vertical Rows
    gs_master = GridSpec(2, 1, height_ratios=[1.0, 1.05], hspace=0.34, top=0.94, bottom=0.07, left=0.038, right=0.985)
    
    # Top Row: 4 Blocks (a and b narrower, c and d wider)
    gs_top = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_master[0], width_ratios=[0.82, 0.88, 1.30, 1.10], wspace=0.28)
    
    # Bottom Row: 3 Blocks (matching lower part of 3x2 grid figure)
    gs_bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_master[1], width_ratios=[0.85, 0.85, 1.45], wspace=0.26)

    # =========================================================================
    # BLOCK (A): First and Second Round TSE Official Results
    # =========================================================================
    ax_a = fig.add_subplot(gs_top[0, 0])
    
    x_a = np.array([0.0, 1.8])
    w_a = 0.26
    
    # Round 1 Bars (4 bars: Lula, Flavio, Others, Non-Voters)
    pos_l1 = x_a[0] - 1.5 * w_a
    pos_f1 = x_a[0] - 0.5 * w_a
    pos_o1 = x_a[0] + 0.5 * w_a
    pos_nv1 = x_a[0] + 1.5 * w_a

    ax_a.bar(pos_l1, tse1_lula, w_a, color=COLORS["lula"], edgecolor="#333333", lw=0.8, zorder=3, label="Lula (Valid %)")
    ax_a.text(pos_l1, tse1_lula + 0.8, f"{tse1_lula:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=COLORS["lula"])

    ax_a.bar(pos_f1, tse1_flavio, w_a, color=COLORS["flavio"], edgecolor="#333333", lw=0.8, zorder=3, label="Flávio (Valid %)")
    ax_a.text(pos_f1, tse1_flavio + 0.8, f"{tse1_flavio:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=COLORS["flavio"])

    ax_a.bar(pos_o1, tse1_others, w_a, color=COLORS["others"], edgecolor="#333333", lw=0.8, zorder=3, label="Outros (Valid %)")
    ax_a.text(pos_o1, tse1_others + 0.8, f"{tse1_others:.1f}%", ha="center", va="bottom", fontsize=7.8, color=COLORS["others"])

    ax_a.bar(pos_nv1, 24.70, w_a, color=COLORS["nonvoters"], edgecolor="#333333", lw=0.8, zorder=3, label="Não-Votantes (%)")
    ax_a.text(pos_nv1, 24.70 + 0.8, "24.7%", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=COLORS["nonvoters"])

    # Round 2 Bars (3 bars: Lula, Flavio, Non-Voters)
    pos_l2 = x_a[1] - w_a
    pos_f2 = x_a[1]
    pos_nv2 = x_a[1] + w_a

    ax_a.bar(pos_l2, tse2_lula, w_a, color=COLORS["lula"], edgecolor="#333333", lw=0.8, zorder=3)
    ax_a.text(pos_l2, tse2_lula + 0.8, f"★ {tse2_lula:.2f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=COLORS["lula"])

    ax_a.bar(pos_f2, tse2_flavio, w_a, color=COLORS["flavio"], edgecolor="#333333", lw=0.8, zorder=3)
    ax_a.text(pos_f2, tse2_flavio + 0.8, f"{tse2_flavio:.2f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=COLORS["flavio"])

    ax_a.bar(pos_nv2, 21.79, w_a, color=COLORS["nonvoters"], edgecolor="#333333", lw=0.8, zorder=3)
    ax_a.text(pos_nv2, 21.79 + 0.8, "21.8%", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=COLORS["nonvoters"])

    # Divider between R1 and R2
    ax_a.axvline(x=0.9, color="#9E9E9E", linestyle="--", lw=1.0, alpha=0.7)

    # Annotate turnout surge / abstention drop
    ax_a.annotate("Mobilização no 2ºT:\nNão-Votantes caem\n24.70% -> 21.79%\n(-2.91% Turnout Surge)",
                  xy=(pos_nv2, 21.79), xytext=(0.58, 33.0),
                  arrowprops=dict(arrowstyle="->", color=COLORS["nonvoters"], lw=1.1),
                  fontsize=7.0, fontweight="bold", color=COLORS["nonvoters"],
                  bbox=dict(boxstyle="round,pad=0.25", fc="#ECEFF1", ec=COLORS["nonvoters"], alpha=0.92))

    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(tse_rounds, fontsize=8.2, fontweight="bold")
    ax_a.set_ylabel("Participação Eleitoral (%)", fontsize=8.8, fontweight="bold")
    ax_a.set_ylim(0, 62)
    ax_a.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_a.legend(loc="upper left", frameon=True, fontsize=6.8)
    ax_a.set_title("a  Resultados Oficiais TSE (1º e 2º Turnos)", fontweight="bold", loc="left", fontsize=9.8)

    # =========================================================================
    # BLOCK (B): Pre-Campaign Electorate Conviction Segmentation (Datafolha 1-5)
    # =========================================================================
    ax_b = fig.add_subplot(gs_top[0, 1])
    
    x_indices_b = np.arange(len(segment_categories))
    bar_width_b = 0.52

    # Lulista bar (Stacked)
    ax_b.bar(x_indices_b[0], convict_vals[0], width=bar_width_b, color=COLORS["lula_dark"],
             edgecolor='#333333', linewidth=0.8, label='Convicto (Cristalizado)')
    ax_b.bar(x_indices_b[0], non_convict_vals[0], bottom=convict_vals[0], width=bar_width_b,
             color=COLORS["lula_light"], edgecolor='#333333', linewidth=0.8, label='Não Convicto (Simpatizante)')

    # Bolsonarista bar (Stacked)
    ax_b.bar(x_indices_b[1], convict_vals[1], width=bar_width_b, color=COLORS["flavio_dark"],
             edgecolor='#333333', linewidth=0.8)
    ax_b.bar(x_indices_b[1], non_convict_vals[1], bottom=convict_vals[1], width=bar_width_b,
             color=COLORS["flavio_light"], edgecolor='#333333', linewidth=0.8)

    # Neutro bar (Single)
    ax_b.bar(x_indices_b[2], single_vals[2], width=bar_width_b, color=COLORS["others"],
             edgecolor='#333333', linewidth=0.8)

    # Nenhum bar (Single)
    ax_b.bar(x_indices_b[3], single_vals[3], width=bar_width_b, color=COLORS["neutral"],
             edgecolor='#333333', linewidth=0.8)

    # Values inside & top
    ax_b.text(x_indices_b[0], convict_vals[0] / 2, "28.0%\n(Raiz)", ha='center', va='center',
              color='white', fontsize=7.8, fontweight='bold')
    ax_b.text(x_indices_b[0], convict_vals[0] + non_convict_vals[0] / 2, "12.0%",
              ha='center', va='center', color='#212121', fontsize=7.5, fontweight='bold')
    ax_b.text(x_indices_b[0], total_vals[0] + 1.0, "40.0%", ha='center', va='bottom',
              color=COLORS["lula_dark"], fontsize=9.2, fontweight='bold')

    ax_b.text(x_indices_b[1], convict_vals[1] / 2, "23.0%\n(Raiz)", ha='center', va='center',
              color='white', fontsize=7.8, fontweight='bold')
    ax_b.text(x_indices_b[1], convict_vals[1] + non_convict_vals[1] / 2, "11.0%",
              ha='center', va='center', color='#212121', fontsize=7.5, fontweight='bold')
    ax_b.text(x_indices_b[1], total_vals[1] + 1.0, "34.0%", ha='center', va='bottom',
              color=COLORS["flavio_dark"], fontsize=9.2, fontweight='bold')

    ax_b.text(x_indices_b[2], single_vals[2] / 2, "19.0%\n(Centro)", ha='center', va='center',
              color='white', fontsize=7.8, fontweight='bold')
    ax_b.text(x_indices_b[2], total_vals[2] + 1.0, "19.0%", ha='center', va='bottom',
              color=COLORS["others"], fontsize=9.2, fontweight='bold')

    ax_b.text(x_indices_b[3], single_vals[3] / 2, "7.0%", ha='center', va='center',
              color='white', fontsize=7.8, fontweight='bold')
    ax_b.text(x_indices_b[3], total_vals[3] + 1.0, "7.0%", ha='center', va='bottom',
              color='#616161', fontsize=9.2, fontweight='bold')

    ax_b.set_xticks(x_indices_b)
    ax_b.set_xticklabels(['Lulista\n(PT)', 'Bolsonarista\n(PL)', 'Neutro\n(Centro)', 'Nenhum\n(Despolit.)'],
                         fontsize=7.8, fontweight='bold')
    ax_b.set_ylabel("Fração do Eleitorado (%)", fontsize=8.8, fontweight='bold')
    ax_b.set_ylim(0, 48)
    ax_b.grid(axis='y', linestyle=':', alpha=0.5)
    ax_b.set_title("b  Convicção Ideológica (Datafolha 1 a 5)", fontweight="bold", loc="left", fontsize=9.8)

    legend_elements_b = [
        Patch(facecolor='#424242', edgecolor='#333333', label='Convicto (Notas 5 e 1)'),
        Patch(facecolor='#E0E0E0', edgecolor='#333333', label='Não Convicto (Notas 4 e 2)'),
        Patch(facecolor=COLORS["others"], edgecolor='#333333', label='Neutro (Nota 3)'),
        Patch(facecolor=COLORS["neutral"], edgecolor='#333333', label='Nenhum (Fora da régua)')
    ]
    ax_b.legend(handles=legend_elements_b, loc='upper right', framealpha=0.92, fontsize=6.5)

    # =========================================================================
    # BLOCK (C): Vote Intention Plot (Pure Dots, Distinct Shapes: Circle vs Triangle)
    # =========================================================================
    ax_c = fig.add_subplot(gs_top[0, 2])
    
    # Check for ties in R2 (Oct 22: Lula 50% vs Flavio 50%) to offset slightly for dual visibility
    r1_lula_x_c = r1_poll_x.copy()
    r1_flavio_x_c = r1_poll_x.copy()
    
    r2_lula_x_c = r2_poll_x.copy()
    r2_flavio_x_c = r2_poll_x.copy()
    # On Oct 22 (index 2), tie at 50%
    r2_lula_x_c[2] -= 0.35
    r2_flavio_x_c[2] += 0.35

    # Dashed guide lines connecting dots directly (No Spline Curves)
    ax_c.plot(np.append(r1_poll_x, tse1_x), np.append(r1_poll_lula, tse1_lula),
              color=COLORS["lula"], lw=1.0, linestyle="--", alpha=0.70)
    ax_c.plot(np.append(r1_poll_x, tse1_x), np.append(r1_poll_flavio, tse1_flavio),
              color=COLORS["flavio"], lw=1.0, linestyle="--", alpha=0.70)
    ax_c.plot(np.append(r1_poll_x, tse1_x), np.append(r1_poll_others, tse1_others),
              color=COLORS["others"], lw=1.0, linestyle="--", alpha=0.70)

    ax_c.plot(np.append(r2_poll_x, tse2_x), np.append(r2_poll_lula, tse2_lula),
              color=COLORS["lula"], lw=1.0, linestyle="--", alpha=0.70)
    ax_c.plot(np.append(r2_poll_x, tse2_x), np.append(r2_poll_flavio, tse2_flavio),
              color=COLORS["flavio"], lw=1.0, linestyle="--", alpha=0.70)

    # 1st Round Datafolha Dots: Lula = Circle ('o'), Flávio = Triangle ('^'), Outros = Square ('s')
    ax_c.scatter(r1_lula_x_c, r1_poll_lula, color=COLORS["lula"], marker='o', s=55, zorder=5,
                 edgecolor='black', linewidth=0.8, label="Lula (PT): Círculo")
    ax_c.scatter(r1_flavio_x_c, r1_poll_flavio, color=COLORS["flavio"], marker='^', s=60, zorder=5,
                 edgecolor='black', linewidth=0.8, label="Flávio Bolsonaro (PL): Triângulo")
    ax_c.scatter(r1_poll_x, r1_poll_others, color=COLORS["others"], marker='s', s=48, zorder=5,
                 edgecolor='black', linewidth=0.8, label="Outros / 3ª Via: Quadrado")

    # 1st Round TSE Urnas Dots
    ax_c.scatter(tse1_x, tse1_lula, color=COLORS["lula"], marker='o', s=95, zorder=6,
                 edgecolor='white', linewidth=1.2)
    ax_c.scatter(tse1_x, tse1_flavio, color=COLORS["flavio"], marker='^', s=100, zorder=6,
                 edgecolor='white', linewidth=1.2)
    ax_c.scatter(tse1_x, tse1_others, color=COLORS["others"], marker='s', s=80, zorder=6,
                 edgecolor='white', linewidth=1.2)

    # 2nd Round Datafolha Dots
    ax_c.scatter(r2_lula_x_c, r2_poll_lula, color=COLORS["lula"], marker='o', s=55, zorder=5,
                 edgecolor='black', linewidth=0.8)
    ax_c.scatter(r2_flavio_x_c, r2_poll_flavio, color=COLORS["flavio"], marker='^', s=60, zorder=5,
                 edgecolor='black', linewidth=0.8)

    # 2nd Round TSE Urnas Dots
    ax_c.scatter(tse2_x, tse2_lula, color=COLORS["lula"], marker='o', s=130, zorder=6,
                 edgecolor='yellow', linewidth=1.5)
    ax_c.scatter(tse2_x, tse2_flavio, color=COLORS["flavio"], marker='^', s=120, zorder=6,
                 edgecolor='white', linewidth=1.2)

    # Phase Separator
    sep_xc = (tse1_x + r2_poll_x[0]) / 2.0
    ax_c.axvline(sep_xc, color='#757575', linestyle='--', linewidth=1.1, alpha=0.8)

    ax_c.text((r1_poll_x[0] + tse1_x) / 2, 58.5, "1º TURNO (6 Pesquisas & TSE)",
              ha='center', va='center', fontsize=8.0, fontweight='bold', color='#424242',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.9))

    ax_c.text((r2_poll_x[0] + tse2_x) / 2, 58.5, "2º TURNO (Head-to-Head)",
              ha='center', va='center', fontsize=8.0, fontweight='bold', color='#424242',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.9))

    # Precise Callouts
    ax_c.annotate(f"TSE: {tse1_flavio:.1f}%\n(01/Out: 41%)", xy=(tse1_x, tse1_flavio),
                  xytext=(tse1_x - 12.5, tse1_flavio + 3.2),
                  fontsize=7.5, fontweight='bold', color=COLORS["flavio"],
                  arrowprops=dict(arrowstyle="->", color=COLORS["flavio"], lw=0.8))
    ax_c.annotate(f"TSE: {tse1_lula:.1f}%", xy=(tse1_x, tse1_lula),
                  xytext=(tse1_x - 11, tse1_lula - 3.4),
                  fontsize=7.8, fontweight='bold', color=COLORS["lula"],
                  arrowprops=dict(arrowstyle="->", color=COLORS["lula"], lw=0.8))
    ax_c.annotate(f"3ª Via: {tse1_others:.1f}%", xy=(tse1_x, tse1_others),
                  xytext=(tse1_x - 11, tse1_others - 2.8),
                  fontsize=7.2, fontweight='bold', color=COLORS["others"],
                  arrowprops=dict(arrowstyle="->", color=COLORS["others"], lw=0.7))

    eve_xc = r2_poll_x[3]
    ax_c.annotate("Véspera: 51.0%", xy=(eve_xc, r2_poll_flavio[3]),
                  xytext=(eve_xc - 13.5, r2_poll_flavio[3] + 2.0),
                  fontsize=7.8, fontweight='bold', color=COLORS["flavio"],
                  arrowprops=dict(arrowstyle="->", color=COLORS["flavio"], lw=0.8))
    ax_c.annotate("Véspera: 49.0%", xy=(eve_xc, r2_poll_lula[3]),
                  xytext=(eve_xc - 13.5, r2_poll_lula[3] - 3.5),
                  fontsize=7.8, fontweight='bold', color=COLORS["lula"],
                  arrowprops=dict(arrowstyle="->", color=COLORS["lula"], lw=0.8))

    ax_c.annotate(f"★ Eleito: {tse2_lula:.2f}%", xy=(tse2_x, tse2_lula),
                  xytext=(tse2_x + 1.8, tse2_lula + 0.6),
                  fontsize=8.2, fontweight='bold', color=COLORS["lula"],
                  bbox=dict(boxstyle='square,pad=0.2', facecolor='#FFEBEE', edgecolor=COLORS["lula"], lw=0.7))
    ax_c.annotate(f"TSE: {tse2_flavio:.2f}%", xy=(tse2_x, tse2_flavio),
                  xytext=(tse2_x + 1.8, tse2_flavio - 1.2),
                  fontsize=7.8, fontweight='bold', color=COLORS["flavio"],
                  bbox=dict(boxstyle='square,pad=0.2', facecolor='#E3F2FD', edgecolor=COLORS["flavio"], lw=0.7))

    ax_c.set_xticks([r1_poll_x[0], r1_poll_x[1], r1_poll_x[3], tse1_x, r2_poll_x[1], tse2_x])
    ax_c.set_xticklabels(["19/Ago\n(Início)", "02/Set", "17/Set", "04/Out\n(TSE 1ºT)", "15/Out", "25/Out\n(TSE 2ºT)"],
                         fontsize=7.8)

    ax_c.set_xlim(-3, tse2_x + 16)
    ax_c.set_ylim(2, 62)
    ax_c.set_ylabel("Votos Válidos (%)", fontsize=8.8, fontweight='bold')
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(loc="lower left", frameon=True, fontsize=6.8)
    ax_c.set_title("c  Intenção de Voto: Pesquisas & Urnas (Sem Interpolação)", fontweight="bold", loc="left", fontsize=9.8)

    # =========================================================================
    # BLOCK (D): Candidate Rejection Dynamics (Dashed Line, Circle vs Triangle)
    # =========================================================================
    ax_d = fig.add_subplot(gs_top[0, 3])

    # Offset tied points in R1 (10/Set, 17/Set, 23/Set, 01/Out) so both circle and triangle appear side-by-side
    rej_r1_lula_x = rej_r1_x.copy()
    rej_r1_flavio_x = rej_r1_x.copy()
    
    for i in range(len(rej_r1_dates)):
        if rej_r1_lula[i] == rej_r1_flavio[i]:
            rej_r1_lula_x[i] -= 0.38
            rej_r1_flavio_x[i] += 0.38

    # Dashed line connecting rejection points directly (No Spline Curves!)
    ax_d.plot(rej_r1_x, rej_r1_lula, color=COLORS["lula"], linewidth=1.1, linestyle="--", alpha=0.75)
    ax_d.plot(rej_r1_x, rej_r1_flavio, color=COLORS["flavio"], linewidth=1.1, linestyle="--", alpha=0.75)

    ax_d.plot(rej_r2_x, rej_r2_lula, color=COLORS["lula"], linewidth=1.1, linestyle="--", alpha=0.75)
    ax_d.plot(rej_r2_x, rej_r2_flavio, color=COLORS["flavio"], linewidth=1.1, linestyle="--", alpha=0.75)

    # Scatter dots: Lula = Circle ('o'), Flávio = Triangle ('^')
    ax_d.scatter(rej_r1_lula_x, rej_r1_lula, color=COLORS["lula"], marker='o', s=55, zorder=5,
                 edgecolor='black', linewidth=0.8, label='Rejeição Lula (Círculo)')
    ax_d.scatter(rej_r1_flavio_x, rej_r1_flavio, color=COLORS["flavio"], marker='^', s=60, zorder=5,
                 edgecolor='black', linewidth=0.8, label='Rejeição Flávio (Triângulo)')

    ax_d.scatter(rej_r2_x, rej_r2_lula, color=COLORS["lula"], marker='o', s=55, zorder=5,
                 edgecolor='black', linewidth=0.8)
    ax_d.scatter(rej_r2_x, rej_r2_flavio, color=COLORS["flavio"], marker='^', s=60, zorder=5,
                 edgecolor='black', linewidth=0.8)

    sep_xd = (rej_r1_x[-1] + rej_r2_x[0]) / 2.0
    ax_d.axvline(sep_xd, color='#757575', linestyle='--', linewidth=1.1, alpha=0.8)

    ax_d.text((rej_r1_x[0] + rej_r1_x[-1]) / 2, 50.8, "1º TURNO",
              ha='center', va='center', fontsize=8.0, fontweight='bold', color='#424242',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.9))

    ax_d.text((rej_r2_x[0] + rej_r2_x[-1]) / 2, 50.8, "2º TURNO",
              ha='center', va='center', fontsize=8.0, fontweight='bold', color='#424242',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.9))

    ax_d.annotate("Flávio: 47%", xy=(rej_r1_x[1], rej_r1_flavio[1]),
                  xytext=(rej_r1_x[1] - 8, rej_r1_flavio[1] + 1.6),
                  fontsize=7.5, fontweight='bold', color=COLORS["flavio"],
                  arrowprops=dict(arrowstyle="->", color=COLORS["flavio"], lw=0.7))

    ax_d.annotate("Pós-1ºT: Flávio cai p/ 43%\n(Lula em 47%)", xy=(rej_r2_x[0], rej_r2_flavio[0]),
                  xytext=(rej_r2_x[0] - 11, rej_r2_flavio[0] - 3.2),
                  fontsize=7.5, fontweight='bold', color=COLORS["flavio"],
                  arrowprops=dict(arrowstyle="->", color=COLORS["flavio"], lw=0.7))

    ax_d.annotate("WhatsApp: Flávio sobe p/ 46%", xy=(rej_r2_x[2], rej_r2_flavio[2]),
                  xytext=(rej_r2_x[2] - 13, rej_r2_flavio[2] + 2.0),
                  fontsize=7.5, fontweight='bold', color=COLORS["flavio"],
                  arrowprops=dict(arrowstyle="->", color=COLORS["flavio"], lw=0.7))

    eve_xd = rej_r2_x[3]
    ax_d.annotate("Véspera: 46%", xy=(eve_xd, rej_r2_lula[3]),
                  xytext=(eve_xd + 1.2, rej_r2_lula[3] + 0.3),
                  fontsize=7.8, fontweight='bold', color=COLORS["lula"])
    ax_d.annotate("Véspera: 45%", xy=(eve_xd, rej_r2_flavio[3]),
                  xytext=(eve_xd + 1.2, rej_r2_flavio[3] - 0.4),
                  fontsize=7.8, fontweight='bold', color=COLORS["flavio"])

    ax_d.axhspan(44, 48, color='#EEEEEE', alpha=0.5, zorder=1)

    ax_d.set_xticks([rej_r1_x[0], rej_r1_x[1], rej_r1_x[3], rej_r1_x[5], rej_r2_x[0], rej_r2_x[1], rej_r2_x[3]])
    ax_d.set_xticklabels(["19/Ago\n(Início)", "02/Set", "17/Set", "01/Out", "08/Out\n(2ºT)", "15/Out", "24/Out\n(Véspera)"],
                         fontsize=7.6)

    ax_d.set_xlim(-3, rej_r2_x[3] + 15)
    ax_d.set_ylim(38, 52.5)
    ax_d.set_ylabel("Rejeição Eleitoral (%)", fontsize=8.8, fontweight='bold')
    ax_d.grid(True, linestyle=":", alpha=0.5)
    ax_d.legend(loc='lower left', framealpha=0.92, fontsize=6.8)
    ax_d.set_title("d  Evolução da Rejeição: 'Não Votaria'", fontweight="bold", loc="left", fontsize=9.8)
    ax_d.set_ylabel("Rejeição Eleitoral (%)", fontsize=8.8, fontweight='bold')
    ax_d.grid(True, linestyle=":", alpha=0.5)
    ax_d.legend(loc='lower left', framealpha=0.92, fontsize=6.8)
    ax_d.set_title("d  Evolução da Rejeição: 'Não Votaria'", fontweight="bold", loc="left", fontsize=9.8)

    # =========================================================================
    # BOTTOM ROW - LOWER PART OF 3x2 GRID FIGURE
    # =========================================================================

    # -------------------------------------------------------------------------
    # Panel E: Quaest Digital Popularity Index (IPD) Benchmark Waves
    # -------------------------------------------------------------------------
    ax_e = fig.add_subplot(gs_bottom[0, 0])
    waves = ["Oct 4\n(Post-R1 Urnas)", "Oct 11\n(Debate Eve)", "Oct 19\n(Post-Debate)"]
    x_e = np.arange(len(waves))
    w_e = 0.28

    flavio_ipd = [80.0, 76.4, 76.6]
    lula_ipd = [70.7, 82.6, 83.3]

    ax_e.bar(x_e - w_e/2, flavio_ipd, w_e, color=COLORS["flavio"], label="Flávio Bolsonaro", zorder=3)
    ax_e.bar(x_e + w_e/2, lula_ipd, w_e, color=COLORS["lula"], label="Lula da Silva", zorder=3)

    ax_e.text(x_e[0] - w_e/2, 80.5, "80.0 pts\n(Flávio +9.3)", ha="center", va="bottom", fontweight="bold", color=COLORS["flavio"], fontsize=7.2)
    ax_e.text(x_e[0] + w_e/2, 71.2, "70.7 pts", ha="center", va="bottom", color=COLORS["lula"], fontsize=7.2)

    ax_e.text(x_e[1] - w_e/2, 76.9, "76.4 pts", ha="center", va="bottom", color=COLORS["flavio"], fontsize=7.2)
    ax_e.text(x_e[1] + w_e/2, 83.1, "82.6 pts\n(Lula +6.2)", ha="center", va="bottom", fontweight="bold", color=COLORS["lula"], fontsize=7.2)

    ax_e.text(x_e[2] - w_e/2, 77.1, "76.6 pts", ha="center", va="bottom", color=COLORS["flavio"], fontsize=7.2)
    ax_e.text(x_e[2] + w_e/2, 83.8, "83.3 pts\n(Lula +6.7)", ha="center", va="bottom", fontweight="bold", color=COLORS["lula"], fontsize=7.2)

    net_box = (
        "NET CHANGE (Oct 4 -> Oct 19):\n"
        "• Lula:   +12.6 pts (70.7 -> 83.3)\n"
        "• Flávio:  -3.4 pts (80.0 -> 76.6)\n"
        "• Net Swing: +16.0 pts Inversion!"
    )
    ax_e.annotate(net_box, xy=(0.02, 0.52), xycoords="axes fraction",
                  fontsize=6.8, family="sans-serif",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#FF9800", alpha=0.95))

    ax_e.text(0.48, 0.04, "Sources: Quaest IPD (175 metrics); reports by Poder360, Vero Notícias, ICL Notícias.",
              transform=ax_e.transAxes, fontsize=6.5, color="#424242", style="italic")

    ax_e.set_xticks(x_e)
    ax_e.set_xticklabels(waves, fontsize=7.5)
    ax_e.set_ylabel("IPD Score (Points)", fontsize=8.2)
    ax_e.set_ylim(66, 88)
    ax_e.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_e.legend(loc="upper left", frameon=True, fontsize=6.8)
    ax_e.set_title("e  Quaest Digital Popularity Index (IPD) - Benchmark Waves", fontweight="bold", loc="left", fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel F: Average Daily Social Interactions (Normalized by Day)
    # -------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs_bottom[0, 1])
    cat_inter = [
        "Oct 4 (1 day)\n[Vero/Bites]",
        "Debate (1 day)\n[Poder360/B]",
        "Oct 5–17 (13d Avg)\n[Bites]",
    ]
    x_f = np.arange(len(cat_inter))
    w_f = 0.28

    flavio_inter = [3.90, 5.00, 2.41]
    lula_inter = [0.17, 25.50, 8.52]

    ax_f.bar(x_f - w_f/2, flavio_inter, w_f, color=COLORS["flavio"], label="Flávio", zorder=3)
    ax_f.bar(x_f + w_f/2, lula_inter, w_f, color=COLORS["lula"], label="Lula", zorder=3)

    ax_f.text(x_f[0] - w_f/2, 4.4, "3.90M/d\n(22.9x)", ha="center", va="bottom", fontweight="bold", color=COLORS["flavio"], fontsize=6.8)
    ax_f.text(x_f[0] + w_f/2, 0.7, "0.17M/d", ha="center", va="bottom", color=COLORS["lula"], fontsize=6.5)

    ax_f.text(x_f[1] - w_f/2, 5.5, "5.0M/d", ha="center", va="bottom", color=COLORS["flavio"], fontsize=6.5)
    ax_f.text(x_f[1] + w_f/2, 26.0, "25.5M/d\n(5.1x)", ha="center", va="bottom", fontweight="bold", color=COLORS["lula"], fontsize=6.8)

    ax_f.text(x_f[2] - w_f/2, 2.9, "2.41M/d", ha="center", va="bottom", color=COLORS["flavio"], fontsize=6.5)
    ax_f.text(x_f[2] + w_f/2, 9.0, "8.52M/d\n(3.5x)", ha="center", va="bottom", fontweight="bold", color=COLORS["lula"], fontsize=6.8)

    sept_box = "Sept Daily Wins:\nFlávio 14 v Lula 3 (82.4%)\n[ICL Notícias / Bites]"
    ax_f.annotate(sept_box, xy=(0.04, 0.62), xycoords="axes fraction",
                  fontsize=6.5, bbox=dict(boxstyle="round,pad=0.25", fc="#F5F5F5", ec="#BDBDBD", alpha=0.95))

    ax_f.set_xticks(x_f)
    ax_f.set_xticklabels(cat_inter, fontsize=7.2)
    ax_f.set_ylabel("Avg. Interactions/Day (Millions)", fontsize=8.0)
    ax_f.set_ylim(0, 31)
    ax_f.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_f.legend(loc="upper left", frameon=True, fontsize=6.8)
    ax_f.set_title("f  Average Daily Social Interactions (Rates)", fontweight="bold", loc="left", fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel G: Follower Ecosystem: Candidate Base, Party, Closed & Bench Reach
    # -------------------------------------------------------------------------
    ax_g = fig.add_subplot(gs_bottom[0, 2])
    cats_g = [
        "Candidate Base\n[Flávio v Lula]",
        "Party Official\n[PL v PT]",
        "WhatsApp Army\n[Tropa v Exército]",
        "Surrogate Top-5 Avg\n[Excl. Candidate]",
        "Surrogates Unique\n[Excl. Candidate]",
    ]
    x_g = np.arange(len(cats_g))
    w_g = 0.32

    lula_plats = [18.2, 9.1, 5.2, 4.6, 1.9]
    flavio_plats = [13.5, 4.0, 3.2, 2.0, 1.4]
    plat_shades = [1.0, 0.78, 0.60, 0.45, 0.30]

    bot_f = 0
    for val, s in zip(flavio_plats, plat_shades):
        ax_g.bar(x_g[0] - w_g/2, val, w_g, bottom=bot_f, color=COLORS["flavio"], alpha=s, edgecolor="white", lw=0.5, zorder=3)
        bot_f += val

    bot_l = 0
    for val, s in zip(lula_plats, plat_shades):
        ax_g.bar(x_g[0] + w_g/2, val, w_g, bottom=bot_l, color=COLORS["lula"], alpha=s, edgecolor="white", lw=0.5, zorder=3)
        bot_l += val

    ax_g.text(x_g[0] - w_g/2, bot_f + 1.0, f"{bot_f:.1f}M", ha="center", va="bottom", fontweight="bold", color="#0D47A1", fontsize=7.2)
    ax_g.text(x_g[0] + w_g/2, bot_l + 1.0, f"{bot_l:.1f}M (1.6x)", ha="center", va="bottom", fontweight="bold", color="#B71C1C", fontsize=7.2)

    ax_g.bar(x_g[1] - w_g/2, 3.5, w_g, color=COLORS["flavio"], edgecolor="white", lw=0.5, zorder=3)
    ax_g.bar(x_g[1] + w_g/2, 4.8, w_g, color=COLORS["lula"], edgecolor="white", lw=0.5, zorder=3)
    ax_g.text(x_g[1] - w_g/2, 3.5 + 1.2, "3.5M", ha="center", va="bottom", fontweight="bold", color="#0D47A1", fontsize=7.5)
    ax_g.text(x_g[1] + w_g/2, 4.8 + 1.2, "4.8M\n(1.4x)", ha="center", va="bottom", fontweight="bold", color="#B71C1C", fontsize=7.5)

    ax_g.bar(x_g[2] - w_g/2, 2.0, w_g, color=COLORS["flavio"], edgecolor="white", lw=0.5, zorder=3)
    ax_g.bar(x_g[2] + w_g/2, 10.0, w_g, color=COLORS["lula"], edgecolor="white", lw=0.5, zorder=3)
    ax_g.text(x_g[2] - w_g/2, 2.0 + 1.2, "2.0M\n(Tropa)", ha="center", va="bottom", color="#0D47A1", fontsize=7.0)
    ax_g.text(x_g[2] + w_g/2, 10.0 + 1.2, "10.0M\n(5.0x)", ha="center", va="bottom", fontweight="bold", color="#B71C1C", fontsize=7.5)

    ax_g.bar(x_g[3] - w_g/2, 17.36, w_g, color="#0D47A1", hatch="//", edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    ax_g.bar(x_g[3] + w_g/2, 7.12, w_g, color="#B71C1C", hatch="//", edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    ax_g.text(x_g[3] - w_g/2, 17.36 + 1.2, "17.4M\n(2.4x)", ha="center", va="bottom", fontweight="bold", color="#0D47A1", fontsize=7.5)
    ax_g.text(x_g[3] + w_g/2, 7.12 + 1.2, "7.1M", ha="center", va="bottom", fontweight="bold", color="#B71C1C", fontsize=7.5)

    ax_g.bar(x_g[4] - w_g/2, 82.0, w_g, color="#0D47A1", hatch="..", edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    ax_g.bar(x_g[4] + w_g/2, 32.5, w_g, color="#B71C1C", hatch="..", edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    ax_g.text(x_g[4] - w_g/2, 82.0 + 1.2, "82.0M\n(2.5x)", ha="center", va="bottom", fontweight="bold", color="#0D47A1", fontsize=7.5)
    ax_g.text(x_g[4] + w_g/2, 32.5 + 1.2, "32.5M", ha="center", va="bottom", fontweight="bold", color="#B71C1C", fontsize=7.5)

    ax_g.set_xticks(x_g)
    ax_g.set_xticklabels(cats_g, fontsize=7.3)
    ax_g.set_ylabel("Followers / Reach (Millions)", fontsize=8.5)
    ax_g.set_ylim(0, 108)
    ax_g.grid(True, linestyle=":", alpha=0.5, axis="y")

    legend_elements_g = [
        Patch(facecolor="#1E88E5", edgecolor="white", label="Flávio Bolsonaro / PL"),
        Patch(facecolor="#E53935", edgecolor="white", label="Lula da Silva / PT"),
        Patch(facecolor="#90CAF9", hatch="//", edgecolor="#0D47A1", label="Surrogate Top-5 Avg (Excl. Candidate)"),
        Patch(facecolor="#90CAF9", hatch="..", edgecolor="#0D47A1", label="Surrogates Unique Reach (Excl. Candidate)"),
    ]
    ax_g.legend(handles=legend_elements_g, loc="upper left", frameon=True, fontsize=7.0)

    subdiv_box = (
        "Follower Ecosystem & Deduplication Architecture (Option B Clarification):\n"
        "• Surrogate Bench (Excl. Candidate): Measures top-5 allied influencers without the presidential nominee.\n"
        "  - PL Bench: Nikolas (52M), Eduardo (13.2M), Gayer (7.8M), Zambelli (7.4M), Damares (6.4M) [Avg: 17.4M, Jaccard 22.4% -> 82.0M Unique]\n"
        "  - PT/Allies Bench: Janones (17.2M), Boulos (8.4M), Erika (4.2M), Gleisi (3.1M), Randolfe (2.7M) [Avg: 7.1M, Jaccard 18.2% -> 32.5M Unique]\n"
        "• Note on Scope: Lula's 39.0M is gross multi-platform followers (IG+X+FB+TT+YT); the 32.5M is deduplicated citizens reached by his 5 allies alone."
    )
    ax_g.annotate(subdiv_box, xy=(0.02, 0.49), xycoords="axes fraction",
                  fontsize=6.4, family="sans-serif",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#FAFAFA", ec="#BDBDBD", alpha=0.95))

    ax_g.set_title("g  Follower Ecosystem: Official Base, Closed WhatsApp & Surrogate Reach (Excl. Candidate)", fontweight="bold", loc="left", fontsize=9.2)

    # -------------------------------------------------------------------------
    # Save & Export
    # -------------------------------------------------------------------------
    out_pdf = PROJECT_ROOT / "figures" / "election_2026_final_observational_figure.pdf"
    out_svg = PROJECT_ROOT / "figures" / "election_2026_final_observational_figure.svg"
    out_png = PROJECT_ROOT / "figures" / "election_2026_final_observational_figure.png"

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated Final Observational Figure:")
    print(f" - PDF: {out_pdf}")
    print(f" - SVG: {out_svg}")
    print(f" - PNG: {out_png}")

    artifact_dir = Path("/Users/rennocosta/.gemini/antigravity/brain/9a018403-8aec-4ff7-aadc-19a18baf7af1")
    if artifact_dir.exists():
        shutil.copy(out_pdf, artifact_dir / "election_2026_final_observational_figure.pdf")
        shutil.copy(out_svg, artifact_dir / "election_2026_final_observational_figure.svg")
        shutil.copy(out_png, artifact_dir / "election_2026_final_observational_figure.png")
        print(f"Copied to artifact directory: {artifact_dir}")


if __name__ == "__main__":
    main()
