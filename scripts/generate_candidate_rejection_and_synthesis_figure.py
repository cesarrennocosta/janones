#!/usr/bin/env python3
"""
Generate High-Resolution Publication Figures for Candidate Rejection:
1. Standalone Rejection Figure (election_2026_candidate_rejection_trajectory)
2. Unified 3-Graph Synthesis Figure (election_2026_full_electoral_and_rejection_synthesis):
   - Left: Electorate Conviction Segmentation Bar Plot (Datafolha 1-5 Scale)
   - Center: Longitudinal Polling Trajectory (6 Datafolha 1ºT waves + 2ºT + TSE actuals)
   - Right: Candidate Rejection Trajectory (Datafolha "Não Votaria de Jeito Nenhum": Lula vs. Flávio)

Adobe Illustrator Compatibility:
- FontType 42 (TrueType)
- Single text object layout patch to prevent split glyphs.
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
# Colors
# -----------------------------------------------------------------------------
COLOR_LULA_DARK = "#C62828"       # Convict Lulista / Lula (Red)
COLOR_LULA_LIGHT = "#EF9A9A"      # Non-convict Lulista (Light Red)
COLOR_BOLSO_DARK = "#1565C0"      # Convict Bolsonarista / Flávio (Blue)
COLOR_BOLSO_LIGHT = "#90CAF9"     # Non-convict Bolsonarista (Light Blue)
COLOR_NEUTRO = "#7E57C2"          # Neutro / Centro / 3ª Via (Purple)
COLOR_NONE = "#9E9E9E"            # Nenhum / Despolitizado (Grey)

# -----------------------------------------------------------------------------
# Data Definition
# -----------------------------------------------------------------------------
# 1. Electorate Segmentation
segment_categories = ['Lulista', 'Bolsonarista', 'Neutro', 'Nenhum']
convict_vals = [28.0, 23.0, 0.0, 0.0]
non_convict_vals = [12.0, 11.0, 0.0, 0.0]
single_vals = [0.0, 0.0, 19.0, 7.0]
total_vals = [40.0, 34.0, 19.0, 7.0]

# 2. Polling & Election Results (Valid Votes %)
r1_dates = [
    datetime.date(2026, 8, 19),
    datetime.date(2026, 9, 2),
    datetime.date(2026, 9, 10),
    datetime.date(2026, 9, 17),
    datetime.date(2026, 9, 23),
    datetime.date(2026, 10, 1),
    datetime.date(2026, 10, 4)
]
r1_lula = [44.3, 41.8, 43.3, 43.0, 44.9, 45.0, 45.16]
r1_flavio = [37.5, 36.3, 37.8, 39.6, 40.4, 41.0, 47.03]
r1_others = [18.2, 22.0, 18.9, 17.4, 14.7, 14.0, 7.81]

r2_dates = [
    datetime.date(2026, 10, 8),
    datetime.date(2026, 10, 15),
    datetime.date(2026, 10, 22),
    datetime.date(2026, 10, 24),
    datetime.date(2026, 10, 25)
]
r2_flavio = [52.0, 51.0, 50.0, 51.0, 48.21]
r2_lula = [48.0, 49.0, 50.0, 49.0, 51.79]

# 3. Datafolha Rejection Rates ("Não Votaria de Jeito Nenhum", % of Electorate)
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

# Reference base date
base_date = datetime.date(2026, 8, 19)
r1_x = np.array([(d - base_date).days for d in r1_dates], dtype=float)
r2_x = np.array([(d - base_date).days for d in r2_dates], dtype=float)

rej_r1_x = np.array([(d - base_date).days for d in rej_r1_dates], dtype=float)
rej_r2_x = np.array([(d - base_date).days for d in rej_r2_dates], dtype=float)

# Splines for Valid Votes
x_dense_r1 = np.linspace(r1_x[0], r1_x[-1], 200)
spline_r1_lula = PchipInterpolator(r1_x, r1_lula)(x_dense_r1)
spline_r1_flavio = PchipInterpolator(r1_x, r1_flavio)(x_dense_r1)
spline_r1_others = PchipInterpolator(r1_x, r1_others)(x_dense_r1)

x_dense_r2 = np.linspace(r2_x[0], r2_x[-1], 200)
spline_r2_lula = PchipInterpolator(r2_x, r2_lula)(x_dense_r2)
spline_r2_flavio = PchipInterpolator(r2_x, r2_flavio)(x_dense_r2)

# Splines for Rejection
rej_x_dense_r1 = np.linspace(rej_r1_x[0], rej_r1_x[-1], 200)
spline_rej_r1_lula = PchipInterpolator(rej_r1_x, rej_r1_lula)(rej_x_dense_r1)
spline_rej_r1_flavio = PchipInterpolator(rej_r1_x, rej_r1_flavio)(rej_x_dense_r1)

rej_x_dense_r2 = np.linspace(rej_r2_x[0], rej_r2_x[-1], 200)
spline_rej_r2_lula = PchipInterpolator(rej_r2_x, rej_r2_lula)(rej_x_dense_r2)
spline_rej_r2_flavio = PchipInterpolator(rej_r2_x, rej_r2_flavio)(rej_x_dense_r2)


def render_rejection_plot(ax):
    """Render the rejection trajectory onto an axes object."""
    # 1st Round Splines
    ax.plot(rej_x_dense_r1, spline_rej_r1_lula, color=COLOR_LULA_DARK, linewidth=2.4, zorder=3,
            label='Rejeição a Lula (PT)')
    ax.plot(rej_x_dense_r1, spline_rej_r1_flavio, color=COLOR_BOLSO_DARK, linewidth=2.4, zorder=3,
            label='Rejeição a Flávio Bolsonaro (PL)')

    # 2nd Round Splines
    ax.plot(rej_x_dense_r2, spline_rej_r2_lula, color=COLOR_LULA_DARK, linewidth=2.4, zorder=3)
    ax.plot(rej_x_dense_r2, spline_rej_r2_flavio, color=COLOR_BOLSO_DARK, linewidth=2.4, zorder=3)

    # Data Points (Circles)
    ax.scatter(rej_r1_x, rej_r1_lula, color=COLOR_LULA_DARK, s=65, zorder=5, edgecolor='black', linewidth=0.8)
    ax.scatter(rej_r1_x, rej_r1_flavio, color=COLOR_BOLSO_DARK, s=65, zorder=5, edgecolor='black', linewidth=0.8)

    ax.scatter(rej_r2_x, rej_r2_lula, color=COLOR_LULA_DARK, s=65, zorder=5, edgecolor='black', linewidth=0.8)
    ax.scatter(rej_r2_x, rej_r2_flavio, color=COLOR_BOLSO_DARK, s=65, zorder=5, edgecolor='black', linewidth=0.8)

    # Phase Separator
    sep_x = (rej_r1_x[-1] + rej_r2_x[0]) / 2.0
    ax.axvline(sep_x, color='#757575', linestyle='--', linewidth=1.2, alpha=0.8)

    # Phase Banners
    ax.text((rej_r1_x[0] + rej_r1_x[-1]) / 2, 50.8, "1º TURNO (6 Ondas Datafolha)",
            ha='center', va='center', fontsize=9.0, fontweight='bold', color='#424242',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.9))

    ax.text((rej_r2_x[0] + rej_r2_x[-1]) / 2, 50.8, "2º TURNO (Reta Final)",
            ha='center', va='center', fontsize=9.0, fontweight='bold', color='#424242',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.9))

    # Annotations for Key Rejection Dynamics
    # 1. 02/Set: Flávio peak rejection in R1
    ax.annotate("Flávio: 47.0%", xy=(rej_r1_x[1], rej_r1_flavio[1]),
                xytext=(rej_r1_x[1] - 8, rej_r1_flavio[1] + 1.8),
                fontsize=8.2, fontweight='bold', color=COLOR_BOLSO_DARK,
                arrowprops=dict(arrowstyle="->", color=COLOR_BOLSO_DARK, lw=0.8))

    # 2. 17/Set: Tie at 47%
    ax.annotate("Empate em 47.0%", xy=(rej_r1_x[3], 47.0),
                xytext=(rej_r1_x[3] - 7, 48.6),
                fontsize=8.2, fontweight='bold', color='#424242',
                arrowprops=dict(arrowstyle="->", color='#424242', lw=0.8))

    # 3. 08/Out: Post-R1 Dip for Flávio (43%) vs Lula (47%)
    ax.annotate("Pós-1ºT: Flávio cai p/ 43%\n(Lula em 47%)", xy=(rej_r2_x[0], rej_r2_flavio[0]),
                xytext=(rej_r2_x[0] - 11, rej_r2_flavio[0] - 3.2),
                fontsize=8.2, fontweight='bold', color=COLOR_BOLSO_DARK,
                arrowprops=dict(arrowstyle="->", color=COLOR_BOLSO_DARK, lw=0.8))

    # 4. 22/Out: Janones impact (Flávio rejection rises to 46%)
    ax.annotate("Janonismo no WhatsApp:\nFlávio sobe p/ 46%", xy=(rej_r2_x[2], rej_r2_flavio[2]),
                xytext=(rej_r2_x[2] - 12, rej_r2_flavio[2] + 2.2),
                fontsize=8.2, fontweight='bold', color=COLOR_BOLSO_DARK,
                arrowprops=dict(arrowstyle="->", color=COLOR_BOLSO_DARK, lw=0.8))

    # 5. 24/Out (Véspera): Lula 46% vs Flávio 45%
    eve_x = rej_r2_x[3]
    ax.annotate("Véspera: 46.0%", xy=(eve_x, rej_r2_lula[3]),
                xytext=(eve_x + 1.5, rej_r2_lula[3] + 0.4),
                fontsize=8.5, fontweight='bold', color=COLOR_LULA_DARK)
    ax.annotate("Véspera: 45.0%", xy=(eve_x, rej_r2_flavio[3]),
                xytext=(eve_x + 1.5, rej_r2_flavio[3] - 0.4),
                fontsize=8.5, fontweight='bold', color=COLOR_BOLSO_DARK)

    # Shaded band for parity zone (44-48%)
    ax.axhspan(44, 48, color='#EEEEEE', alpha=0.5, zorder=1)
    ax.text(rej_r1_x[0] + 1, 44.3, "Faixa de Cristalização Polarizada (44% - 48%)",
            fontsize=7.8, color='#757575', style='italic')

    # X-axis ticks
    ax.set_xticks([
        rej_r1_x[0],
        rej_r1_x[1],
        rej_r1_x[3],
        rej_r1_x[5],
        rej_r2_x[0],
        rej_r2_x[1],
        rej_r2_x[3]
    ])
    ax.set_xticklabels([
        "19/Ago\n(Início)",
        "02/Set",
        "17/Set",
        "01/Out",
        "08/Out\n(2º Turno)",
        "15/Out",
        "24/Out\n(Véspera)"
    ], fontsize=8.4)

    ax.set_xlim(-3, rej_r2_x[3] + 15)
    ax.set_ylim(38, 52.5)
    ax.set_ylabel("Rejeição Eleitoral (%)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.set_title("Evolução da Rejeição: 'Não Votaria de Jeito Nenhum'\n(Série Histórica Datafolha 2026)",
                 fontsize=11.5, fontweight='bold', pad=12)

    # Subtext
    ax.text(0.5, -0.17, "Datafolha: Eleitores que declaram que não votariam no candidato de jeito nenhum | Linhas: Interpolação Spline (PCHIP)",
            ha='center', va='top', transform=ax.transAxes, fontsize=8.2, color='#555555', style='italic')

    legend_elements = [
        Line2D([0], [0], color=COLOR_LULA_DARK, lw=2.4, label='Rejeição a Lula (PT)'),
        Line2D([0], [0], color=COLOR_BOLSO_DARK, lw=2.4, label='Rejeição a Flávio Bolsonaro (PL)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#424242', markeredgecolor='black',
               markersize=7.5, label='Pesquisa Datafolha')
    ]
    ax.legend(handles=legend_elements, loc='lower left', framealpha=0.92, fontsize=8.5)


def generate_standalone_rejection_figure():
    """Generate standalone rejection figure."""
    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=300)
    render_rejection_plot(ax)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])

    out_pdf = PROJECT_ROOT / "figures" / "election_2026_candidate_rejection_trajectory.pdf"
    out_svg = PROJECT_ROOT / "figures" / "election_2026_candidate_rejection_trajectory.svg"
    out_png = PROJECT_ROOT / "figures" / "election_2026_candidate_rejection_trajectory.png"

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated standalone rejection figure:")
    print(f" - PDF: {out_pdf}")
    print(f" - SVG: {out_svg}")
    print(f" - PNG: {out_png}")

    artifact_dir = Path("/Users/rennocosta/.gemini/antigravity/brain/9a018403-8aec-4ff7-aadc-19a18baf7af1")
    if artifact_dir.exists():
        shutil.copy(out_pdf, artifact_dir / "election_2026_candidate_rejection_trajectory.pdf")
        shutil.copy(out_svg, artifact_dir / "election_2026_candidate_rejection_trajectory.svg")
        shutil.copy(out_png, artifact_dir / "election_2026_candidate_rejection_trajectory.png")


def generate_unified_3_graph_synthesis_figure():
    """Generate unified 3-graph horizontal matrix: Segmentation | Vote Intention | Rejection."""
    fig, (ax_bar, ax_poll, ax_rej) = plt.subplots(1, 3, figsize=(21.5, 6.4), dpi=300)

    # -------------------------------------------------------------------------
    # Panel 1: Electorate Segmentation Bar Plot
    # -------------------------------------------------------------------------
    x_indices = np.arange(len(segment_categories))
    bar_width = 0.52

    ax_bar.bar(x_indices[0], convict_vals[0], width=bar_width, color=COLOR_LULA_DARK,
               edgecolor='#333333', linewidth=0.9, label='Convicto / Cristalizado')
    ax_bar.bar(x_indices[0], non_convict_vals[0], bottom=convict_vals[0], width=bar_width,
               color=COLOR_LULA_LIGHT, edgecolor='#333333', linewidth=0.9, label='Não Convicto / Simpatizante')

    ax_bar.bar(x_indices[1], convict_vals[1], width=bar_width, color=COLOR_BOLSO_DARK,
               edgecolor='#333333', linewidth=0.9)
    ax_bar.bar(x_indices[1], non_convict_vals[1], bottom=convict_vals[1], width=bar_width,
               color=COLOR_BOLSO_LIGHT, edgecolor='#333333', linewidth=0.9)

    ax_bar.bar(x_indices[2], single_vals[2], width=bar_width, color=COLOR_NEUTRO,
               edgecolor='#333333', linewidth=0.9)

    ax_bar.bar(x_indices[3], single_vals[3], width=bar_width, color=COLOR_NONE,
               edgecolor='#333333', linewidth=0.9)

    # Inner & Top Annotations
    ax_bar.text(x_indices[0], convict_vals[0] / 2, "28.0%\n(Convicto)", ha='center', va='center',
                color='white', fontsize=9.5, fontweight='bold')
    ax_bar.text(x_indices[0], convict_vals[0] + non_convict_vals[0] / 2, "12.0%\n(Não conv.)",
                ha='center', va='center', color='#212121', fontsize=8.8, fontweight='bold')
    ax_bar.text(x_indices[0], total_vals[0] + 1.2, "40.0%", ha='center', va='bottom',
                color=COLOR_LULA_DARK, fontsize=12.0, fontweight='bold')

    ax_bar.text(x_indices[1], convict_vals[1] / 2, "23.0%\n(Convicto)", ha='center', va='center',
                color='white', fontsize=9.5, fontweight='bold')
    ax_bar.text(x_indices[1], convict_vals[1] + non_convict_vals[1] / 2, "11.0%\n(Não conv.)",
                ha='center', va='center', color='#212121', fontsize=8.8, fontweight='bold')
    ax_bar.text(x_indices[1], total_vals[1] + 1.2, "34.0%", ha='center', va='bottom',
                color=COLOR_BOLSO_DARK, fontsize=12.0, fontweight='bold')

    ax_bar.text(x_indices[2], single_vals[2] / 2, "19.0%\n(Centro)", ha='center', va='center',
                color='white', fontsize=9.5, fontweight='bold')
    ax_bar.text(x_indices[2], total_vals[2] + 1.2, "19.0%", ha='center', va='bottom',
                color=COLOR_NEUTRO, fontsize=12.0, fontweight='bold')

    ax_bar.text(x_indices[3], single_vals[3] / 2, "7.0%\n(Nenhum)", ha='center', va='center',
                color='white', fontsize=9.0, fontweight='bold')
    ax_bar.text(x_indices[3], total_vals[3] + 1.2, "7.0%", ha='center', va='bottom',
                color='#616161', fontsize=12.0, fontweight='bold')

    ax_bar.set_xticks(x_indices)
    ax_bar.set_xticklabels(['Lulista\n(PT / Aliados)', 'Bolsonarista\n(PL / Aliados)', 'Neutro\n(Centro)', 'Nenhum\n(Despolitizado)'],
                           fontsize=9.5, fontweight='bold')
    ax_bar.set_ylabel("Fração do Eleitorado (%)", fontsize=11, fontweight='bold')
    ax_bar.set_ylim(0, 48)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.45)
    ax_bar.set_title("Identificação & Convicção do Eleitorado\n(Estudo Pré-Eleitoral Datafolha 2026)",
                     fontsize=11.2, fontweight='bold', pad=12)

    ax_bar.text(0.5, -0.17, "Datafolha Escala 1 a 5: Lulista (Notas 4 e 5), Bolsonarista (Notas 1 e 2), Neutro (Nota 3), Nenhum (Fora da régua)",
                ha='center', va='top', transform=ax_bar.transAxes, fontsize=8.0, color='#555555', style='italic')

    legend_elements_bar = [
        Patch(facecolor='#424242', edgecolor='#333333', label='Convicto (Cristalizado: Notas 5 e 1)'),
        Patch(facecolor='#E0E0E0', edgecolor='#333333', label='Não Convicto (Simpatizante: Notas 4 e 2)'),
        Patch(facecolor=COLOR_NEUTRO, edgecolor='#333333', label='Neutro / Independente (Nota 3)'),
        Patch(facecolor=COLOR_NONE, edgecolor='#333333', label='Nenhum / Desengajado (Fora da régua)')
    ]
    ax_bar.legend(handles=legend_elements_bar, loc='upper right', framealpha=0.92, fontsize=8.2)

    # -------------------------------------------------------------------------
    # Panel 2: Polling Trajectory + TSE Actuals
    # -------------------------------------------------------------------------
    ax_poll.plot(x_dense_r1, spline_r1_lula, color=COLOR_LULA_DARK, linewidth=2.4, zorder=3)
    ax_poll.plot(x_dense_r1, spline_r1_flavio, color=COLOR_BOLSO_DARK, linewidth=2.4, zorder=3)
    ax_poll.plot(x_dense_r1, spline_r1_others, color=COLOR_NEUTRO, linewidth=2.2, zorder=3)

    ax_poll.plot(x_dense_r2, spline_r2_lula, color=COLOR_LULA_DARK, linewidth=2.4, zorder=3)
    ax_poll.plot(x_dense_r2, spline_r2_flavio, color=COLOR_BOLSO_DARK, linewidth=2.4, zorder=3)

    ax_poll.scatter(r1_x[:6], r1_lula[:6], color=COLOR_LULA_DARK, s=55, zorder=5, edgecolor='black', linewidth=0.8)
    ax_poll.scatter(r1_x[:6], r1_flavio[:6], color=COLOR_BOLSO_DARK, s=55, zorder=5, edgecolor='black', linewidth=0.8)
    ax_poll.scatter(r1_x[:6], r1_others[:6], color=COLOR_NEUTRO, s=50, zorder=5, edgecolor='black', linewidth=0.8)

    ax_poll.scatter(r2_x[:4], r2_lula[:4], color=COLOR_LULA_DARK, s=55, zorder=5, edgecolor='black', linewidth=0.8)
    ax_poll.scatter(r2_x[:4], r2_flavio[:4], color=COLOR_BOLSO_DARK, s=55, zorder=5, edgecolor='black', linewidth=0.8)

    # TSE Urnas Markers
    tse1_idx = 6
    ax_poll.scatter(r1_x[tse1_idx], r1_flavio[tse1_idx], color=COLOR_BOLSO_DARK, marker='D', s=95, zorder=6,
                    edgecolor='white', linewidth=1.2)
    ax_poll.scatter(r1_x[tse1_idx], r1_lula[tse1_idx], color=COLOR_LULA_DARK, marker='D', s=95, zorder=6,
                    edgecolor='white', linewidth=1.2)
    ax_poll.scatter(r1_x[tse1_idx], r1_others[tse1_idx], color=COLOR_NEUTRO, marker='D', s=80, zorder=6,
                    edgecolor='white', linewidth=1.2)

    ax_poll.scatter(r2_x[4], r2_lula[4], color=COLOR_LULA_DARK, marker='D', s=120, zorder=6,
                    edgecolor='yellow', linewidth=1.5)
    ax_poll.scatter(r2_x[4], r2_flavio[4], color=COLOR_BOLSO_DARK, marker='D', s=110, zorder=6,
                    edgecolor='white', linewidth=1.2)

    sep_x = (r1_x[-1] + r2_x[0]) / 2.0
    ax_poll.axvline(sep_x, color='#757575', linestyle='--', linewidth=1.2, alpha=0.8)

    ax_poll.text((r1_x[0] + r1_x[-1]) / 2, 59.0, "1º TURNO (6 Pesquisas & TSE)",
                 ha='center', va='center', fontsize=8.8, fontweight='bold', color='#424242',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.9))

    ax_poll.text((r2_x[0] + r2_x[-1]) / 2, 59.0, "2º TURNO (Head-to-Head)",
                 ha='center', va='center', fontsize=8.8, fontweight='bold', color='#424242',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.9))

    tse1_x = r1_x[tse1_idx]
    ax_poll.annotate(f"TSE: {r1_flavio[tse1_idx]:.1f}%\n(01/Out: 41%)", xy=(tse1_x, r1_flavio[tse1_idx]),
                     xytext=(tse1_x - 12.5, r1_flavio[tse1_idx] + 3.2),
                     fontsize=8.0, fontweight='bold', color=COLOR_BOLSO_DARK,
                     arrowprops=dict(arrowstyle="->", color=COLOR_BOLSO_DARK, lw=0.9))
    ax_poll.annotate(f"TSE: {r1_lula[tse1_idx]:.1f}%", xy=(tse1_x, r1_lula[tse1_idx]),
                     xytext=(tse1_x - 11, r1_lula[tse1_idx] - 3.4),
                     fontsize=8.2, fontweight='bold', color=COLOR_LULA_DARK,
                     arrowprops=dict(arrowstyle="->", color=COLOR_LULA_DARK, lw=0.9))
    ax_poll.annotate(f"3ª Via: {r1_others[tse1_idx]:.1f}%", xy=(tse1_x, r1_others[tse1_idx]),
                     xytext=(tse1_x - 11, r1_others[tse1_idx] - 2.8),
                     fontsize=7.8, fontweight='bold', color=COLOR_NEUTRO,
                     arrowprops=dict(arrowstyle="->", color=COLOR_NEUTRO, lw=0.8))

    eve_x = r2_x[3]
    ax_poll.annotate("Véspera: 51.0%", xy=(eve_x, r2_flavio[3]),
                     xytext=(eve_x - 13.5, r2_flavio[3] + 2.2),
                     fontsize=8.2, fontweight='bold', color=COLOR_BOLSO_DARK,
                     arrowprops=dict(arrowstyle="->", color=COLOR_BOLSO_DARK, lw=0.9))
    ax_poll.annotate("Véspera: 49.0%", xy=(eve_x, r2_lula[3]),
                     xytext=(eve_x - 13.5, r2_lula[3] - 3.5),
                     fontsize=8.2, fontweight='bold', color=COLOR_LULA_DARK,
                     arrowprops=dict(arrowstyle="->", color=COLOR_LULA_DARK, lw=0.9))

    tse2_x = r2_x[4]
    ax_poll.annotate(f"★ Eleito: {r2_lula[4]:.2f}%", xy=(tse2_x, r2_lula[4]),
                     xytext=(tse2_x + 1.8, r2_lula[4] + 0.6),
                     fontsize=8.8, fontweight='bold', color=COLOR_LULA_DARK,
                     bbox=dict(boxstyle='square,pad=0.2', facecolor='#FFEBEE', edgecolor=COLOR_LULA_DARK, lw=0.7))
    ax_poll.annotate(f"TSE: {r2_flavio[4]:.2f}%", xy=(tse2_x, r2_flavio[4]),
                     xytext=(tse2_x + 1.8, r2_flavio[4] - 1.2),
                     fontsize=8.4, fontweight='bold', color=COLOR_BOLSO_DARK,
                     bbox=dict(boxstyle='square,pad=0.2', facecolor='#E3F2FD', edgecolor=COLOR_BOLSO_DARK, lw=0.7))

    ax_poll.set_xticks([r1_x[0], r1_x[1], r1_x[3], r1_x[6], r2_x[1], r2_x[4]])
    ax_poll.set_xticklabels(["19/Ago\n(Início)", "02/Set", "17/Set", "04/Out\n(TSE 1ºT)", "15/Out", "25/Out\n(TSE 2ºT)"],
                           fontsize=8.5)

    ax_poll.set_xlim(-3, r2_x[4] + 16)
    ax_poll.set_ylim(2, 62)
    ax_poll.set_ylabel("Votos Válidos (%)", fontsize=11, fontweight='bold')
    ax_poll.grid(True, linestyle='--', alpha=0.45)
    ax_poll.set_title("Trajetória de Votos Válidos: Pesquisas & TSE\n(Série Histórica Datafolha 2026)",
                      fontsize=11.2, fontweight='bold', pad=12)

    ax_poll.text(0.5, -0.17, "Círculos: Pesquisas Datafolha | Losangos: Resultados TSE Urnas | Linhas: Spline PCHIP",
                 ha='center', va='top', transform=ax_poll.transAxes, fontsize=8.0, color='#555555', style='italic')

    legend_elements_poll = [
        Line2D([0], [0], color=COLOR_LULA_DARK, lw=2.4, label='Lula (PT)'),
        Line2D([0], [0], color=COLOR_BOLSO_DARK, lw=2.4, label='Flávio Bolsonaro (PL)'),
        Line2D([0], [0], color=COLOR_NEUTRO, lw=2.2, label='Outros / 3ª Via (1ºT)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#424242', markeredgecolor='black',
               markersize=7, label='Pesquisa Datafolha'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#424242', markeredgecolor='white',
               markersize=7.5, label='Resultado Oficial TSE')
    ]
    ax_poll.legend(handles=legend_elements_poll, loc='lower left', framealpha=0.92, fontsize=8.0)

    # -------------------------------------------------------------------------
    # Panel 3: Candidate Rejection Trajectory
    # -------------------------------------------------------------------------
    render_rejection_plot(ax_rej)

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])

    out_pdf = PROJECT_ROOT / "figures" / "election_2026_full_electoral_and_rejection_synthesis.pdf"
    out_svg = PROJECT_ROOT / "figures" / "election_2026_full_electoral_and_rejection_synthesis.svg"
    out_png = PROJECT_ROOT / "figures" / "election_2026_full_electoral_and_rejection_synthesis.png"

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated unified 3-graph synthesis figure:")
    print(f" - PDF: {out_pdf}")
    print(f" - SVG: {out_svg}")
    print(f" - PNG: {out_png}")

    artifact_dir = Path("/Users/rennocosta/.gemini/antigravity/brain/9a018403-8aec-4ff7-aadc-19a18baf7af1")
    if artifact_dir.exists():
        shutil.copy(out_pdf, artifact_dir / "election_2026_full_electoral_and_rejection_synthesis.pdf")
        shutil.copy(out_svg, artifact_dir / "election_2026_full_electoral_and_rejection_synthesis.svg")
        shutil.copy(out_png, artifact_dir / "election_2026_full_electoral_and_rejection_synthesis.png")


def main():
    generate_standalone_rejection_figure()
    generate_unified_3_graph_synthesis_figure()


if __name__ == "__main__":
    main()
