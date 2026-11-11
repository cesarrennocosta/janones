#!/usr/bin/env python3
"""
Generate Two-Graph High-Resolution Publication Figure:
1. Left Graph: Electorate Segmentation Bar Plot (Datafolha 1-to-5 Scale)
   - Lulista (Red), Bolsonarista (Blue), Neutro (Purple), None (Grey)
   - Non-convict segments integrated as lighter-colored sub-bars within each ideological column.
2. Right Graph: Datafolha Polling Trajectory + TSE Actual Results
   - Valid votes for 1st Round (Lula, Flávio, Others) and 2nd Round (Lula, Flávio)
   - Data points plotted as distinct marks (Polls vs. TSE Urnas)
   - Spline interpolation curves (Lula: Red, Flávio: Blue, Others: Purple)

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
# Data Definition
# -----------------------------------------------------------------------------
# 1. Electorate Conviction Study (Datafolha 2026 Escala 1 a 5)
segment_categories = ['Lulista', 'Bolsonarista', 'Neutro', 'Nenhum']
convict_vals = [28.0, 23.0, 0.0, 0.0]
non_convict_vals = [12.0, 11.0, 0.0, 0.0]
single_vals = [0.0, 0.0, 19.0, 7.0]
total_vals = [40.0, 34.0, 19.0, 7.0]

# Palette requested by user:
COLOR_LULA_DARK = "#C62828"       # Convict Lulista (Nota 5)
COLOR_LULA_LIGHT = "#EF9A9A"      # Non-convict Lulista (Nota 4)
COLOR_BOLSO_DARK = "#1565C0"      # Convict Bolsonarista (Nota 1)
COLOR_BOLSO_LIGHT = "#90CAF9"     # Non-convict Bolsonarista (Nota 2)
COLOR_NEUTRO = "#7E57C2"          # Neutro / Centro / Others (Nota 3)
COLOR_NONE = "#9E9E9E"            # Nenhum / Despolitizado

# 2. Polling and Election Results (Valid Votes %)
# 1st Round Campaign: Exactly the 6 Datafolha Polls + 04/Out TSE Urnas
r1_dates = [
    datetime.date(2026, 8, 19),   # 1ª Rodada Datafolha
    datetime.date(2026, 9, 2),    # 2ª Rodada Datafolha
    datetime.date(2026, 9, 10),   # 3ª Rodada Datafolha
    datetime.date(2026, 9, 17),   # 4ª Rodada Datafolha
    datetime.date(2026, 9, 23),   # 5ª Rodada Datafolha
    datetime.date(2026, 10, 1),   # 6ª Rodada Datafolha (Reta final)
    datetime.date(2026, 10, 4)    # TSE Oficial Urnas 1ºT
]
r1_lula = [44.3, 41.8, 43.3, 43.0, 44.9, 45.0, 45.16]
r1_flavio = [37.5, 36.3, 37.8, 39.6, 40.4, 41.0, 47.03]
r1_others = [18.2, 22.0, 18.9, 17.4, 14.7, 14.0, 7.81]

# 2nd Round Campaign
r2_dates = [
    datetime.date(2026, 10, 8),
    datetime.date(2026, 10, 15),
    datetime.date(2026, 10, 22),
    datetime.date(2026, 10, 24),  # Véspera Datafolha 2ºT (Flávio 51% x Lula 49%)
    datetime.date(2026, 10, 25)   # TSE Oficial Urnas 2ºT (Lula 51.79% x Flávio 48.21%)
]
r2_flavio = [52.0, 51.0, 50.0, 51.0, 48.21]
r2_lula = [48.0, 49.0, 50.0, 49.0, 51.79]

# Reference date for numerical x-axis (days from Aug 19)
base_date = datetime.date(2026, 8, 19)
r1_x = np.array([(d - base_date).days for d in r1_dates], dtype=float)
r2_x = np.array([(d - base_date).days for d in r2_dates], dtype=float)

# -----------------------------------------------------------------------------
# Generate Spline Interpolations
# -----------------------------------------------------------------------------
# 1st Round Splines (PCHIP monotonic interpolation)
x_dense_r1 = np.linspace(r1_x[0], r1_x[-1], 200)
spline_r1_lula = PchipInterpolator(r1_x, r1_lula)(x_dense_r1)
spline_r1_flavio = PchipInterpolator(r1_x, r1_flavio)(x_dense_r1)
spline_r1_others = PchipInterpolator(r1_x, r1_others)(x_dense_r1)

# 2nd Round Splines (PCHIP monotonic interpolation)
x_dense_r2 = np.linspace(r2_x[0], r2_x[-1], 200)
spline_r2_lula = PchipInterpolator(r2_x, r2_lula)(x_dense_r2)
spline_r2_flavio = PchipInterpolator(r2_x, r2_flavio)(x_dense_r2)


def main():
    fig, (ax_bar, ax_poll) = plt.subplots(1, 2, figsize=(15.2, 6.4), dpi=300)

    # =========================================================================
    # LEFT GRAPH: Electorate Segmentation Bar Plot
    # =========================================================================
    x_indices = np.arange(len(segment_categories))
    bar_width = 0.52

    # Lulista bar (Stacked: bottom=Convict, top=Non-convict)
    ax_bar.bar(x_indices[0], convict_vals[0], width=bar_width, color=COLOR_LULA_DARK,
               edgecolor='#333333', linewidth=0.9, label='Convicto / Cristalizado')
    ax_bar.bar(x_indices[0], non_convict_vals[0], bottom=convict_vals[0], width=bar_width,
               color=COLOR_LULA_LIGHT, edgecolor='#333333', linewidth=0.9, label='Não Convicto / Simpatizante')

    # Bolsonarista bar (Stacked: bottom=Convict, top=Non-convict)
    ax_bar.bar(x_indices[1], convict_vals[1], width=bar_width, color=COLOR_BOLSO_DARK,
               edgecolor='#333333', linewidth=0.9)
    ax_bar.bar(x_indices[1], non_convict_vals[1], bottom=convict_vals[1], width=bar_width,
               color=COLOR_BOLSO_LIGHT, edgecolor='#333333', linewidth=0.9)

    # Neutro bar (Single)
    ax_bar.bar(x_indices[2], single_vals[2], width=bar_width, color=COLOR_NEUTRO,
               edgecolor='#333333', linewidth=0.9)

    # Nenhum bar (Single)
    ax_bar.bar(x_indices[3], single_vals[3], width=bar_width, color=COLOR_NONE,
               edgecolor='#333333', linewidth=0.9)

    # Inner & Top Annotations
    # Lulista
    ax_bar.text(x_indices[0], convict_vals[0] / 2, "28.0%\n(Convicto)", ha='center', va='center',
                color='white', fontsize=10.0, fontweight='bold')
    ax_bar.text(x_indices[0], convict_vals[0] + non_convict_vals[0] / 2, "12.0%\n(Não conv.)",
                ha='center', va='center', color='#212121', fontsize=9.0, fontweight='bold')
    ax_bar.text(x_indices[0], total_vals[0] + 1.2, "40.0%", ha='center', va='bottom',
                color=COLOR_LULA_DARK, fontsize=12.5, fontweight='bold')

    # Bolsonarista
    ax_bar.text(x_indices[1], convict_vals[1] / 2, "23.0%\n(Convicto)", ha='center', va='center',
                color='white', fontsize=10.0, fontweight='bold')
    ax_bar.text(x_indices[1], convict_vals[1] + non_convict_vals[1] / 2, "11.0%\n(Não conv.)",
                ha='center', va='center', color='#212121', fontsize=9.0, fontweight='bold')
    ax_bar.text(x_indices[1], total_vals[1] + 1.2, "34.0%", ha='center', va='bottom',
                color=COLOR_BOLSO_DARK, fontsize=12.5, fontweight='bold')

    # Neutro
    ax_bar.text(x_indices[2], single_vals[2] / 2, "19.0%\n(Centro)", ha='center', va='center',
                color='white', fontsize=10.0, fontweight='bold')
    ax_bar.text(x_indices[2], total_vals[2] + 1.2, "19.0%", ha='center', va='bottom',
                color=COLOR_NEUTRO, fontsize=12.5, fontweight='bold')

    # Nenhum
    ax_bar.text(x_indices[3], single_vals[3] / 2, "7.0%\n(Nenhum)", ha='center', va='center',
                color='white', fontsize=9.2, fontweight='bold')
    ax_bar.text(x_indices[3], total_vals[3] + 1.2, "7.0%", ha='center', va='bottom',
                color='#616161', fontsize=12.5, fontweight='bold')

    ax_bar.set_xticks(x_indices)
    ax_bar.set_xticklabels(['Lulista\n(PT / Aliados)', 'Bolsonarista\n(PL / Aliados)', 'Neutro\n(Centro)', 'Nenhum\n(Despolitizado)'],
                           fontsize=10.0, fontweight='bold')
    ax_bar.set_ylabel("Fração do Eleitorado (%)", fontsize=11, fontweight='bold')
    ax_bar.set_ylim(0, 48)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.45)
    ax_bar.set_title("Identificação & Convicção Ideológica do Eleitorado\n(Estudo Pré-Eleitoral Datafolha 2026)",
                     fontsize=11.5, fontweight='bold', pad=12)

    # Explanatory subtext below x-axis
    ax_bar.text(0.5, -0.17, "Datafolha Escala 1 a 5: Lulista (Notas 4 e 5), Bolsonarista (Notas 1 e 2), Neutro (Nota 3), Nenhum (Fora da régua)",
                ha='center', va='top', transform=ax_bar.transAxes, fontsize=8.2, color='#555555', style='italic')

    # Custom legend for conviction status
    legend_elements_bar = [
        Patch(facecolor='#424242', edgecolor='#333333', label='Convicto (Cristalizado: Notas 5 e 1)'),
        Patch(facecolor='#E0E0E0', edgecolor='#333333', label='Não Convicto (Simpatizante: Notas 4 e 2)'),
        Patch(facecolor=COLOR_NEUTRO, edgecolor='#333333', label='Neutro / Independente (Nota 3)'),
        Patch(facecolor=COLOR_NONE, edgecolor='#333333', label='Nenhum / Desengajado (Fora da régua)')
    ]
    ax_bar.legend(handles=legend_elements_bar, loc='upper right', framealpha=0.92, fontsize=8.5)

    # =========================================================================
    # RIGHT GRAPH: Polling Data Points + TSE Actuals + Spline Curves
    # =========================================================================
    # 1st Round Splines
    ax_poll.plot(x_dense_r1, spline_r1_lula, color=COLOR_LULA_DARK, linewidth=2.4, zorder=3)
    ax_poll.plot(x_dense_r1, spline_r1_flavio, color=COLOR_BOLSO_DARK, linewidth=2.4, zorder=3)
    ax_poll.plot(x_dense_r1, spline_r1_others, color=COLOR_NEUTRO, linewidth=2.2, zorder=3)

    # 2nd Round Splines
    ax_poll.plot(x_dense_r2, spline_r2_lula, color=COLOR_LULA_DARK, linewidth=2.4, zorder=3)
    ax_poll.plot(x_dense_r2, spline_r2_flavio, color=COLOR_BOLSO_DARK, linewidth=2.4, zorder=3)

    # Datafolha Poll Markers (Circles)
    # R1 Polls (All 6 Datafolha surveys: 19/08, 02/09, 10/09, 17/09, 23/09, 01/10)
    ax_poll.scatter(r1_x[:6], r1_lula[:6], color=COLOR_LULA_DARK, s=60, zorder=5, edgecolor='black', linewidth=0.8)
    ax_poll.scatter(r1_x[:6], r1_flavio[:6], color=COLOR_BOLSO_DARK, s=60, zorder=5, edgecolor='black', linewidth=0.8)
    ax_poll.scatter(r1_x[:6], r1_others[:6], color=COLOR_NEUTRO, s=55, zorder=5, edgecolor='black', linewidth=0.8)

    # R2 Polls (Oct 8, Oct 15, Oct 22, Oct 24)
    ax_poll.scatter(r2_x[:4], r2_lula[:4], color=COLOR_LULA_DARK, s=60, zorder=5, edgecolor='black', linewidth=0.8)
    ax_poll.scatter(r2_x[:4], r2_flavio[:4], color=COLOR_BOLSO_DARK, s=60, zorder=5, edgecolor='black', linewidth=0.8)

    # TSE Urnas Markers (Diamonds)
    # R1 TSE (Oct 4, Index 6)
    tse1_idx = 6
    ax_poll.scatter(r1_x[tse1_idx], r1_flavio[tse1_idx], color=COLOR_BOLSO_DARK, marker='D', s=100, zorder=6,
                    edgecolor='white', linewidth=1.2)
    ax_poll.scatter(r1_x[tse1_idx], r1_lula[tse1_idx], color=COLOR_LULA_DARK, marker='D', s=100, zorder=6,
                    edgecolor='white', linewidth=1.2)
    ax_poll.scatter(r1_x[tse1_idx], r1_others[tse1_idx], color=COLOR_NEUTRO, marker='D', s=85, zorder=6,
                    edgecolor='white', linewidth=1.2)

    # R2 TSE (Oct 25, Index 4)
    ax_poll.scatter(r2_x[4], r2_lula[4], color=COLOR_LULA_DARK, marker='D', s=130, zorder=6,
                    edgecolor='yellow', linewidth=1.5)
    ax_poll.scatter(r2_x[4], r2_flavio[4], color=COLOR_BOLSO_DARK, marker='D', s=120, zorder=6,
                    edgecolor='white', linewidth=1.2)

    # Phase separator vertical line between R1 and R2
    sep_x = (r1_x[-1] + r2_x[0]) / 2.0
    ax_poll.axvline(sep_x, color='#757575', linestyle='--', linewidth=1.2, alpha=0.8)

    # Phase banners positioned cleanly at the top without collision
    ax_poll.text((r1_x[0] + r1_x[-1]) / 2, 59.0, "1º TURNO (6 Pesquisas Datafolha & TSE)",
                 ha='center', va='center', fontsize=9.2, fontweight='bold', color='#424242',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.9))

    ax_poll.text((r2_x[0] + r2_x[-1]) / 2, 59.0, "2º TURNO (Head-to-Head)",
                 ha='center', va='center', fontsize=9.2, fontweight='bold', color='#424242',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.9))

    # Precise Callouts & Annotations with non-overlapping geometry
    # 1. TSE 1º Turno Callouts
    tse1_x = r1_x[tse1_idx]
    ax_poll.annotate(f"TSE: {r1_flavio[tse1_idx]:.1f}%\n(01/Out: 41.0%)", xy=(tse1_x, r1_flavio[tse1_idx]),
                     xytext=(tse1_x - 12.5, r1_flavio[tse1_idx] + 3.2),
                     fontsize=8.2, fontweight='bold', color=COLOR_BOLSO_DARK,
                     arrowprops=dict(arrowstyle="->", color=COLOR_BOLSO_DARK, lw=0.9))
    ax_poll.annotate(f"TSE: {r1_lula[tse1_idx]:.1f}%", xy=(tse1_x, r1_lula[tse1_idx]),
                     xytext=(tse1_x - 11, r1_lula[tse1_idx] - 3.4),
                     fontsize=8.5, fontweight='bold', color=COLOR_LULA_DARK,
                     arrowprops=dict(arrowstyle="->", color=COLOR_LULA_DARK, lw=0.9))
    ax_poll.annotate(f"3ª Via: {r1_others[tse1_idx]:.1f}%", xy=(tse1_x, r1_others[tse1_idx]),
                     xytext=(tse1_x - 11, r1_others[tse1_idx] - 2.8),
                     fontsize=8.0, fontweight='bold', color=COLOR_NEUTRO,
                     arrowprops=dict(arrowstyle="->", color=COLOR_NEUTRO, lw=0.8))

    # 2. Véspera 2º Turno (Oct 24: Flávio 51% vs Lula 49%) - Callout to the LEFT
    eve_x = r2_x[3]
    ax_poll.annotate("Véspera: 51.0%", xy=(eve_x, r2_flavio[3]),
                     xytext=(eve_x - 13.5, r2_flavio[3] + 2.2),
                     fontsize=8.5, fontweight='bold', color=COLOR_BOLSO_DARK,
                     arrowprops=dict(arrowstyle="->", color=COLOR_BOLSO_DARK, lw=0.9))
    ax_poll.annotate("Véspera: 49.0%", xy=(eve_x, r2_lula[3]),
                     xytext=(eve_x - 13.5, r2_lula[3] - 3.5),
                     fontsize=8.5, fontweight='bold', color=COLOR_LULA_DARK,
                     arrowprops=dict(arrowstyle="->", color=COLOR_LULA_DARK, lw=0.9))

    # 3. TSE 2º Turno (Oct 25: Lula 51.79% vs Flávio 48.21%) - Badges to the RIGHT
    tse2_x = r2_x[4]
    ax_poll.annotate(f"★ Eleito: {r2_lula[4]:.2f}%", xy=(tse2_x, r2_lula[4]),
                     xytext=(tse2_x + 1.8, r2_lula[4] + 0.6),
                     fontsize=9.2, fontweight='bold', color=COLOR_LULA_DARK,
                     bbox=dict(boxstyle='square,pad=0.2', facecolor='#FFEBEE', edgecolor=COLOR_LULA_DARK, lw=0.7))
    ax_poll.annotate(f"TSE: {r2_flavio[4]:.2f}%", xy=(tse2_x, r2_flavio[4]),
                     xytext=(tse2_x + 1.8, r2_flavio[4] - 1.2),
                     fontsize=8.8, fontweight='bold', color=COLOR_BOLSO_DARK,
                     bbox=dict(boxstyle='square,pad=0.2', facecolor='#E3F2FD', edgecolor=COLOR_BOLSO_DARK, lw=0.7))

    # X-axis formatting: evenly spaced major calendar dates and election days
    ax_poll.set_xticks([
        r1_x[0],  # 19/Ago
        r1_x[1],  # 02/Set
        r1_x[3],  # 17/Set
        r1_x[6],  # 04/Out (TSE 1ºT)
        r2_x[1],  # 15/Out (Meio 2ºT)
        r2_x[4]   # 25/Out (TSE 2ºT)
    ])
    ax_poll.set_xticklabels([
        "19/Ago\n(Início)",
        "02/Set\n(Pesquisa 2)",
        "17/Set\n(Pesquisa 4)",
        "04/Out\n(TSE 1ºT)",
        "15/Out",
        "25/Out\n(TSE 2ºT)"
    ], fontsize=8.8)

    ax_poll.set_xlim(-3, r2_x[4] + 16)
    ax_poll.set_ylim(2, 62)
    ax_poll.set_ylabel("Votos Válidos (%)", fontsize=11, fontweight='bold')
    ax_poll.grid(True, linestyle='--', alpha=0.45)
    ax_poll.set_title("Trajetória Eleitoral 2026: Pesquisas Datafolha & Resultado Oficial das Urnas",
                      fontsize=11.5, fontweight='bold', pad=12)

    # Subtitle note for right plot
    ax_poll.text(0.5, -0.17, "Círculos: Pesquisas Datafolha | Losangos: Resultados Oficiais TSE nas Urnas | Linhas: Interpolação Spline (PCHIP)",
                 ha='center', va='top', transform=ax_poll.transAxes, fontsize=8.2, color='#555555', style='italic')

    # Legend for Right Plot
    legend_elements_poll = [
        Line2D([0], [0], color=COLOR_LULA_DARK, lw=2.4, label='Lula (PT)'),
        Line2D([0], [0], color=COLOR_BOLSO_DARK, lw=2.4, label='Flávio Bolsonaro (PL)'),
        Line2D([0], [0], color=COLOR_NEUTRO, lw=2.2, label='Outros / 3ª Via (1ºT)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#424242', markeredgecolor='black',
               markersize=7.5, label='Pesquisa Datafolha'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#424242', markeredgecolor='white',
               markersize=8, label='Resultado Oficial TSE')
    ]
    ax_poll.legend(handles=legend_elements_poll, loc='lower left', framealpha=0.92, fontsize=8.5)

    # -------------------------------------------------------------------------
    # Layout Adjustment and Export
    # -------------------------------------------------------------------------
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])

    out_pdf = PROJECT_ROOT / "figures" / "election_2026_electorate_segmentation_and_polling_trajectory.pdf"
    out_svg = PROJECT_ROOT / "figures" / "election_2026_electorate_segmentation_and_polling_trajectory.svg"
    out_png = PROJECT_ROOT / "figures" / "election_2026_electorate_segmentation_and_polling_trajectory.png"

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Successfully generated refined figures:")
    print(f" - PDF: {out_pdf}")
    print(f" - SVG: {out_svg}")
    print(f" - PNG: {out_png}")

    # Copy to artifacts directory
    artifact_dir = Path("/Users/rennocosta/.gemini/antigravity/brain/9a018403-8aec-4ff7-aadc-19a18baf7af1")
    if artifact_dir.exists():
        shutil.copy(out_pdf, artifact_dir / "election_2026_electorate_segmentation_and_polling_trajectory.pdf")
        shutil.copy(out_svg, artifact_dir / "election_2026_electorate_segmentation_and_polling_trajectory.svg")
        shutil.copy(out_png, artifact_dir / "election_2026_electorate_segmentation_and_polling_trajectory.png")
        print(f"Copied refined figures to artifacts directory: {artifact_dir}")


if __name__ == "__main__":
    main()
