#!/usr/bin/env python3
"""Figure 1 Replication: Empirical Data and Polling Dynamics of the 2026 Brazilian Presidential Election.

Author: César Rennó-Costa (Brain Institute, UFRN, Natal, RN, Brazil)
Paper: "Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections"
Target: Nature Human Behaviour (2026)
Contact: rennocosta@neuro.ufrn.br

Analytical Panels (3x3 Grid):
- Panel a: Official TSE Outcomes & Voter Turnout Mobilization (R1 vs R2)
- Panel b: Longitudinal Polling Trajectory (Datafolha tracking vs official ballot tallies)
- Panel c: Candidate Rejection Dynamics (Datafolha "Não Votaria de Jeito Nenhum")
- Panel d: Statutory Free Broadcast Airtime (HGPE daily blocks on TV & Radio)
- Panel e: Macro Party Public Financing (FEFC R$ 4.96B pool) vs TSE Presidential Spending Ceilings
- Panel f: Presidential Ticket Receipts: Public Injections vs Private Micro-Crowdfunding
- Panel g: Quaest Digital Popularity Index (IPD) Benchmark Waves (Net Turnaround Swing)
- Panel h: Daily Social Interaction Volumes (Millions/day, Bites data across milestones)
- Panel i: Follower Ecosystem & Bench Architecture (Candidate bases, WhatsApp, Top-5 surrogates)
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from matplotlib import _text_helpers

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Publication-grade typography and Illustrator TrueType fonttype settings
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

# Color Palette (consistent with manuscript figures)
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
    "amber": "#FFA000",
}


def generate_figure1(output_dir: Path | str = None) -> tuple[Path, Path]:
    """Render and save Figure 1 (Empirical Observational Master Synthesis) in Nature style."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Empirical Data Definitions
    # (a) TSE Actual Results
    tse_rounds = ["1º Turno TSE\n(04/Out/2026)", "2º Turno TSE\n(25/Out/2026)"]
    r1_shares = [47.03, 45.16, 7.81]
    r2_shares = [48.21, 51.79, 0.0]
    r1_nonvoters = 24.70
    r2_nonvoters = 21.79

    # (b) Polling Trajectories
    base_date = datetime.date(2026, 8, 16)
    r1_poll_dates = [
        datetime.date(2026, 8, 19),
        datetime.date(2026, 9, 2),
        datetime.date(2026, 9, 10),
        datetime.date(2026, 9, 17),
        datetime.date(2026, 9, 23),
        datetime.date(2026, 10, 1),
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
        datetime.date(2026, 10, 24),
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

    # (c) Candidate Rejection
    rej_r1_dates = r1_poll_dates
    rej_r1_flavio = [48.0, 49.0, 47.0, 46.0, 45.0, 44.0]
    rej_r1_lula = [42.0, 44.0, 43.0, 44.0, 43.0, 42.0]
    rej_r2_dates = r2_poll_dates
    rej_r2_flavio = [43.0, 44.0, 46.0, 46.0]
    rej_r2_lula = [47.0, 46.0, 45.0, 45.0]

    # (d) TV Airtime
    airtime_rounds = ["1º Turno (HGPE)", "2º Turno (HGPE)"]
    airtime_lula = [5.52, 5.0]
    airtime_flavio = [4.33, 5.0]
    airtime_others = [2.65, 0.0]

    # (e) Macro Finance
    parties = ["PL", "PT/Fed.", "UNIÃO", "PSD", "MDB", "REP"]
    fefc_vals = [886.8, 619.9, 536.5, 420.0, 404.0, 343.0]

    # (f) Presidential Ticket Receipts
    campaign_entities = ["Flávio (PL)", "Lula (PT)"]
    camp_fefc = [47.0, 128.5]
    camp_private = [43.2, 4.8]

    # (g) Quaest IPD
    ipd_waves = ["Onda 1\n(04-05/Out)", "Onda 2\n(15/Out)", "Onda 3\n(19/Out)"]
    ipd_flavio = [80.0, 81.4, 76.6]
    ipd_lula = [70.7, 74.2, 83.3]

    # (h) Bites Interactions
    bites_periods = ["Pré-Blitz 1ºT\n(20-30/Set)", "Pós-Debate\n(16/Out)", "Janela 2ºT\n(11-23/Out)"]
    bites_flavio = [6.84, 5.0, 2.41]
    bites_lula = [1.21, 25.5, 8.52]

    # (i) Follower Ecosystem
    cats_i = ["Bases Diretas\n(5 Redes)", "Partidos\n(Canais Oficiais)", "WhatsApp\n(Grupos Fechados)", "Bancada Aliada\n(Top-5 Média)", "Alcance Único\n(Deduplicado)"]
    flavio_i = [24.1, 3.5, 2.0, 17.36, 82.0]
    lula_i = [39.0, 4.8, 10.0, 7.12, 32.5]

    # 2. Render Figure (3x3 Grid)
    fig = plt.figure(figsize=(7.17, 9.13), dpi=300)
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.32,
                  left=0.07, right=0.97, top=0.96, bottom=0.04)

    # Panel a: TSE Outcomes
    ax_a = fig.add_subplot(gs[0, 0])
    x_a = np.array([0, 1])
    w_a = 0.22
    ax_a.bar(x_a - w_a, [r1_shares[0], r2_shares[0]], w_a, color=COLORS["flavio"], label="Flávio Bolsonaro", edgecolor="white", lw=0.4)
    ax_a.bar(x_a, [r1_shares[1], r2_shares[1]], w_a, color=COLORS["lula"], label="Lula da Silva", edgecolor="white", lw=0.4)
    ax_a.bar(x_a + w_a, [r1_shares[2], r2_shares[2]], w_a, color=COLORS["others"], label="Outros / 3ª Via", edgecolor="white", lw=0.4)
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(tse_rounds, fontsize=5.8)
    ax_a.set_ylabel("Votos Válidos (%)", fontsize=6.2)
    ax_a.set_ylim(0, 62)
    ax_a.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_a.legend(loc="upper left", fontsize=4.8, frameon=True)
    ax_a.text(0 - w_a, 47.03 + 1.0, "47.03%", ha="center", fontsize=5.2, fontweight="bold", color=COLORS["flavio_dark"])
    ax_a.text(0, 45.16 + 1.0, "45.16%", ha="center", fontsize=5.2, fontweight="bold", color=COLORS["lula_dark"])
    ax_a.text(0 + w_a, 7.81 + 1.0, "7.81%", ha="center", fontsize=5.0, color=COLORS["others"])
    ax_a.text(1 - w_a, 48.21 + 1.0, "48.21%", ha="center", fontsize=5.2, fontweight="bold", color=COLORS["flavio_dark"])
    ax_a.text(1, 51.79 + 1.0, "51.79%", ha="center", fontsize=5.2, fontweight="bold", color=COLORS["lula_dark"])
    ax_a.set_title("a  Resultados Oficiais TSE & Mobilização", fontweight="bold", loc="left", fontsize=6.5)

    # Panel b: Polling Trajectory
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(r1_poll_x, r1_poll_lula, "o-", color=COLORS["lula"], ms=3.5, lw=1.2, label="Lula (Datafolha)")
    ax_b.plot(r1_poll_x, r1_poll_flavio, "s-", color=COLORS["flavio"], ms=3.5, lw=1.2, label="Flávio (Datafolha)")
    ax_b.plot(r1_poll_x, r1_poll_others, "^-", color=COLORS["others"], ms=3.0, lw=1.0, label="Outros")
    ax_b.scatter([tse1_x], [tse1_lula], color=COLORS["lula"], marker="*", s=55, zorder=5, edgecolor="black", lw=0.4, label="TSE 1ºT")
    ax_b.scatter([tse1_x], [tse1_flavio], color=COLORS["flavio"], marker="*", s=55, zorder=5, edgecolor="black", lw=0.4)
    ax_b.plot(r2_poll_x, r2_poll_lula, "o--", color=COLORS["lula"], ms=3.5, lw=1.2)
    ax_b.plot(r2_poll_x, r2_poll_flavio, "s--", color=COLORS["flavio"], ms=3.5, lw=1.2)
    ax_b.scatter([tse2_x], [tse2_lula], color=COLORS["lula"], marker="*", s=75, zorder=5, edgecolor="gold", lw=0.8, label="TSE 2ºT (51.79%)")
    ax_b.scatter([tse2_x], [tse2_flavio], color=COLORS["flavio"], marker="*", s=75, zorder=5, edgecolor="gold", lw=0.8)
    ax_b.axvline(tse1_x, color="grey", ls=":", lw=0.8, alpha=0.7)
    ax_b.set_ylabel("Intenção de Voto Válido (%)", fontsize=6.2)
    ax_b.set_ylim(5, 58)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    ax_b.legend(loc="lower left", fontsize=4.8, frameon=True)
    ax_b.set_title("b  Trajetória Longitudinal (Pesquisas vs Urnas)", fontweight="bold", loc="left", fontsize=6.5)

    # Panel c: Rejection Dynamics
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.plot(r1_poll_x, rej_r1_flavio, "s-", color=COLORS["flavio"], ms=3.5, lw=1.2, label="Rejeição Flávio")
    ax_c.plot(r1_poll_x, rej_r1_lula, "o-", color=COLORS["lula"], ms=3.5, lw=1.2, label="Rejeição Lula")
    ax_c.plot(r2_poll_x, rej_r2_flavio, "s--", color=COLORS["flavio"], ms=3.5, lw=1.2)
    ax_c.plot(r2_poll_x, rej_r2_lula, "o--", color=COLORS["lula"], ms=3.5, lw=1.2)
    ax_c.axvline(tse1_x, color="grey", ls=":", lw=0.8, alpha=0.7)
    ax_c.text(r2_poll_x[-1] + 0.5, rej_r2_flavio[-1], "46.0%", color=COLORS["flavio_dark"], fontsize=5.2, fontweight="bold")
    ax_c.text(r2_poll_x[-1] + 0.5, rej_r2_lula[-1] - 1.0, "45.0%", color=COLORS["lula_dark"], fontsize=5.2, fontweight="bold")
    ax_c.set_ylabel("Rejeição Não Votaria (%)", fontsize=6.2)
    ax_c.set_ylim(35, 55)
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(loc="upper left", fontsize=4.8, frameon=True)
    ax_c.set_title("c  Dinâmica de Rejeição de Candidatos", fontweight="bold", loc="left", fontsize=6.5)

    # Panel d: Free TV & Radio Airtime (HGPE)
    ax_d = fig.add_subplot(gs[1, 0])
    x_d = np.array([0, 1])
    w_d = 0.22
    ax_d.bar(x_d - w_d, airtime_lula, w_d, color=COLORS["lula"], label="Lula / Coligação", edgecolor="white", lw=0.4)
    ax_d.bar(x_d, airtime_flavio, w_d, color=COLORS["flavio"], label="Flávio / Coligação", edgecolor="white", lw=0.4)
    ax_d.bar(x_d + w_d, airtime_others, w_d, color=COLORS["others"], label="Outros", edgecolor="white", lw=0.4)
    ax_d.text(0 - w_d, 5.52 + 0.1, "5m31s\n(+27%)", ha="center", fontsize=4.8, color=COLORS["lula_dark"], fontweight="bold")
    ax_d.text(0, 4.33 + 0.1, "4m20s", ha="center", fontsize=4.8, color=COLORS["flavio_dark"])
    ax_d.text(1 - w_d/2, 5.0 + 0.1, "5m00s\n(Paridade)", ha="center", fontsize=4.8, color="#333333", fontweight="bold")
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(airtime_rounds, fontsize=5.8)
    ax_d.set_ylabel("Tempo Diário por Bloco (min)", fontsize=6.2)
    ax_d.set_ylim(0, 7.5)
    ax_d.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_d.legend(loc="upper right", fontsize=4.8, frameon=True)
    ax_d.set_title("d  Horário Gratuito de TV & Rádio (HGPE)", fontweight="bold", loc="left", fontsize=6.5)

    # Panel e: Macro Party Public Funding (FEFC)
    ax_e = fig.add_subplot(gs[1, 1])
    x_e = np.arange(len(parties))
    p_colors = [COLORS["flavio"], COLORS["lula"], "#7E57C2", "#FF7043", "#26A69A", "#42A5F5"]
    ax_e.bar(x_e, fefc_vals, width=0.55, color=p_colors, edgecolor="white", lw=0.4)
    for i, v in enumerate(fefc_vals):
        ax_e.text(i, v + 15, f"R${v:.0f}M", ha="center", fontsize=4.8, fontweight="bold")
    ax_e.axhline(88.9, color="#666666", ls="--", lw=0.8, label="Teto Presidencial 1ºT (R$ 88.9M)")
    ax_e.axhline(133.4, color="#333333", ls=":", lw=0.8, label="Teto Acumulado 2ºT (R$ 133.4M)")
    ax_e.set_xticks(x_e)
    ax_e.set_xticklabels(parties, fontsize=5.8)
    ax_e.set_ylabel("Fundo Eleitoral (R$ Milhões)", fontsize=6.2)
    ax_e.set_ylim(0, 1050)
    ax_e.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_e.legend(loc="upper right", fontsize=4.6, frameon=True)
    ax_e.set_title("e  Fundo Especial de Financiamento (FEFC)", fontweight="bold", loc="left", fontsize=6.5)

    # Panel f: Presidential Campaign Receipts
    ax_f = fig.add_subplot(gs[1, 2])
    x_f = np.array([0, 1])
    w_f = 0.40
    ax_f.bar(x_f, camp_fefc, w_f, label="Fundo Público (FEFC)", color="#43A047", edgecolor="white", lw=0.4)
    ax_f.bar(x_f, camp_private, w_f, bottom=camp_fefc, label="Doações / Vaquinhas (Privado)", color="#FB8C00", edgecolor="white", lw=0.4)
    ax_f.text(0, camp_fefc[0] + camp_private[0] + 1.5, f"R$ {camp_fefc[0]+camp_private[0]:.1f}M\n(48% Privado: 9.0x)", ha="center", fontsize=4.8, color="#BF360C", fontweight="bold")
    ax_f.text(1, camp_fefc[1] + camp_private[1] + 1.5, f"R$ {camp_fefc[1]+camp_private[1]:.1f}M\n(96% FEFC)", ha="center", fontsize=4.8, color="#1B5E20", fontweight="bold")
    ax_f.set_xticks(x_f)
    ax_f.set_xticklabels(campaign_entities, fontsize=5.8)
    ax_f.set_ylabel("Receitas da Campanha (R$ Milhões)", fontsize=6.2)
    ax_f.set_ylim(0, 155)
    ax_f.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_f.legend(loc="upper left", fontsize=4.8, frameon=True)
    ax_f.set_title("f  Receitas de Campanha: Público vs Privado", fontweight="bold", loc="left", fontsize=6.5)

    # Panel g: Quaest IPD Benchmark Waves
    ax_g = fig.add_subplot(gs[2, 0])
    x_g = np.array([0, 1, 2])
    w_g = 0.28
    ax_g.bar(x_g - w_g/2, ipd_flavio, w_g, color=COLORS["flavio"], label="Flávio", edgecolor="white", lw=0.4)
    ax_g.bar(x_g + w_g/2, ipd_lula, w_g, color=COLORS["lula"], label="Lula", edgecolor="white", lw=0.4)
    ax_g.text(0, 83, "+9.3 pts\n(Flávio)", ha="center", fontsize=4.8, color=COLORS["flavio_dark"], fontweight="bold")
    ax_g.text(2, 85, "+6.7 pts\n(Lula: +16.0 Net)", ha="center", fontsize=4.8, color=COLORS["lula_dark"], fontweight="bold")
    ax_g.set_xticks(x_g)
    ax_g.set_xticklabels(ipd_waves, fontsize=5.6)
    ax_g.set_ylabel("Índice IPD (Escala 0–100)", fontsize=6.2)
    ax_g.set_ylim(60, 92)
    ax_g.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_g.legend(loc="lower left", fontsize=4.8, frameon=True)
    ax_g.set_title("g  Popularidade Digital Quaest (IPD)", fontweight="bold", loc="left", fontsize=6.5)

    # Panel h: Average Daily Social Interactions
    ax_h = fig.add_subplot(gs[2, 1])
    x_h = np.array([0, 1, 2])
    w_h = 0.28
    ax_h.bar(x_h - w_h/2, bites_flavio, w_h, color=COLORS["flavio"], label="Flávio", edgecolor="white", lw=0.4)
    ax_h.bar(x_h + w_h/2, bites_lula, w_h, color=COLORS["lula"], label="Lula", edgecolor="white", lw=0.4)
    ax_h.text(0, 7.5, "5.6x Flávio", ha="center", fontsize=4.8, color=COLORS["flavio_dark"], fontweight="bold")
    ax_h.text(1, 26.5, "25.5M (5.1x)", ha="center", fontsize=4.8, color=COLORS["lula_dark"], fontweight="bold")
    ax_h.text(2, 9.3, "3.5x Lula", ha="center", fontsize=4.8, color=COLORS["lula_dark"], fontweight="bold")
    ax_h.set_xticks(x_h)
    ax_h.set_xticklabels(bites_periods, fontsize=5.6)
    ax_h.set_ylabel("Interações Diárias (Milhões/dia)", fontsize=6.2)
    ax_h.set_ylim(0, 30)
    ax_h.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_h.legend(loc="upper left", fontsize=4.8, frameon=True)
    ax_h.set_title("h  Volume de Interações Sociais (Bites)", fontweight="bold", loc="left", fontsize=6.5)

    # Panel i: Follower Ecosystem & Bench Architecture
    ax_i = fig.add_subplot(gs[2, 2])
    x_i = np.arange(len(cats_i))
    w_i = 0.32
    ax_i.bar(x_i - w_i/2, flavio_i, w_i, color=COLORS["flavio"], label="Flávio / PL", edgecolor="white", lw=0.4)
    ax_i.bar(x_i + w_i/2, lula_i, w_i, color=COLORS["lula"], label="Lula / PT", edgecolor="white", lw=0.4)
    ax_i.text(0, 41, "39.0M\n(1.6x)", ha="center", fontsize=4.8, color=COLORS["lula_dark"], fontweight="bold")
    ax_i.text(2, 11, "10.0M\n(5.0x)", ha="center", fontsize=4.8, color=COLORS["lula_dark"], fontweight="bold")
    ax_i.text(3, 19, "17.4M\n(2.4x)", ha="center", fontsize=4.8, color=COLORS["flavio_dark"], fontweight="bold")
    ax_i.text(4, 84, "82.0M\n(2.5x)", ha="center", fontsize=4.8, color=COLORS["flavio_dark"], fontweight="bold")
    ax_i.set_xticks(x_i)
    ax_i.set_xticklabels(cats_i, fontsize=5.0)
    ax_i.set_ylabel("Seguidores / Alcance (Milhões)", fontsize=6.2)
    ax_i.set_ylim(0, 96)
    ax_i.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_i.legend(loc="upper left", fontsize=4.8, frameon=True)
    ax_i.set_title("i  Ecossistema Digital & Bancada Aliada", fontweight="bold", loc="left", fontsize=6.5)

    pdf_path = output_dir / "figure0_jan_electiondata.pdf"
    png_path = output_dir / "figure0_jan_electiondata.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 1] Saved: {pdf_path}")
    print(f"[Figure 1] Saved: {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 1 for the 2026 election manuscript.")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "figures"), help="Output directory for generated figures")
    args = parser.parse_args()
    generate_figure1(args.output_dir)
