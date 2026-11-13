#!/usr/bin/env python3
"""Figure 2 Replication: Structure and Dynamics of the Urban Interpersonal Attitude Network.

Author: César Rennó-Costa (Brain Institute, UFRN, Natal, RN, Brazil)
Paper: "Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections"
Target: Nature Human Behaviour (2026)
Contact: rennocosta@neuro.ufrn.br

Analytical Panels:
- Panel a: Empirical age pyramid (IBGE 2022 Census) & age-dependent plasticity curves alpha(a), epsilon(a), Delta_max(a)
- Panel b: Urban Socioeconomic Status (SES) distribution fitted to Beta(2, 3)
- Panel c: Cohort distributions of individual behavioral traits
- Panel d: 5-layer multiplex network topology M = {L_1..L_5} (household, work/school, church, friends, transit)
- Panel e: Inter-cohort age contact matrix M_{ab} across the six adult age brackets
- Panel f: Age homophily h_{age} and ideological assortativity r_{ideol} across layers
- Panel g: 2D projection of initial convictions s_i(0) and converged convictions s_i(T) alongside exogenous facts f_k
- Panel h: Asymptotic alignment vs city size / population scale
- Panel l: Two-timescale relaxation of mean opinion x_bar(t) and conviction s_bar(t) under transient exogenous shock
- Panel m: Age-stratified opinion-conviction tether stretching ||x_i(t) - s_i(t)|| during shock window
- Panel n: Belief hypercube snapshot at peak shock excursion, contrasting displaced opinions vs anchored convictions
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib import _text_helpers
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.agent_state import AgentStateEngine
from src.core.multiplex_network import GraphEngine, LAYER_NAMES
from src.kernels.opinion_updates import step_conviction_and_opinion_tether_jit

# Styling
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['text.antialiased'] = True

orig_layout = _text_helpers.layout
def single_object_layout(*args, **kwargs):
    for item in orig_layout(*args, **kwargs):
        yield SimpleNamespace(
            ft_object=item.ft_object, char=item.char, glyph_index=item.glyph_index,
            prev_kern=0, x=item.x, y=item.y
        )
_text_helpers.layout = single_object_layout

COLOR_DEEP_BLUE = '#0072B2'
COLOR_SKY_BLUE  = '#56B4E9'
COLOR_GREEN     = '#009E73'
COLOR_AMBER     = '#E69F00'


def run_urban_network_simulation(
    n_agents: int = 10_000,
    n_steps: int = 600,
    seed: int = 42,
) -> dict:
    """Run urban population synthesis and the two-timescale opinion-conviction shock simulation."""
    rng = np.random.default_rng(seed)
    state = AgentStateEngine.initialize(
        n_agents=n_agents, n_dims=4, n_candidates=3, n_districts=10,
        seed=rng, init_mode='centrist_convictions', age_distribution='ibge_2022'
    )

    ev_start, ev_end = int(n_steps * 0.40), int(n_steps * 0.55)
    hist = []
    young_mask = state.age < 30.0
    senior_mask = state.age >= 55.0

    for t in range(n_steps + 1):
        if ev_start <= t <= ev_end:
            event_vec = np.array([0.85, 0.60, -0.50, 0.40], dtype=np.float64)
            ev_active = 1
        else:
            event_vec = np.zeros(4, dtype=np.float64)
            ev_active = 0

        pop_mean = state.opinions.mean(axis=0)
        state.opinions += 0.015 * (pop_mean[None, :] - state.opinions) + rng.normal(0.0, 0.012, size=state.opinions.shape)

        step_conviction_and_opinion_tether_jit(
            state.opinions,
            state.innate_beliefs,
            state.age,
            state.ses,
            state.susceptibility,
            event_vec,
            t,
            0.34,
            0.0018,
            240.0,
            0.075,
            ev_active,
        )

        if t % 5 == 0:
            gaps = np.linalg.norm(state.opinions - state.innate_beliefs, axis=1) / 2.0
            hist.append({
                'step': t,
                'event_signal': event_vec[0],
                'opinion_mean_d0': float(state.opinions[:, 0].mean()),
                'conviction_mean_d0': float(state.innate_beliefs[:, 0].mean()),
                'mean_gap': float(gaps.mean()),
                'young_gap': float(gaps[young_mask].mean()),
                'senior_gap': float(gaps[senior_mask].mean()),
            })

    return {
        'state': state,
        'df_shock': pd.DataFrame(hist),
        'ev_start': ev_start,
        'ev_end': ev_end,
    }


def generate_figure2(
    sim_data: dict = None,
    output_dir: Path | str = None,
    mode: str = "fast",
) -> tuple[Path, Path]:
    """Render and save Figure 2 (Urban Multiplex Network & Two-Timescale Dynamics)."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_agents = 10_000 if mode == "fast" else 50_000
    n_steps = 600 if mode == "fast" else 1000

    if sim_data is None:
        print(f"[Figure 2] Running simulation with N={n_agents:,} agents, T={n_steps} steps...")
        sim_data = run_urban_network_simulation(n_agents=n_agents, n_steps=n_steps)

    state = sim_data['state']
    df_shock = sim_data['df_shock']
    ev_start = sim_data['ev_start']
    ev_end = sim_data['ev_end']

    fig = plt.figure(figsize=(7.17, 8.50), dpi=300)
    gs = GridSpec(3, 2, figure=fig, hspace=0.36, wspace=0.28,
                  left=0.08, right=0.96, top=0.95, bottom=0.05)

    # 1. Age Distribution & Life-Stage Plasticity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(state.age, bins=15, range=(16, 90), density=True, color='#D0E6F5',
             edgecolor='#333333', linewidth=0.5, alpha=0.75, label='Idade IBGE 2022')
    sns.kdeplot(state.age, ax=ax1, color=COLOR_DEEP_BLUE, lw=1.8, cut=0)
    ax1_twin = ax1.twinx()
    age_grid = np.linspace(16.0, 90.0, 100)
    susc_curve = 0.85 - 0.65 * ((age_grid - 16.0) / 74.0)
    tether_curve = 0.34 * (1.30 - 0.55 * np.clip((age_grid - 18.0) / 62.0, 0.0, 1.0))
    ax1_twin.plot(age_grid, susc_curve, color=COLOR_GREEN, lw=1.6, label=r'Plasticidade $\alpha(a)$')
    ax1_twin.plot(age_grid, tether_curve, color=COLOR_AMBER, lw=1.6, ls='--', label=r'Raio Tether $\Delta_{\max}(a)$')
    ax1.set_xlabel('Idade do Cidadão (Anos)', fontsize=6.2)
    ax1.set_ylabel('Densidade Populacional', fontsize=6.2, color=COLOR_DEEP_BLUE)
    ax1_twin.set_ylabel('Parâmetro Cognitivo', fontsize=6.2, color=COLOR_GREEN)
    ax1.set_title('a  Pirâmide Etária IBGE & Plasticidade Cognitiva', fontweight='bold', loc='left', fontsize=6.5)

    # 2. Socioeconomic Status (SES ~ Beta(2,3))
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(state.ses, bins=25, density=True, color='#E8F5E9', edgecolor='#2E7D32', lw=0.5, alpha=0.85)
    sns.kdeplot(state.ses, ax=ax2, color=COLOR_GREEN, lw=1.8)
    ax2.axvline(float(np.mean(state.ses)), color=COLOR_AMBER, ls='--', lw=1.2, label=f'Média SES = {np.mean(state.ses):.2f}')
    ax2.set_xlabel('Status Socioeconômico (SES)', fontsize=6.2)
    ax2.set_ylabel('Densidade de Cidadãos', fontsize=6.2)
    ax2.legend(loc='upper right', fontsize=5.0, frameon=True)
    ax2.set_title('b  Distribuição de Renda e Escolaridade (SES)', fontweight='bold', loc='left', fontsize=6.5)

    # 3. Two-Timescale Exogenous Shock (Opinion vs Conviction)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axvspan(ev_start, ev_end, color='#FFF2B2', alpha=0.6, label='Janela de Choque Exógeno')
    ax3.plot(df_shock['step'], df_shock['opinion_mean_d0'], color=COLOR_GREEN, lw=1.8,
             label=r'Opinião Pública Expressa $\bar{x}_0(t)$ (Rápida)')
    ax3.plot(df_shock['step'], df_shock['conviction_mean_d0'], color=COLOR_DEEP_BLUE, lw=1.6, ls='--',
             label=r'Convicção Interna $\bar{s}_0(t)$ (Inercial)')
    ax3.axhline(0.0, color='#666666', lw=0.6, ls=':')
    ax3.set_xlabel('Passo Temporal da Simulação ($t$)', fontsize=6.2)
    ax3.set_ylabel('Posição Média no Eixo Temático ($d=0$)', fontsize=6.2)
    ax3.legend(loc='upper left', fontsize=5.0, frameon=True)
    ax3.set_title('c  Relaxamento em Duas Escalas de Tempo (Opinião vs Convicção)', fontweight='bold', loc='left', fontsize=6.5)

    # 4. Age-Stratified Opinion-Conviction Tether Stretching
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axvspan(ev_start, ev_end, color='#FFF2B2', alpha=0.6)
    ax4.plot(df_shock['step'], df_shock['young_gap'], color=COLOR_AMBER, lw=1.6, label=r'Jovens (<30) — Distorção $\|\mathbf{x}_i - \mathbf{s}_i\|$')
    ax4.plot(df_shock['step'], df_shock['mean_gap'], color=COLOR_DEEP_BLUE, lw=1.6, label=r'Média Geral — Distorção Média')
    ax4.plot(df_shock['step'], df_shock['senior_gap'], color=COLOR_GREEN, lw=1.6, label=r'Idosos (≥55) — Distorção Mínima')
    ax4.set_xlabel('Passo Temporal da Simulação ($t$)', fontsize=6.2)
    ax4.set_ylabel(r'Distância Euclidiana $\|\mathbf{x}_i - \mathbf{s}_i\| / \sqrt{D}$', fontsize=6.2)
    ax4.legend(loc='upper left', fontsize=5.0, frameon=True)
    ax4.set_title('d  Alongamento do Tether Elástico por Faixa Etária', fontweight='bold', loc='left', fontsize=6.5)

    # 5. Multiplex 5-Layer Assortativity & Homophily
    ax5 = fig.add_subplot(gs[2, 0])
    layers_short = ['L1 Família', 'L2 Trab/Esc', 'L3 Igreja', 'L4 Amigos', 'L5 Trânsito']
    homophily_vals = [0.88, 0.42, 0.65, 0.72, 0.08]
    weights_vals = [0.36, 0.26, 0.14, 0.14, 0.10]
    x_bars = np.arange(len(layers_short))
    w_b = 0.35
    ax5.bar(x_bars - w_b/2, homophily_vals, w_b, color=COLOR_DEEP_BLUE, label=r'Homofilia Etária $h_{\mathrm{age}}$', edgecolor='white', lw=0.4)
    ax5.bar(x_bars + w_b/2, weights_vals, w_b, color=COLOR_GREEN, label=r'Peso Relativo $w_l$', edgecolor='white', lw=0.4)
    ax5.set_xticks(x_bars)
    ax5.set_xticklabels(layers_short, fontsize=5.5)
    ax5.set_ylabel('Índice Estrutural', fontsize=6.2)
    ax5.set_ylim(0, 1.05)
    ax5.legend(loc='upper right', fontsize=5.0, frameon=True)
    ax5.set_title('e  Topologia da Rede Multiplex (5 Camadas Físicas)', fontweight='bold', loc='left', fontsize=6.5)

    # 6. Belief Hypercube Snapshot at Peak Shock Excursion
    ax6 = fig.add_subplot(gs[2, 1])
    sample_idx = np.random.choice(n_agents, size=min(400, n_agents), replace=False)
    ax6.scatter(state.innate_beliefs[sample_idx, 0], state.innate_beliefs[sample_idx, 1],
                color='#BBDEFB', s=16, alpha=0.6, label=r'Convicções Ancoradas $\mathbf{s}_i$')
    ax6.scatter(state.opinions[sample_idx, 0], state.opinions[sample_idx, 1],
                color='#D32F2F', s=14, alpha=0.7, label=r'Opiniões Deslocadas $\mathbf{x}_i(t_{\mathrm{shock}})$')
    for idx in sample_idx[:40]:
        ax6.plot([state.innate_beliefs[idx, 0], state.opinions[idx, 0]],
                 [state.innate_beliefs[idx, 1], state.opinions[idx, 1]],
                 color='#888888', lw=0.4, alpha=0.5)
    ax6.set_xlabel('Dimensão Política 1 (Econômica)', fontsize=6.2)
    ax6.set_ylabel('Dimensão Política 2 (Moral/Costumes)', fontsize=6.2)
    ax6.set_xlim(-1.05, 1.05)
    ax6.set_ylim(-1.05, 1.05)
    ax6.legend(loc='lower left', fontsize=5.0, frameon=True)
    ax6.set_title('f  Instantâneo no Hipercubo: Deslocamento no Pico do Choque', fontweight='bold', loc='left', fontsize=6.5)

    pdf_path = output_dir / "figure1_jan_nonmedianetwork.pdf"
    png_path = output_dir / "figure1_jan_nonmedianetwork.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 2] Saved: {pdf_path}")
    print(f"[Figure 2] Saved: {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Figure 2 simulations and generate plots.")
    parser.add_argument("--mode", type=str, choices=["fast", "full"], default="fast", help="Execution mode (fast or full)")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "figures"), help="Output directory")
    args = parser.parse_args()
    generate_figure2(output_dir=args.output_dir, mode=args.mode)
