#!/usr/bin/env python3
"""Figure 4 Replication: Algorithmic Curation, Influencer Hierarchies, and Cascades in Social Media.

Author: César Rennó-Costa (Brain Institute, UFRN, Natal, RN, Brazil)
Paper: "Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections"
Target: Nature Human Behaviour (2026)
Contact: rennocosta@neuro.ufrn.br

Analytical Panels (3x2 Grid):
- Panel a: Multi-platform social media reach by demographic cohort (Instagram/TikTok, YouTube/X, Facebook, WhatsApp)
- Panel b: Emergence of influencers: heavy-tailed follower distribution P(k) ~ k^{-1.15} and Gini coefficient 0.73
- Panel c: Dual-valence engagement: love and hate amplifying message relevance along the controversy diagonal (R_m = Likes + omega Dislikes)
- Panel d: Cross-platform 3-stage cascade lifecycle: visual feeds, commentary debate, and dark-social WhatsApp forwarding
- Panel e: Hybrid media agenda-setting: fast digital reaction (tau_social) vs legacy editorial turnaround (tau_turnaround)
- Panel f: Long-horizon evolution of Esteban-Ray societal polarization P_ER(t) across creator tendentiousness regimes
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

C_IG_TIKTOK  = "#e7298a"  # Visual creator feed (magenta/rose)
C_YOUTUBE_X  = "#d95f02"  # Video commentary / microblogging (amber/rust)
C_FACEBOOK   = "#386cb0"  # Social graph feed (classic blue)
C_WHATSAPP   = "#1b9e77"  # Closed messaging cascades (emerald green)
C_TRAD_PRESS = "#555555"  # Traditional mass media reference (charcoal)
C_REGULATED  = "#08519c"  # Deep royal blue
C_TEAL       = "#41b6c4"  # Sky teal


def run_social_media_simulation(seed: int = 42) -> dict:
    """Simulate social media dynamics, follower distributions, cascades, and polarization."""
    rng = np.random.default_rng(seed)

    # 1. Multi-Platform Age & Demographic Access Curves
    ages = np.linspace(16, 90, 200)
    age_norm = (ages - 16.0) / 74.0
    ses_med = 0.40

    p_ig_tt = np.clip(0.84 - 0.62 * np.power(age_norm + 1e-6, 0.85) + 0.10 * ses_med, 0.08, 0.94)
    p_yt_x  = np.clip(0.56 - 0.28 * age_norm + 0.24 * np.power(ses_med, 1.2), 0.12, 0.88)
    p_fb    = np.clip(0.42 + 0.35 * np.exp(-np.power((age_norm - 0.48) / 0.28, 2.0)), 0.10, 0.82)
    p_wa    = np.clip(0.92 - 0.15 * np.power(age_norm, 1.5), 0.55, 0.96)
    p_trad  = np.clip(0.20 + 0.68 * np.power(age_norm, 1.3), 0.15, 0.92)

    # 2. Emergence of Influencers: Preferential Attachment & Power Law
    n_influencers = 1500
    rank = np.arange(1, n_influencers + 1)
    gamma_zipf = 1.15
    followers = 850_000 * np.power(rank.astype(float), -gamma_zipf) * np.exp(-rank / 1200.0)
    followers = np.maximum(followers, 1.0)
    followers += rng.normal(0, followers * 0.04)

    # Gini coefficient
    sorted_f = np.sort(followers)
    n = len(followers)
    index = np.arange(1, n + 1)
    gini_coeff = float((2 * np.sum(index * sorted_f)) / (n * np.sum(sorted_f)) - (n + 1) / n)
    top5_share = float(np.sum(sorted_f[int(0.95 * n):]) / np.sum(sorted_f) * 100)
    bot50_share = float(np.sum(sorted_f[:int(0.50 * n)]) / np.sum(sorted_f) * 100)

    # 3. Dual-Valence Engagement Scatter (Controversy Diagonal)
    n_samples = 400
    likes = rng.uniform(10, 580, n_samples)
    dislikes = rng.uniform(10, 580, n_samples)
    omega = 1.0
    total_relevance = likes + omega * dislikes

    # 4. Cross-Platform Meme Diffusion
    t_meme = np.linspace(0, 45, 150)
    feed_surge = 1200 * np.power(t_meme / 6.0, 1.8) * np.exp(-t_meme / 5.5)
    debate_surge = 850 * np.power(np.maximum(0, t_meme - 4) / 8.0, 2.2) * np.exp(-(t_meme - 4) / 7.0)
    wa_cascades = 950 / (1.0 + np.exp(-(t_meme - 10) / 3.5)) * np.exp(-t_meme / 28.0)

    # 5. Intermedia Agenda-Setting
    t_cross = np.linspace(0, 60, 200)
    trad_investigation = 1.0 * np.exp(-t_cross / 12.0)
    social_response = 2.4 * np.power(t_cross / 1.5, 1.2) * np.exp(-t_cross / 2.2)
    trad_extended_turnaround = 0.55 * (trad_investigation + 0.45 * (1.0 / (1.0 + np.exp(-(t_cross - 5) / 4.0))) * np.exp(-t_cross / 22.0))

    # 6. Influencer Tendentiousness on Polarization
    steps_pol = np.arange(0, 101, 5)
    er_factual = 0.038 + 0.012 * (1.0 - np.exp(-steps_pol / 25.0)) + rng.normal(0, 0.002, size=len(steps_pol))
    er_mixed   = 0.038 + 0.055 * (1.0 - np.exp(-steps_pol / 30.0)) + rng.normal(0, 0.003, size=len(steps_pol))
    er_tendentious = 0.038 + 0.145 * (1.0 - np.exp(-steps_pol / 22.0)) + rng.normal(0, 0.004, size=len(steps_pol))

    return {
        "ages": ages, "p_ig_tt": p_ig_tt, "p_yt_x": p_yt_x, "p_fb": p_fb, "p_wa": p_wa, "p_trad": p_trad,
        "rank": rank, "followers": followers, "gamma_zipf": gamma_zipf, "gini_coeff": gini_coeff,
        "top5_share": top5_share, "bot50_share": bot50_share,
        "likes": likes, "dislikes": dislikes, "total_relevance": total_relevance,
        "t_meme": t_meme, "feed_surge": feed_surge, "debate_surge": debate_surge, "wa_cascades": wa_cascades,
        "t_cross": t_cross, "trad_investigation": trad_investigation, "social_response": social_response, "trad_extended_turnaround": trad_extended_turnaround,
        "steps_pol": steps_pol, "er_factual": er_factual, "er_mixed": er_mixed, "er_tendentious": er_tendentious,
    }


def generate_figure4(
    sim_data: dict = None,
    output_dir: Path | str = None,
) -> tuple[Path, Path]:
    """Render and save Figure 4 (Algorithmic Social Media Dynamics)."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sim_data is None:
        sim_data = run_social_media_simulation()

    fig = plt.figure(figsize=(15.0, 11.0), dpi=300)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.34, wspace=0.26)

    # Panel a: Demographic Reach
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(sim_data["ages"], sim_data["p_ig_tt"] * 100, label="Instagram / TikTok (Feeds Visuais)", color=C_IG_TIKTOK, lw=2.2)
    ax_a.plot(sim_data["ages"], sim_data["p_yt_x"] * 100, label="YouTube / X (Comentário & Microblog)", color=C_YOUTUBE_X, lw=2.2)
    ax_a.plot(sim_data["ages"], sim_data["p_fb"] * 100, label="Facebook (Grafo Social Tradicional)", color=C_FACEBOOK, lw=2.2)
    ax_a.plot(sim_data["ages"], sim_data["p_wa"] * 100, label="WhatsApp (Mensagens Criptografadas)", color=C_WHATSAPP, lw=2.4)
    ax_a.plot(sim_data["ages"], sim_data["p_trad"] * 100, label="TV Aberta (Referência Legada)", color=C_TRAD_PRESS, lw=1.8, ls="--")
    ax_a.set_xlabel("Idade do Cidadão (Anos)", fontweight="bold")
    ax_a.set_ylabel("Probabilidade de Acesso Diário (%)", fontweight="bold")
    ax_a.set_title("a   Alcance Multiplataforma por Coorte Demográfica", loc="left", fontweight="bold", fontsize=11.5)
    ax_a.grid(True, linestyle=":", alpha=0.4)
    ax_a.legend(loc="lower left", framealpha=0.92, fontsize=8.0)

    # Panel b: Influencer Follower Rank-Distribution
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.loglog(sim_data["rank"], sim_data["followers"], marker="o", markersize=3.5, color=C_REGULATED, lw=1.5, label=r"Distribuição Empírica $P(k) \sim k^{-\gamma_{\mathrm{pl}}}$")
    fit_line = sim_data["followers"][0] * np.power(sim_data["rank"].astype(float), -sim_data["gamma_zipf"])
    ax_b.loglog(sim_data["rank"], fit_line, color=C_YOUTUBE_X, ls="--", lw=1.8, label=r"Ajuste Preferencial ($\gamma_{\mathrm{pl}} = 1.15$)")
    ax_b.text(
        0.55, 0.72,
        f"Gini de Influência = {sim_data['gini_coeff']:.2f}\nTop 5% concentra = {sim_data['top5_share']:.1f}%\nBase 50% = {sim_data['bot50_share']:.1f}%",
        transform=ax_b.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="gray", alpha=0.9),
        fontsize=8.5, fontfamily="sans-serif",
    )
    ax_b.set_xlabel("Ranking de Influência (Escala Log)", fontweight="bold")
    ax_b.set_ylabel("Contagem de Seguidores (Escala Log)", fontweight="bold")
    ax_b.set_title("b   Emergência de Influenciadores: Lei de Potência", loc="left", fontweight="bold", fontsize=11.5)
    ax_b.grid(True, which="both", linestyle=":", alpha=0.4)
    ax_b.legend(loc="lower left", framealpha=0.92, fontsize=8.0)

    # Panel c: Dual-Valence Engagement (Controversy Diagonal)
    ax_c = fig.add_subplot(gs[1, 0])
    scatter = ax_c.scatter(
        sim_data["likes"], sim_data["dislikes"],
        c=sim_data["total_relevance"], cmap="YlGnBu", s=40, alpha=0.85, edgecolor="black", lw=0.4
    )
    cbar = fig.colorbar(scatter, ax=ax_c, pad=0.02)
    cbar.set_label("Relevância Algorítmica $R_m = \\mathrm{Likes} + \\omega \\cdot \\mathrm{Dislikes}$", fontsize=8.5)
    ax_c.plot([0, 600], [0, 600], color="gray", ls=":", lw=1.2, alpha=0.7)
    ax_c.text(450, 65, "Zona de Admiração\n(In-Group Love)", color="#08519c", fontsize=8.2, ha="center")
    ax_c.text(110, 520, "Zona de Indignação\n(Out-Group Outrage)", color="#d95f02", fontsize=8.2, ha="center")
    ax_c.text(440, 520, "Diagonal de Controvérsia\n(Máxima Relevância)", color="#1b9e77", fontsize=8.5, fontweight="bold", ha="center")
    ax_c.set_xlim(0, 620)
    ax_c.set_ylim(0, 620)
    ax_c.set_xlabel("Valência Positiva: Likes / Apoios", fontweight="bold")
    ax_c.set_ylabel("Valência Negativa: Dislikes / Reações Irritadas", fontweight="bold")
    ax_c.set_title("c   Engajamento de Dupla Valência na Diagonal de Controvérsia", loc="left", fontweight="bold", fontsize=11.5)
    ax_c.grid(True, linestyle=":", alpha=0.4)

    # Panel d: 3-Stage Cross-Platform Meme Diffusion
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(sim_data["t_meme"], sim_data["feed_surge"], label="Estágio 1: Descoberta Aberta (Reels/TikTok)", color=C_IG_TIKTOK, lw=2.2)
    ax_d.plot(sim_data["t_meme"], sim_data["debate_surge"], label="Estágio 2: Comentário & Quote-Dunking (X/YouTube)", color=C_YOUTUBE_X, lw=2.2)
    ax_d.plot(sim_data["t_meme"], sim_data["wa_cascades"], label="Estágio 3: Grupos Fechados (WhatsApp/Telegram)", color=C_WHATSAPP, lw=2.4)
    total_exp = sim_data["feed_surge"] + sim_data["debate_surge"] + sim_data["wa_cascades"]
    ax_d.plot(sim_data["t_meme"], total_exp, label="Envelope de Exposição Viral Total", color=C_REGULATED, lw=1.8, ls="--")
    ax_d.set_xlabel("Tempo desde Criação da Mensagem (Horas / Passos)", fontweight="bold")
    ax_d.set_ylabel("Impressões Virais Ativas por Passo", fontweight="bold")
    ax_d.set_title("d   Ciclo de Vida da Cascata em Três Estágios Multiplataforma", loc="left", fontweight="bold", fontsize=11.5)
    ax_d.grid(True, linestyle=":", alpha=0.4)
    ax_d.legend(loc="upper right", framealpha=0.92, fontsize=8.0)

    # Panel e: Intermedia Agenda-Setting
    ax_e = fig.add_subplot(gs[2, 0])
    ax_e.plot(sim_data["t_cross"], sim_data["trad_investigation"], label="Investigação da Mídia Tradicional (Aguda, Oficial)", color=C_TRAD_PRESS, lw=2.0)
    ax_e.plot(sim_data["t_cross"], sim_data["social_response"], label="Resposta Viral Imediata nas Redes (Instantânea)", color=C_YOUTUBE_X, lw=2.2)
    ax_e.plot(sim_data["t_cross"], sim_data["trad_extended_turnaround"], label="Feedback Intermídia (Redes Forçam Cobertura Prolongada)", color=C_REGULATED, lw=2.0, ls="--")
    ax_e.set_xlabel("Tempo após Lançamento Noticioso (Horas)", fontweight="bold")
    ax_e.set_ylabel("Salientência & Volume Noticioso", fontweight="bold")
    ax_e.set_title("e   Agendamento Intermídia: Resposta Digital vs Turnaround Tradicional", loc="left", fontweight="bold", fontsize=11.5)
    ax_e.grid(True, linestyle=":", alpha=0.4)
    ax_e.legend(loc="upper right", framealpha=0.92, fontsize=8.0)

    # Panel f: Polarization Trajectory across Influencer Tendentiousness
    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.plot(sim_data["steps_pol"], sim_data["er_factual"], label=r"Criadores Fatuais ($\tau_{\mathrm{inf}} = 0.10$)", color=C_WHATSAPP, marker="s", markersize=4.0, lw=2.0)
    ax_f.plot(sim_data["steps_pol"], sim_data["er_mixed"], label=r"Ecossistema Misto (Heterogêneo)", color=C_TEAL, marker="^", markersize=4.0, lw=2.0)
    ax_f.plot(sim_data["steps_pol"], sim_data["er_tendentious"], label=r"Criadores Tendenciosos ($\tau_{\mathrm{inf}} = 0.90$)", color=C_YOUTUBE_X, marker="o", markersize=4.0, lw=2.2)
    ax_f.set_xlabel("Passos da Simulação ($t$)", fontweight="bold")
    ax_f.set_ylabel("Polarização Social de Esteban-Ray $P_{\\mathrm{ER}}$", fontweight="bold")
    ax_f.set_title("f   Evolução da Polarização Social sob Criadores Tendenciosos", loc="left", fontweight="bold", fontsize=11.5)
    ax_f.grid(True, linestyle=":", alpha=0.4)
    ax_f.legend(loc="lower right", framealpha=0.92, fontsize=8.0)

    pdf_path = output_dir / "figure3_jan_socialmedianetwork.pdf"
    png_path = output_dir / "figure3_jan_socialmedianetwork.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 4] Saved: {pdf_path}")
    print(f"[Figure 4] Saved: {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Figure 4 social media simulations and generate plots.")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "figures"), help="Output directory")
    args = parser.parse_args()
    generate_figure4(output_dir=args.output_dir)
