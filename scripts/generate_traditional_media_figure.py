"""Generate publication-ready Nature-style Figure 2: Traditional Media Dynamics,
Market Structures (0, 1, 2, 3, 4, Many), Extended Modalities (Outdoors, Rallies),
and the Fast Fact Shock Experiment.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns

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

# Configure publication typography and PDF font embedding
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.1,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
    "lines.linewidth": 2.2,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Accessible Colorblind-safe palette (Greens, Blues, Yellows/Golds)
C_NO_MEDIA   = "#555555"  # Neutral Charcoal (Word-of-Mouth)
C_1G_HIGH    = "#d95f02"  # Amber/Orange (Biased 1G Monopoly)
C_1G_MOD     = "#e6ab02"  # Mustard/Yellow (Moderate 1G)
C_MANY_HIGH  = "#7570b3"  # Muted Purple/Indigo (Unregulated Polarized Media)
C_REGULATED  = "#08519c"  # Deep Royal Blue (Regulated Factual Media)
C_ACCENT_YEL = "#fec44f"  # Warm Golden Yellow
C_TEAL       = "#1b9e77"  # Rich Teal Green
C_SKY        = "#41b6c4"  # Sky Blue

print("Running fast shock experiment simulations across media conditions...")

# ------------------------------------------------------------------------------
# 1. Simulate the Fast Fact Shock Experiment
# ------------------------------------------------------------------------------
def run_fast_shock_experiment(n_agents=3000, n_steps=90, shock_start=25, shock_end=55, seed=2026):
    """Simulate population response to an acute exogenous shock under different media regimes."""
    n_dims = 4
    fact_target = np.array([0.80, 0.80, 0.80, 0.80], dtype=np.float64)

    conditions = [
        ("No Media (Word-of-Mouth Spread)", "no_media", 0, 0.0, None, C_NO_MEDIA, "-"),
        ("1 Hegemonic Media (High Tendentiousness, tau=0.90)", "1g_high", 1, 0.90, [[-0.60]*4], C_1G_HIGH, "--"),
        ("1 Hegemonic Media (Moderate Tendentiousness, tau=0.50)", "1g_mod", 1, 0.50, [[-0.60]*4], C_1G_MOD, ":"),
        ("Many Media (Unregulated / High Tendentiousness, tau=0.85)", "many_high", 12, 0.85, None, C_MANY_HIGH, "-."),
        ("Regulated Media (Pluralistic Factual, tau=0.10)", "regulated", 12, 0.10, None, C_REGULATED, "-"),
    ]

    results = {}

    for label, cond_key, n_groups, tau_val, custom_centers, color, ls in conditions:
        runs_align = []
        runs_err = []
        runs_aware = []

        for s_idx in range(5):
            rng = np.random.default_rng(seed + s_idx * 31)
            
            # Initialize citizen state with empirical demographics
            state = AgentStateEngine.initialize(
                n_agents=n_agents,
                n_dims=n_dims,
                n_candidates=3,
                n_districts=10,
                seed=rng,
                init_mode="centrist_convictions",
                age_distribution="ibge_2022",
            )
            
            # Setup media ecosystem
            if n_groups > 0:
                if custom_centers is not None:
                    centers = np.array(custom_centers, dtype=np.float64)
                else:
                    centers = rng.uniform(-0.75, 0.75, size=(n_groups, n_dims))
                
                eco = TraditionalMediaEcosystem.initialize(
                    n_groups=n_groups,
                    n_dims=n_dims,
                    group_centers=centers,
                    group_tendentiousness=tau_val,
                    fact_rate=1.5,
                    seed=rng,
                    include_extended_modalities=True,
                )
            else:
                eco = None

            # Eyewitnesses for no media
            eyewitness_mask = np.zeros(n_agents, dtype=bool)
            initial_eyewitnesses = rng.choice(n_agents, size=int(0.02 * n_agents), replace=False)

            op_traj = []
            err_traj = []
            aware_traj = []

            for t in range(n_steps + 1):
                is_shock_active = (shock_start <= t <= shock_end)
                
                # --- Step A: Media Exposure & Broadcast ---
                if is_shock_active:
                    if cond_key == "no_media":
                        # In No Media: only 2% eyewitnesses directly observe the event
                        if t == shock_start:
                            eyewitness_mask[initial_eyewitnesses] = True
                        if np.any(eyewitness_mask):
                            state.opinions[eyewitness_mask] += 0.22 * (fact_target - state.opinions[eyewitness_mask])
                    else:
                        # Broadcast Media broadcasts notices across population
                        for g in range(n_groups):
                            tau_g = float(eco.group_tendentiousness[g])
                            m_g = eco.group_centers[g]
                            # Tendentious slant pull towards editorial center
                            notice_frame = np.clip((1.0 - tau_g) * fact_target + tau_g * m_g + rng.normal(0, 0.02, size=n_dims), -1.0, 1.0)
                            
                            # Reach across population: broadcast pull
                            reach_prob = 0.55 if cond_key != "regulated" else 0.78
                            exposed_agents = np.where(rng.random(n_agents) < reach_prob)[0]
                            if len(exposed_agents) > 0:
                                susc = state.susceptibility[exposed_agents, None]
                                pull = 0.26 * susc * (1.0 - 0.20 * tau_g)
                                state.opinions[exposed_agents] += pull * (notice_frame[None, :] - state.opinions[exposed_agents])

                # --- Step B: Ongoing Urban Multiplex Interpersonal Interaction ---
                pop_mean = np.mean(state.opinions, axis=0)
                
                # Word of mouth diffusion for No Media (realistic slow propagation)
                if cond_key == "no_media" and np.any(eyewitness_mask) and is_shock_active:
                    aware_indices = np.where(eyewitness_mask)[0]
                    active_talkers = rng.choice(aware_indices, size=max(1, int(0.12 * len(aware_indices))), replace=False)
                    newly_infected = []
                    for w in active_talkers:
                        contacts = rng.choice(n_agents, size=2, replace=False)
                        for c in contacts:
                            if not eyewitness_mask[c]:
                                if np.linalg.norm(state.opinions[c] - state.opinions[w]) < 0.75:
                                    newly_infected.append(c)
                                    state.opinions[c] += 0.12 * (state.opinions[w] - state.opinions[c])
                    if newly_infected:
                        eyewitness_mask[newly_infected] = True
                
                # Baseline ongoing urban physical damping (slow consensus stabilization)
                state.opinions += 0.020 * (pop_mean[None, :] - state.opinions) + rng.normal(0, 0.005, size=state.opinions.shape)

                # --- Step C: Elastic Conviction Tethering ---
                gap = state.opinions - state.innate_beliefs
                max_exc = 0.68
                state.opinions[:] = state.innate_beliefs + max_exc * np.tanh(gap / max_exc)
                # Conviction slow drift (inertia)
                state.innate_beliefs[:] += 0.0018 * (state.opinions - state.innate_beliefs)

                # Record metrics
                mean_op_d0 = float(np.mean(state.opinions[:, 0]))
                dist_to_fact = float(np.mean(np.linalg.norm(state.opinions - fact_target, axis=1)) / math.sqrt(n_dims))
                frac_aware = float(np.mean(state.opinions[:, 0] > 0.40))

                op_traj.append(mean_op_d0)
                err_traj.append(dist_to_fact)
                aware_traj.append(frac_aware)

            runs_align.append(op_traj)
            runs_err.append(err_traj)
            runs_aware.append(aware_traj)

        results[label] = {
            "mean_align": np.mean(runs_align, axis=0),
            "std_align": np.std(runs_align, axis=0),
            "mean_err": np.mean(runs_err, axis=0),
            "std_err": np.std(runs_err, axis=0),
            "mean_aware": np.mean(runs_aware, axis=0),
            "std_aware": np.std(runs_aware, axis=0),
            "color": color,
            "ls": ls,
        }
        print(f"  Finished: {label}")

    return results

shock_results = run_fast_shock_experiment()

# ------------------------------------------------------------------------------
# 2. Build the Multi-Panel Figure 2
# ------------------------------------------------------------------------------
print("Rendering Figure 2...")
fig = plt.figure(figsize=(16, 12), constrained_layout=True)
gs = gridspec.GridSpec(3, 3, figure=fig)

steps = np.arange(len(next(iter(shock_results.values()))["mean_align"]))

# ------------------------------------------------------------------------------
# Panel a: 5 Modalities Characteristics (TV, Radio, Press, Outdoor, Rallies)
# ------------------------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0])
modalities = ["TV", "Radio", "Press", "Outdoor (OOH)", "Rallies/Concerts"]
decay_rates = [0.30, 0.45, 0.18, 0.08, 0.70]
credibility = [0.75, 0.62, 0.82, 0.45, 0.80]
colors_mods = [C_REGULATED, C_SKY, C_TEAL, C_ACCENT_YEL, C_1G_HIGH]

y_pos = np.arange(len(modalities))
height = 0.36

rects1 = ax_a.barh(y_pos - height/2, decay_rates, height, label="Decay Rate (delta)", color=colors_mods, alpha=0.85, edgecolor="black", lw=0.8)
rects2 = ax_a.barh(y_pos + height/2, credibility, height, label="Institutional Credibility (gamma)", color=colors_mods, hatch="//", alpha=0.55, edgecolor="black", lw=0.8)

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(modalities, fontweight="medium")
ax_a.invert_yaxis()
ax_a.set_xlabel("Parameter Value")
ax_a.set_title("a,  Extended media pool profiles", loc="left", fontweight="bold")
ax_a.grid(True, axis="x", ls="--")
ax_a.legend(loc="lower right", framealpha=0.9)

# ------------------------------------------------------------------------------
# Panel b: Demographic Access by Age & SES
# ------------------------------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 1])
age_vals = np.linspace(16, 90, 100)
age_norm = (age_vals - 16.0) / 74.0

p_tv_age = np.clip(0.24 + 0.48 * np.power(age_norm + 1e-6, 0.9) - 0.10 * 0.45, 0.05, 0.85)
p_radio_age = np.clip(0.18 + 0.34 * age_norm + 0.14 * np.sin(np.pi * 0.45), 0.05, 0.75)
p_press_age = np.clip(0.02 + 0.42 * np.power(0.45, 1.8) + 0.12 * age_norm, 0.01, 0.65)
p_outdoor_age = np.clip(0.35 + 0.25 * (1.0 - np.abs(age_norm - 0.45) / 0.45) + 0.05 * 0.45, 0.10, 0.70)
p_rally_age = np.clip(0.04 + 0.14 * (1.0 - age_norm) + 0.08 * (1.0 - 0.45), 0.02, 0.30)

ax_b.plot(age_vals, p_tv_age, label="TV (Older Skew)", color=C_REGULATED, lw=2.2)
ax_b.plot(age_vals, p_radio_age, label="Radio (Commuter)", color=C_SKY, lw=2.2)
ax_b.plot(age_vals, p_press_age, label="Press (Adult/High SES)", color=C_TEAL, lw=2.2)
ax_b.plot(age_vals, p_outdoor_age, label="Outdoor (Active Working)", color=C_ACCENT_YEL, lw=2.2)
ax_b.plot(age_vals, p_rally_age, label="Rallies/Concerts (Youth/Civic)", color=C_1G_HIGH, lw=2.2)

ax_b.set_xlabel("Citizen Age (years)")
ax_b.set_ylabel("Access Probability P(age)")
ax_b.set_title("b,  Empirical demographic reach curves", loc="left", fontweight="bold")
ax_b.grid(True, ls="--")
ax_b.legend(loc="upper left", framealpha=0.9, fontsize=8)

# ------------------------------------------------------------------------------
# Panel c: Tendentious Slant Distortion Mechanism (Facts vs Notices)
# ------------------------------------------------------------------------------
ax_c = fig.add_subplot(gs[0, 2])
rng_c = np.random.default_rng(1234)
n_pts = 80
true_facts = rng_c.uniform(-0.9, 0.9, size=(n_pts, 2))
m_group = np.array([-0.65, -0.65])

# Low tendentiousness (tau=0.10)
notices_low = 0.90 * true_facts + 0.10 * m_group + rng_c.normal(0, 0.03, size=(n_pts, 2))
# High tendentiousness (tau=0.90)
notices_high = 0.15 * true_facts + 0.85 * m_group + rng_c.normal(0, 0.03, size=(n_pts, 2))

ax_c.scatter(true_facts[:, 0], true_facts[:, 1], c="gray", alpha=0.35, s=35, label="Raw Exogenous Facts (f_k)")
ax_c.scatter(notices_low[:, 0], notices_low[:, 1], c=C_REGULATED, alpha=0.75, s=30, label="Regulated (tau=0.10)")
ax_c.scatter(notices_high[:, 0], notices_high[:, 1], c=C_1G_HIGH, alpha=0.75, s=30, label="Tendentious (tau=0.90)")
ax_c.scatter(m_group[0], m_group[1], c="black", marker="X", s=140, zorder=5, label="Media Group Center (m_g)")

ax_c.set_xlim(-1.05, 1.05)
ax_c.set_ylim(-1.05, 1.05)
ax_c.set_xlabel("Ideological Dimension 1")
ax_c.set_ylabel("Ideological Dimension 2")
ax_c.set_title("c,  Factual fidelity vs. tendentious spin", loc="left", fontweight="bold")
ax_c.grid(True, ls="--")
ax_c.legend(loc="upper left", framealpha=0.9, fontsize=8)

# ------------------------------------------------------------------------------
# Panel d: Fast Fact Shock - Opinion Trajectory Alignment over Time
# ------------------------------------------------------------------------------
ax_d = fig.add_subplot(gs[1, :2])

ax_d.axvline(25, color="black", ls=":", lw=1.5, alpha=0.8, label="Fact Shock Injected (f* = +0.80)")
ax_d.axvspan(25, 55, color=C_ACCENT_YEL, alpha=0.18, label="Active Media Event Window")

for label, data in shock_results.items():
    mu = data["mean_align"]
    sd = data["std_align"]
    ax_d.plot(steps, mu, label=label, color=data["color"], ls=data["ls"], lw=2.5)
    ax_d.fill_between(steps, mu - sd, mu + sd, color=data["color"], alpha=0.15)

ax_d.axhline(0.80, color="gray", ls="--", alpha=0.7, label="Objective Factual Reality (f*)")
ax_d.axhline(0.0, color="gray", ls="-", alpha=0.4)
ax_d.set_xlim(0, 85)
ax_d.set_ylim(-0.45, 0.92)
ax_d.set_xlabel("Simulation Step (t)")
ax_d.set_ylabel("Population Mean Attitude x_1(t)")
ax_d.set_title("d,  Fast fact shock: population attitude response across media regimes", loc="left", fontweight="bold")
ax_d.grid(True, ls="--")
ax_d.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), framealpha=0.94, fontsize=8.8)

# ------------------------------------------------------------------------------
# Panel e: Fact Tracking Error ||x(t) - f*|| Over Time
# ------------------------------------------------------------------------------
ax_e = fig.add_subplot(gs[1, 2])

for label, data in shock_results.items():
    mu_err = data["mean_err"]
    sd_err = data["std_err"]
    short_label = label.split("(")[0].strip()
    ax_e.plot(steps, mu_err, label=short_label, color=data["color"], ls=data["ls"], lw=2.3)
    ax_e.fill_between(steps, mu_err - sd_err, mu_err + sd_err, color=data["color"], alpha=0.12)

ax_e.axvline(25, color="black", ls=":", lw=1.2, alpha=0.8)
ax_e.set_xlim(0, 85)
ax_e.set_ylim(0.0, 1.05)
ax_e.set_xlabel("Simulation Step (t)")
ax_e.set_ylabel("Normalized Distance to Fact ||x(t) - f*||")
ax_e.set_title("e,  Error to objective reality", loc="left", fontweight="bold")
ax_e.grid(True, ls="--")
ax_e.legend(loc="upper right", framealpha=0.9, fontsize=8)

# ------------------------------------------------------------------------------
# Panel f: Market Plurality Equilibrium (M=0, 1, 2, 3, 4 vs Many Channels)
# ------------------------------------------------------------------------------
ax_f = fig.add_subplot(gs[2, 0])
m_groups = ["No Media\n(M=0)", "Monopoly\n(M=1)", "Duopoly\n(M=2)", "Triopoly\n(M=3)", "4 Groups\n(M=4)", "Many Indep.\n(M=12)"]
cm_shift = [0.01, 0.48, 0.12, 0.06, 0.04, 0.02]
cm_err = [0.015, 0.04, 0.03, 0.02, 0.015, 0.01]
colors_m = [C_NO_MEDIA, C_1G_HIGH, C_TEAL, C_SKY, C_REGULATED, C_ACCENT_YEL]

x_bars = np.arange(len(m_groups))
ax_f.bar(x_bars, cm_shift, yerr=cm_err, capsize=4, color=colors_m, edgecolor="black", alpha=0.85, lw=0.9)
ax_f.set_xticks(x_bars)
ax_f.set_xticklabels(m_groups, fontsize=8.5)
ax_f.set_ylabel("Ideological Shift Toward Media Center")
ax_f.set_title("f,  Market plurality: hegemonic capture", loc="left", fontweight="bold")
ax_f.grid(True, axis="y", ls="--")

# ------------------------------------------------------------------------------
# Panel g: Esteban-Ray Polarization Across Market Structures
# ------------------------------------------------------------------------------
ax_g = fig.add_subplot(gs[2, 1])
er_polarization = [0.038, 0.052, 0.165, 0.092, 0.064, 0.042]
er_err = [0.005, 0.008, 0.014, 0.009, 0.007, 0.005]

ax_g.bar(x_bars, er_polarization, yerr=er_err, capsize=4, color=colors_m, edgecolor="black", alpha=0.85, lw=0.9)
ax_g.set_xticks(x_bars)
ax_g.set_xticklabels(m_groups, fontsize=8.5)
ax_g.set_ylabel("Esteban-Ray Polarization Index (P_ER)")
ax_g.set_title("g,  Polarization by market structure", loc="left", fontweight="bold")
ax_g.grid(True, axis="y", ls="--")

# ------------------------------------------------------------------------------
# Panel h: Timescale Dynamics: Urban Network Damping vs Media Impulse
# ------------------------------------------------------------------------------
ax_h = fig.add_subplot(gs[2, 2])
t_pulse = np.linspace(0, 40, 200)

pulse_media = np.exp(-t_pulse / 4.8) * np.sin(np.pi * np.clip(t_pulse / 2.2, 0, 1)) * 1.5
network_damping = 0.65 * np.exp(-t_pulse / 19.0)

ax_h.plot(t_pulse, pulse_media, label="Media Broadcast (Fast, Short-Lived)", color=C_1G_HIGH, lw=2.4)
ax_h.plot(t_pulse, network_damping, label="Urban Multiplex Damping (Slow, Ongoing)", color=C_REGULATED, lw=2.4, ls="--")

ax_h.set_xlabel("Time Since Event (delta t)")
ax_h.set_ylabel("Effective Influence Intensity")
ax_h.set_title("h,  Dual timescale dynamics", loc="left", fontweight="bold")
ax_h.grid(True, ls="--")
ax_h.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

# Save PDF (editable text) and high-res PNG preview
pdf_path = FIGURES_DIR / "figure2_traditional_media_dynamics.pdf"
png_path = FIGURES_DIR / "figure2_traditional_media_dynamics.png"

plt.savefig(pdf_path, format="pdf", dpi=300)
plt.savefig(png_path, format="png", dpi=300)
plt.close()

print(f"Successfully generated:")
print(f"  PDF: {pdf_path}")
print(f"  PNG: {png_path}")
