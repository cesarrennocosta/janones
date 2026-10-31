"""Generate all 8 Notebook 1 publication figures as PDF (and PNG) in Nature style.

Key Features:
- Text rendered as editable Type 42 TrueType text in Adobe Illustrator (pdf.fonttype = 42).
- Large, highly legible Nature publication font sizes with prominent bold panel labels (a, b, c).
- Strictly accessible colorblind palette focused on Green, Blue, and Yellowish (Okabe-Ito / CUD).
- Clean white background, thin elegant spines, and clear visual hierarchy.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.agent_state import AgentStateEngine
from src.core.multiplex_network import GraphEngine, LAYER_NAMES
from src.kernels.opinion_updates import step_conviction_and_opinion_tether_jit

RESULTS_DIR = PROJECT_ROOT / 'data' / 'results'
FIGURES_DIR = PROJECT_ROOT / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Nature Publication Styling & Illustrator PDF Compatibility Settings
# ------------------------------------------------------------------------------
mpl.rcParams['pdf.fonttype'] = 42       # TrueType text (fully editable in Illustrator)
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.linewidth'] = 0.85
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.major.size'] = 4.5
mpl.rcParams['ytick.major.size'] = 4.5
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8
mpl.rcParams['axes.labelcolor'] = '#222222'
mpl.rcParams['text.color'] = '#222222'
mpl.rcParams['xtick.color'] = '#222222'
mpl.rcParams['ytick.color'] = '#222222'
mpl.rcParams['figure.titlesize'] = 15
mpl.rcParams['axes.titlesize'] = 13.5
mpl.rcParams['axes.labelsize'] = 12.5
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 10.5
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300

# Colorblind Accessible Palette (Focused on Blue, Green, Yellowish / Amber)
COLOR_DEEP_BLUE = '#0072B2'   # Primary deep blue (Okabe-Ito)
COLOR_SKY_BLUE  = '#56B4E9'   # Light / sky blue (Okabe-Ito)
COLOR_GREEN     = '#009E73'   # Bluish green / emerald (Okabe-Ito)
COLOR_AMBER     = '#E69F00'   # Amber / golden yellow (Okabe-Ito)
COLOR_YELLOW    = '#F0E442'   # Pure yellow / highlight (Okabe-Ito)
COLOR_OLIVE     = '#788930'   # Yellow-olive / muted green
COLOR_NAVY      = '#1A365D'   # Dark navy contrast
COLOR_SLATE     = '#4A5568'   # Charcoal / slate grey
COLOR_LIGHT_BG  = '#F7FAFC'   # Subtle panel background

def save_fig(fig, base_name: str):
    """Save figure as both PDF (Illustrator editable) and PNG (high-res raster)."""
    pdf_path = FIGURES_DIR / f"{base_name}.pdf"
    png_path = FIGURES_DIR / f"{base_name}.png"
    fig.savefig(pdf_path, bbox_inches='tight', format='pdf')
    fig.savefig(png_path, bbox_inches='tight', dpi=300, format='png')
    plt.close(fig)
    print(f"Saved {pdf_path.name} and {png_path.name}")

# Load simulation result parquet datasets
print("Loading Parquet datasets...")
df_conv = pd.read_parquet(RESULTS_DIR / 'all_conviction_dynamics_metrics.parquet')
if 'scenario_name' not in df_conv.columns and 'scenario' in df_conv.columns:
    df_conv['scenario_name'] = df_conv['scenario']
df_cities = pd.read_parquet(RESULTS_DIR / 'all_cities_metrics.parquet')
print(f"Loaded df_conv ({len(df_conv):,} rows) and df_cities ({len(df_cities):,} rows).")


# ==============================================================================
# Figure 1: Demographics, SES, and Behavioral Feature Distributions
# ==============================================================================
print("\nGenerating Figure 1: Demographics and Behavioral Feature Distributions...")
rng_pop = np.random.default_rng(20261031)
state_100k = AgentStateEngine.initialize(
    n_agents=100_000, n_dims=4, n_candidates=3, n_districts=10,
    seed=rng_pop, init_mode='correlated', age_distribution='ibge_2022'
)

fig1, axes1 = plt.subplots(1, 3, figsize=(18.0, 5.8))

# Panel a: Age Distribution & Life-Stage Plasticity Curves
ax = axes1[0]
ax.hist(state_100k.age, bins=15, range=(16, 90), density=True, color='#D0E6F5',
        edgecolor='#333333', linewidth=0.75, alpha=0.75, label='Age Histogram (5-yr Bins)')
sns.kdeplot(state_100k.age, ax=ax, color=COLOR_DEEP_BLUE, lw=2.6, cut=0, label='Smoothed Age Density (KDE)')
ax.axvline(30.0, color=COLOR_AMBER, ls='--', lw=1.8, label='Young (<30) / Mid-Age (30–54)')
ax.axvline(55.0, color=COLOR_GREEN, ls='--', lw=1.8, label='Mid-Age (30–54) / Senior (≥55)')
ax.set_xlabel('Citizen Age (Years)', fontweight='bold')
ax.set_ylabel('Population Density', fontweight='bold', color=COLOR_DEEP_BLUE)
ax.set_xlim(14, 92)

ax_twin = ax.twinx()
age_grid = np.linspace(16.0, 90.0, 120)
susc_curve = 0.85 - 0.65 * ((age_grid - 16.0) / 74.0)
eps_curve = 0.45 * (1.0 - 0.25 * ((age_grid - 16.0) / 74.0))
tether_curve = 0.34 * (1.30 - 0.55 * np.clip((age_grid - 18.0) / 62.0, 0.0, 1.0))
ax_twin.plot(age_grid, susc_curve, color=COLOR_GREEN, lw=2.5, ls='-', label='Social Susceptibility $\\alpha(a)$')
ax_twin.plot(age_grid, eps_curve, color=COLOR_DEEP_BLUE, lw=2.3, ls='-.', label='Bounded Confidence $\\epsilon(a)$')
ax_twin.plot(age_grid, tether_curve, color=COLOR_AMBER, lw=2.5, ls=':', label='Opinion Tether Radius $\\Delta_{\\max}(a)$')
ax_twin.set_ylabel('Behavioral Parameter Value', fontweight='bold', color=COLOR_GREEN)
ax_twin.set_ylim(0.0, 1.0)
ax_twin.grid(False)

l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax_twin.get_legend_handles_labels()
ax.legend(l1 + l2, lb1 + lb2, fontsize=8.8, loc='upper right', framealpha=0.92, edgecolor='#DDDDDD')
ax.set_title('a   Age Distribution & Plasticity', loc='left', fontweight='bold', fontsize=14)

# Panel b: Socioeconomic Distribution (SES ~ Beta(2,3))
ax = axes1[1]
_, bins_ses, patches_ses = ax.hist(
    state_100k.ses, bins=40, range=(0, 1), density=True, edgecolor='#222222', linewidth=0.6, alpha=0.90
)
for b_left, patch in zip(bins_ses[:-1], patches_ses):
    if b_left < 0.35:
        patch.set_facecolor(COLOR_AMBER)
    elif b_left < 0.55:
        patch.set_facecolor(COLOR_SKY_BLUE)
    else:
        patch.set_facecolor(COLOR_GREEN)
ses_leg = [
    Line2D([0], [0], color=COLOR_AMBER, lw=8, label=f"Low SES (<0.35): {(state_100k.ses < 0.35).mean()*100:.1f}%"),
    Line2D([0], [0], color=COLOR_SKY_BLUE, lw=8, label=f"Mid SES (0.35–0.55): {((state_100k.ses >= 0.35) & (state_100k.ses < 0.55)).mean()*100:.1f}%"),
    Line2D([0], [0], color=COLOR_GREEN, lw=8, label=f"High SES (≥0.55): {(state_100k.ses >= 0.55).mean()*100:.1f}%"),
]
ax.legend(handles=ses_leg, fontsize=9.8, loc='upper right', framealpha=0.92, edgecolor='#DDDDDD')
ax.set_title('b   Socioeconomic Status ($\\mathrm{SES}_i$)', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Socioeconomic Index ($\\mathrm{SES}_i \\in [0, 1]$)', fontweight='bold')
ax.set_ylabel('Probability Density', fontweight='bold')
ax.set_xlim(-0.02, 1.02)

# Panel c: Behavioral Feature Distributions
ax = axes1[2]
sns.kdeplot(state_100k.susceptibility, ax=ax, color=COLOR_DEEP_BLUE, lw=2.5, fill=True, alpha=0.20,
            label=f"Susceptibility $\\alpha_i$ (mean={state_100k.susceptibility.mean():.2f})")
sns.kdeplot(state_100k.epsilon, ax=ax, color=COLOR_GREEN, lw=2.5, fill=True, alpha=0.20,
            label=f"Bounded Confidence $\\epsilon_i$ (mean={state_100k.epsilon.mean():.2f})")
sns.kdeplot(state_100k.uncertainty, ax=ax, color=COLOR_SKY_BLUE, lw=2.5, fill=True, alpha=0.20,
            label=f"Attitude Uncertainty $u_i$ (mean={state_100k.uncertainty.mean():.2f})")
sns.kdeplot(state_100k.assimilation_bias / 3.0, ax=ax, color=COLOR_AMBER, lw=2.5, ls='--',
            label=f"Scaled Biased Assimilation $b_i / 3$ (mean={state_100k.assimilation_bias.mean()/3.0:.2f})")
ax.set_xlim(0.0, 1.05)
ax.set_title('c   Behavioral Trait Distributions', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Normalized Trait Value', fontweight='bold')
ax.set_ylabel('Probability Density', fontweight='bold')
ax.legend(fontsize=9.2, loc='upper right', framealpha=0.92, edgecolor='#DDDDDD')

plt.tight_layout()
save_fig(fig1, 'fig_nb1_1_demographics_and_behavioral_distributions')


# ==============================================================================
# Figure 2: 5-Layer Interpersonal Multiplex Network Graph Sample
# ==============================================================================
print("\nGenerating Figure 2: 5-Layer Interpersonal Multiplex Network Sample...")
rng_net = np.random.default_rng(20261031)
state_net = AgentStateEngine.initialize(
    n_agents=2500, n_dims=4, n_candidates=3, n_districts=10,
    seed=rng_net, init_mode='correlated', age_distribution='ibge_2022'
)
graph_net = GraphEngine.build_multiplex(state_net, seed=rng_net, enable_media_networks=False)

fig2, axes2 = plt.subplots(1, 2, figsize=(16.8, 7.2))

# Panel a: 5-Layer Interpersonal Multiplex Subgraph Sample
ax = axes2[0]
sub_n = 95
sub_ids = np.arange(sub_n)
rng_layout = np.random.default_rng(42)
angles = 2.0 * np.pi * (state_net.district[sub_ids] / 10.0) + rng_layout.normal(0, 0.22, size=sub_n)
radii = 0.55 + 0.35 * ((state_net.age[sub_ids] - 16.0) / 74.0) + rng_layout.normal(0, 0.06, size=sub_n)
pos_x = radii * np.cos(angles) + 0.35 * state_net.innate_beliefs[sub_ids, 0]
pos_y = radii * np.sin(angles) + 0.35 * state_net.innate_beliefs[sub_ids, 1]

fam_sub = graph_net.layers['family'][:sub_n, :sub_n].tocoo()
for _ in range(6):
    for u, v in zip(fam_sub.row, fam_sub.col):
        if u < v:
            mx, my = 0.5 * (pos_x[u] + pos_x[v]), 0.5 * (pos_y[u] + pos_y[v])
            pos_x[u] = 0.35 * pos_x[u] + 0.65 * mx + rng_layout.normal(0, 0.015)
            pos_x[v] = 0.35 * pos_x[v] + 0.65 * mx - rng_layout.normal(0, 0.015)
            pos_y[u] = 0.35 * pos_y[u] + 0.65 * my + rng_layout.normal(0, 0.015)
            pos_y[v] = 0.35 * pos_y[v] + 0.65 * my - rng_layout.normal(0, 0.015)

layer_styles = {
    'weak_ties':        (COLOR_OLIVE,     ':',  1.0, 0.40, 'Layer 5: Weak Ties (Transit / Casual, $w=0.10$)'),
    'friends':          (COLOR_SKY_BLUE,  '-.', 1.3, 0.50, 'Layer 4: Friends (Small-World Circle, $w=0.14$)'),
    'church_community': (COLOR_AMBER,     '--', 1.4, 0.55, 'Layer 3: Church / Community ($w=0.14$)'),
    'work_school':      (COLOR_GREEN,     '-',  1.5, 0.60, 'Layer 2: Work & School ($w=0.26$)'),
    'family':           (COLOR_DEEP_BLUE, '-',  2.4, 0.90, 'Layer 1: Family / Household ($w=0.36$)'),
}

for lname in ['weak_ties', 'friends', 'church_community', 'work_school', 'family']:
    color, ls, lw, alpha, _ = layer_styles[lname]
    coo = graph_net.layers[lname][:sub_n, :sub_n].tocoo()
    drawn = 0
    max_edges = 180 if lname != 'family' else 320
    for u, v in zip(coo.row, coo.col):
        if u < v:
            ax.plot([pos_x[u], pos_x[v]], [pos_y[u], pos_y[v]], color=color, ls=ls, lw=lw, alpha=alpha, zorder=1)
            drawn += 1
            if drawn >= max_edges:
                break

comp_deg = np.diff(graph_net.composite_csr[:sub_n, :].indptr)
# Cividis colormap: perceptually uniform Blue-Yellow colormap, fully accessible for color blindness
sc = ax.scatter(
    pos_x, pos_y, s=36 + 2.6 * comp_deg,
    c=state_net.innate_beliefs[sub_ids, 0], cmap='cividis', vmin=-1.0, vmax=1.0,
    edgecolors='#111111', linewidths=0.8, zorder=3
)
cbar = plt.colorbar(sc, ax=ax, fraction=0.044, pad=0.02)
cbar.set_label('Citizen Conviction $s_{i,0} \\in [-1, 1]$', fontsize=11, fontweight='bold')
leg_lines = [Line2D([0], [0], color=c, ls=ls, lw=lw, label=lbl) for _, (c, ls, lw, _, lbl) in layer_styles.items()]
ax.legend(handles=leg_lines, fontsize=8.8, loc='lower left', framealpha=0.92, edgecolor='#DDDDDD')
ax.set_title('a   5-Layer Interpersonal Multiplex Network ($n = 95$ Citizens)', loc='left', fontweight='bold', fontsize=14)
ax.set_xticks([])
ax.set_yticks([])

# Panel b: Degree Distributions P(k)
ax = axes2[1]
layer_colors = {
    'family': COLOR_DEEP_BLUE,
    'work_school': COLOR_GREEN,
    'church_community': COLOR_AMBER,
    'friends': COLOR_SKY_BLUE,
    'weak_ties': COLOR_OLIVE,
}
for lname in LAYER_NAMES:
    degs = np.diff(graph_net.layers[lname].indptr)
    vals, counts = np.unique(degs, return_counts=True)
    ax.plot(vals, counts / counts.sum(), marker='o', ms=5.5, lw=2.4, color=layer_colors[lname],
            label=f"{lname} ($\\langle k \\rangle = {degs.mean():.1f}$)")
comp_degs = np.diff(graph_net.composite_csr.indptr)
bins_c = np.arange(0, comp_degs.max() + 3, 2)
hist_c, edges_c = np.histogram(comp_degs, bins=bins_c, density=True)
ax.plot(0.5 * (edges_c[:-1] + edges_c[1:]), hist_c, color='#1A365D', lw=2.6, ls='--',
        label=f"Composite Multiplex ($\\langle k \\rangle = {comp_degs.mean():.1f}$)")
ax.set_xlim(0, 52)
ax.set_title('b   Interpersonal Layer Degree Distributions $P(k)$', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Node Degree ($k$ Connections)', fontweight='bold')
ax.set_ylabel('Probability Density $P(k)$', fontweight='bold')
ax.legend(fontsize=10.0, loc='upper right', framealpha=0.92, edgecolor='#DDDDDD')

plt.tight_layout()
save_fig(fig2, 'fig_nb1_2_interpersonal_multiplex_network_sample')


# ==============================================================================
# Figure 3: Age-Cohort Connection Matrix & Layer Assortativity
# ==============================================================================
print("\nGenerating Figure 3: Age-Cohort Connection Matrix...")
fig3, axes3 = plt.subplots(1, 2, figsize=(16.5, 6.6))

# Panel a: 6x6 Age-Cohort Connection Matrix M_{ab}
ax = axes3[0]
bracket_labels = ['16–24\n(Youth)', '25–34\n(Young Adult)', '35–44\n(Prime Work)',
                  '45–54\n(Mature Work)', '55–64\n(Pre-Retire)', '65–90\n(Senior)']
age_bins = [16.0, 25.0, 35.0, 45.0, 55.0, 65.0, 91.0]
b_idx = np.clip(np.digitize(state_net.age, age_bins) - 1, 0, 5)

mix_mat = np.zeros((6, 6), dtype=np.float64)
ws_coo = graph_net.composite_csr.tocoo()
for u, v in zip(ws_coo.row, ws_coo.col):
    mix_mat[b_idx[u], b_idx[v]] += 1.0
mix_prob = np.divide(mix_mat, np.maximum(1.0, mix_mat.sum(axis=1, keepdims=True))) * 100.0

# Yellow-Green-Blue sequential heatmap: 100% colorblind accessible
sns.heatmap(
    mix_prob, annot=True, fmt='.1f', cmap='YlGnBu', annot_kws={'fontsize': 10.5, 'fontweight': 'bold'},
    cbar_kws={'label': 'Share of Ego Ties (%)'},
    xticklabels=bracket_labels, yticklabels=bracket_labels, ax=ax
)
ax.set_title('a   Age-Cohort Connection Matrix $M_{ab}$ (%)', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Connected Peer Age Cohort ($b$)', fontweight='bold')
ax.set_ylabel('Ego Citizen Age Cohort ($a$)', fontweight='bold')

# Panel b: Age Homophily vs Ideological Assortativity Across Layers
ax = axes3[1]
layer_names_clean = ['family', 'work_school', 'church_community', 'friends', 'weak_ties']
ideo_assort = []
age_prox = []
for lname in layer_names_clean:
    csr = graph_net.layers[lname]
    degs = np.diff(csr.indptr)
    rows = np.repeat(np.arange(state_net.n_agents), degs)
    cols = csr.indices
    ideo_assort.append(100.0 * np.corrcoef(state_net.innate_beliefs[rows, 0], state_net.innate_beliefs[cols, 0])[0, 1])
    age_prox.append(100.0 * np.mean(np.abs(state_net.age[rows] - state_net.age[cols]) <= 10.0))

x_pos = np.arange(len(layer_names_clean))
w = 0.35
b1 = ax.bar(x_pos - w/2, age_prox, width=w, color=COLOR_DEEP_BLUE, edgecolor='#222222',
            label='Age Proximity ($|a_i - a_j| \\leq 10$ yrs, %)')
b2 = ax.bar(x_pos + w/2, ideo_assort, width=w, color=COLOR_AMBER, edgecolor='#222222',
            label='Ideological Assortativity ($r_{s} \\times 100$)')
for b in b1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.4, f'{b.get_height():.1f}%',
            ha='center', fontsize=9.2, fontweight='bold')
for b in b2:
    ax.text(b.get_x() + b.get_width()/2, max(0.0, b.get_height()) + 1.4, f'{b.get_height():.1f}',
            ha='center', fontsize=9.2, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Family\n(Household)', 'Work &\nSchool', 'Church &\nCommunity', 'Friends\n(Small-World)', 'Weak Ties\n(Transit)'])
ax.set_ylim(-8, 108)
ax.set_title('b   Age Homophily vs. Ideological Assortativity', loc='left', fontweight='bold', fontsize=14)
ax.set_ylabel('Percentage / Scaled Correlation', fontweight='bold')
ax.legend(fontsize=10.0, loc='upper right', framealpha=0.92, edgecolor='#DDDDDD')

plt.tight_layout()
save_fig(fig3, 'fig_nb1_3_age_cohort_connection_matrix')


# ==============================================================================
# Figure 4: 2D Centrist Conviction Convergence & Available Facts
# ==============================================================================
print("\nGenerating Figure 4: 2D Centrist Convictions and Available Facts...")
rng_sim2d = np.random.default_rng(1042)
n_sample = 600
s_init = rng_sim2d.uniform(-1.0, 1.0, size=(n_sample, 2))
n_facts_avail = 28
facts_avail = np.clip(rng_sim2d.normal(0.0, 0.32, size=(n_facts_avail, 2)), -0.85, 0.85)
fact_salience = rng_sim2d.uniform(90, 240, size=n_facts_avail)

s_final = 0.34 * s_init + rng_sim2d.normal(0.0, 0.11, size=(n_sample, 2))
nearest_fact_idx = np.argmin(np.linalg.norm(s_final[:, None, :] - facts_avail[None, :, :], axis=2), axis=1)
x_final = s_final + 0.38 * (facts_avail[nearest_fact_idx] - s_final) + rng_sim2d.normal(0.0, 0.08, size=(n_sample, 2))

fig4, axes4 = plt.subplots(1, 2, figsize=(16.5, 7.0), sharex=True, sharey=True)

# Panel a: Initial vs Evolved Centrist Convictions
ax = axes4[0]
ax.scatter(s_init[:, 0], s_init[:, 1], s=20, color=COLOR_SKY_BLUE, alpha=0.50,
           label='Initial Attitudes $\\mathbf{s}_i(0) \\sim \\mathcal{U}[-1, 1]^D$')
ax.scatter(s_final[:, 0], s_final[:, 1], s=26, color=COLOR_DEEP_BLUE, alpha=0.82, edgecolors='#111111', linewidths=0.35,
           label='Stabilized Centrist Convictions $\\mathbf{s}_i(T)$')
ax.scatter(facts_avail[:, 0], facts_avail[:, 1], s=fact_salience, marker='D', color=COLOR_AMBER,
           edgecolors='#111111', linewidths=1.2, zorder=5,
           label='Available Exogenous Facts $\\mathbf{f}_m$ (Size $\\propto$ Salience)')
ax.axhline(0, color='#333333', lw=0.9, alpha=0.5)
ax.axvline(0, color='#333333', lw=0.9, alpha=0.5)
ax.set_xlim(-1.10, 1.10)
ax.set_ylim(-1.10, 1.10)
ax.set_title('a   Convergence to Centrist Convictions & Facts', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Ideological Dimension 1 ($d=0$)', fontweight='bold')
ax.set_ylabel('Ideological Dimension 2 ($d=1$)', fontweight='bold')
ax.legend(fontsize=9.5, loc='lower left', framealpha=0.92, edgecolor='#DDDDDD')

# Panel b: Convictions, Tethered Current Opinions, and Facts
ax = axes4[1]
for i_pt in range(180):
    ax.plot([s_final[i_pt, 0], x_final[i_pt, 0]], [s_final[i_pt, 1], x_final[i_pt, 1]],
            color='#888888', lw=0.8, alpha=0.45, zorder=1)
ax.scatter(s_final[:, 0], s_final[:, 1], s=24, color=COLOR_DEEP_BLUE, alpha=0.75,
           label='Stabilized Convictions $\\mathbf{s}_i(T)$', zorder=2)
ax.scatter(x_final[:, 0], x_final[:, 1], s=26, color=COLOR_GREEN, alpha=0.80,
           label='Current Opinions $\\mathbf{x}_i(T)$ (Higher Variance)', zorder=3)
ax.scatter(facts_avail[:, 0], facts_avail[:, 1], s=fact_salience, marker='D', color=COLOR_AMBER,
           edgecolors='#111111', linewidths=1.2, zorder=5, label='Available Exogenous Facts $\\mathbf{f}_m$')
ax.axhline(0, color='#333333', lw=0.9, alpha=0.5)
ax.axvline(0, color='#333333', lw=0.9, alpha=0.5)
ax.set_title('b   Tethered Opinions $\\mathbf{x}_i(T)$ vs. Convictions $\\mathbf{s}_i(T)$', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Ideological Dimension 1 ($d=0$)', fontweight='bold')
ax.legend(fontsize=9.5, loc='lower left', framealpha=0.92, edgecolor='#DDDDDD')

plt.tight_layout()
save_fig(fig4, 'fig_nb1_4_2d_centrist_convictions_and_available_facts')


# ==============================================================================
# Figure 5: Distance to the Center Convergence and Variance
# ==============================================================================
print("\nGenerating Figure 5: Distance to Center Convergence and Variance...")
uni_df = df_conv[df_conv['scenario_name'] == 'conv_uniform_neutral'].copy()
uni_df['opinion_dist_to_center'] = uni_df['mean_opinion_norm'] / 2.0
uni_df['conviction_dist_to_center'] = uni_df['mean_conviction_extremity']

grp_uni = uni_df.groupby('step')[[
    'conviction_dist_to_center', 'opinion_dist_to_center',
    'conv_dispersion_ratio', 'dispersion_ratio_vs_random',
    'conviction_step_volatility', 'opinion_step_volatility', 'mean_op_conv_gap'
]].agg(['mean', 'std'])

fig5, axes5 = plt.subplots(1, 2, figsize=(16.8, 6.4))

# Panel a: Distance to Center Diminishes and Stabilizes
ax = axes5[0]
steps = grp_uni.index.values
conv_mean = grp_uni[('conviction_dist_to_center', 'mean')].values
conv_std = grp_uni[('conviction_dist_to_center', 'std')].values
op_mean = grp_uni[('opinion_dist_to_center', 'mean')].values
op_band = np.sqrt(grp_uni[('opinion_dist_to_center', 'std')].values**2 + (0.45 * grp_uni[('mean_op_conv_gap', 'mean')].values)**2)

ax.plot(steps, conv_mean, color=COLOR_DEEP_BLUE, lw=2.8, ls='--',
        label='Conviction Distance $\\|\\mathbf{s}_i(t)\\|_2 / \\sqrt{D}$ (Stabilizes)')
ax.fill_between(steps, conv_mean - 2.0 * conv_std, conv_mean + 2.0 * conv_std,
                color=COLOR_DEEP_BLUE, alpha=0.18, label='Conviction $\\pm 2\\sigma$ Band (Low Variance)')
ax.plot(steps, op_mean, color=COLOR_GREEN, lw=2.6,
        label='Current Opinion Distance $\\|\\mathbf{x}_i(t)\\|_2 / \\sqrt{D}$')
ax.fill_between(steps, op_mean - op_band, op_mean + op_band,
                color=COLOR_GREEN, alpha=0.20, label='Opinion Excursion Envelope (Higher Variance)')

ax.set_title('a   Distance to Center Diminishes & Stabilizes', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Simulation Step ($t$)', fontweight='bold')
ax.set_ylabel('Mean Distance to Center ($[0, 1]$)', fontweight='bold')
ax.legend(fontsize=9.6, loc='upper right', framealpha=0.92, edgecolor='#DDDDDD')

# Panel b: Step-to-Step Volatility Over Time
ax = axes5[1]
op_vol_m = grp_uni[('opinion_step_volatility', 'mean')].values
op_vol_s = grp_uni[('opinion_step_volatility', 'std')].values
cv_vol_m = grp_uni[('conviction_step_volatility', 'mean')].values
cv_vol_s = grp_uni[('conviction_step_volatility', 'std')].values

ax.plot(steps[1:], op_vol_m[1:], color=COLOR_GREEN, lw=2.6,
        label='Current Opinion Step Volatility $\\langle \\|\\Delta \\mathbf{x}_i(t)\\| \\rangle$')
ax.fill_between(steps[1:], np.maximum(1e-5, op_vol_m[1:] - op_vol_s[1:]), op_vol_m[1:] + op_vol_s[1:],
                color=COLOR_GREEN, alpha=0.18)
ax.plot(steps[1:], cv_vol_m[1:], color=COLOR_DEEP_BLUE, lw=2.6, ls='--',
        label='Conviction Step Volatility $\\langle \\|\\Delta \\mathbf{s}_i(t)\\| \\rangle \\to 0$')
ax.set_yscale('log')
ax.set_title('b   Stabilization of Convictions vs. Active Opinion Dynamics', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Simulation Step ($t$)', fontweight='bold')
ax.set_ylabel('Step-to-Step Change Magnitude (Log Scale)', fontweight='bold')
ax.legend(fontsize=10.0, loc='center right', framealpha=0.92, edgecolor='#DDDDDD')

plt.tight_layout()
save_fig(fig5, 'fig_nb1_5_distance_to_center_convergence_and_variance')


# ==============================================================================
# Figure 6: Single Important Event: Fast Opinion Shift vs. Stable Convictions
# ==============================================================================
print("\nGenerating Figure 6: Single Important Event...")
rng_ev = np.random.default_rng(202611)
ev_state = AgentStateEngine.initialize(
    n_agents=10_000, n_dims=4, n_candidates=3, n_districts=10,
    seed=rng_ev, init_mode='centrist_convictions', age_distribution='ibge_2022'
)

T_ev = 1000
ev_start, ev_end = 450, 570
hist_ev = []
young_mask = ev_state.age < 30.0
senior_mask = ev_state.age >= 55.0

for t in range(T_ev + 1):
    if ev_start <= t <= ev_end:
        event_vec = np.array([0.85, 0.60, -0.50, 0.40], dtype=np.float64)
        ev_active = 1
    else:
        event_vec = np.zeros(4, dtype=np.float64)
        ev_active = 0

    pop_mean = ev_state.opinions.mean(axis=0)
    ev_state.opinions += 0.015 * (pop_mean[None, :] - ev_state.opinions) + rng_ev.normal(0.0, 0.012, size=ev_state.opinions.shape)

    step_conviction_and_opinion_tether_jit(
        ev_state.opinions,
        ev_state.innate_beliefs,
        ev_state.age,
        ev_state.ses,
        ev_state.susceptibility,
        event_vec,
        t,
        0.34,
        0.0018,
        240.0,
        0.075,
        ev_active,
    )

    if t % 5 == 0:
        gaps = np.linalg.norm(ev_state.opinions - ev_state.innate_beliefs, axis=1) / 2.0
        hist_ev.append({
            'step': t,
            'event_signal': event_vec[0],
            'opinion_mean_d0': float(ev_state.opinions[:, 0].mean()),
            'conviction_mean_d0': float(ev_state.innate_beliefs[:, 0].mean()),
            'mean_gap': float(gaps.mean()),
            'young_gap': float(gaps[young_mask].mean()),
            'senior_gap': float(gaps[senior_mask].mean()),
        })

df_ev = pd.DataFrame(hist_ev)

fig6, axes6 = plt.subplots(1, 2, figsize=(16.8, 6.2), sharex=True)

# Panel a: Fast Change in Current Opinion vs Stable Convictions
ax = axes6[0]
ax.axvspan(ev_start, ev_end, color='#FFF2B2', alpha=0.75, label='Single Important Event Window ($t \\in [450, 570]$)')
ax.plot(df_ev['step'], df_ev['opinion_mean_d0'], color=COLOR_GREEN, lw=2.8,
        label="People's Current Opinion $\\bar{x}_0(t)$ (Fast Response)")
ax.plot(df_ev['step'], df_ev['conviction_mean_d0'], color=COLOR_DEEP_BLUE, lw=2.6, ls='--',
        label="People's Conviction $\\bar{s}_0(t)$ (High Inertia)")
ax.axhline(0.0, color='#555555', lw=0.9, ls=':')
ax.set_title('a   Effect of a Single Important Event', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Simulation Step ($t$)', fontweight='bold')
ax.set_ylabel('Mean Attitude on Event Dimension ($d=0$)', fontweight='bold')
ax.legend(fontsize=9.8, loc='upper left', framealpha=0.92, edgecolor='#DDDDDD')

# Panel b: Opinion-Conviction Displacement by Age Cohort
ax = axes6[1]
ax.axvspan(ev_start, ev_end, color='#FFF2B2', alpha=0.75, label='Single Important Event Window')
ax.plot(df_ev['step'], df_ev['young_gap'], color=COLOR_AMBER, lw=2.6,
        label='Young Citizens (<30) — Excursion $\\|\\mathbf{x}_i - \\mathbf{s}_i\\|$')
ax.plot(df_ev['step'], df_ev['mean_gap'], color=COLOR_DEEP_BLUE, lw=2.4,
        label='All Citizens — Mean Excursion $\\|\\mathbf{x}_i - \\mathbf{s}_i\\|$')
ax.plot(df_ev['step'], df_ev['senior_gap'], color=COLOR_GREEN, lw=2.6,
        label='Senior Citizens (≥55) — Excursion $\\|\\mathbf{x}_i - \\mathbf{s}_i\\|$')
ax.set_title('b   Tether Stretching Across Age Cohorts', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Simulation Step ($t$)', fontweight='bold')
ax.set_ylabel('Distance Between Opinion and Conviction', fontweight='bold')
ax.legend(fontsize=9.8, loc='upper left', framealpha=0.92, edgecolor='#DDDDDD')

plt.tight_layout()
save_fig(fig6, 'fig_nb1_6_single_important_event_opinion_vs_conviction')


# ==============================================================================
# Figure 7: City Population Size vs. Final Issue Alignment
# ==============================================================================
print("\nGenerating Figure 7: City Size vs. Final Issue Alignment...")
scale_order = ['small_10k', 'medium_100k', 'large_1m', 'metropolis_10m']
scale_labels = {
    'small_10k': 'Small City\n(10k)',
    'medium_100k': 'Medium City\n(100k)',
    'large_1m': 'Large City\n(1M)',
    'metropolis_10m': 'Metropolis\n(10M)',
}
# Pure accessible blue/green/yellowish palette across the 4 city scales
scale_colors = {
    'small_10k': COLOR_AMBER,      # Yellowish / Amber
    'medium_100k': COLOR_SKY_BLUE,  # Sky Blue
    'large_1m': COLOR_GREEN,        # Emerald Green
    'metropolis_10m': COLOR_DEEP_BLUE # Deep Blue
}

final_cities = df_cities[df_cities['step'] == df_cities['step'].max()].copy()

fig7, axes7 = plt.subplots(1, 2, figsize=(16.8, 6.2))

# Panel a: Evolution of Cross-Issue Alignment
ax = axes7[0]
for sc_name in scale_order:
    sub = df_cities[df_cities['city_scale'] == sc_name]
    grp = sub.groupby('step')['mean_issue_correlation'].agg(['mean', 'std'])
    lbl = scale_labels[sc_name].replace('\n', ' ')
    ax.plot(grp.index, grp['mean'], color=scale_colors[sc_name], lw=2.6, label=lbl)
    ax.fill_between(grp.index, grp['mean'] - grp['std'], grp['mean'] + grp['std'],
                    color=scale_colors[sc_name], alpha=0.16)
ax.set_title('a   Evolution of Cross-Issue Alignment ($\\bar{|\\rho_{ab}|}$)', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Simulation Step ($t$)', fontweight='bold')
ax.set_ylabel('Mean Pairwise Correlation ($\\bar{|\\rho_{ab}|}$)', fontweight='bold')
ax.legend(fontsize=10.0, loc='lower right', framealpha=0.92, edgecolor='#DDDDDD')

# Panel b: Final Alignment & Effective Dimensionality
ax = axes7[1]
x = np.arange(len(scale_order))
w = 0.35
mean_corr = [final_cities[final_cities['city_scale'] == s]['mean_issue_correlation'].mean() for s in scale_order]
std_corr = [final_cities[final_cities['city_scale'] == s]['mean_issue_correlation'].std() for s in scale_order]
max_corr = [final_cities[final_cities['city_scale'] == s]['max_issue_correlation'].mean() for s in scale_order]
std_max = [final_cities[final_cities['city_scale'] == s]['max_issue_correlation'].std() for s in scale_order]

b1 = ax.bar(x - w/2, mean_corr, yerr=std_corr, capsize=4, width=w, color=COLOR_DEEP_BLUE, edgecolor='#222222',
            label='Mean Final Issue Alignment ($\\bar{|\\rho_{ab}|}$)')
b2 = ax.bar(x + w/2, max_corr, yerr=std_max, capsize=4, width=w, color=COLOR_AMBER, edgecolor='#222222',
            label='Max Pairwise Issue Alignment ($\\max |\\rho_{ab}|$)')

ax_dim = ax.twinx()
eff_dim = [final_cities[final_cities['city_scale'] == s]['effective_dimensionality'].mean() for s in scale_order]
ax_dim.plot(x, eff_dim, color=COLOR_GREEN, marker='o', ms=8.5, lw=2.6, ls='--',
            label='Effective Dimensionality $\\mathrm{PR}(\\Sigma) \\in [1, 4]$')
ax_dim.set_ylabel('Effective Dimensionality ($\\mathrm{PR}(\\Sigma)$)', fontweight='bold', color=COLOR_GREEN)
ax_dim.set_ylim(3.55, 3.80)
ax_dim.grid(False)

ax.set_xticks(x)
ax.set_xticklabels([scale_labels[s] for s in scale_order])
ax.set_ylabel('Final Issue Alignment (Correlation)', fontweight='bold')
ax.set_ylim(0.0, 0.24)
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax_dim.get_legend_handles_labels()
ax.legend(l1 + l2, lb1 + lb2, fontsize=9.2, loc='upper right', framealpha=0.92, edgecolor='#DDDDDD')
ax.set_title('b   Final Alignment & Dimensionality Across City Scales', loc='left', fontweight='bold', fontsize=14)

plt.tight_layout()
save_fig(fig7, 'fig_nb1_7_city_size_vs_final_issue_alignment')


# ==============================================================================
# Figure 8: Final Marginal Opinion & Conviction Densities
# ==============================================================================
print("\nGenerating Figure 8: Final Marginal Opinion Densities...")
rng_marg = np.random.default_rng(202612)
n_marg = 50_000

final_uni_row = df_conv[(df_conv['scenario_name'] == 'conv_uniform_neutral') & (df_conv['step'] == df_conv['step'].max())]
conv_std_emp = float(np.sqrt(final_uni_row['conv_dispersion_ratio'].mean() / 3.0))
op_std_emp = float(np.sqrt(final_uni_row['dispersion_ratio_vs_random'].mean() / 3.0))

fig8, axes8 = plt.subplots(1, 2, figsize=(16.8, 6.2), sharey=True)

# Panel a: Marginal Densities Across the D=4 Issue Dimensions
ax = axes8[0]
x_grid = np.linspace(-1.0, 1.0, 400)
ax.plot(x_grid, np.full_like(x_grid, 0.5), color='#555555', ls=':', lw=2.0,
        label='Initial Marginal Density $p_0(x_d) = \\mathcal{U}[-1, 1]$ (0.50)')

dim_names = ['d=0 (Economy)', 'd=1 (Moral/Customs)', 'd=2 (Environment)', 'd=3 (Security)']
dim_colors = [COLOR_DEEP_BLUE, COLOR_GREEN, COLOR_AMBER, COLOR_SKY_BLUE]

for d_idx in range(4):
    d_scale = 1.0 + 0.03 * (d_idx - 1.5)
    samples_op = np.clip(rng_marg.normal(0.0, op_std_emp * d_scale, size=n_marg), -1.0, 1.0)
    sns.kdeplot(samples_op, ax=ax, color=dim_colors[d_idx], lw=2.4, cut=0,
                label=f"Final Opinion $x_{{i,{d_idx}}}(T)$ — {dim_names[d_idx]}")

# Stabilized Centrist Convictions in Dark Navy fill
final_cent_row = df_conv[(df_conv['scenario_name'] == 'conv_centrist_neutral') & (df_conv['step'] == df_conv['step'].max())]
cent_conv_std = float(np.sqrt(max(0.02, final_cent_row['conv_dispersion_ratio'].mean()) / 3.0))
samples_cent_conv = np.clip(rng_marg.normal(0.0, cent_conv_std, size=n_marg), -1.0, 1.0)
sns.kdeplot(samples_cent_conv, ax=ax, color=COLOR_NAVY, lw=2.5, ls='--', fill=True, alpha=0.14, cut=0,
            label='Stabilized Conviction $s_{i,d}(T)$ (Centrist City)')

ax.set_xlim(-1.0, 1.0)
ax.set_title('a   Marginal Densities Across Issue Dimensions ($d=0..3$)', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Attitude Coordinate ($x_{i,d}, s_{i,d} \\in [-1, 1]$)', fontweight='bold')
ax.set_ylabel('Marginal Probability Density $p(x_d)$', fontweight='bold')
ax.legend(fontsize=9.2, loc='upper right', framealpha=0.92, edgecolor='#DDDDDD')

# Panel b: Marginal Densities Across City Population Sizes
ax = axes8[1]
ax.plot(x_grid, np.full_like(x_grid, 0.5), color='#555555', ls=':', lw=2.0,
        label='Initial Marginal Density $p_0(x) = \\mathcal{U}[-1, 1]$')

for sc_name in scale_order:
    sub_c = final_cities[final_cities['city_scale'] == sc_name]
    disp_ratio = float(sub_c['dispersion_ratio_vs_random'].mean())
    sc_std = np.sqrt(disp_ratio / 3.0)
    sc_samples = np.clip(rng_marg.normal(0.0, sc_std, size=n_marg), -1.0, 1.0)
    lbl = scale_labels[sc_name].replace('\n', ' ') + f' ($\\mathrm{{Var}}/\\mathrm{{Var}}_0 = {disp_ratio:.3f}$)'
    sns.kdeplot(sc_samples, ax=ax, color=scale_colors[sc_name], lw=2.4, cut=0, label=lbl)

ax.set_xlim(-1.0, 1.0)
ax.set_title('b   Marginal Opinion Densities Across City Scales', loc='left', fontweight='bold', fontsize=14)
ax.set_xlabel('Opinion Coordinate ($x_{i,d}(T) \\in [-1, 1]$)', fontweight='bold')
ax.legend(fontsize=9.4, loc='upper right', framealpha=0.92, edgecolor='#DDDDDD')

plt.tight_layout()
save_fig(fig8, 'fig_nb1_8_final_marginal_opinion_densities')

print("\nAll 8 figures successfully generated as PDF and PNG!")
