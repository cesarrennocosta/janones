# Replication Package: Late-Campaign Opinion Volatility and Algorithmic Counter-Mobilization in Polarized Elections

Official computational reproduction suite and replication scripts for the research paper:

> **Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections**  
> **Author:** César Rennó-Costa$^{1,*}$  
> $^{1}$*Brain Institute, Federal University of Rio Grande do Norte (UFRN), Natal, RN, Brazil*  
> $^*$Corresponding author: `rennocosta@neuro.ufrn.br`  
> **Target Journal:** *Nature Human Behaviour* (2026)

---

## 1. Overview

This repository contains the complete, self-contained codebase for reproducing all empirical analyses, agent-based simulations, parametric sensitivity sweeps, and figures presented in the manuscript.

The study develops an empirical digital twin of the 2026 Brazilian presidential election—a high-stakes quasi-natural experiment featuring a preliminary first-round conservative surge followed by a historic runoff reversal. The model couples:
1. A **census-calibrated urban multiplex network** across 5 physical layers (household cliques, workplace/school cohorts, religious congregations, small-world friendship circles, and transit weak ties);
2. An **institutional traditional mass media ecosystem** (television, radio, print, and outdoor signage) with modality-specific reach curves and editorial tendentiousness;
3. A **multi-platform algorithmic social media environment** with power-law influencer follower hierarchies, dual-valence engagement recommendation scoring along the diagonal of controversy ($R_m = \text{Likes}_m + \omega \cdot \text{Dislikes}_m$, $\omega = 1.0$), and dark-social encrypted micro-clique cascades (WhatsApp/Telegram);
4. A **two-timescale cognitive opinion dynamics engine** bridging Philip Converse's belief constraint paradigm with John Zaller's Receive-Accept-Sample model, where volatile expressed public opinions $\mathbf{x}_i(t)$ are elastically tethered to slow-crystallizing internal convictions $\mathbf{s}_i(t)$ via a radial hyperbolic-tangent bound;
5. An **asymmetric candidate evaluation calculus** where candidate aversion $R_{ik}(t)$ accumulates faster and decays slower than affinity $A_{ik}(t)$ ($(\lambda_R / \delta_R) / (\lambda_A / \delta_A) = 1.75$), generating cognitive hysteresis;
6. An operational translation of **Janonismo Cultural** into 6 graph-theoretic counter-tactics that reverse rival momentum under strict 50/50 broadcast parity.

---

## 2. Directory Structure

```
replication/
├── README.md                      # Replication guide, parameter mappings, and citations
├── run_simulations.py             # Master CLI runner for all simulations and figures
├── sim_figure1_empirical_data.py  # Figure 1: Empirical 2026 election data and polling trajectories
├── sim_figure2_urban_network.py   # Figure 2: Urban multiplex contact network and two-timescale tether
├── sim_figure3_traditional_media.py # Figure 3: Traditional mass media gatekeeping and polarization
├── sim_figure4_social_media.py    # Figure 4: Algorithmic social media, influencer power-law, and cascades
├── sim_figure5_first_round.py     # Figure 5: First-round ABM calibration and 5 Janones mechanisms
├── sim_figure6_second_round.py    # Figure 6: Runoff Janonismo counter-tactics evaluation
└── paper_reproduction_figures.ipynb # Master interactive reproduction Jupyter Notebook
```

---

## 3. Installation & Environment Setup

The simulation engine is implemented in Python 3.10+ using Numba JIT compiler with parallel vectorization, NumPy, SciPy, and Matplotlib.

### Using `uv` (Recommended)
```bash
# Clone the repository
git clone https://github.com/cesarrennocosta/janones.git
cd janones

# Create environment and sync dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Using standard `pip`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Key Dependencies
- `numpy >= 1.24`
- `scipy >= 1.10`
- `numba >= 0.58` (Parallel JIT execution)
- `pandas >= 2.0`
- `pyarrow >= 14.0`
- `matplotlib >= 3.8`
- `seaborn >= 0.13`
- `jupyter >= 1.0`

---

## 4. Running the Simulations

You can execute the entire simulation suite or target individual figures via the master CLI script `replication/run_simulations.py`.

### Fast Mode (Quick verification in ~2 minutes)
Fast mode runs with downscaled agent populations ($N=5{,}000$ to $10{,}000$) and 5 Monte Carlo realizations, producing equivalent trajectories:
```bash
python replication/run_simulations.py --all --mode fast
```

### Full Mode (Publication grade with full Monte Carlo sweeps)
Full mode reproduces the exact publication-grade figures ($N=30{,}000$ to $100{,}000$, 100 seeds):
```bash
python replication/run_simulations.py --all --mode full
```

### Running Individual Figures
```bash
# Figure 1: Empirical election data and polling timelines
python replication/sim_figure1_empirical_data.py

# Figure 2: Interpersonal multiplex network & two-timescale opinion-conviction dynamics
python replication/sim_figure2_urban_network.py --mode fast

# Figure 3: Traditional mass media gatekeeping and market polarization
python replication/sim_figure3_traditional_media.py

# Figure 4: Social media feeds, controversy diagonal, and dark-social cascades
python replication/sim_figure4_social_media.py

# Figure 5: First-round calibration (RMSE = 0.85%) & 5 Janones mechanism sensitivity
python replication/sim_figure5_first_round.py --mode fast

# Figure 6: Second-round runoff & 6 Janones counter-tactics (RMSE = 0.56%)
python replication/sim_figure6_second_round.py --mode fast
```

Outputs are automatically saved as high-resolution PDFs (Illustrator editable, TrueType fonttype 42) and PNGs (300 DPI) in `figures/`.

---

## 5. Interactive Jupyter Notebook

For an interactive, step-by-step walkthrough of the simulations with live visualizations and narrative commentary:
```bash
jupyter notebook replication/paper_reproduction_figures.ipynb
```
The notebook executes each paper figure sequentially, displaying high-resolution plots side-by-side with theoretical derivations and empirical validation metrics.

---

## 6. Mathematical Parameters & Equations Mapping

| Parameter | Symbol | Paper Equation | Code Variable | Default Value | Empirical / Theoretical Reference |
|---|---|---|---|---|---|
| Issue Dimensions | $D$ | Eq. 1, Eq. S1 | `n_dims` | 4 ($\Omega = [-1, 1]^4$) | Converse (1964) |
| Integration Step | $\Delta t$ | Eq. 1, Eq. S14 | `dt` | 0.15 steps | Friedkin & Johnsen (1990) |
| Susceptibility | $\alpha_i$ | Eq. 1, Eq. S8 | `susceptibility` | $[0.05, 0.98]$ | Alwin & Krosnick (1991) |
| Bounded Confidence | $\epsilon_i$ | Eq. 3, Eq. S9 | `epsilon` | $\bar{\epsilon} = 0.45$ | Deffuant et al. (2000) |
| Biased Assimilation | $b_i$ | Eq. 4, Eq. S11 | `assimilation_bias`| $\bar{b} = 1.25$ | Lord, Ross & Lepper (1979) |
| Fact Pull | $\gamma_0$ | Eq. 6, Eq. S19 | `gamma_0` | $0.055 \text{ step}^{-1}$ | Mullainathan & Shleifer (2005) |
| Opinion Tether Radius | $\Delta_0$ | Eq. 8, Eq. S22 | `delta_0` | $0.34$ | Sherif & Hovland (1961) |
| Conviction Crystallization | $\eta_0$ | Eq. 10, Eq. S25 | `eta_0` | $0.0024 \text{ step}^{-1}$ | Kiley & Vaisey (2020) |
| Crystallization Timescale | $\tau_c$ | Eq. 10, Eq. S25 | `tau_c` | 280 steps | Alwin & Krosnick (1991) |
| Power-Law In-Degree | $\gamma_{\mathrm{pl}}$ | Eq. S40 | `alpha` | $1.15 \pm 0.04$ | Barabási & Albert (1999) |
| Algorithmic Homophily | $\beta_{\mathrm{cur}}$ | Eq. S41 | `beta` | $2.0$ | Brady et al. (2017) |
| Relevance Dislike Weight | $\omega$ | Eq. S46 | `omega` | $1.0$ (Symmetric) | Crockett (2017) |
| WhatsApp Trust Weight | $\tau_{\text{WA}}$ | Eq. S47 | `tau_wa` | $0.85$ ($2.8\times$) | Resende et al. (2019) |
| Affinity Rate | $\lambda_A, \delta_A$ | Eq. 13, Eq. S55 | `lambda_A, delta_A` | $0.12, 0.06$ | Model Calibration |
| Aversion Rate | $\lambda_R, \delta_R$ | Eq. 14, Eq. S56 | `lambda_R, delta_R` | $0.14, 0.04$ | Model Calibration |
| Aversion Exponent | -- | Eq. 14, Eq. S56 | Exponent | $1.35$ | Code congruency |
| Rejection Utility Penalty | $\theta_{\mathrm{rej}}$ | Eq. 15, Eq. S57 | `theta_rej` | $1.65$ | Model Calibration |
| Logit Choice Sensitivity | $\beta_{\mathrm{vote}}$ | Eq. S58 | `beta_vote` | $3.2$ | Train (2009) |
| Free Transit Subsidy | $\delta_{\mathrm{transit}}$ | Eq. S59 | `delta_transit` | $-0.15$ | TSE / Passe Livre decrees |

---

## 7. Citation

If you use this codebase or replication suite in your research, please cite:

```bibtex
@article{rennocosta2026latecampaign,
  title={Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections},
  author={Renn{\'o}-Costa, C{\'e}sar},
  journal={Nature Human Behaviour},
  year={2026},
  publisher={Nature Publishing Group},
  institution={Brain Institute, Federal University of Rio Grande do Norte (UFRN), Natal, RN, Brazil}
}
```

---

## 8. License

This project is licensed under the MIT License. See `LICENSE` for details.
