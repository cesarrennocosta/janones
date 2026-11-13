# OmniOpinion: Sociopolitical Digital Twin & Replication Package

**Accompanying computational replication repository for the research paper:**

> **Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections**  
> **Author:** César Rennó-Costa$^{1,*}$  
> $^{1}$*Brain Institute, Federal University of Rio Grande do Norte (UFRN), Natal, RN, Brazil*  
> $^*$Corresponding author: `rennocosta@neuro.ufrn.br`  
> **Target Journal:** *Nature Human Behaviour* (2026)

---

## Quick Replication Guide

All six figures from the manuscript can be reproduced directly via the dedicated replication scripts in `replication/` or interactively in the Jupyter notebook:

```bash
# 1. Environment Setup
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Fast Replication (Runs all 6 figures in ~20 seconds)
python replication/run_simulations.py --all --mode fast

# 3. Interactive Reproduction Notebook (Renders Figures 1 to 6 inline)
jupyter notebook notebooks/paper_reproduction_figures.ipynb
```

### Individual Replication Scripts (`replication/`)
- `replication/sim_figure1_empirical_data.py`: Figure 1 (TSE ballot outcomes, Datafolha polls, HGPE airtime, campaign finances, Bites interactions)
- `replication/sim_figure2_urban_network.py`: Figure 2 (Urban multiplex contact network & two-timescale opinion-conviction elastic tether)
- `replication/sim_figure3_traditional_media.py`: Figure 3 (Traditional mass media gatekeeping & market polarization)
- `replication/sim_figure4_social_media.py`: Figure 4 (Algorithmic social media feeds, influencer power law & dark-social cascades)
- `replication/sim_figure5_first_round.py`: Figure 5 (First-round calibration: RMSE = 0.85% & 5 Janones mechanisms sensitivity)
- `replication/sim_figure6_second_round.py`: Figure 6 (Second-round runoff: RMSE = 0.56% & 6 Janonismo counter-tactics)

For complete details on parameter mappings, see [`replication/README.md`](replication/README.md).

---

## Overview & Framework

`OmniOpinion` is a stochastic computational statistical physics and complex networks framework developed in Python and compiled with **Numba** (`@njit(parallel=True, fastmath=True)`). It simulates multidimensional attitude evolution, cognitive issue constraints, organic interaction across 5-layer sparse CSR multiplex networks, competitive algorithmic and private messaging media platforms, misinformation cascades, and multinomial logit electoral geography on high-performance computing (HPC) clusters.

---

## 1. Mathematical & Dynamical Formulation

### A. Multidimensional Attitude Space & Cognitive Coupling
Each agent $i \in \{1, \dots, N\}$ holds a continuous attitude vector $\mathbf{x}_i(t) \in [-1, 1]^d$ across $d \ge 3$ latent thematic dimensions (e.g., Economy/State Intervention, Moral/Customs, Environment/Climate, Public Security).
Thematic interdependence (*issue constraint*) is enforced via a cognitive coupling matrix $C \in \mathbb{R}^{d \times d}$, so that an attitudinal shift $\Delta \mathbf{x}_i$ along one dimension induces correlated updates across coupled issues:

$$
\mathbf{x}_i(t + \Delta t) = \Pi_{[-1, 1]^d}\Big( \mathbf{x}_i(t) + \Delta t \big[ (1 - \alpha_i)(\mathbf{s}_i - \mathbf{x}_i(t)) + \alpha_i \, C \, \Phi_i^{\text{social}}(t) \big] + \boldsymbol{\xi}_i(t) \Big)
$$

where:
- $\mathbf{s}_i \in [-1, 1]^d$ is the innate Friedkin-Johnsen psychological anchor;
- $\alpha_i \in [0, 1]$ is the basal susceptibility calibrated inversely with age $a_i \in [16, 90]$ (modeling higher cognitive inertia in older cohorts);
- $\epsilon_i > 0$ is the bounded-confidence tolerance threshold modulated by attitudinal uncertainty $u_i(t) \in (0, 1]$ (*Relative Agreement*);
- $b_i \ge 0$ is the individual biased assimilation coefficient amplifying confirmatory interactions.

### B. 5-Layer Multiplex Interpersonal Network (`scipy.sparse.csr_matrix`)
Organic interpersonal interactions are structured across five sparse CSR layers (`GraphEngine`):
1. **Family (`family`):** Small dense cliques with daily contact frequency and maximum mutual trust.
2. **School / Work (`work_school`):** Modular graphs structured by age-cohort and socioeconomic ($ses_i \in [0, 1]$) homophily.
3. **Church / Communities (`church_community`):** High-density community clusters with strong moral/attitudinal conformity.
4. **Social Circle / Friends (`friends`):** Watts-Strogatz small-world graphs ordered by multidimensional attitudinal homophily.
5. **Weak / Casual Ties (`weak_ties`):** Sparse random long-range connections acting as global information bridges.

### C. Competitive Media Vectors & Coevolution (`MediaEngine`)
1. **Traditional Mass Media (Broadcast $1 \to N$):** Institutional channels with editorial orientation $\mathbf{m}_k \in [-1, 1]^d$ and credibility $\gamma_k \in [0, 1]$, with reach probability modulated by age and socioeconomic status.
2. **Algorithmic Social Media Feeds (Open Scale-Free X/TikTok):** Directed scale-free graphs where high eigenvector-centrality hubs act as primary content influencers. The closed-loop recommendation engine displays content item $j$ to user $i$ with probability:
   $$
   q_{ij} \propto \|\mathbf{x}_i - \mathbf{c}_j\|^{-\beta} \, (1 + \kappa_e e_j) \, \ln(1 + \text{pop}_j)
   $$
   where $\beta \ge 0$ controls algorithmic homophily bias and $e_j \in [0, 1]$ is emotional sensationalism.
3. **Closed Private Messaging (WhatsApp Groups $K \le 256$):** Decentralized group diffusion without algorithmic curation, driven by high interpersonal trust and viral cross-group forwarding cascades dependent on member attitudinal concordance.

### D. Electoral Dynamics & District Geography (`ElectoralEngine`)
At election time $T_{\text{election}}$, each candidate $k \in \{1, \dots, K\}$ has an ideological profile $\boldsymbol{\theta}_k \in [-1, 1]^d$. Agent $i$'s utility is:

$$
U_{ik} = -\|\mathbf{x}_i - \boldsymbol{\theta}_k\|^2 + \eta_{ik}
$$

Choice probabilities $P(v_i = k)$ follow a Multinomial Logit formulation. Agents whose maximum utility $\max_k U_{ik}$ falls below the alienation/indifference threshold opt for **abstention**. Agents are distributed across electoral districts to quantify the discrepancy between direct majoritarian popular vote and district-based representation.

---

## 2. Repository Structure

```text
omni_opinion/
├── ANTIGRAVITY_CONTEXT.md          # Complete architectural, mathematical, & HPC cluster reference
├── configs/
│   ├── base_simulation.yaml
│   ├── local_cities_nomedia.yaml   # Empirical Brazilian interpersonal baseline (N=10k..10M, 100 steps)
│   ├── traditional_media_cities.yaml # Traditional media (TV, Radio, Press) @ N=100k, 1500 steps
│   ├── networks/
│   │   └── multiplex_default.yaml
│   ├── media/
│   │   └── platform_profiles.yaml
│   └── slurm/
│       └── cluster_hpc.yaml
├── notebooks/
│   ├── analyze_local_cities.ipynb      # Multi-scale city baseline analysis (10k, 100k, 1M, 10M)
│   └── analyze_traditional_media.ipynb # Traditional media (M=0..4) & tendentiousness analysis
├── scripts/
│   ├── run_single_simulation.py
│   ├── postprocess_results.py
│   ├── submit_slurm_array.sh
│   └── slurm/
│       ├── city_10k_{intel128,amd512}.sh
│       ├── city_100k_{intel128,amd512}.sh
│       ├── city_1m_{intel128,amd512}.sh
│       ├── city_10m_{intel128,amd512}.sh
│       ├── trad_media_100k_{intel128,amd512}.sh
│       ├── tendentious_100k_{intel128,amd512}.sh
│       └── consolidate_results.sh
├── src/
│   ├── core/
│   │   ├── agent_state.py          # IBGE 2022 demographics, FJ anchors, coupling matrix C
│   │   ├── multiplex_network.py    # 5-layer CSR interpersonal graph + social/WA graphs
│   │   └── news_entity.py          # TraditionalMediaEcosystem (TV, Radio, Press) & NewsEntityPool
│   ├── kernels/
│   │   ├── opinion_updates.py      # Numba @njit(parallel=True, fastmath=True) FJ/RA/BA kernels
│   │   ├── recommendation_algos.py # Traditional media exposure & beta-biased feed curation
│   │   └── whatsapp_cascades.py    # Viral stochastic group cascade kernel
│   ├── engines/
│   │   ├── simulation_engine.py    # Orchestrator with preemption & horizon-extension resume
│   │   ├── media_engine.py
│   │   └── electoral_engine.py
│   ├── metrics/
│   │   ├── polarization.py         # O(N) Esteban-Ray, Sarle BC, IPR/Entropy clusters, PR(Sigma)
│   │   └── network_metrics.py
│   └── io/
│       ├── hdf5_writer.py
│       └── parquet_writer.py
├── tests/
│   ├── test_kernels.py
│   ├── test_multiplex.py
│   ├── test_electoral.py
│   └── test_traditional_media.py
├── pyproject.toml
└── README.md
```

---

## 3. Installation & Testing (Modern `uv` + `pyproject.toml` + `uv.lock`)

This repository uses [**`uv`**](https://docs.astral.sh/uv/) with `.python-version` (`3.11`) and `uv.lock` for deterministic, cross-platform package and Python version management on both macOS and Linux HPC clusters:

```bash
# 1. Install uv once (if not already installed on the host/cluster)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Provision Python 3.11, create .venv, and install all locked packages in seconds
uv sync

# 3. Run the full unit test suite
uv run pytest -v
```

---

## 4. Running Simulations, SLURM Job Arrays, & Analysis Notebooks

### A. Single Simulation Run (Local or Traditional Media)
```bash
# Pure interpersonal baseline (no media)
python3 scripts/run_single_simulation.py \
    --config configs/local_cities_nomedia.yaml \
    --n-agents 100000 \
    --n-steps 100 \
    --init-mode uniform_random \
    --no-media \
    --output-dir outputs/local_100k \
    --task-id 0

# Traditional Media (M=1..4 vehicle groups, 1500 steps, configurable tendentiousness)
python3 scripts/run_single_simulation.py \
    --config configs/traditional_media_cities.yaml \
    --n-agents 100000 \
    --n-steps 1500 \
    --init-mode uniform_random \
    --traditional-media-only \
    --n-vehicle-groups 2 \
    --tendentiousness "0.95,0.10" \
    --symmetric-duopoly \
    --output-dir outputs/trad_2g_asym \
    --task-id 0
```

### B. Submitting SLURM Suites on NPAD (`intel-128` & `amd-512`)
```bash
# Traditional Media (M = 0..4, N = 100k, 1500 steps)
sbatch scripts/slurm/trad_media_100k_intel128.sh
sbatch scripts/slurm/trad_media_100k_amd512.sh

# Media Tendentiousness Sweep & Asymmetric Duopoly (N = 100k, 1500 steps)
sbatch scripts/slurm/tendentious_100k_intel128.sh
sbatch scripts/slurm/tendentious_100k_amd512.sh

# Consolidate all Parquet shards on the cluster
sbatch scripts/slurm/consolidate_results.sh
```

### C. Executing Analysis Notebooks
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/analyze_local_cities.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/analyze_traditional_media.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/paper_reproduction_figures.ipynb
```

---

## 4. Citation & Academic Attribution

If you use this codebase, replication scripts, or digital twin framework in your research, please cite:

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

## 5. Author & Contact
- **Author:** César Rennó-Costa
- **Affiliation:** Brain Institute, Federal University of Rio Grande do Norte (UFRN), Natal, RN, Brazil
- **Email:** `rennocosta@neuro.ufrn.br`
