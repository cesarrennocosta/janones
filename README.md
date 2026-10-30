# OmniOpinion

**High-Performance Multidimensional Opinion Dynamics, Multiplex Media Ecosystems, and Electoral Projection Framework for SLURM HPC Clusters**

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
├── configs/
│   ├── base_simulation.yaml
│   ├── networks/
│   │   └── multiplex_default.yaml
│   ├── media/
│   │   └── platform_profiles.yaml
│   └── slurm/
│       └── cluster_hpc.yaml
├── scripts/
│   ├── run_single_simulation.py
│   ├── submit_slurm_array.sh
│   └── postprocess_results.py
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent_state.py
│   │   ├── multiplex_network.py
│   │   └── news_entity.py
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── opinion_updates.py      # Numba @njit(parallel=True, fastmath=True)
│   │   ├── recommendation_algos.py # Closed-loop beta-biased recommendation
│   │   └── whatsapp_cascades.py    # Viral stochastic group cascade kernel
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── simulation_engine.py
│   │   ├── media_engine.py
│   │   └── electoral_engine.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── polarization.py         # Multidimensional Esteban-Ray & Sarle bimodality
│   │   └── network_metrics.py
│   └── io/
│       ├── __init__.py
│       ├── hdf5_writer.py
│       └── parquet_writer.py
├── tests/
│   ├── test_kernels.py
│   ├── test_multiplex.py
│   └── test_electoral.py
├── pyproject.toml
└── README.md
```

---

## 3. Installation & Testing (Modern `uv` + `pyproject.toml` + `uv.lock`)

This repository uses [**`uv`**](https://docs.astral.sh/uv/) with `.python-version` (`3.11`) and `uv.lock` for deterministic, cross-platform package and Python version management on both macOS and Linux HPC clusters (even when the login node's system Python is older):

```bash
# 1. Install uv once (if not already installed on the host/cluster)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Provision Python 3.11, create .venv, and install all locked packages in seconds
uv sync

# 3. Run the test suite
uv run pytest
```

---

## 4. Running Simulations & SLURM Job Arrays

### Single Simulation Run
```bash
python3 scripts/run_single_simulation.py \
    --config configs/base_simulation.yaml \
    --network-config configs/networks/multiplex_default.yaml \
    --media-config configs/media/platform_profiles.yaml \
    --output-dir outputs/local_run \
    --task-id 0
```

### Submitting 1,000 Tasks on SLURM
```bash
sbatch scripts/submit_slurm_array.sh
```

### Consolidating Parquet Metric Shards
```bash
python3 scripts/postprocess_results.py \
    --input-dir outputs/slurm_job_12345 \
    --output-file outputs/consolidated_metrics.parquet
```
