# Antigravity Agent Context & Project Guide (`OmniOpinion`)

This document serves as the comprehensive architectural, mathematical, and operational reference for future Antigravity sessions working in `/Users/rennocosta/janones`.

---

## 1. Project Identity & Repository Setup

- **Framework Name:** `OmniOpinion` (`omni-opinion` in `pyproject.toml`)
- **Workspace Root:** `/Users/rennocosta/janones`
- **Remote Git Repository:** `ssh://git@projetos.imd.ufrn.br:4422/cesarrennocosta/janones.git`
- **Primary Branch:** `main`
- **Python Environment & Auto-Execution Setup (`.venv` & `requirements.txt`):**
  - Always create and use a local `.venv` inside `/Users/rennocosta/janones/.venv` so Antigravity can execute Python inside the workspace sandbox without prompting for user permission:
    ```bash
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install -e .
    ```
  - All project dependencies are tracked in both [`requirements.txt`](file:///Users/rennocosta/janones/requirements.txt) and [`pyproject.toml`](file:///Users/rennocosta/janones/pyproject.toml) for reproducible deployment on GitLab and the NPAD/IMD cluster.
  - Run tests via `.venv/bin/pytest` (or `/Users/rennocosta/miniconda3/bin/pytest` if using the host Conda environment).

### Resolving Remote Git Push Issues (`fetch first` or `remote: GitLab: 401 Unauthorized (pre-receive hook declined)`)
1. **Why `git push --force` triggers `remote: GitLab: 401 Unauthorized (pre-receive hook declined)`:**
   - GitLab protects the `main` branch by default against **force pushes**, or the `projetos.imd.ufrn.br` GitLab server has an internal pre-receive hook / Reverse Proxy discrepancy (or requires the commit `user.email` to match your verified GitLab email).
2. **Step-by-step Fix:**
   - **Check SSH identity & Git email:**
     ```bash
     ssh -p 4422 git@projetos.imd.ufrn.br
     git config user.email   # Ensure this matches your verified email on projetos.imd.ufrn.br
     ```
   - **Integrate remote's initial commit WITHOUT `--force` (avoids protected-branch rejection):**
     ```bash
     git fetch origin main
     git pull origin main --allow-unrelated-histories --no-edit -X ours
     git push -u origin main
     ```
   - **Or push via HTTPS + Personal Access Token (if the server's internal SSH `gitlab-shell` hook returns 401):**
     Create a Personal Access Token (with `write_repository` scope) on `https://projetos.imd.ufrn.br/-/user_settings/personal_access_tokens` and run:
     ```bash
     git remote set-url origin https://projetos.imd.ufrn.br/cesarrennocosta/janones.git
     git push -u origin main
     ```

---

## 2. Mathematical & Physical Models Implemented

### A. Multidimensional Attitude Space & Cognitive Coupling
- $N$ agents hold continuous attitude vectors $\mathbf{x}_i(t) \in [-1, 1]^d$ across $d \ge 3$ latent thematic dimensions (e.g., Economy/State Intervention, Moral/Customs, Environment/Climate, Public Security).
- **Issue Constraint (Cognitive Coupling):** A row-normalized matrix $C \in \mathbb{R}^{d \times d}$ (`agent_state.coupling_matrix`) couples attitude shifts across themes so that changes in one issue induce correlated updates in related dimensions.
- **Friedkin-Johnsen + Relative Agreement + Biased Assimilation:**
  - Each agent has a fixed innate belief anchor $\mathbf{s}_i \in [-1, 1]^d$ (`innate_beliefs`).
  - Basal susceptibility $\alpha_i \in [0, 1]$ (`susceptibility`) is calibrated inversely with age $a_i \in [16, 90]$ (older agents exhibit stronger cognitive inertia).
  - Bounded confidence threshold $\epsilon_i > 0$ (`epsilon`) interacts with dynamic attitudinal uncertainty $u_i(t) \in (0, 1]$ (`uncertainty`).
  - Biased assimilation parameter $b_i \ge 0$ (`assimilation_bias`) exponentially boosts assimilation when interacting with aligned attitudes or confirmatory news framing.

### B. 5-Layer Multiplex Interpersonal Network (`GraphEngine`)
All interpersonal layers are stored as `scipy.sparse.csr_matrix` of shape `(N, N)` and combined into a row-normalized weighted composite CSR matrix (`composite_csr`) for zero-copy Numba traversal:
1. `family`: Small dense cliques, daily contact frequency, highest mutual trust weight (`0.38`).
2. `work_school`: Modular graph structured by age cohort and socioeconomic status ($ses_i \in [0, 1]$) homophily (`0.22`).
3. `church_community`: High-density clusters with strong conformity along the moral dimension (`0.18`).
4. `friends`: Watts-Strogatz small-world network ordered by multidimensional attitudinal projection (`0.15`).
5. `weak_ties`: Sparse random Erdos-Renyi bridges providing global information shortcuts (`0.07`).

### C. Competitive Media Vectors & Coevolution (`MediaEngine` & `NewsEntityPool`)
1. **Traditional Mass Media (Broadcast $1 \to N$):**
   - Channels with editorial orientation $\mathbf{m}_k \in [-1, 1]^d$ and institutional credibility $\gamma_k \in [0, 1]$.
   - Reach probability is modulated by age and socioeconomic status (`broadcast_exposure_kernel_jit`).
2. **Algorithmic Social Media Feeds (Open Scale-Free X/TikTok):**
   - Directed scale-free follower-to-creator graph (`social_feed_csr`). Top eigenvector-centrality nodes are flagged as primary content influencers (`is_influencer`, `influencer_ids`).
   - Closed-loop recommendation (`compute_closed_loop_recommendations_jit`) selects item $j$ for agent $i$ with probability:
     $$q_{ij} \propto (\|\mathbf{x}_i - \mathbf{c}_j\| + \delta)^{-\beta} \, (1 + \kappa_e e_j) \, \ln(1 + \text{pop}_j)$$
   - Bidirectional feedback: agent engagement increments `news_pool.popularity`, and influencers periodically adapt their content framing (`refresh_influencer_content`) to maximize engagement/polarization.
3. **Closed Private Messaging (WhatsApp Groups $K \le 256$):**
   - Stored in flat CSR-like arrays (`wa_group_indptr`, `wa_group_members`, `wa_agent_group_indptr`, `wa_agent_groups`) with strict group size $K \le 256$.
   - No centralized algorithmic curation; high interpersonal trust widens effective confidence bounds, and viral cross-group forwarding cascades (`propagate_whatsapp_cascades_jit`) depend on attitudinal concordance and sensationalism $e_n$.

### D. Electoral Dynamics & District Geography (`ElectoralEngine`)
- Candidates $k \in \{0, \dots, K-1\}$ have profiles $\boldsymbol{\theta}_k \in [-1, 1]^d$.
- Individual utility: $U_{ik} = -\|\mathbf{x}_i - \boldsymbol{\theta}_k\|^2 + \eta_{ik}$ where $\eta_{ik}$ is Gumbel noise.
- If $\max_k U_{ik} < \tau_{\text{indiff}}$, the agent abstains (`vote_choice[i] = -1`, `vote_probs[i, K] = 1.0`). Otherwise, probabilities follow a Multinomial Logit softmax (`compute_multinomial_logit_votes_jit`).
- Agents are assigned to electoral districts (`district`), allowing `evaluate_election` to compute popular vote shares, district seat shares, winner mismatch flags, and the total variation electoral discrepancy index.

---

## 3. Codebase Map & Module Responsibilities (Skills)

| Module / Skill | File Path | Key Responsibilities |
| :--- | :--- | :--- |
| **`AgentStateEngine`** | [`src/core/agent_state.py`](file:///Users/rennocosta/janones/src/core/agent_state.py) | C-contiguous 1D/2D NumPy allocation for demographics, FJ anchors, cognitive coupling matrix $C$, and mutable state. |
| **`GraphEngine`** | [`src/core/multiplex_network.py`](file:///Users/rennocosta/janones/src/core/multiplex_network.py) | 5-layer CSR interpersonal multiplex builder, scale-free feed graph, eigenvector centrality, and WhatsApp group CSR buffers. |
| **`NewsEntityPool`** | [`src/core/news_entity.py`](file:///Users/rennocosta/janones/src/core/news_entity.py) | Contiguous news attributes ($\mathbf{y}_n, e_n, v_n$), broadcast channels, and influencer co-evolution logic. |
| **`DynamicsKernel`** | [`src/kernels/opinion_updates.py`](file:///Users/rennocosta/janones/src/kernels/opinion_updates.py) | `@njit(parallel=True, fastmath=True)` kernels for interpersonal FJ/RA/BA updates and media assimilation. |
| **`RecommendationAlgos`** | [`src/kernels/recommendation_algos.py`](file:///Users/rennocosta/janones/src/kernels/recommendation_algos.py) | `@njit(parallel=True, fastmath=True)` closed-loop $\beta$-biased feed curation and broadcast exposure. |
| **`WhatsAppCascades`** | [`src/kernels/whatsapp_cascades.py`](file:///Users/rennocosta/janones/src/kernels/whatsapp_cascades.py) | `@njit(parallel=True, fastmath=True)` viral cascade diffusion inside closed groups ($K \le 256$). |
| **`MediaEngine`** | [`src/engines/media_engine.py`](file:///Users/rennocosta/janones/src/engines/media_engine.py) | Concurrent orchestration of broadcast, algorithmic feed, and WhatsApp cross-group cascade routing. |
| **`ElectoralEngine`** | [`src/engines/electoral_engine.py`](file:///Users/rennocosta/janones/src/engines/electoral_engine.py) | `@njit` Multinomial Logit vote choice, abstention thresholding, and popular vs. district apuration. |
| **`SimulationEngine`** | [`src/engines/simulation_engine.py`](file:///Users/rennocosta/janones/src/engines/simulation_engine.py) | Master simulation loop, `SeedSequence(base_seed, spawn_key=(task_id,))` derivation, and I/O coordination. |
| **`AnalyticsEngine`** | [`src/metrics/polarization.py`](file:///Users/rennocosta/janones/src/metrics/polarization.py) & [`src/metrics/network_metrics.py`](file:///Users/rennocosta/janones/src/metrics/network_metrics.py) | Multidimensional Esteban-Ray polarization index ($P_{ER}$), Sarle's bimodality coefficient, and CSR ideological segregation. |
| **`HPCStorage`** | [`src/io/hdf5_writer.py`](file:///Users/rennocosta/janones/src/io/hdf5_writer.py) & [`src/io/parquet_writer.py`](file:///Users/rennocosta/janones/src/io/parquet_writer.py) | Buffered HDF5 trajectory/state writer (`gzip`/`lzf`) and Apache Parquet (`pyarrow`) tabular metrics writer. |

---

## 4. HPC, SLURM, and I/O Design Rules

1. **No Small Textual Files:** Never generate per-step or per-agent CSV/JSON files. Use `HDF5TrajectoryWriter` for trajectories/matrices and `ParquetMetricsWriter` for time-series metrics on the BeeGFS parallel filesystem.
2. **Contiguous Memory & Numba Constraints:** All arrays passed into `src/kernels/*.py` must remain 1D or 2D C-contiguous NumPy arrays (`np.ascontiguousarray`) and sparse graphs must pass raw CSR buffers (`indptr`, `indices`, `data`).
3. **Reproducible Parallel Streams:** Always use `np.random.SeedSequence(base_seed, spawn_key=(task_id,))` so each SLURM task (`SLURM_ARRAY_TASK_ID` or MPI/task rank) has statistically independent, deterministic RNG sub-streams.
4. **Mid-Simulation Checkpointing & Preemption Resilience (`preempt_strict`):**
   - Because jobs run under `--qos=preempt_strict`, simulations **can be interrupted/preempted at any time**.
   - Every simulation **must** implement periodic mid-run state saving (checkpointing) to HDF5/Parquet (via `HDF5TrajectoryWriter.flush()`, `ParquetMetricsWriter.flush()`, and periodic `checkpoint_step` state dumps, plus `SIGTERM`/`SIGUSR1` signal trapping in SLURM scripts) and support resuming from the latest saved step if restarted.

---

## 5. Cluster Hardware Inventory & SLURM Script Generation Standards

Whenever the user asks to generate SLURM submission scripts or run experiments on the cluster, **strictly follow these hardware specifications and mandatory rules**:

### A. Mandatory SLURM Generation Rules
1. **Dual Partition Generation (`intel-128` and `amd-512`):**
   - **Always generate two versions** of every SLURM job script (or a paired template): one optimized for `intel-128` (55 nodes available) and one optimized for `amd-512` (56 nodes available).
2. **Compact & Informative `--job-name` (5–7 characters):**
   - The `#SBATCH --job-name` **must be short and informative (strictly 5 to 7 characters)** encoding key parameters (e.g., `b20_t10`, `f07_r1`, `op_b25`), while `--output` and `--error` carry the descriptive full experiment name (e.g., `slurm_omni_b20_n10k_t100.out`).
3. **Mandatory QoS (`preempt_strict`):**
   - Always include `#SBATCH --qos=preempt_strict` in the SLURM header.
4. **Node & Thread Optimization:**
   - Optimize `--nodes`, `--ntasks-per-node`, and `--cpus-per-task` (`NUMBA_NUM_THREADS`) for the target partition's physical core/thread topology so all cores on allocated nodes are saturated without oversubscription.
5. **Preemption-Safe Checkpointing:**
   - Every SLURM run must flush intermediate HDF5/Parquet buffers periodically during execution and check for existing checkpoint files before starting from step 0.

### B. Cluster Hardware Specifications (NPAD / IMD-UFRN Cluster)

#### 1. Primary CPU Partitions (Always target both)
- **`intel-128` (55 nodes) — Intel Xeon E5:**
  - **CPUs:** 2 × Intel Xeon E5-2698v3 (16 physical cores / 32 threads per socket $\implies$ **32 physical cores / 64 hardware threads per node**, 2.3 GHz)
  - **RAM:** 128 GB DDR4-2133 (also `intel-256` with 7 nodes @ 256 GB, `intel-512` with 6 nodes @ 512 GB)
  - **Local Storage:** 120 GB Local SSD
  - **Interconnect:** 2 × InfiniBand ConnectX-3 FDR (56 Gb/s)
- **`amd-512` (56 nodes) — AMD EPYC 7713 (Milan):**
  - **CPUs:** 2 × AMD EPYC 7713 (**128 physical cores / 256 threads per node**)
  - **RAM:** 512 GB DDR4 RDIMM (16 × 32 GB)
  - **Local Storage:** 240 GB Local SSD
  - **Interconnect:** 1 × InfiniBand ConnectX-6 HDR (100 Gb/s)

#### 2. High-Memory & Accelerator Partitions (Reference)
- **`amd-3tb` (7 nodes) — AMD EPYC Genoa:**
  - 2 × AMD EPYC 9654 (96C/192T per socket $\implies$ 192C/384T per node), 3 TB DDR5 RAM, 8 × 3.8 TB NVMe RAID0, ConnectX-6 HDR 100 Gb/s.
- **`gpu-8-h100` (1 node):** 2 × AMD EPYC 9654, 3 TB DDR5, 8 × NVIDIA H100 80 GB SXM5, 8 × 3.8 TB NVMe RAID0.
- **`gpu-4-h200` (2 nodes):** 2 × Intel Xeon Gold 6548N, 1 TB DDR5, 4 × NVIDIA H200 141 GB HBM3e, 5.8 TB NVMe RAID0 (`/scratch/local`).
- **`gpu-4-a100` (1 node):** 2 × AMD EPYC 7713 (128 cores), 512 GB DDR4, 4 × NVIDIA A100 80 GB NVLink.
- **`gpu-8-v100` (2 nodes):** 2 × Intel Xeon E5-2683 v4, 512 GB DDR4, 8 × NVIDIA Tesla V100-SXM2 16 GB.
- **`fpga-3-u50` (1 node):** 2 × AMD EPYC 7713 (128 cores), 512 GB DDR4, 3 × Xilinx Alveo U50.
- **Login Node (`service0`):** 2 × Intel Xeon Silver 4114 (2.20 GHz), 192 GB DDR4.
- **Parallel Storage:** **BeeGFS** parallel filesystem (60 TB total capacity).

---

### C. Canonical SLURM Templates (`intel-128` vs. `amd-512`)

#### Template 1: `intel-128` (64 hardware threads / 32 physical cores per node)
```bash
#!/bin/bash
#SBATCH --job-name=op_b20i
#SBATCH --output=slurm_omni_intel128_b20_n10k_t100.out
#SBATCH --error=slurm_omni_intel128_b20_n10k_t100.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=00-23:59
#SBATCH --partition=intel-128
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# Trap preemption/timeout signal to flush checkpoints cleanly
trap 'echo "[SLURM] Preemption signal received, flushing checkpoints..."; wait' USR1 TERM

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Run 64 parallel task replicas across the 64 hardware threads of the intel-128 node
for TASK_OFFSET in $(seq 0 63); do
    srun --ntasks=1 --nodes=1 --exclusive python3 scripts/run_single_simulation.py \
        --config configs/base_simulation.yaml \
        --output-dir outputs/intel128_b20 \
        --task-id "${TASK_OFFSET}" &
done
wait
```
*(Note: For fewer, larger multi-threaded simulations per node on `intel-128`, use e.g. `--ntasks-per-node=8` and `--cpus-per-task=8` with `export NUMBA_NUM_THREADS=8`).*

#### Template 2: `amd-512` (128 physical cores / 256 threads per node)
```bash
#!/bin/bash
#SBATCH --job-name=op_b20a
#SBATCH --output=slurm_omni_amd512_b20_n10k_t100.out
#SBATCH --error=slurm_omni_amd512_b20_n10k_t100.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=00-23:59
#SBATCH --partition=amd-512
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# Trap preemption/timeout signal to flush checkpoints cleanly
trap 'echo "[SLURM] Preemption signal received, flushing checkpoints..."; wait' USR1 TERM

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Run 128 parallel task replicas across the 128 physical cores of the amd-512 node
for TASK_OFFSET in $(seq 0 127); do
    srun --ntasks=1 --nodes=1 --exclusive python3 scripts/run_single_simulation.py \
        --config configs/base_simulation.yaml \
        --output-dir outputs/amd512_b20 \
        --task-id "${TASK_OFFSET}" &
done
wait
```
*(Note: For multi-node jobs `--nodes=N`, scale the loop across `SLURM_NTASKS` or use `--array` paired with node-packed `srun` steps so all 64 (`intel-128`) or 128 (`amd-512`) slots per node are fully utilized).*

---

## 6. Empirical Demographic Calibration, Reference Metrics & City-Scale Experiments (No-Media Baseline)

### A. Empirical Brazilian Demographic & Urban Calibration (`configs/local_cities_nomedia.yaml`)
All local interpersonal layers and agent age distributions are calibrated against official Brazilian public data:
1. **Age Pyramid (`IBGE_2022_ADULT_AGE_PYRAMID` in `src/core/agent_state.py`):**
   - **Reference:** IBGE, *Censo Demográfico 2022: População por idade e sexo* (Instituto Brasileiro de Geografia e Estatística, 2023).
   - **Parameters:** National median age = $35.0$ years; seniors ($\ge 65$ years) = $10.9\%$ of total population ($13.6\%$ of the $\ge 16$ voting-age population). Adult brackets ($16\text{--}24$: $16.8\%$, $25\text{--}34$: $19.8\%$, $35\text{--}44$: $19.6\%$, $45\text{--}54$: $16.4\%$, $55\text{--}64$: $13.8\%$, $65\text{--}90$: $13.6\%$).
2. **Household / Family Size (`family` layer, weight `0.36`):**
   - **Reference:** IBGE, *Censo Demográfico 2022: Características dos Domicílios* (2023).
   - **Parameters:** Mean household density = **$2.79$ residents per private permanent household** in Brazil (down from $3.31$ in 2010; ranging regionally from $2.54$ in RS to $3.64$ in AM). Modeled as dense household cliques of $2\text{--}5$ members averaging $\sim 2.79$.
3. **Schools & Workplaces (`work_school` layer, weight `0.26`):**
   - **References:**
     - INEP/MEC, *Censo Escolar da Educação Básica 2023* ($47.3\text{M}$ enrollments across $178.5\text{k}$ schools): Mean class size is **$22\text{--}25$ students** in Primary (Fundamental I) and **$25\text{--}30$ students/class** in Lower/Upper Secondary (Fundamental II & Ensino Médio).
     - MTE, *RAIS (Relação Anual de Informações Sociais)* & SEBRAE: Formal Brazilian establishments are predominantly micro and small units ($<19$ employees), creating daily close-contact coworker clusters of **$8\text{--}12$ peers** ($mean\_deg = 10$).
4. **Urban Transportation & Commuting Corridors (`weak_ties` layer, weight `0.10`):**
   - **Reference:** IBGE, *Censo Demográfico 2022: Deslocamento para Trabalho e Estudo* (2024).
   - **Parameters:** $88.4\%$ of workers and $92.7\%$ of students commute within their municipality of residence. Primary work commuting modes: Automobile ($32.3\%$), Public Bus ($21.4\%$), Walking ($17.8\%$), Motorcycle ($16.4\%$), High-capacity Rail/Metro ($1.6\%$), Vans ($1.4\%$) — totaling **$24.4\%$ daily shared public transit usage**, modeled via degree-$4$ transit/urban encounter bridges.

---

### B. Reference Metrics for Polarization, Clustering & Dimensional Correlation (`src/metrics/polarization.py`)
To quantify how opinions evolve from a **purely uniform random initial distribution** (`init_mode: "uniform_random"`):

1. **Multidimensional Polarization ($P_{ER}$):**
   - **Reference:** Esteban, J., & Ray, D. (1994). *"On the Measurement of Polarization."* **Econometrica**, 62(4), 819–851.
   - **Metric:** `esteban_ray` ($P_{ER} = \kappa \sum_i \sum_j \pi_i^{1+\alpha} \pi_j \|\mathbf{y}_i - \mathbf{y}_j\|_2$).
2. **Distance from Uniform Random Baseline (Dispersion Ratio):**
   - **Reference:** Bramson, A., et al. (2016). *"Disambiguation of social polarization concepts and measures."* **The Journal of Mathematical Sociology**, 40(2), 80–111.
   - **Metric:** `dispersion_ratio_vs_random` = $\frac{1}{d}\sum_{m=1}^d \text{Var}(x_{\cdot, m}) / (1/3)$, where $1/3$ is the exact analytical variance of $\mathcal{U}(-1, 1)$. Equals **$1.0$** at $t=0$ (uniform random), approaches **$0.0$** under global consensus, and exceeds **$1.0$** under extreme bimodal polarization.
3. **Bimodality Coefficient ($BC$):**
   - **Reference:** Sarle, W. S. (1983) / Pfister, R., et al. (2013). *"Good things peak in pairs: a note on the bimodality coefficient."* **Frontiers in Psychology**, 4, 700.
   - **Metric:** `sarle_bimodality` ($BC = \frac{\gamma^2 + 1}{\kappa_{\text{kurt}} + 3\frac{(n-1)^2}{(n-2)(n-3)}}$). A uniform random distribution has $BC = 5/9 \approx 0.555$; values $> 0.555$ indicate bimodal/multimodal splitting.
4. **Effective Number of Opinion Clusters ($N_{\text{eff}}$):**
   - **References:**
     - Laakso, M., & Taagepera, R. (1979). *"Effective Number of Parties: A Measure with Application to West Europe."* **Comparative Political Studies**, 12(1), 3–27 (Inverse Participation Ratio / Herfindahl index).
     - Jost, L. (2006). *"Entropy and diversity."* **Oikos**, 113(2), 363–375 (True diversity of order $q=1, 2$).
   - **Metrics:**
     - `effective_clusters_ipr`: $N_{\text{eff}}^{\text{IPR}} = \left(\sum_{c=1}^K \pi_c^2\right)^{-1}$ ($1.0$ = single consensus cluster; $2.0$ = two equal-sized opposite camps; $\sim 4^d$ = uniform random across grid cells).
     - `effective_clusters_entropy`: $N_{\text{eff}}^{H} = \exp\left(-\sum_{c=1}^K \pi_c \ln \pi_c\right)$.
     - `occupied_clusters`: Number of ideological cells containing $\ge 1\%$ of the city population.
5. **Cross-Dimensional Issue Correlation & Dimensionality Collapse:**
   - **References:**
     - Baldassarri, D., & Gelman, A. (2008). *"Partisans without Constraint: Political Polarization and Trends in American Public Opinion."* **American Journal of Sociology**, 114(2), 408–446.
     - Roy, A. D. (1953) / Abbott, L. F., et al. (2011). Participation Ratio of the covariance spectrum.
   - **Metrics:**
     - `mean_issue_correlation` ($\frac{2}{d(d-1)}\sum_{a<b} |\rho_{ab}|$), `max_issue_correlation`, and pairwise `corr_d0_d1`, `corr_d0_d2`, etc. Starts near **$0.0$** under random initialization and rises as cognitive coupling $C$ and local homophily align separate issues.
     - `effective_dimensionality`: $\text{PR}(\Sigma) = \frac{(\text{Tr}\,\Sigma)^2}{\text{Tr}(\Sigma^2)} \in [1, d]$. Starts at **$d$** ($4.0$) for uncorrelated random dimensions and drops toward **$1.0$** when opinions collapse onto a single ideological axis.

---

### C. Multi-Scale City SLURM Scripts (`scripts/slurm/`, Seeds $0\dots 100$)
All scripts run $101$ independent seeds (`0..100`) with `--init-mode uniform_random --no-media`, `--qos=preempt_strict`, and automatic checkpoint recovery:

| City Scale | Population ($N$) | `intel-128` Script (`--job-name`) | `amd-512` Script (`--job-name`) | Node & Thread Packing |
| :--- | :---: | :--- | :--- | :--- |
| **Small City** | $10,000$ | [`scripts/slurm/city_10k_intel128.sh`](file:///Users/rennocosta/janones/scripts/slurm/city_10k_intel128.sh) (`c10k_i`) | [`scripts/slurm/city_10k_amd512.sh`](file:///Users/rennocosta/janones/scripts/slurm/city_10k_amd512.sh) (`c10k_a`) | `intel-128`: 2 nodes $\times$ 64 tasks; `amd-512`: 1 node $\times$ 128 tasks |
| **Medium City** | $100,000$ | [`scripts/slurm/city_100k_intel128.sh`](file:///Users/rennocosta/janones/scripts/slurm/city_100k_intel128.sh) (`c100k_i`) | [`scripts/slurm/city_100k_amd512.sh`](file:///Users/rennocosta/janones/scripts/slurm/city_100k_amd512.sh) (`c100k_a`) | `intel-128`: 16 tasks $\times$ 4 threads (array 0–6); `amd-512`: 32 tasks $\times$ 4 threads (array 0–3) |
| **Large City** | $1,000,000$ | [`scripts/slurm/city_1m_intel128.sh`](file:///Users/rennocosta/janones/scripts/slurm/city_1m_intel128.sh) (`c1m_i`) | [`scripts/slurm/city_1m_amd512.sh`](file:///Users/rennocosta/janones/scripts/slurm/city_1m_amd512.sh) (`c1m_a`) | `intel-128`: 4 tasks $\times$ 16 threads (array 0–25); `amd-512`: 8 tasks $\times$ 16 threads (array 0–12) |
| **Metropolis** | $10,000,000$ | [`scripts/slurm/city_10m_intel128.sh`](file:///Users/rennocosta/janones/scripts/slurm/city_10m_intel128.sh) (`c10m_i`) | [`scripts/slurm/city_10m_amd512.sh`](file:///Users/rennocosta/janones/scripts/slurm/city_10m_amd512.sh) (`c10m_a`) | `intel-128`: 1 task $\times$ 64 threads/node (array 0–100); `amd-512`: 4 tasks $\times$ 32 threads/node (array 0–25) |


