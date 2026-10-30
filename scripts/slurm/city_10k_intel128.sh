#!/bin/bash
#SBATCH --job-name=c10k_i
#SBATCH --output=slurm_city_small_10k_nomedia_s0_100_intel128_%A_%a.out
#SBATCH --error=slurm_city_small_10k_nomedia_s0_100_intel128_%A_%a.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --mem=0
#SBATCH --time=00-23:59
#SBATCH --partition=intel-128
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Small City (N = 10,000) | Local Connections Only (No Media) | Seeds 0..100
# Partition: intel-128 (2 nodes x 64 hardware threads = 128 parallel slots)
# Uses --mem=0 to allocate full 128GB RAM per node without 128MB/task cgroup OOM
# ==============================================================================

set -euo pipefail
trap 'echo "[SLURM] Preemption signal received, waiting for child tasks to flush checkpoints..."; wait' USR1 TERM

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export PATH="$HOME/.local/bin:$PATH"
if [ ! -f ".venv/bin/activate" ] && command -v uv >/dev/null 2>&1; then
    uv sync --frozen
fi
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/global/crcosta/omni_opinion}"
OUTPUT_DIR="${SCRATCH_BASE}/local_nomedia/small_10k"
mkdir -p "${OUTPUT_DIR}"

# Pre-compile Numba JIT cache in a single process before spawning parallel workers
python3 -c "import src.kernels, src.engines.electoral_engine, src.metrics.polarization, src.metrics.network_metrics"

for SEED_ID in $(seq 0 100); do
    srun --ntasks=1 --nodes=1 --cpus-per-task=1 --mem=1800M --exclusive \
        python3 scripts/run_single_simulation.py \
        --config configs/local_cities_nomedia.yaml \
        --n-agents 10000 \
        --n-steps 100 \
        --checkpoint-interval 10 \
        --init-mode uniform_random \
        --no-media \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${SEED_ID}" &
done

wait
