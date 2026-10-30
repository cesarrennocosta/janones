#!/bin/bash
#SBATCH --job-name=c10m_i
#SBATCH --output=slurm_city_metropolis_10m_nomedia_s0_100_intel128_%A_%a.out
#SBATCH --error=slurm_city_metropolis_10m_nomedia_s0_100_intel128_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --array=0-100%25
#SBATCH --time=00-23:59
#SBATCH --partition=intel-128
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Metropolis (N = 10,000,000) | Local Connections Only (No Media) | Seeds 0..100
# Partition: intel-128 (128GB RAM, 64 threads -> 1 full-node task x 64 threads)
# Array 0..100%25: each seed uses 1 dedicated node (~85-100GB peak RAM)
# ==============================================================================

set -euo pipefail
trap 'echo "[SLURM] Preemption signal received, waiting for simulation to flush checkpoint..."; wait' USR1 TERM

export NUMBA_NUM_THREADS=64
export OMP_NUM_THREADS=64
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
OUTPUT_DIR="${SCRATCH_BASE}/local_nomedia/metropolis_10m"
mkdir -p "${OUTPUT_DIR}"

SEED_ID="${SLURM_ARRAY_TASK_ID:-0}"

srun --ntasks=1 --nodes=1 --cpus-per-task=64 --exclusive \
    python3 scripts/run_single_simulation.py \
    --config configs/local_cities_nomedia.yaml \
    --n-agents 10000000 \
    --n-steps 100 \
    --checkpoint-interval 2 \
    --init-mode uniform_random \
    --no-media \
    --output-dir "${OUTPUT_DIR}" \
    --task-id "${SEED_ID}" &

wait
