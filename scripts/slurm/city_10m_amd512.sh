#!/bin/bash
#SBATCH --job-name=c10m_a
#SBATCH --output=slurm_city_metropolis_10m_nomedia_s0_100_amd512_%A_%a.out
#SBATCH --error=slurm_city_metropolis_10m_nomedia_s0_100_amd512_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --array=0-25%15
#SBATCH --time=00-23:59
#SBATCH --partition=amd-512
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Metropolis (N = 10,000,000) | Local Connections Only (No Media) | Seeds 0..100
# Partition: amd-512 (512GB RAM via --mem=0, 128 cores -> 4 tasks x 32 cores/node)
# Array 0..25: each node runs 4 seeds concurrently
# ==============================================================================

set -euo pipefail
trap 'echo "[SLURM] Preemption signal received, waiting for child tasks to flush checkpoints..."; wait' USR1 TERM

export NUMBA_NUM_THREADS=32
export OMP_NUM_THREADS=32
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

BLOCK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_SEED=$(( BLOCK_ID * 4 ))
END_SEED=$(( START_SEED + 3 ))
if [ "${END_SEED}" -gt 100 ]; then
    END_SEED=100
fi

for SEED_ID in $(seq "${START_SEED}" "${END_SEED}"); do
    python3 scripts/run_single_simulation.py \
        --config configs/local_cities_nomedia.yaml \
        --n-agents 10000000 \
        --n-steps 100 \
        --checkpoint-interval 2 \
        --init-mode uniform_random \
        --no-media \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${SEED_ID}" &
done

wait
