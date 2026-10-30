#!/bin/bash
#SBATCH --job-name=c1m_a
#SBATCH --output=slurm_city_large_1m_nomedia_s0_100_amd512_%A_%a.out
#SBATCH --error=slurm_city_large_1m_nomedia_s0_100_amd512_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --array=0-12
#SBATCH --time=00-23:59
#SBATCH --partition=amd-512
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Large City (N = 1,000,000) | Local Connections Only (No Media) | Seeds 0..100
# Partition: amd-512 (512GB RAM via --mem=0, 128 cores -> 8 tasks x 16 threads)
# Array 0..12 (13 nodes): each node runs 8 seeds concurrently
# ==============================================================================

set -euo pipefail
trap 'echo "[SLURM] Preemption signal received, waiting for child tasks to flush checkpoints..."; wait' USR1 TERM

export NUMBA_NUM_THREADS=16
export OMP_NUM_THREADS=16
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
OUTPUT_DIR="${SCRATCH_BASE}/local_nomedia/large_1m"
mkdir -p "${OUTPUT_DIR}"

BLOCK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_SEED=$(( BLOCK_ID * 8 ))
END_SEED=$(( START_SEED + 7 ))
if [ "${END_SEED}" -gt 100 ]; then
    END_SEED=100
fi

for SEED_ID in $(seq "${START_SEED}" "${END_SEED}"); do
    python3 scripts/run_single_simulation.py \
        --config configs/local_cities_nomedia.yaml \
        --n-agents 1000000 \
        --n-steps 100 \
        --checkpoint-interval 5 \
        --init-mode uniform_random \
        --no-media \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${SEED_ID}" &
done

wait
