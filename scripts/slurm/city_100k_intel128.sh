#!/bin/bash
#SBATCH --job-name=c100k_i
#SBATCH --output=slurm_city_medium_100k_nomedia_s0_100_intel128_%A_%a.out
#SBATCH --error=slurm_city_medium_100k_nomedia_s0_100_intel128_%A_%a.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0-6
#SBATCH --time=00-23:59
#SBATCH --partition=intel-128
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Medium City (N = 100,000) | Local Connections Only (No Media) | Seeds 0..100
# Partition: intel-128 (128GB RAM via --mem=0, full node via --exclusive)
# Array 0..6 (7 nodes total out of 38 idle): each node runs 16 seeds (8 at a time)
# ==============================================================================

set -euo pipefail
trap 'echo "[SLURM] Preemption signal received, waiting for child tasks to flush checkpoints..."; wait' USR1 TERM

export NUMBA_NUM_THREADS=4
export OMP_NUM_THREADS=4
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
OUTPUT_DIR="${SCRATCH_BASE}/local_nomedia/medium_100k"
mkdir -p "${OUTPUT_DIR}"

python3 -c "import src.kernels, src.engines.electoral_engine, src.metrics.polarization, src.metrics.network_metrics"

BLOCK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_SEED=$(( BLOCK_ID * 16 ))
END_SEED=$(( START_SEED + 15 ))
if [ "${END_SEED}" -gt 100 ]; then
    END_SEED=100
fi

RUNNING=0
for SEED_ID in $(seq "${START_SEED}" "${END_SEED}"); do
    python3 scripts/run_single_simulation.py \
        --config configs/local_cities_nomedia.yaml \
        --n-agents 100000 \
        --n-steps 100 \
        --checkpoint-interval 10 \
        --init-mode uniform_random \
        --no-media \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${SEED_ID}" &
    RUNNING=$(( RUNNING + 1 ))
    if [ "${RUNNING}" -ge 8 ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$(( RUNNING - 1 ))
    fi
done

wait
