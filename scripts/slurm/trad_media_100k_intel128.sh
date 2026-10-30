#!/bin/bash
#SBATCH --job-name=tm100ki
#SBATCH --output=slurm_trad_media_medium_100k_g0_4_t800_intel128_%A_%a.out
#SBATCH --error=slurm_trad_media_medium_100k_g0_4_t800_intel128_%A_%a.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0-19
#SBATCH --time=00-23:59
#SBATCH --partition=intel-128
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Traditional Media (TV, Radio, Press) | Medium City (N = 100,000) | M = 0, 1, 2, 3, 4
# Horizon: 800 steps (full damped convergence) | 32 seeds per M (Seeds 0..31)
# Partition: intel-128 (Array 0..19 = 5 group configs x 4 blocks of 8 seeds)
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

TASK_IDX="${SLURM_ARRAY_TASK_ID:-0}"
M_GROUPS=$(( TASK_IDX / 4 ))
BLOCK_ID=$(( TASK_IDX % 4 ))

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/global/crcosta/omni_opinion}"
OUTPUT_DIR="${SCRATCH_BASE}/trad_media/medium_100k_g${M_GROUPS}"
mkdir -p "${OUTPUT_DIR}"

python3 -c "import src.kernels, src.engines.electoral_engine, src.metrics.polarization, src.metrics.network_metrics"

START_SEED=$(( BLOCK_ID * 8 ))
END_SEED=$(( START_SEED + 7 ))

for SEED_ID in $(seq "${START_SEED}" "${END_SEED}"); do
    python3 scripts/run_single_simulation.py \
        --config configs/traditional_media_cities.yaml \
        --n-agents 100000 \
        --n-steps 800 \
        --checkpoint-interval 50 \
        --init-mode uniform_random \
        --traditional-media-only \
        --n-vehicle-groups "${M_GROUPS}" \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${SEED_ID}" &
done

wait
