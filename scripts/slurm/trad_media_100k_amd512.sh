#!/bin/bash
#SBATCH --job-name=tm100ka
#SBATCH --output=slurm_trad_media_medium_100k_g1_4_s0_100_amd512_%A_%a.out
#SBATCH --error=slurm_trad_media_medium_100k_g1_4_s0_100_amd512_%A_%a.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0-15
#SBATCH --time=00-23:59
#SBATCH --partition=amd-512
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Traditional Media (TV, Radio, Press) | Medium City (N = 100,000) | M = 1, 2, 3, 4
# Partition: amd-512 (Array 0..15 = 4 group configs x 4 seed blocks)
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
M_GROUPS=$(( (TASK_IDX / 4) + 1 ))
BLOCK_ID=$(( TASK_IDX % 4 ))

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/global/crcosta/omni_opinion}"
OUTPUT_DIR="${SCRATCH_BASE}/trad_media/medium_100k_g${M_GROUPS}"
mkdir -p "${OUTPUT_DIR}"

python3 -c "import src.kernels, src.engines.electoral_engine, src.metrics.polarization, src.metrics.network_metrics"

START_SEED=$(( BLOCK_ID * 32 ))
END_SEED=$(( START_SEED + 31 ))
if [ "${END_SEED}" -gt 100 ]; then
    END_SEED=100
fi

RUNNING=0
for SEED_ID in $(seq "${START_SEED}" "${END_SEED}"); do
    python3 scripts/run_single_simulation.py \
        --config configs/traditional_media_cities.yaml \
        --n-agents 100000 \
        --n-steps 100 \
        --checkpoint-interval 10 \
        --init-mode uniform_random \
        --traditional-media-only \
        --n-vehicle-groups "${M_GROUPS}" \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${SEED_ID}" &
    RUNNING=$(( RUNNING + 1 ))
    if [ "${RUNNING}" -ge 16 ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$(( RUNNING - 1 ))
    fi
done

wait
