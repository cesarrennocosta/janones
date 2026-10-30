#!/bin/bash
#SBATCH --job-name=tm10k_a
#SBATCH --output=slurm_trad_media_small_10k_g1_4_s0_100_amd512_%A_%a.out
#SBATCH --error=slurm_trad_media_small_10k_g1_4_s0_100_amd512_%A_%a.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=1-4
#SBATCH --time=00-12:00
#SBATCH --partition=amd-512
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Traditional Media (TV, Radio, Press) | Small City (N = 10,000) | M = 1, 2, 3, 4
# Partition: amd-512 (1 node per vehicle-group count M=1..4, 101 seeds each)
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

M_GROUPS="${SLURM_ARRAY_TASK_ID:-1}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/global/crcosta/omni_opinion}"
OUTPUT_DIR="${SCRATCH_BASE}/trad_media/small_10k_g${M_GROUPS}"
mkdir -p "${OUTPUT_DIR}"

python3 -c "import src.kernels, src.engines.electoral_engine, src.metrics.polarization, src.metrics.network_metrics"

RUNNING=0
for SEED_ID in $(seq 0 100); do
    python3 scripts/run_single_simulation.py \
        --config configs/traditional_media_cities.yaml \
        --n-agents 10000 \
        --n-steps 100 \
        --checkpoint-interval 10 \
        --init-mode uniform_random \
        --traditional-media-only \
        --n-vehicle-groups "${M_GROUPS}" \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${SEED_ID}" &
    RUNNING=$(( RUNNING + 1 ))
    if [ "${RUNNING}" -ge 64 ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$(( RUNNING - 1 ))
    fi
done

wait
