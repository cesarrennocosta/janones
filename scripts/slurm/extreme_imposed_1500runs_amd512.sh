#!/bin/bash
#SBATCH --job-name=ex1500a
#SBATCH --output=slurm_extreme_imposed_1500runs_amd512_%A_%a.out
#SBATCH --error=slurm_extreme_imposed_1500runs_amd512_%A_%a.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0-71
#SBATCH --time=00-23:59
#SBATCH --partition=amd-512
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Extreme Imposed Ensemble: 1500 Runs x 1500 Steps per Scenario
# Reference Scale: Medium City (N = 100,000) | 1500 Steps | 1500 Seeds/Scenario (Seeds 0..1499)
# Partition: amd-512 (Array 0..71 = 6 scenarios x 12 blocks of 128 seeds)
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

TASK_IDX="${SLURM_ARRAY_TASK_ID:-0}"
SCENARIO_ID=$(( TASK_IDX / 12 ))
BLOCK_ID=$(( TASK_IDX % 12 ))
MODE_FLAGS=()

case "${SCENARIO_ID}" in
    0)
        SCENARIO_NAME="ext_nopress_baseline"
        MODE_FLAGS=(--social-media-only --extreme-reach-boost 1.0 --extreme-popularity-boost 1.0)
        ;;
    1)
        SCENARIO_NAME="ext_nopress_moderate_boost"
        MODE_FLAGS=(--social-media-only --extreme-imposed --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0)
        ;;
    2)
        SCENARIO_NAME="ext_nopress_high_boost"
        MODE_FLAGS=(--social-media-only --extreme-imposed --extreme-reach-boost 7.0 --extreme-popularity-boost 8.0)
        ;;
    3)
        SCENARIO_NAME="ext_withpress_m3_baseline"
        MODE_FLAGS=(--full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65" --extreme-reach-boost 1.0 --extreme-popularity-boost 1.0)
        ;;
    4)
        SCENARIO_NAME="ext_withpress_m3_moderate_boost"
        MODE_FLAGS=(--full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65" --extreme-imposed --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0)
        ;;
    5)
        SCENARIO_NAME="ext_withpress_m3_high_boost"
        MODE_FLAGS=(--full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65" --extreme-imposed --extreme-reach-boost 7.0 --extreme-popularity-boost 8.0)
        ;;
    *)
        echo "Invalid SCENARIO_ID=${SCENARIO_ID}"
        exit 1
        ;;
esac

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/global/crcosta/omni_opinion}"
OUTPUT_DIR="${SCRATCH_BASE}/extreme_imposed_1500runs/${SCENARIO_NAME}"
mkdir -p "${OUTPUT_DIR}"

python3 -c "import src.kernels, src.engines.electoral_engine, src.metrics.polarization, src.metrics.network_metrics"

START_SEED=$(( (11 - BLOCK_ID) * 128 ))
END_SEED=$(( START_SEED + 127 ))

for SEED_ID in $(seq "${START_SEED}" "${END_SEED}"); do
    if [ "${SEED_ID}" -lt 1500 ]; then
        flock -n "${OUTPUT_DIR}/.lock_seed_${SEED_ID}" \
            python3 scripts/run_single_simulation.py \
                --config configs/extreme_imposed.yaml \
                --n-agents 100000 \
                --n-steps 1500 \
                --checkpoint-interval 50 \
                --init-mode uniform_random \
                "${MODE_FLAGS[@]}" \
                --output-dir "${OUTPUT_DIR}" \
                --task-id "${SEED_ID}" &
    fi
done

wait
