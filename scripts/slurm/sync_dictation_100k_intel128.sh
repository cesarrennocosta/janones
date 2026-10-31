#!/bin/bash
#SBATCH --job-name=sy100ki
#SBATCH --output=slurm_sync_dictation_medium_100k_t1500_intel128_%A_%a.out
#SBATCH --error=slurm_sync_dictation_medium_100k_t1500_intel128_%A_%a.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0-23
#SBATCH --time=00-23:59
#SBATCH --partition=intel-128
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Long-Term Synchronization Tactics, Influencer Co-Working & Fact Dictation
# Reference Scale: Medium City (N = 100,000) | 1500 Steps | 100 Seeds/Scenario
# Partition: intel-128 (Array 0..23 = 6 scenarios x 4 blocks of 25 seeds = 100 seeds/scenario):
#   0: unsync_nopress        -> M=0, Unsynchronized Extreme Imposed (3.5x Reach / 4x Pop)
#   1: sync_cowork_nopress   -> M=0, Synchronized Facts + Influencer Co-Working
#   2: unsync_withpress_m3   -> M=3, Unsynchronized Extreme Imposed + Tripolar Press
#   3: sync_only_withpress   -> M=3, Synchronized Facts Only + Tripolar Press
#   4: sync_cowork_withpress -> M=3, Synchronized Facts + Co-Working + Tripolar Press
#   5: full_sync_cowork_dict -> M=3, Synchronized + Co-Working + Fact/Subject Dictation to Press
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
SCENARIO_ID=$(( TASK_IDX / 4 ))
BLOCK_ID=$(( TASK_IDX % 4 ))
MODE_FLAGS=()

case "${SCENARIO_ID}" in
    0)
        SCENARIO_NAME="unsync_nopress"
        MODE_FLAGS=(--social-media-only --extreme-imposed --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0)
        ;;
    1)
        SCENARIO_NAME="sync_cowork_nopress"
        MODE_FLAGS=(--social-media-only --extreme-imposed --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0 --sync-tactics --cowork)
        ;;
    2)
        SCENARIO_NAME="unsync_withpress_m3"
        MODE_FLAGS=(--full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65" --extreme-imposed --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0)
        ;;
    3)
        SCENARIO_NAME="sync_only_withpress"
        MODE_FLAGS=(--full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65" --extreme-imposed --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0 --sync-tactics)
        ;;
    4)
        SCENARIO_NAME="sync_cowork_withpress"
        MODE_FLAGS=(--full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65" --extreme-imposed --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0 --sync-tactics --cowork)
        ;;
    5)
        SCENARIO_NAME="full_sync_cowork_dict"
        MODE_FLAGS=(--full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65" --extreme-imposed --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0 --sync-tactics --cowork --fact-dictation)
        ;;
    *)
        echo "Invalid SCENARIO_ID=${SCENARIO_ID}"
        exit 1
        ;;
esac

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/global/crcosta/omni_opinion}"
OUTPUT_DIR="${SCRATCH_BASE}/sync_dictation_100k/${SCENARIO_NAME}"
mkdir -p "${OUTPUT_DIR}"

python3 -c "import src.kernels, src.engines.electoral_engine, src.metrics.polarization, src.metrics.network_metrics"

START_SEED=$(( BLOCK_ID * 25 ))
END_SEED=$(( START_SEED + 24 ))

for SEED_ID in $(seq "${START_SEED}" "${END_SEED}"); do
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
done

wait
