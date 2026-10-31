#!/bin/bash
#SBATCH --job-name=sy100ki
#SBATCH --partition=intel-128
#SBATCH --qos=preempt_strict
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --array=0-23
#SBATCH --requeue
#SBATCH --output=/scratch/global/crcosta/omni_opinion/logs/sy100ki_%A_%a.out
#SBATCH --error=/scratch/global/crcosta/omni_opinion/logs/sy100ki_%A_%a.err

# ==============================================================================
# OmniOpinion SLURM Array Job: Long-Term Synchronization Tactics, Influencer
# Co-Working & Intermedia Fact/Subject Dictation (intel-128)
#
# Scale: N = 100,000 agents (Medium City Reference), 1500 steps, 100 seeds/scenario
# 6 Scenarios (Incremental Ablation of Long-Term Synchronization & Dictation):
#   0: unsync_nopress        -> M=0, Unsynchronized Extreme Imposed (3.5x Reach / 4x Pop)
#   1: sync_cowork_nopress   -> M=0, Synchronized Facts + Influencer Co-Working
#   2: unsync_withpress_m3   -> M=3, Unsynchronized Extreme Imposed + Tripolar Press
#   3: sync_only_withpress   -> M=3, Synchronized Facts Only + Tripolar Press
#   4: sync_cowork_withpress -> M=3, Synchronized Facts + Co-Working + Tripolar Press
#   5: full_sync_cowork_dict -> M=3, Synchronized + Co-Working + Fact/Subject Dictation to Press
# 24 Array Tasks x 25 runs/task = 600 total simulation runs (100 seeds x 6 scenarios)
# ==============================================================================

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/omni_opinion}"
SCRATCH_ROOT="/scratch/global/crcosta/omni_opinion/sync_dictation_100k"
mkdir -p "${SCRATCH_ROOT}" "/scratch/global/crcosta/omni_opinion/logs"

cd "${PROJECT_DIR}"

SCENARIOS=(
  "unsync_nopress"
  "sync_cowork_nopress"
  "unsync_withpress_m3"
  "sync_only_withpress"
  "sync_cowork_withpress"
  "full_sync_cowork_dict"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SCENARIO_IDX=$(( TASK_ID / 4 ))
CHUNK_IDX=$(( TASK_ID % 4 ))
SEED_START=$(( CHUNK_IDX * 25 ))
SEED_END=$(( SEED_START + 24 ))

SCENARIO="${SCENARIOS[$SCENARIO_IDX]}"
OUT_DIR="${SCRATCH_ROOT}/${SCENARIO}"
mkdir -p "${OUT_DIR}"

echo "[$(date)] Starting sy100ki Task ${TASK_ID}: Scenario=${SCENARIO}, Seeds=${SEED_START}..${SEED_END}"

for SEED in $(seq ${SEED_START} ${SEED_END}); do
  case "${SCENARIO}" in
    unsync_nopress)
      uv run python scripts/run_single_simulation.py \
        --config configs/extreme_imposed.yaml \
        --n-agents 100000 --n-steps 1500 \
        --social-media-only --extreme-imposed \
        --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0 \
        --task-id "${SEED}" --output-dir "${OUT_DIR}"
      ;;
    sync_cowork_nopress)
      uv run python scripts/run_single_simulation.py \
        --config configs/extreme_imposed.yaml \
        --n-agents 100000 --n-steps 1500 \
        --social-media-only --extreme-imposed \
        --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0 \
        --sync-tactics --cowork \
        --task-id "${SEED}" --output-dir "${OUT_DIR}"
      ;;
    unsync_withpress_m3)
      uv run python scripts/run_single_simulation.py \
        --config configs/extreme_imposed.yaml \
        --n-agents 100000 --n-steps 1500 \
        --full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65" \
        --extreme-imposed --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0 \
        --task-id "${SEED}" --output-dir "${OUT_DIR}"
      ;;
    sync_only_withpress)
      uv run python scripts/run_single_simulation.py \
        --config configs/extreme_imposed.yaml \
        --n-agents 100000 --n-steps 1500 \
        --full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65" \
        --extreme-imposed --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0 \
        --sync-tactics \
        --task-id "${SEED}" --output-dir "${OUT_DIR}"
      ;;
    sync_cowork_withpress)
      uv run python scripts/run_single_simulation.py \
        --config configs/extreme_imposed.yaml \
        --n-agents 100000 --n-steps 1500 \
        --full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65" \
        --extreme-imposed --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0 \
        --sync-tactics --cowork \
        --task-id "${SEED}" --output-dir "${OUT_DIR}"
      ;;
    full_sync_cowork_dict)
      uv run python scripts/run_single_simulation.py \
        --config configs/extreme_imposed.yaml \
        --n-agents 100000 --n-steps 1500 \
        --full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65" \
        --extreme-imposed --extreme-reach-boost 3.5 --extreme-popularity-boost 4.0 \
        --sync-tactics --cowork --fact-dictation \
        --task-id "${SEED}" --output-dir "${OUT_DIR}"
      ;;
  esac
done

echo "[$(date)] Completed sy100ki Task ${TASK_ID} (${SCENARIO})"
