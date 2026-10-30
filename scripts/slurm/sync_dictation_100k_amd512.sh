#!/bin/bash
#SBATCH --job-name=sy100ka
#SBATCH --partition=amd-512
#SBATCH --qos=preempt_strict
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --requeue
#SBATCH --output=/scratch/global/crcosta/omni_opinion/logs/sy100ka_%A_%a.out
#SBATCH --error=/scratch/global/crcosta/omni_opinion/logs/sy100ka_%A_%a.err

# ==============================================================================
# OmniOpinion SLURM Array Job: Long-Term Synchronization Tactics, Influencer
# Co-Working & Intermedia Fact/Subject Dictation (amd-512)
#
# Scale: N = 100,000 agents (Medium City Reference), 1500 steps, 32 seeds/scenario
# 12 Array Tasks x 16 runs/task = 192 total simulation runs (32 seeds x 6 scenarios)
# ==============================================================================

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

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
SCENARIO_IDX=$(( TASK_ID / 2 ))
CHUNK_IDX=$(( TASK_ID % 2 ))
SEED_START=$(( CHUNK_IDX * 16 ))
SEED_END=$(( SEED_START + 15 ))

SCENARIO="${SCENARIOS[$SCENARIO_IDX]}"
OUT_DIR="${SCRATCH_ROOT}/${SCENARIO}"
mkdir -p "${OUT_DIR}"

echo "[$(date)] Starting sy100ka Task ${TASK_ID}: Scenario=${SCENARIO}, Seeds=${SEED_START}..${SEED_END}"

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

echo "[$(date)] Completed sy100ka Task ${TASK_ID} (${SCENARIO})"
