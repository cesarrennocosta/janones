#!/bin/bash
#SBATCH --job-name=sm100ki
#SBATCH --output=slurm_social_media_medium_100k_t1500_intel128_%A_%a.out
#SBATCH --error=slurm_social_media_medium_100k_t1500_intel128_%A_%a.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0-23
#SBATCH --time=00-23:59
#SBATCH --partition=intel-128
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Multi-Platform Social Media Layer WITH vs WITHOUT Traditional Press
# Reference Scale: Medium City (N = 100,000) | 1500 Steps | 32 Seeds/Scenario
# Most Accurate Brazilian Traditional Press Baseline: M = 3 Tripolar Conglomerates
#   (Group 1: +0.65, tau=0.80 | Group 2: +0.15, tau=0.25 | Group 3: -0.50, tau=0.65)
# Partition: intel-128 (Array 0..23 = 6 scenarios x 4 blocks of 8 seeds = 32 seeds/scenario):
#   Scenario 0: Social Media ONLY (No Press, M=0) — Factual Influencers
#   Scenario 1: Social Media ONLY (No Press, M=0) — Heterogeneous Influencers (Mixed)
#   Scenario 2: Social Media ONLY (No Press, M=0) — Tendentious Influencers
#   Scenario 3: Social Media + Traditional Press (M=3) — Factual Influencers
#   Scenario 4: Social Media + Traditional Press (M=3) — Heterogeneous Influencers (Mixed)
#   Scenario 5: Social Media + Traditional Press (M=3) — Tendentious Influencers
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
        SCENARIO_NAME="sm_nopress_factual"
        INF_TEND="factual"
        MODE_FLAGS=(--social-media-only)
        ;;
    1)
        SCENARIO_NAME="sm_nopress_mixed"
        INF_TEND="heterogeneous"
        MODE_FLAGS=(--social-media-only)
        ;;
    2)
        SCENARIO_NAME="sm_nopress_tendentious"
        INF_TEND="tendentious"
        MODE_FLAGS=(--social-media-only)
        ;;
    3)
        SCENARIO_NAME="sm_withpress_m3_factual"
        INF_TEND="factual"
        MODE_FLAGS=(--full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65")
        ;;
    4)
        SCENARIO_NAME="sm_withpress_m3_mixed"
        INF_TEND="heterogeneous"
        MODE_FLAGS=(--full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65")
        ;;
    5)
        SCENARIO_NAME="sm_withpress_m3_tendentious"
        INF_TEND="tendentious"
        MODE_FLAGS=(--full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65")
        ;;
    *)
        echo "Invalid SCENARIO_ID=${SCENARIO_ID}"
        exit 1
        ;;
esac

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/global/crcosta/omni_opinion}"
OUTPUT_DIR="${SCRATCH_BASE}/social_media/${SCENARIO_NAME}"
mkdir -p "${OUTPUT_DIR}"

python3 -c "import src.kernels, src.engines.electoral_engine, src.metrics.polarization, src.metrics.network_metrics"

START_SEED=$(( BLOCK_ID * 8 ))
END_SEED=$(( START_SEED + 7 ))

for SEED_ID in $(seq "${START_SEED}" "${END_SEED}"); do
    python3 scripts/run_single_simulation.py \
        --config configs/social_media_ecosystem.yaml \
        --n-agents 100000 \
        --n-steps 1500 \
        --checkpoint-interval 50 \
        --init-mode uniform_random \
        --influencer-tendentiousness "${INF_TEND}" \
        "${MODE_FLAGS[@]}" \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${SEED_ID}" &
done

wait
