#!/bin/bash
#SBATCH --job-name=cv100ki
#SBATCH --output=slurm_conviction_dynamics_medium_100k_t1500_intel128_%A_%a.out
#SBATCH --error=slurm_conviction_dynamics_medium_100k_t1500_intel128_%A_%a.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0-23
#SBATCH --time=00-23:59
#SBATCH --partition=intel-128
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Two-Timescale Conviction vs. Current Opinion Dynamics & Fact-Bias Shifts
# Reference Scale: Medium City (N = 100,000) | 1500 Steps | 100 Seeds/Scenario
# Partition: intel-128 (Array 0..23 = 6 scenarios x 4 blocks of 25 seeds = 100 seeds/scenario):
#   0: conv_uniform_neutral     -> Uniform Convictions | Neutral Facts (Stabilization & Cohort Dynamics)
#   1: conv_centrist_neutral    -> Centrist Convictions | Neutral Facts (High Swing & Electoral Mobility)
#   2: conv_extremist_neutral   -> Extremist Convictions | Neutral Facts (Rigid Anchoring & Low Swing)
#   3: conv_uniform_factshift   -> Uniform Convictions | Strong Fact-Bias Shift (0 -> +0.78 -> -0.82)
#   4: conv_centrist_factshift  -> Centrist Convictions | Strong Fact-Bias Shift (0 -> +0.78 -> -0.82)
#   5: conv_extremist_factshift -> Extremist Convictions | Strong Fact-Bias Shift (0 -> +0.78 -> -0.82)
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
        SCENARIO_NAME="conv_uniform_neutral"
        MODE_FLAGS=(--conviction-dynamics --init-mode uniform_random --fact-bias-mode neutral --full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65")
        ;;
    1)
        SCENARIO_NAME="conv_centrist_neutral"
        MODE_FLAGS=(--conviction-dynamics --init-mode centrist_convictions --fact-bias-mode neutral --full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65")
        ;;
    2)
        SCENARIO_NAME="conv_extremist_neutral"
        MODE_FLAGS=(--conviction-dynamics --init-mode extremist_convictions --fact-bias-mode neutral --full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65")
        ;;
    3)
        SCENARIO_NAME="conv_uniform_factshift"
        MODE_FLAGS=(--conviction-dynamics --init-mode uniform_random --fact-bias-mode regime_shift --full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65")
        ;;
    4)
        SCENARIO_NAME="conv_centrist_factshift"
        MODE_FLAGS=(--conviction-dynamics --init-mode centrist_convictions --fact-bias-mode regime_shift --full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65")
        ;;
    5)
        SCENARIO_NAME="conv_extremist_factshift"
        MODE_FLAGS=(--conviction-dynamics --init-mode extremist_convictions --fact-bias-mode regime_shift --full-media-ecosystem --n-vehicle-groups 3 --tendentiousness "0.80,0.25,0.65")
        ;;
    *)
        echo "Invalid SCENARIO_ID=${SCENARIO_ID}"
        exit 1
        ;;
esac

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/global/crcosta/omni_opinion}"
OUTPUT_DIR="${SCRATCH_BASE}/conviction_dynamics_100k/${SCENARIO_NAME}"
mkdir -p "${OUTPUT_DIR}"

python3 -c "import src.kernels, src.engines.electoral_engine, src.metrics.polarization, src.metrics.network_metrics"

START_SEED=$(( BLOCK_ID * 25 ))
END_SEED=$(( START_SEED + 24 ))

for SEED_ID in $(seq "${START_SEED}" "${END_SEED}"); do
    flock -n "${OUTPUT_DIR}/.lock_seed_${SEED_ID}" \
        python3 scripts/run_single_simulation.py \
            --config configs/conviction_opinion_dynamics.yaml \
            --n-agents 100000 \
            --n-steps 1500 \
            --checkpoint-interval 50 \
            "${MODE_FLAGS[@]}" \
            --output-dir "${OUTPUT_DIR}" \
            --task-id "${SEED_ID}" &
done

wait
