#!/bin/bash
#SBATCH --job-name=jan100ki
#SBATCH --output=slurm_janones_2026_medium_100k_intel128_%A_%a.out
#SBATCH --error=slurm_janones_2026_medium_100k_intel128_%A_%a.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0-7
#SBATCH --time=00-23:59
#SBATCH --partition=intel-128
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# 2026 Brazilian Presidential Election: Multi-Layer ABM Fit & Sensitivity Suite
# Reference Scale: Medium City (N = 100,000) | 100 Seeds/Scenario (Seeds 0..99)
# Partition: intel-128 (Array 0..7 = 2 scenarios x 4 blocks of 25 seeds = 100 seeds/scenario):
#   Scenario 0: round1_diagnosis      -> 50-Day First Round Fit (23/08 Debate Shock) & 5-Mechanism Sensitivity
#   Scenario 1: round2_countertactics -> 21-Day Second Round Fit (Janonismo vs Same Strategy) & 6-Tactic Sensitivity
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

case "${SCENARIO_ID}" in
    0)
        SCENARIO_NAME="round1_diagnosis"
        TARGET_SCRIPT="scripts/generate_first_round_fit_and_sensibility_figure.py"
        BASE_SEED="20261004"
        ;;
    1)
        SCENARIO_NAME="round2_countertactics"
        TARGET_SCRIPT="scripts/generate_second_round_fit_and_sensibility_figure.py"
        BASE_SEED="20261025"
        ;;
    *)
        echo "Invalid SCENARIO_ID=${SCENARIO_ID}"
        exit 1
        ;;
esac

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/global/crcosta/omni_opinion}"
OUTPUT_DIR="${SCRATCH_BASE}/janones_2026_100k/${SCENARIO_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Pre-warm Numba JIT cache once on the node before spawning parallel seed workers
python3 -c "import src.kernels, src.core.agent_state, src.core.multiplex_network, src.kernels.opinion_updates"

START_SEED=$(( BLOCK_ID * 25 ))
END_SEED=$(( START_SEED + 24 ))

RUNNING=0
for SEED_ID in $(seq "${START_SEED}" "${END_SEED}"); do
    flock -n "${OUTPUT_DIR}/.lock_seed_${SEED_ID}" \
        python3 "${TARGET_SCRIPT}" \
            --n-agents 100000 \
            --base-seed "${BASE_SEED}" \
            --output-dir "${OUTPUT_DIR}" \
            --task-id "${SEED_ID}" &
    RUNNING=$(( RUNNING + 1 ))
    if [ "${RUNNING}" -ge 8 ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$(( RUNNING - 1 ))
    fi
done

wait

# Automatically aggregate into multi-seed ensemble JSON + figures once all 100 seeds complete
COMPLETED_SEEDS=$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'seed_*.json' | wc -l | tr -d ' ')
if [ "${COMPLETED_SEEDS}" -ge 100 ]; then
    flock -n "${OUTPUT_DIR}/.lock_aggregate" \
        python3 "${TARGET_SCRIPT}" --aggregate-dir "${OUTPUT_DIR}" --n-agents 100000 || true
fi
