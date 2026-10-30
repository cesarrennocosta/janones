#!/bin/bash
#SBATCH --job-name=td100ka
#SBATCH --output=slurm_tendentious_medium_100k_t1500_amd512_%A_%a.out
#SBATCH --error=slurm_tendentious_medium_100k_t1500_amd512_%A_%a.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0-13
#SBATCH --time=00-23:59
#SBATCH --partition=amd-512
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Media Tendentiousness Sweep (M=1) & Asymmetric Duopoly (M=2) | N = 100,000 | 1500 Steps
# Partition: amd-512 (Array 0..13 = 7 scenarios x 2 blocks of 16 seeds = 32 seeds/scenario)
# ==============================================================================

set -euo pipefail
trap 'echo "[SLURM] Preemption signal received, waiting for child tasks to flush checkpoints..."; wait' USR1 TERM

export NUMBA_NUM_THREADS=8
export OMP_NUM_THREADS=8
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
SCENARIO_ID=$(( TASK_IDX / 2 ))
BLOCK_ID=$(( TASK_IDX % 2 ))
EXTRA_FLAGS=()

case "${SCENARIO_ID}" in
    0)
        SCENARIO_NAME="1g_tend_010"
        M_GROUPS=1
        TEND_ARG="0.10"
        ;;
    1)
        SCENARIO_NAME="1g_tend_035"
        M_GROUPS=1
        TEND_ARG="0.35"
        ;;
    2)
        SCENARIO_NAME="1g_tend_065"
        M_GROUPS=1
        TEND_ARG="0.65"
        ;;
    3)
        SCENARIO_NAME="1g_tend_095"
        M_GROUPS=1
        TEND_ARG="0.95"
        ;;
    4)
        SCENARIO_NAME="2g_sym_low_015_015"
        M_GROUPS=2
        TEND_ARG="0.15,0.15"
        EXTRA_FLAGS=(--symmetric-duopoly)
        ;;
    5)
        SCENARIO_NAME="2g_sym_high_090_090"
        M_GROUPS=2
        TEND_ARG="0.90,0.90"
        EXTRA_FLAGS=(--symmetric-duopoly)
        ;;
    6)
        SCENARIO_NAME="2g_asym_095_010"
        M_GROUPS=2
        TEND_ARG="0.95,0.10"
        EXTRA_FLAGS=(--symmetric-duopoly)
        ;;
    *)
        echo "Invalid SCENARIO_ID=${SCENARIO_ID}"
        exit 1
        ;;
esac

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/global/crcosta/omni_opinion}"
OUTPUT_DIR="${SCRATCH_BASE}/tendentious/${SCENARIO_NAME}"
mkdir -p "${OUTPUT_DIR}"

python3 -c "import src.kernels, src.engines.electoral_engine, src.metrics.polarization, src.metrics.network_metrics"

START_SEED=$(( BLOCK_ID * 16 ))
END_SEED=$(( START_SEED + 15 ))

for SEED_ID in $(seq "${START_SEED}" "${END_SEED}"); do
    python3 scripts/run_single_simulation.py \
        --config configs/traditional_media_cities.yaml \
        --n-agents 100000 \
        --n-steps 1500 \
        --checkpoint-interval 50 \
        --init-mode uniform_random \
        --traditional-media-only \
        --n-vehicle-groups "${M_GROUPS}" \
        --tendentiousness "${TEND_ARG}" \
        "${EXTRA_FLAGS[@]}" \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${SEED_ID}" &
done

wait
