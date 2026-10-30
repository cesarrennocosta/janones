#!/bin/bash
#SBATCH --job-name=td10k_a
#SBATCH --output=slurm_tendentious_small_10k_s0_100_amd512_%A_%a.out
#SBATCH --error=slurm_tendentious_small_10k_s0_100_amd512_%A_%a.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --array=0-6
#SBATCH --time=00-12:00
#SBATCH --partition=amd-512
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Media Tendentiousness Sweep (M=1) & Asymmetric Duopoly (M=2) | N = 10,000
# Partition: amd-512 (7 array tasks x 101 seeds each):
#   0: M=1, tau=0.10 (Low tendentiousness - fact-centered)
#   1: M=1, tau=0.35 (Moderate tendentiousness)
#   2: M=1, tau=0.65 (High tendentiousness)
#   3: M=1, tau=0.95 (Extreme tendentiousness - media-CM-centered)
#   4: M=2, tau=0.15,0.15 (Symmetric low-tendentiousness duopoly)
#   5: M=2, tau=0.90,0.90 (Symmetric high-tendentiousness duopoly)
#   6: M=2, tau=0.95,0.10 (Asymmetric duopoly: Group 1 much more tendentious than Group 2)
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

SCENARIO_ID="${SLURM_ARRAY_TASK_ID:-0}"
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
        --tendentiousness "${TEND_ARG}" \
        "${EXTRA_FLAGS[@]}" \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${SEED_ID}" &
    RUNNING=$(( RUNNING + 1 ))
    if [ "${RUNNING}" -ge 64 ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$(( RUNNING - 1 ))
    fi
done

wait
