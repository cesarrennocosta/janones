#!/bin/bash
#SBATCH --job-name=c100k_i
#SBATCH --output=slurm_city_medium_100k_nomedia_s0_100_intel128_%A_%a.out
#SBATCH --error=slurm_city_medium_100k_nomedia_s0_100_intel128_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --array=0-6
#SBATCH --time=00-23:59
#SBATCH --partition=intel-128
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Medium City (N = 100,000) | Local Connections Only (No Media) | Seeds 0..100
# Partition: intel-128 (64 hardware threads / node -> 16 tasks x 4 Numba threads)
# Array 0..6 (7 nodes total): each array task runs 16 seeds concurrently
# ==============================================================================

set -euo pipefail
trap 'echo "[SLURM] Preemption signal received, waiting for child tasks to flush checkpoints..."; wait' USR1 TERM

export NUMBA_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/global/crcosta/omni_opinion}"
OUTPUT_DIR="${SCRATCH_BASE}/local_nomedia/medium_100k"
mkdir -p "${OUTPUT_DIR}"

BLOCK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_SEED=$(( BLOCK_ID * 16 ))
END_SEED=$(( START_SEED + 15 ))
if [ "${END_SEED}" -gt 100 ]; then
    END_SEED=100
fi

for SEED_ID in $(seq "${START_SEED}" "${END_SEED}"); do
    srun --ntasks=1 --nodes=1 --cpus-per-task=4 --exclusive \
        python3 scripts/run_single_simulation.py \
        --config configs/local_cities_nomedia.yaml \
        --n-agents 100000 \
        --n-steps 100 \
        --checkpoint-interval 10 \
        --init-mode uniform_random \
        --no-media \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${SEED_ID}" &
done

wait
