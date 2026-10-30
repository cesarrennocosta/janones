#!/bin/bash
#SBATCH --job-name=c10k_i
#SBATCH --output=slurm_city_small_10k_nomedia_s0_100_intel128_%A_%a.out
#SBATCH --error=slurm_city_small_10k_nomedia_s0_100_intel128_%A_%a.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --time=00-23:59
#SBATCH --partition=intel-128
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Small City (N = 10,000) | Local Connections Only (No Media) | Seeds 0..100
# Partition: intel-128 (2 nodes x 64 hardware threads = 128 parallel slots)
# Runs all 101 seeds (0..100) concurrently across 2 nodes with checkpointing
# ==============================================================================

set -euo pipefail
trap 'echo "[SLURM] Preemption signal received, waiting for child tasks to flush checkpoints..."; wait' USR1 TERM

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

OUTPUT_DIR="outputs/local_nomedia/small_10k"
mkdir -p "${OUTPUT_DIR}"

for SEED_ID in $(seq 0 100); do
    srun --ntasks=1 --nodes=1 --cpus-per-task=1 --exclusive \
        python3 scripts/run_single_simulation.py \
        --config configs/local_cities_nomedia.yaml \
        --n-agents 10000 \
        --n-steps 100 \
        --checkpoint-interval 10 \
        --init-mode uniform_random \
        --no-media \
        --output-dir "${OUTPUT_DIR}" \
        --task-id "${SEED_ID}" &
done

wait
