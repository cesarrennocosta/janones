#!/usr/bin/env bash
#SBATCH --job-name=omni_opinion_array
#SBATCH --partition=cpu_hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --array=0-999%100
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

# ==============================================================================
# OmniOpinion SLURM Job Array Submission Script (Up to 1,000 Concurrent Tasks)
# ==============================================================================

set -euo pipefail

# Configure Numba threading to match SLURM allocated CPUs per task
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-local}"
OUTPUT_DIR="${SCRATCH_OUTPUT_DIR:-outputs/slurm_job_${JOB_ID}}"
BASE_SEED="${OMNI_BASE_SEED:-20261029}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "OmniOpinion HPC Simulation Array"
echo "SLURM Job ID       : ${JOB_ID}"
echo "SLURM Array Task ID: ${TASK_ID}"
echo "Numba Threads      : ${NUMBA_NUM_THREADS}"
echo "Output Directory   : ${OUTPUT_DIR}"
echo "============================================================"

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python3 scripts/run_single_simulation.py \
    --config configs/base_simulation.yaml \
    --network-config configs/networks/multiplex_default.yaml \
    --media-config configs/media/platform_profiles.yaml \
    --output-dir "${OUTPUT_DIR}" \
    --base-seed "${BASE_SEED}" \
    --task-id "${TASK_ID}"
