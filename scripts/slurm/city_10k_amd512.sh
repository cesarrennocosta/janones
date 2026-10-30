#!/bin/bash
#SBATCH --job-name=c10k_a
#SBATCH --output=slurm_city_small_10k_nomedia_s0_100_amd512_%j.out
#SBATCH --error=slurm_city_small_10k_nomedia_s0_100_amd512_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=0
#SBATCH --time=00-23:59
#SBATCH --partition=amd-512
#SBATCH --qos=preempt_strict
#SBATCH --signal=B:USR1@120

# ==============================================================================
# Small City (N = 10,000) | Local Connections Only (No Media) | Seeds 0..100
# Partition: amd-512 (1 node x 128 physical cores, 512GB RAM via --mem=0)
# Runs all 101 seeds (0..100) simultaneously on a single AMD EPYC 7713 node
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

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/global/crcosta/omni_opinion}"
OUTPUT_DIR="${SCRATCH_BASE}/local_nomedia/small_10k"
mkdir -p "${OUTPUT_DIR}"

# Pre-compile Numba JIT cache once before spawning 101 parallel seeds
python3 -c "import src.kernels, src.engines.electoral_engine, src.metrics.polarization, src.metrics.network_metrics"

for SEED_ID in $(seq 0 100); do
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
