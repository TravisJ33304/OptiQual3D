#!/bin/bash
#SBATCH --job-name=oq3d-smoke
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

# ==========================================================================
# OptiQual3D — End-to-end smoke test on a single T4 GPU
# Runs pretrain → train → evaluate with 32 synthetic samples, 2 batches
# each.  Should complete in under 10 minutes.
# ==========================================================================

set -euo pipefail

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Start time:    $(date)"
echo "=========================================="


mkdir -p logs outputs/smoke_test

python scripts/smoke_test.py \
    --samples    32  \
    --batches    2   \
    --batch-size 4   \
    --device     cuda \
    --output-dir outputs/smoke_test

echo "Finished at $(date)"
