#!/bin/bash
#SBATCH --job-name=oq3d-eval
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ==========================================================================
# OptiQual3D — Evaluation on MVTec 3D-AD / Real3D-AD
# ==========================================================================
#
# Submit with:
#   sbatch slurm/evaluate.sbatch
#
# Single-GPU job (inference only).
# ==========================================================================

set -euo pipefail

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "GPU:           ${SLURM_GPUS_ON_NODE:-1}"
echo "Start time:    $(date)"
echo "=========================================="

# -- Environment setup -------------------------------------------------------
export PATH="/home/ad.msoe.edu/jankowskitb/.conda/envs/optiqual3d/bin:$PATH"

mkdir -p logs outputs/eval

CHECKPOINT=${1:-outputs/train/checkpoints/anomaly_best.pt}
DATASET=${2:-both}

# -- Launch -------------------------------------------------------------------
python -m scripts.evaluate \
    --config configs/evaluate.yaml \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    --output-dir outputs/eval

echo "Finished at $(date)"
