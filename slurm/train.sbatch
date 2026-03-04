#!/bin/bash
#SBATCH --job-name=oq3d-train
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ==========================================================================
# OptiQual3D — Phase 2: Anomaly Detection Training on Rosie DGX H100
# ==========================================================================
#
# Submit with:
#   sbatch slurm/train.sbatch
#
# Requires a Phase 1 checkpoint at outputs/pretrain/best.pt.
# ==========================================================================

set -euo pipefail

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "GPUs:          ${SLURM_GPUS_ON_NODE:-4}"
echo "Start time:    $(date)"
echo "=========================================="

# -- Environment setup -------------------------------------------------------
source /usr/local/miniforge/miniforge3/etc/profile.d/conda.sh
conda activate optiqual3d

mkdir -p logs outputs/train

# -- Distributed config -------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29501
export NCCL_DEBUG=INFO

NGPUS=${SLURM_GPUS_ON_NODE:-4}

PRETRAINED=${1:-outputs/pretrain/best.pt}

# -- Launch -------------------------------------------------------------------
srun python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NGPUS}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m scripts.train \
    --config configs/train.yaml \
    --pretrained "${PRETRAINED}" \
    --output-dir outputs/train

echo "Finished at $(date)"
