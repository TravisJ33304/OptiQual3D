#!/bin/bash
#SBATCH --job-name=oq3d-pretrain
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err

# ==========================================================================
# OptiQual3D — Phase 1: Self-supervised Pre-training on Rosie DGX H100
# ==========================================================================
#
# Submit with:
#   sbatch slurm/pretrain.sbatch
#
# Adjust --partition, --time, and GPU count as needed.
# ==========================================================================

set -euo pipefail

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "GPUs:          ${SLURM_GPUS_ON_NODE:-4}"
echo "Start time:    $(date)"
echo "=========================================="

# -- Environment setup -------------------------------------------------------
export PATH="/home/ad.msoe.edu/jankowskitb/.conda/envs/optiqual3d/bin:$PATH"

# Ensure logging directory exists
mkdir -p logs outputs/pretrain

# -- Distributed config -------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))
export NCCL_DEBUG=INFO

NGPUS=${SLURM_GPUS_ON_NODE:-4}

# -- Auto-resume from latest checkpoint if one exists -----------------------
LATEST_CKPT="outputs/pretrain/checkpoints/pretrain_latest.pt"
RESUME_FLAG=""
if [ -f "${LATEST_CKPT}" ]; then
    RESUME_FLAG="--resume ${LATEST_CKPT}"
    echo "Resuming from ${LATEST_CKPT}"
fi

# -- Launch -------------------------------------------------------------------
srun python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NGPUS}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m scripts.pretrain \
    --config configs/pretrain.yaml \
    --output-dir outputs/pretrain \
    ${RESUME_FLAG} \
    training.pretrain.batch_size=256

echo "Finished at $(date)"
