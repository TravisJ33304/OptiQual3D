#!/bin/bash
#SBATCH --job-name=oq3d-pretrain
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
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
source /usr/local/miniforge/miniforge3/etc/profile.d/conda.sh
conda activate optiqual3d

# Ensure logging directory exists
mkdir -p logs outputs/pretrain

# -- Distributed config -------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

NGPUS=${SLURM_GPUS_ON_NODE:-4}

# -- Launch -------------------------------------------------------------------
srun python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NGPUS}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m scripts.pretrain \
    --config configs/pretrain.yaml \
    --output-dir outputs/pretrain

echo "Finished at $(date)"
