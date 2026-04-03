#!/bin/bash
#SBATCH --job-name=oq3d-train
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ==========================================================================
# OptiQual3D — Phase 2: Anomaly Detection Training on Rosie DGX H100
# ==========================================================================
#
# Submit with:
#   sbatch slurm/train.sbatch [PRETRAINED_CKPT] [RESUME_CKPT|auto] [OUTPUT_DIR]
#
# Uses a Phase 1 checkpoint and optionally resumes Phase 2 from the latest
# anomaly checkpoint. This enables safe chaining under a 24h wall-clock limit.
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

mkdir -p logs outputs/train

# -- Distributed config -------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29501
export NCCL_DEBUG=INFO

NGPUS=${SLURM_GPUS_ON_NODE:-4}

PRETRAINED=${1:-outputs/pretrain/checkpoints/pretrain_latest.pt}
RESUME_ARG=${2:-auto}
OUTPUT_DIR=${3:-outputs/train}

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/checkpoints"

if [[ "${RESUME_ARG}" == "auto" ]]; then
    LATEST_RESUME=$(ls -1t "${OUTPUT_DIR}"/checkpoints/anomaly_*.pt 2>/dev/null | head -n1 || true)
    if [[ -n "${LATEST_RESUME}" ]]; then
        RESUME_ARG="${LATEST_RESUME}"
        echo "Auto-resume checkpoint found: ${RESUME_ARG}"
    else
        RESUME_ARG=""
        echo "No existing Phase 2 checkpoint found; starting fresh from pretrained."
    fi
fi

CMD=(
    srun python -m torch.distributed.run
    --nnodes=1
    --nproc_per_node="${NGPUS}"
    --rdzv_id="${SLURM_JOB_ID}"
    --rdzv_backend=c10d
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
    -m scripts.train
    --config configs/train.yaml
    --pretrained "${PRETRAINED}"
    --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${RESUME_ARG}" ]]; then
    CMD+=(--resume "${RESUME_ARG}")
fi

# -- Launch -------------------------------------------------------------------
"${CMD[@]}"

echo "Finished at $(date)"
