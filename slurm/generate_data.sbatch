#!/bin/bash
#SBATCH --job-name=oq3d-gendata
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/generate_data_%j.out
#SBATCH --error=logs/generate_data_%j.err

# ==========================================================================
# OptiQual3D — Synthetic Anomaly Data Generation from ShapeNetCore
# ==========================================================================
#
# Submit with:
#   sbatch slurm/generate_data.sbatch
#
# CPU-only job: spawns many processes via ProcessPoolExecutor to convert
# ShapeNet meshes into point clouds with synthetic anomalies.
#
# Produces ~400 K NPZ files written to datasets/generated/{normal,anomalous}/
# ==========================================================================

set -euo pipefail

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "CPUs:          ${SLURM_CPUS_ON_NODE:-64}"
echo "Start time:    $(date)"
echo "=========================================="

# -- Environment setup -------------------------------------------------------
export PATH="/home/ad.msoe.edu/jankowskitb/.conda/envs/optiqual3d/bin:$PATH"

mkdir -p logs datasets/generated

WORKERS=${SLURM_CPUS_ON_NODE:-32}

# -- Launch -------------------------------------------------------------------
python -m scripts.generate_data \
    --shapenet-root datasets/ShapeNetCore \
    --output-dir    datasets/generated \
    --num-workers   "${WORKERS}" \
    --anomaly-ratio 0.5

echo "Finished at $(date)"
