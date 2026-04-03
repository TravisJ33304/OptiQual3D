#!/bin/bash
# Submit one or more Phase 2 training jobs under 24h wall-clock constraints.
#
# Modes:
#   sequential - chain jobs with afterany dependency, auto-resume same output dir.
#   concurrent - submit independent jobs immediately, each with isolated output dir.
#
# Usage:
#   bash slurm/submit_train_jobs.sh [mode] [num_jobs] [pretrained_ckpt]
#
# Examples:
#   bash slurm/submit_train_jobs.sh sequential 4 outputs/pretrain/checkpoints/pretrain_latest.pt
#   bash slurm/submit_train_jobs.sh concurrent 2 outputs/pretrain/checkpoints/pretrain_latest.pt

set -euo pipefail

MODE=${1:-sequential}
NUM_JOBS=${2:-2}
PRETRAINED=${3:-outputs/pretrain/checkpoints/pretrain_latest.pt}

if [[ "${MODE}" != "sequential" && "${MODE}" != "concurrent" ]]; then
    echo "Invalid mode: ${MODE}. Use 'sequential' or 'concurrent'." >&2
    exit 1
fi

if ! [[ "${NUM_JOBS}" =~ ^[0-9]+$ ]] || [[ "${NUM_JOBS}" -lt 1 ]]; then
    echo "NUM_JOBS must be a positive integer." >&2
    exit 1
fi

echo "Submitting ${NUM_JOBS} train job(s) in ${MODE} mode"

declare -a JOB_IDS=()

if [[ "${MODE}" == "sequential" ]]; then
    DEP=""
    for i in $(seq 1 "${NUM_JOBS}"); do
        if [[ -n "${DEP}" ]]; then
            OUT=$(sbatch --parsable --dependency=afterany:${DEP} \
                slurm/train.sbatch "${PRETRAINED}" auto outputs/train)
        else
            OUT=$(sbatch --parsable slurm/train.sbatch "${PRETRAINED}" auto outputs/train)
        fi
        JOB_ID=${OUT%%;*}
        JOB_IDS+=("${JOB_ID}")
        DEP="${JOB_ID}"
        echo "  submitted sequential job ${i}/${NUM_JOBS}: ${JOB_ID}"
    done
else
    for i in $(seq 1 "${NUM_JOBS}"); do
        OUT_DIR="outputs/train_run${i}"
        OUT=$(sbatch --parsable slurm/train.sbatch "${PRETRAINED}" "" "${OUT_DIR}")
        JOB_ID=${OUT%%;*}
        JOB_IDS+=("${JOB_ID}")
        echo "  submitted concurrent job ${i}/${NUM_JOBS}: ${JOB_ID} -> ${OUT_DIR}"
    done
fi

echo "Job IDs: ${JOB_IDS[*]}"
