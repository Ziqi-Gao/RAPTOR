#!/bin/bash
#SBATCH --job-name=raptor-layer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/layer_%A_%a.out
#SBATCH --error=logs/layer_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-$PWD}"
CONDA_ENV="${CONDA_ENV:-raptor}"
TASK_LIST="${TASK_LIST:-logs/layer_tasks.tsv}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ROOT}"
mkdir -p logs

line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${TASK_LIST}")"
IFS=$'\t' read -r model dataset layer <<< "${line}"

python scripts/run_layer_task.py \
  --model "${model}" \
  --dataset "${dataset}" \
  --layer "${layer}" \
  --methods "${METHODS:-singlelr,xrfm,gcs}"
