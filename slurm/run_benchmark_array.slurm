#!/bin/bash
#SBATCH --job-name=raptor-benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-$PWD}"
CONDA_ENV="${CONDA_ENV:-raptor}"
MODEL_PATH="${MODEL_PATH:-.}"
CUDA_ID="${CUDA_ID:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ROOT}"
mkdir -p logs

python scripts/run_experiments.py \
  --model_path "${MODEL_PATH}" \
  --cuda "${CUDA_ID}" \
  --quant "${QUANT:-32}" \
  --methods "${METHODS:-raptor,xrfm,gcs}" \
  --skip_embeddings
