#!/bin/bash
#SBATCH --job-name=graphai-tune
#SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# ═══════════════════════════════════════════════════════════════
# 02_tune.sh
# Optuna hyperparameter search
# PHASE=1: random sampler, 3-fold CV (~2hr)
# PHASE=2: TPE sampler, 5-fold CV (~4hr)
#
# Submit:  PHASE=1 sbatch slurm/02_tune.sh
#          PHASE=2 sbatch slurm/02_tune.sh
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
cd ~/dualr-graph

module purge
module load Python/3.10.8
module load CUDA/12.2.0
source .venv/bin/activate

PHASE=${PHASE:-1}
ARCH=${ARCH:-transformer}
EDGE=${EDGE:-spearman}

echo "=== $(date) — Optuna Phase $PHASE ==="
echo "  ARCH=$ARCH  EDGE=$EDGE  PHASE=$PHASE"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python tune.py \
    --phase $PHASE \
    --arch $ARCH \
    --edge_method $EDGE \
    --wandb

echo "=== $(date) — Done ==="
