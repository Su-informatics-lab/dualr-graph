#!/bin/bash
#SBATCH --job-name=graphai-heatmap
#SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# ═══════════════════════════════════════════════════════════════
# 04_heatmap.sh
# Extract attention weights from trained model → heatmap figures
# Run AFTER training (01 or 02) has saved a checkpoint
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
cd ~/dualr-graph

module purge
module load Python/3.10.8
module load CUDA/12.2.0
source .venv/bin/activate

ARCH=${ARCH:-transformer}
EDGE=${EDGE:-spearman}

echo "=== $(date) — Attention Heatmap ==="
echo "  ARCH=$ARCH  EDGE=$EDGE"

python heatmap.py \
    --arch $ARCH \
    --edge_method $EDGE

echo "=== $(date) — Done ==="
echo "  The heatmap IS the paper's story."
