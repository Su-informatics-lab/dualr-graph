#!/bin/bash -l
#SBATCH --job-name=graphai-heatmap
#SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

set -euo pipefail
cd ~/dualr-graph

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
source .venv/bin/activate

mkdir -p logs

ARCH=${ARCH:-transformer}
EDGE=${EDGE:-spearman}

echo "=== $(date) — Attention Heatmap ==="
echo "  ARCH=$ARCH  EDGE=$EDGE"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python heatmap.py \
    --arch $ARCH \
    --edge_method $EDGE

echo "=== $(date) — Done ==="
