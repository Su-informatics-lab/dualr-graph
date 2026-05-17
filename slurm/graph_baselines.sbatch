#!/bin/bash -l
#SBATCH --job-name=graphai-prep
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

echo "=== $(date) — Graph construction + Baselines ==="
echo "  Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

for METHOD in spearman mi; do
    echo ""
    echo "--- Graph: $METHOD, k=8 ---"
    python graph.py --method $METHOD --k 8
done

echo ""
echo "--- Baselines ---"
python baselines.py --wandb

echo ""
echo "=== $(date) — Done ==="
