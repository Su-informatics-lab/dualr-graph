#!/bin/bash -l
#SBATCH --job-name=graphai-llm-edges
#SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

echo "=== $(date) — LLM Edge Estimation ==="
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python llm_edges.py

echo ""
echo "=== $(date) — Done ==="
echo "  Output: data/llm_adj.npy"
echo "  Next:   python graph.py --method llm --k 8"
