#!/bin/bash
#SBATCH --job-name=graphai-llm-edges
#SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# ═══════════════════════════════════════════════════════════════
# 03_llm_edges.sh
# LLM pairwise edge estimation: P(AKI | feature_i, feature_j)
# Mirrors DualR's [ESTIMATION] tag protocol
# API-bound, not GPU-bound — GPU partition just for availability
#
# After completion:
#   python 02_graph.py --method llm --k 8
#   EDGE=llm sbatch slurm/01_train_debug.sh
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
cd ~/dualr-graph

module purge
module load Python/3.10.8
module load CUDA/12.2.0
source .venv/bin/activate

echo "=== $(date) — LLM Edge Estimation ==="
echo "  Node: $(hostname)"
echo "  Features: $(python3 -c 'import json; print(len(json.load(open("data/feature_names.json"))))') nodes"
echo "  Pairs: C(n,2) queries"

python llm_edges.py

echo ""
echo "=== $(date) — Done ==="
echo "  Output: data/llm_adj.npy"
echo "  Next:   python 02_graph.py --method llm --k 8"
