#!/bin/bash -l
#SBATCH --job-name=llm-edges-v6
#SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd ~/dualr-graph

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
source .venv/bin/activate

mkdir -p logs data

echo "=== $(date) — LLM Edge Estimation v6 ==="
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Features: $(python -c 'import json; print(len(json.load(open("data/feature_names.json"))))')"

# Dry run first to verify feature descriptions
echo ""
echo "── Dry run (verify prompts) ──"
python llm_edges.py --dry_run 2>&1 | head -60

# Actual run: 703 pairs × 10 reps
echo ""
echo "── Running LLM queries ──"
python llm_edges.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --n_reps 10 \
    --temperature 0.6 \
    --batch_size 100

# Build graph from LLM adjacency
echo ""
echo "── Building LLM graph ──"
python graph.py --method llm --k 8 --llm_prior data/llm_adj.npy

echo ""
echo "=== $(date) — DONE ==="
