#!/bin/bash
#SBATCH --job-name=graphai-prep
#SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# ═══════════════════════════════════════════════════════════════
# 00_graph_baselines.sh
# Light tasks bundled: build 3 graph types + XGBoost/LogReg/RF
# ~5 min total on 2,600 × 25 features
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
cd ~/dualr-graph

module purge
module load Python/3.10.8
module load CUDA/12.2.0
source .venv/bin/activate

mkdir -p slurm/logs

echo "=== $(date) — Graph construction + Baselines ==="
echo "  Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# ── Build graphs (3 methods, ~seconds each) ───────────────────
for METHOD in spearman mi; do
    echo ""
    echo "--- Graph: $METHOD, k=8 ---"
    python graph.py --method $METHOD --k 8
done

# ── Classical baselines (XGBoost, LogReg, RF on same CV folds) ─
echo ""
echo "--- Baselines ---"
python baselines.py --wandb

echo ""
echo "=== $(date) — Done ==="
