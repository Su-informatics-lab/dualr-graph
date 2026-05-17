#!/bin/bash -l
#SBATCH --job-name=gnn_grid
#SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/gnn_grid_%j.out
#SBATCH --error=logs/gnn_grid_%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
source .venv/bin/activate
cd ~/dualr-graph
mkdir -p logs results

echo "=== $(date) — GNN grid sweep ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# ── Build MI graph if not already done ──
if [ ! -f data/graph_mi_k8.pt ]; then
    echo ""
    echo ">>> Building MI graph (k=8)..."
    python graph.py --method mi --k 8
else
    echo ">>> MI graph already exists, skipping."
fi

# ── Spearman × 3 architectures ──
echo ""
echo "=========================================="
echo "  SPEARMAN edges"
echo "=========================================="
for ARCH in transformer gatv2 gcn; do
    echo ""
    echo ">>> ${ARCH} × spearman"
    python train.py --arch ${ARCH} --edge_method spearman --k 8 \
        --epochs 300 --patience 30 --wandb
done

# ── MI × 3 architectures ──
echo ""
echo "=========================================="
echo "  MI edges"
echo "=========================================="
for ARCH in transformer gatv2 gcn; do
    echo ""
    echo ">>> ${ARCH} × mi"
    python train.py --arch ${ARCH} --edge_method mi --k 8 \
        --epochs 300 --patience 30 --wandb
done

# ── LLM edges (conditional — may not be ready yet) ──
echo ""
echo "=========================================="
echo "  LLM edges (conditional)"
echo "=========================================="
if [ -f data/llm_adj.npy ]; then
    if [ ! -f data/graph_llm_k8.pt ]; then
        echo ">>> Building LLM graph (k=8)..."
        python graph.py --method llm --k 8 --llm_prior data/llm_adj.npy
    fi
    for ARCH in transformer gatv2 gcn; do
        echo ""
        echo ">>> ${ARCH} × llm"
        python train.py --arch ${ARCH} --edge_method llm --k 8 \
            --epochs 300 --patience 30 --wandb
    done
else
    echo ">>> llm_adj.npy not found — skipping LLM runs."
    echo ">>> Re-run this block manually after LLM edges complete:"
    echo ">>>   python graph.py --method llm --k 8 --llm_prior data/llm_adj.npy"
    echo ">>>   python train.py --arch transformer --edge_method llm --k 8 --wandb"
    echo ">>>   python train.py --arch gatv2 --edge_method llm --k 8 --wandb"
    echo ">>>   python train.py --arch gcn --edge_method llm --k 8 --wandb"
fi

echo ""
echo "=== $(date) — Grid sweep done ==="
