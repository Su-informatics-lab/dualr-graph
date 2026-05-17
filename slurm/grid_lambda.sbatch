#!/bin/bash -l
#SBATCH --job-name=grid_lambda
#SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/grid_lambda_%j.out
#SBATCH --error=logs/grid_lambda_%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
source .venv/bin/activate
cd ~/dualr-graph
mkdir -p logs results

echo "=== $(date) — Lambda/LR grid search ==="
echo "Goal: rescue transformer_llm_k8 collapse pattern"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# ── Grid: lambda_ce × lr ──
# lambda_ce: reconstruction vs CE balance (higher = more AKI weight)
# lr: slower rates preserve the good LLM-prior initialization
for LAMBDA in 2.0 5.0 10.0 20.0; do
    for LR in 1e-3 5e-4 1e-4; do
        NAME="transformer_llm_k8_lam${LAMBDA}_lr${LR}"
        echo ""
        echo ">>> ${NAME}"
        python train.py --arch transformer --edge_method llm --k 8 \
            --lambda_ce ${LAMBDA} --lr ${LR} \
            --epochs 300 --patience 30 --wandb \
            --run_name "${NAME}"
    done
done

# ── Also sweep the best non-collapsing configs (gatv2) for comparison ──
echo ""
echo "=========================================="
echo "  GATv2 × LLM lambda sweep (control)"
echo "=========================================="
for LAMBDA in 5.0 10.0; do
    for LR in 5e-4 1e-4; do
        NAME="gatv2_llm_k8_lam${LAMBDA}_lr${LR}"
        echo ""
        echo ">>> ${NAME}"
        python train.py --arch gatv2 --edge_method llm --k 8 \
            --lambda_ce ${LAMBDA} --lr ${LR} \
            --epochs 300 --patience 30 --wandb \
            --run_name "${NAME}"
    done
done

echo ""
echo "=== $(date) — Grid search done ==="
