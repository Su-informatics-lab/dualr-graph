#!/bin/bash -l
#SBATCH --job-name=pure_ce
#SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/pure_ce_%j.out
#SBATCH --error=logs/pure_ce_%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
source .venv/bin/activate
cd ~/dualr-graph
mkdir -p logs results

echo "=== $(date) — Pure CE vs reconstruction test ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# ── Test: kill reconstruction, keep only CE (lambda=100,500) ──
# If reconstruction is the enemy, these should be the best runs yet
for ARCH in transformer gatv2; do
    for LAMBDA in 100.0 500.0; do
        for LR in 1e-3 5e-4 1e-4; do
            NAME="${ARCH}_llm_k8_lam${LAMBDA}_lr${LR}"
            echo ""
            echo ">>> ${NAME}"
            python train.py --arch ${ARCH} --edge_method llm --k 8 \
                --lambda_ce ${LAMBDA} --lr ${LR} \
                --epochs 300 --patience 30 --wandb \
                --run_name "${NAME}"
        done
    done
done

# ── Control: also try on spearman to see if it's edge-specific ──
for ARCH in transformer gatv2; do
    NAME="${ARCH}_spearman_k8_lam100_lr5e-4"
    echo ""
    echo ">>> ${NAME}"
    python train.py --arch ${ARCH} --edge_method spearman --k 8 \
        --lambda_ce 100.0 --lr 5e-4 \
        --epochs 300 --patience 30 --wandb \
        --run_name "${NAME}"
done

echo ""
echo "=== $(date) — Done ==="
