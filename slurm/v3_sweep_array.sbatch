#!/bin/bash
#SBATCH -A r01109
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=v10_%a
#SBATCH --output=slurm/v10_%a_%j.out
#SBATCH --error=slurm/v10_%a_%j.err
#SBATCH --array=0-35

# ── v3 sweep (array version) ──
# 36 configs, each runs as a separate job.
# ~20 min per job (5-fold CV).
#
# Usage:
#   mkdir -p slurm results
#   sbatch slurm/v3_sweep_array.sh

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p results

# Sweep grid: 3 × 2 × 3 × 2 = 36
LAMBDA_REC_LIST=(0.5 1.0 2.0)
LAMBDA_AKI_LIST=(1.0 2.0)
LAMBDA_ASSOC_LIST=(0.0 0.5 1.0)
DROPOUT_LIST=(0.0 0.1)

IDX=${SLURM_ARRAY_TASK_ID}

# Decode linear index into grid coordinates.
n_aki=${#LAMBDA_AKI_LIST[@]}
n_assoc=${#LAMBDA_ASSOC_LIST[@]}
n_do=${#DROPOUT_LIST[@]}

i_rec=$((   IDX / (n_aki * n_assoc * n_do) ))
remainder=$((IDX % (n_aki * n_assoc * n_do) ))
i_aki=$((   remainder / (n_assoc * n_do) ))
remainder=$((remainder % (n_assoc * n_do) ))
i_assoc=$(( remainder / n_do ))
i_do=$((    remainder % n_do ))

LR=${LAMBDA_REC_LIST[$i_rec]}
LA=${LAMBDA_AKI_LIST[$i_aki]}
LAS=${LAMBDA_ASSOC_LIST[$i_assoc]}
DO=${DROPOUT_LIST[$i_do]}

NAME="v10_rec${LR}_aki${LA}_assoc${LAS}_do${DO}"

echo "===== Job ${IDX}: ${NAME} at $(date) ====="
echo "lambda_rec=${LR} lambda_aki=${LA} lambda_assoc=${LAS} dropout=${DO}"

python train_v3.py \
    --edge_method llm \
    --use_edge_weights \
    --k 8 \
    --layers 2 \
    --hidden 64 \
    --latent 16 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --epochs 500 \
    --patience 40 \
    --eval_every 5 \
    --grad_clip 5.0 \
    --cv_folds 5 \
    --seed 42 \
    --verbose \
    --wandb \
    --lambda_rec ${LR} \
    --lambda_aki ${LA} \
    --lambda_assoc ${LAS} \
    --dropout ${DO} \
    --run_name ${NAME}

echo "===== ${NAME} finished at $(date) ====="
