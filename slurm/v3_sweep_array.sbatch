#!/bin/bash
#SBATCH -A r01109
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=v10.1_%a
#SBATCH --output=logs/v10.1_%a_%j.out
#SBATCH --error=logs/v10.1_%a_%j.err
#SBATCH --array=0-31

# ── v10.1 sweep: simplified MSE + CE, 3 hops, iterative masking ──
# 2×2×2×2×2 = 32 configs

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p results logs

LAMBDA_REC=(1.0 2.0)
LAMBDA_AKI=(1.0 2.0)
LAMBDA_ASSOC=(0.0 1.0)
LAYERS=(2 3)
NITER=(1 2)

IDX=${SLURM_ARRAY_TASK_ID}
i0=$(( IDX / 16 ));          r0=$(( IDX % 16 ))
i1=$(( r0  / 8  ));          r1=$(( r0  % 8  ))
i2=$(( r1  / 4  ));          r2=$(( r1  % 4  ))
i3=$(( r2  / 2  ));          i4=$(( r2  % 2  ))

LR=${LAMBDA_REC[$i0]}
LA=${LAMBDA_AKI[$i1]}
LAS=${LAMBDA_ASSOC[$i2]}
LY=${LAYERS[$i3]}
NI=${NITER[$i4]}

NAME="v10.1_L${LY}_it${NI}_rec${LR}_aki${LA}_assoc${LAS}"

echo "===== Job ${IDX}: ${NAME} at $(date) ====="

python train_v3.py \
    --edge_method llm \
    --use_edge_weights \
    --k 8 \
    --layers ${LY} \
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
    --n_iter ${NI} \
    --lambda_rec ${LR} \
    --lambda_aki ${LA} \
    --lambda_assoc ${LAS} \
    --dropout 0.0 \
    --run_name ${NAME}

echo "===== ${NAME} finished at $(date) ====="
