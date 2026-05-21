#!/bin/bash
#SBATCH -A r01109
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=v10.2_%a
#SBATCH --output=logs/v10.2_%a_%j.out
#SBATCH --error=logs/v10.2_%a_%j.err
#SBATCH --array=0-23

# ── v10.2: contextual decoder (mean/max/mean_max pool) + GAT ──
# 3×2×2×2 = 24 configs. lambda_rec=1, lambda_aki=2, assoc=0 (v10.1 winners).

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p results logs

POOL=(mean max mean_max)
ENC=(directed gatv2)
LAYERS=(2 3)
DROP=(0.0 0.05)

IDX=${SLURM_ARRAY_TASK_ID}
i0=$(( IDX / 8 ));  r0=$(( IDX % 8 ))
i1=$(( r0  / 4 ));  r1=$(( r0  % 4 ))
i2=$(( r1  / 2 ));  i3=$(( r1  % 2 ))

PL=${POOL[$i0]}
EN=${ENC[$i1]}
LY=${LAYERS[$i2]}
DO=${DROP[$i3]}

NAME="v10.2_${EN}_L${LY}_${PL}_do${DO}"
echo "===== Job ${IDX}: ${NAME} at $(date) ====="

python train_v3.py \
    --edge_method llm \
    --use_edge_weights \
    --k 8 \
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
    --n_iter 1 \
    --lambda_rec 1.0 \
    --lambda_aki 2.0 \
    --lambda_assoc 0.0 \
    --encoder_type ${EN} \
    --pool_type ${PL} \
    --layers ${LY} \
    --dropout ${DO} \
    --run_name ${NAME}

echo "===== ${NAME} finished at $(date) ====="
