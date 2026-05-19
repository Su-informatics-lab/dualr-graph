#!/bin/bash -l
#SBATCH --job-name=gae_v31
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

# ──────────────────────────────────────────────────────────────
#  v3.1 strategy sweep — 6 runs per node × 3 nodes = 18 total
#
#  Node 0: joint dual × 5 λ + joint+uw dual
#  Node 1: joint pooled (2) + pretrain_finetune (4)
#  Node 2: embed_xgb (2) + conv comparison (3) + joint+uw pooled
#
#  Submit:  sbatch slurm/gae_strat.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail
cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

RUN() {
    local NAME=$1; shift
    echo "[$(date +%H:%M)] START ${NAME}"
    python train.py --wandb --verbose --run_name "$NAME" \
        --edge_method llm --use_edge_weights --layers 2 \
        --hidden 64 --latent 32 --heads 4 \
        --drop_edge_rate 0.1 --epochs 500 --patience 40 \
        --warmup_epochs 20 "$@" \
        > logs/${NAME}.log 2>&1
    echo "[$(date +%H:%M)] DONE  ${NAME}"
}

case $SLURM_ARRAY_TASK_ID in
0)
    echo "═══ Node 0: joint dual × 5λ + uw ═══"
    RUN joint_dual_lam0       --strategy joint --aki_mode dual --lambda_rec 0     &
    RUN joint_dual_lam0.017   --strategy joint --aki_mode dual --lambda_rec 0.017 &
    RUN joint_dual_lam0.1     --strategy joint --aki_mode dual --lambda_rec 0.1   &
    RUN joint_dual_lam0.5     --strategy joint --aki_mode dual --lambda_rec 0.5   &
    RUN joint_dual_lam1.0     --strategy joint --aki_mode dual --lambda_rec 1.0   &
    RUN joint_dual_uw_lam0.5  --strategy joint --aki_mode dual --lambda_rec 0.5 --uw &
    wait
    ;;
1)
    echo "═══ Node 1: pooled + pretrain_finetune ═══"
    RUN joint_pooled_lam0.017  --strategy joint --aki_mode pooled --lambda_rec 0.017 &
    RUN joint_pooled_lam0.5    --strategy joint --aki_mode pooled --lambda_rec 0.5   &
    RUN pf_dual_lam0.5         --strategy pretrain_finetune --aki_mode dual --lambda_rec 0.5         &
    RUN pf_pooled_lam0.5       --strategy pretrain_finetune --aki_mode pooled --lambda_rec 0.5       &
    RUN pf_recon_split_lam0.5  --strategy pretrain_finetune --aki_mode recon_split --lambda_rec 0.5  &
    RUN pf_dual_lam1.0         --strategy pretrain_finetune --aki_mode dual --lambda_rec 1.0         &
    wait
    ;;
2)
    echo "═══ Node 2: embed_xgb + conv comparison + uw_pooled ═══"
    RUN embed_xgb_200ep        --strategy embed_xgb --aki_mode dual --lambda_rec 0.5 --pretrain_epochs 200 &
    RUN embed_xgb_400ep        --strategy embed_xgb --aki_mode dual --lambda_rec 0.5 --pretrain_epochs 400 &
    RUN joint_dual_gat_lam0.017       --strategy joint --aki_mode dual --lambda_rec 0.017 --conv_type gat         &
    RUN joint_dual_gcn_lam0.017       --strategy joint --aki_mode dual --lambda_rec 0.017 --conv_type gcn         &
    RUN joint_dual_transformer_lam0.017 --strategy joint --aki_mode dual --lambda_rec 0.017 --conv_type transformer &
    RUN joint_pooled_uw_lam0.5 --strategy joint --aki_mode pooled --lambda_rec 0.5 --uw &
    wait
    ;;
esac

echo "═══ All done on node $SLURM_ARRAY_TASK_ID $(date) ═══"
