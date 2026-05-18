#!/bin/bash -l
#SBATCH --job-name=gae_strat
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-14

# ──────────────────────────────────────────────────────────────
#  Strategy sweep  (15 jobs)
#
#  Based on deep research 2026-05-19:
#    - GRAPE, GraphMAE, T2G-Former: separate head, not recon
#    - Pretrain→finetune: highest expected impact
#    - Embed→XGBoost: highest evidence ceiling
#
#  0-3:   joint       × 4 aki_modes  (dual, pooled, recon_split, recon)
#  4-5:   joint+uw    × 2 aki_modes  (dual, pooled)
#  6-9:   pretrain_finetune × 4 aki_modes
#  10-11: embed_xgb   × 2 pretrain configs (200ep, 400ep)
#  12-14: joint_dual  × 3 λ_rec values (0, 0.1, 1.0)
#
#  All on LLM edge-weighted, gatv2, L=2
#  Submit:  sbatch slurm/gae_strat.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail
cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

IDX=$SLURM_ARRAY_TASK_ID

case $IDX in
    0) STRAT=joint; MODE=dual;        LAM=0.5; EXTRA="" ;;
    1) STRAT=joint; MODE=pooled;      LAM=0.5; EXTRA="" ;;
    2) STRAT=joint; MODE=recon_split; LAM=0.5; EXTRA="" ;;
    3) STRAT=joint; MODE=recon;       LAM=0.5; EXTRA="" ;;
    4) STRAT=joint; MODE=dual;        LAM=0.5; EXTRA="--uw" ;;
    5) STRAT=joint; MODE=pooled;      LAM=0.5; EXTRA="--uw" ;;
    6) STRAT=pretrain_finetune; MODE=dual;        LAM=0.5; EXTRA="" ;;
    7) STRAT=pretrain_finetune; MODE=pooled;      LAM=0.5; EXTRA="" ;;
    8) STRAT=pretrain_finetune; MODE=recon_split; LAM=0.5; EXTRA="" ;;
    9) STRAT=pretrain_finetune; MODE=dual;        LAM=1.0; EXTRA="" ;;
   10) STRAT=embed_xgb; MODE=dual; LAM=0.5; EXTRA="--pretrain_epochs 200" ;;
   11) STRAT=embed_xgb; MODE=dual; LAM=0.5; EXTRA="--pretrain_epochs 400" ;;
   12) STRAT=joint; MODE=dual; LAM=0;   EXTRA="" ;;
   13) STRAT=joint; MODE=dual; LAM=0.1; EXTRA="" ;;
   14) STRAT=joint; MODE=dual; LAM=1.0; EXTRA="" ;;
esac

NAME="${STRAT}_${MODE}_lam${LAM}"
[ -n "$EXTRA" ] && NAME="${NAME}_$(echo $EXTRA | tr ' ' '_' | tr -d '-')"

echo "═══════════════════════════════════════════════"
echo "  Job ${IDX}: ${NAME}"
echo "  strategy=${STRAT} aki_mode=${MODE} λ=${LAM} ${EXTRA}"
echo "  $(date)"
echo "═══════════════════════════════════════════════"

python train.py \
    --strategy "$STRAT" \
    --aki_mode "$MODE" \
    --edge_method llm \
    --use_edge_weights \
    --conv_type gatv2 \
    --k 8 \
    --lambda_rec "$LAM" \
    --layers 2 \
    --hidden 64 \
    --latent 32 \
    --heads 4 \
    --drop_edge_rate 0.1 \
    --epochs 500 \
    --patience 40 \
    --warmup_epochs 20 \
    --wandb \
    --verbose \
    --run_name "$NAME" \
    $EXTRA

echo "Done: ${NAME} $(date)"
