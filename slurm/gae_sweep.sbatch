#!/bin/bash -l
#SBATCH --job-name=gae_sweep
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-30

# ──────────────────────────────────────────────────────────────
#  Feature-Graph AE overnight sweep  (31 jobs)
#
#  BEFORE FIRST USE — fill in account/partition above:
#    1. Run:  sacctmgr show assoc user=$USER format=Account,Partition --noheader
#    2. Run:  sinfo -s
#    3. Replace ___ACCOUNT___   with:  #SBATCH --account=<your_account>
#       Replace ___PARTITION___ with:  #SBATCH --partition=<your_partition>
#       (or delete both lines to use cluster defaults)
#
#  0-23:  core grid   3 edges × 4 λ × 2 depths  (gatv2)
#  24-27: LLM edge-weighted × 4 λ               (gatv2, depth=3)
#  28-30: conv comparison gcn/gat/transformer    (llm, λ=0.5, depth=3)
#
#  Submit:   sbatch slurm/gae_sweep.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

EDGE_METHODS=("spearman" "mi" "llm")
LAMBDA_RECS=(0 0.1 0.5 1.0)
N_LAYERS=(2 3)
CONV_ALTS=("gcn" "gat" "transformer")

IDX=$SLURM_ARRAY_TASK_ID

if [ $IDX -lt 24 ]; then
    n_lam=${#LAMBDA_RECS[@]}
    n_dep=${#N_LAYERS[@]}
    e_idx=$(( IDX / (n_lam * n_dep) ))
    remainder=$(( IDX % (n_lam * n_dep) ))
    l_idx=$(( remainder / n_dep ))
    d_idx=$(( remainder % n_dep ))
    EDGE=${EDGE_METHODS[$e_idx]}
    LAM=${LAMBDA_RECS[$l_idx]}
    DEPTH=${N_LAYERS[$d_idx]}
    CONV="gatv2"
    NAME="gae_${EDGE}_l${DEPTH}_lam${LAM}"
    EXTRA=""
elif [ $IDX -lt 28 ]; then
    w_idx=$(( IDX - 24 ))
    EDGE="llm"
    LAM=${LAMBDA_RECS[$w_idx]}
    DEPTH=3
    CONV="gatv2"
    NAME="gae_llm_w_l3_lam${LAM}"
    EXTRA="--use_edge_weights"
else
    c_idx=$(( IDX - 28 ))
    EDGE="llm"
    LAM=0.5
    DEPTH=3
    CONV=${CONV_ALTS[$c_idx]}
    NAME="gae_llm_${CONV}_l3_lam0.5"
    EXTRA="--use_edge_weights"
fi

echo "═══════════════════════════════════════════════"
echo "  Job ${IDX}: ${NAME}"
echo "  edge=${EDGE} conv=${CONV} λ=${LAM} L=${DEPTH} ${EXTRA}"
echo "  $(date)"
echo "═══════════════════════════════════════════════"

python train.py \
    --edge_method "$EDGE" \
    --conv_type "$CONV" \
    --k 8 \
    --lambda_rec "$LAM" \
    --layers "$DEPTH" \
    --hidden 64 \
    --latent 32 \
    --heads 4 \
    --epochs 500 \
    --patience 40 \
    --wandb \
    --run_name "$NAME" \
    $EXTRA

echo "Done: ${NAME} $(date)"
