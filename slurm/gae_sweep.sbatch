#!/usr/bin/env bash
#SBATCH --job-name=gae_sweep
#SBATCH --account=r00432
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gae_%A_%a.out
#SBATCH --error=logs/gae_%A_%a.err
#SBATCH --array=0-27

# ──────────────────────────────────────────────────────────────
#  Feature-Graph AE overnight sweep (28 jobs)
#
#    0-23:  3 edges × 4 λ_rec × 2 depths  (topology only)
#    24-27: LLM edge-weighted × 4 λ_rec    (depth=3)
#
#  Submit:   sbatch slurm/gae_sweep.sh
#  Monitor:  squeue -u $USER
#  W&B:      https://wandb.ai/<entity>/dualr-graph
# ──────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs results

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate

EDGE_METHODS=("spearman" "mi" "llm")
LAMBDA_RECS=(0 0.1 0.5 1.0)
N_LAYERS=(2 3)

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
    NAME="gae_${EDGE}_l${DEPTH}_lam${LAM}"
    EXTRA=""
else
    w_idx=$(( IDX - 24 ))
    EDGE="llm"
    LAM=${LAMBDA_RECS[$w_idx]}
    DEPTH=3
    NAME="gae_llm_w_l3_lam${LAM}"
    EXTRA="--use_edge_weights"
fi

echo "═══════════════════════════════════════════════"
echo "  Job ${IDX}: ${NAME}"
echo "  edge=${EDGE} λ=${LAM} L=${DEPTH} ${EXTRA}"
echo "═══════════════════════════════════════════════"

python train.py \
    --edge_method "$EDGE" \
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

echo "Done: ${NAME}"
