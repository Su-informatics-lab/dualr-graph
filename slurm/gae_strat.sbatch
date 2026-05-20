#!/bin/bash -l
#SBATCH --job-name=gae_v4.1
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-30

# ──────────────────────────────────────────────────────────────
#  v4 sweep — association decoder architecture  (31 jobs)
#
#  Core grid (0-23): 3 edges × 4 λ_assoc × 2 depths  [gatv2, pooled]
#  Conv ablation (24-26): gcn / gat / graph_transformer  [llm, λ=1.0, L=2]
#  AKI mode (27-28): dual × 2 λ_assoc  [llm, gatv2, L=2]
#  Latent dim (29-30): 16 / 64  [llm, gatv2, L=2, λ=1.0]
#
#  Submit:  sbatch slurm/gae_v4_sweep.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

EDGE_METHODS=("spearman" "mi" "llm")
LAMBDA_ASSOCS=(0.1 0.5 1.0 2.0)
N_LAYERS=(2 3)
CONV_ALTS=("gcn" "gat" "graph_transformer")
LATENT_ALTS=(16 64)

IDX=$SLURM_ARRAY_TASK_ID

if [ $IDX -lt 24 ]; then
    # ── Core grid: 3 edges × 4 λ × 2 depths ──
    n_lam=${#LAMBDA_ASSOCS[@]}
    n_dep=${#N_LAYERS[@]}
    e_idx=$(( IDX / (n_lam * n_dep) ))
    remainder=$(( IDX % (n_lam * n_dep) ))
    l_idx=$(( remainder / n_dep ))
    d_idx=$(( remainder % n_dep ))

    EDGE=${EDGE_METHODS[$e_idx]}
    LAM=${LAMBDA_ASSOCS[$l_idx]}
    DEPTH=${N_LAYERS[$d_idx]}
    CONV="gatv2"
    AKI_MODE="pooled"
    LATENT=32
    NAME="v4_${EDGE}_l${DEPTH}_la${LAM}"
    EXTRA=""
    if [ "$EDGE" = "llm" ]; then EXTRA="--use_edge_weights"; fi

elif [ $IDX -lt 27 ]; then
    # ── Conv ablation: gcn / gat / graph_transformer ──
    c_idx=$(( IDX - 24 ))
    EDGE="llm"
    LAM=1.0
    DEPTH=2
    CONV=${CONV_ALTS[$c_idx]}
    AKI_MODE="pooled"
    LATENT=32
    NAME="v4.1_llm_${CONV}_l2_la1.0"
    EXTRA="--use_edge_weights"

elif [ $IDX -lt 29 ]; then
    # ── AKI mode ablation: dual ──
    d_idx=$(( IDX - 27 ))
    LAMS=(0.5 1.0)
    EDGE="llm"
    LAM=${LAMS[$d_idx]}
    DEPTH=2
    CONV="gatv2"
    AKI_MODE="dual"
    LATENT=32
    NAME="v4_llm_dual_l2_la${LAM}"
    EXTRA="--use_edge_weights"

else
    # ── Latent dim ablation ──
    lat_idx=$(( IDX - 29 ))
    EDGE="llm"
    LAM=1.0
    DEPTH=2
    CONV="gatv2"
    AKI_MODE="pooled"
    LATENT=${LATENT_ALTS[$lat_idx]}
    NAME="v4.1_llm_lat${LATENT}_l2_la1.0"
    EXTRA="--use_edge_weights"
fi

echo "═══════════════════════════════════════════════"
echo "  Job ${IDX}: ${NAME}"
echo "  edge=${EDGE} conv=${CONV} λ_assoc=${LAM} L=${DEPTH}"
echo "  aki_mode=${AKI_MODE} latent=${LATENT} ${EXTRA}"
echo "  $(date)"
echo "═══════════════════════════════════════════════"

python train.py \
    --edge_method "$EDGE" \
    --encoder_type "$CONV" \
    --k 8 \
    --lambda_assoc "$LAM" \
    --lambda_aki 1.0 \
    --layers "$DEPTH" \
    --hidden 64 \
    --latent "$LATENT" \
    --heads 4 \
    --dropout 0.1 \
    --aki_mode "$AKI_MODE" \
    --batch_size 128 \
    --epochs 500 \
    --patience 40 \
    --eval_every 5 \
    --verbose \
    --wandb \
    --run_name "$NAME" \
    $EXTRA

echo "Done: ${NAME} $(date)"
