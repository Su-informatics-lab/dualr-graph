#!/bin/bash -l
#SBATCH --job-name=gae_v43
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-31

# ──────────────────────────────────────────────────────────────
#  v4.3 sweep — bridge v4.1 winners with mild regularisation
#
#  v4.2 lesson: over-regularised. lr=3e-4 + dropout≥0.15 never
#  reaches v4.1's peak. The u-shape was handled by early stopping.
#  Correct approach: keep lr=1e-3, explore MILD regularisation.
#
#  Block A (0-23): lr × dropout × latent around v4.1 sweet spot
#    lr: [7e-4, 1e-3]  ×  dropout: [0.0, 0.05, 0.1, 0.15]  ×  latent: [12, 16, 24]
#    Fixed: GCN, LLM, L=2, hidden=64, λ_assoc=1.0
#
#  Block B (24-26): Architecture at v4.1 settings
#    [gatv2, gcn+ew, graph_transformer]  lat=16, drop=0.1, lr=1e-3
#
#  Block C (27-29): λ balance at v4.1 settings
#    λ_aki=[0.5, 2.0, 3.0]  GCN, lat=16, drop=0.1, lr=1e-3
#
#  Block D (30-31): Edge baselines at v4.1 settings
#    [spearman, mi]  GCN, lat=16, drop=0.1, lr=1e-3
#
#  Total: 32 jobs
#  Submit:  sbatch slurm/gae_v43_sweep.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

# ── Shared defaults ──
EDGE="llm"
LAYERS=2
HEADS=4
HIDDEN=64
LA_ASSOC=1.0
LA_AKI=1.0
AKI_MODE="pooled"
PATIENCE=40
EPOCHS=500
BATCH=128
GRAD_CLIP=5.0
WD=1e-4

IDX=$SLURM_ARRAY_TASK_ID

LRS=(7e-4 1e-3)
DROPOUTS=(0.0 0.05 0.1 0.15)
LATENTS=(12 16 24)

if [ $IDX -lt 24 ]; then
    # ── Block A: lr × dropout × latent (2×4×3=24) ──
    n_drop=${#DROPOUTS[@]}
    n_lat=${#LATENTS[@]}
    lr_idx=$(( IDX / (n_drop * n_lat) ))
    remainder=$(( IDX % (n_drop * n_lat) ))
    drop_idx=$(( remainder / n_lat ))
    lat_idx=$(( remainder % n_lat ))

    LR=${LRS[$lr_idx]}
    DROPOUT=${DROPOUTS[$drop_idx]}
    LATENT=${LATENTS[$lat_idx]}
    CONV="gcn"
    NAME="v4.3_gcn_lat${LATENT}_d${DROPOUT}_lr${LR}"
    EXTRA=""

elif [ $IDX -lt 27 ]; then
    # ── Block B: Architecture (3) ──
    b_idx=$(( IDX - 24 ))
    CONVS=("gatv2" "gcn" "graph_transformer")
    CONV=${CONVS[$b_idx]}

    LATENT=16
    DROPOUT=0.1
    LR=1e-3
    NAME="v4.3_${CONV}_lat16"
    if [ "$CONV" = "gcn" ]; then
        EXTRA="--use_edge_weights"
        NAME="v4.3_gcn_ew_lat16"
    else
        EXTRA=""
    fi

elif [ $IDX -lt 30 ]; then
    # ── Block C: λ_aki reweight (3) ──
    c_idx=$(( IDX - 27 ))
    LAKIS=(0.5 2.0 3.0)
    LA_AKI=${LAKIS[$c_idx]}

    LATENT=16
    DROPOUT=0.1
    LR=1e-3
    CONV="gcn"
    NAME="v4.3_gcn_laAki${LA_AKI}"
    EXTRA=""

else
    # ── Block D: Edge baselines (2) ──
    d_idx=$(( IDX - 30 ))
    EDGES=("spearman" "mi")
    EDGE=${EDGES[$d_idx]}

    LATENT=16
    DROPOUT=0.1
    LR=1e-3
    CONV="gcn"
    NAME="v4.3_${EDGE}_gcn_lat16"
    EXTRA=""
fi

echo "═══════════════════════════════════════════════"
echo "  Job ${IDX}: ${NAME}"
echo "  conv=${CONV} lat=${LATENT} h=${HIDDEN} drop=${DROPOUT}"
echo "  lr=${LR} wd=${WD} λ_assoc=${LA_ASSOC} λ_aki=${LA_AKI}"
echo "  edge=${EDGE} $(date)"
echo "═══════════════════════════════════════════════"

python train.py \
    --edge_method "$EDGE" \
    --encoder_type "$CONV" \
    --k 8 \
    --lambda_assoc "$LA_ASSOC" \
    --lambda_aki "$LA_AKI" \
    --layers "$LAYERS" \
    --hidden "$HIDDEN" \
    --latent "$LATENT" \
    --heads "$HEADS" \
    --dropout "$DROPOUT" \
    --aki_mode "$AKI_MODE" \
    --lr "$LR" \
    --weight_decay "$WD" \
    --batch_size "$BATCH" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --eval_every 5 \
    --grad_clip "$GRAD_CLIP" \
    --verbose \
    --wandb \
    --run_name "$NAME" \
    $EXTRA

echo "Done: ${NAME} $(date)"
