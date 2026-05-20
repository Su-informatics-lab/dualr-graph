#!/bin/bash -l
#SBATCH --job-name=gae_v42
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-29

# ──────────────────────────────────────────────────────────────
#  v4.2 sweep — regularisation-first, guided by v4.1 results
#
#  v4.1 findings:
#    - lat16 > lat32 > lat64 → overfitting, push capacity lower
#    - GCN ≈ GATv2 > GAT > transformer → simpler models generalise
#    - LLM edges >> spearman/MI → fix LLM, drop data-derived
#    - λ_assoc=1.0 sweet spot → keep, sweep λ_aki instead
#    - val_loss u-shape early → lr too high, patience can tighten
#    - pooled ≈ dual → fix pooled
#
#  Block A (0-17):  Regularisation grid  [GCN, LLM]
#    3 latent (8, 12, 16) × 3 dropout (0.15, 0.2, 0.3) × 2 lr (3e-4, 5e-4)
#
#  Block B (18-21): Hidden dim + weight decay  [GCN, LLM, lat=12]
#    hidden (32, 48) × wd (5e-4, 1e-3)
#
#  Block C (22-24): Architecture at best reg  [LLM, lat=12, drop=0.2, lr=3e-4]
#    GATv2, GCN+ew, graph_transformer
#
#  Block D (25-27): λ_aki reweight  [GCN, LLM, lat=12, drop=0.2, lr=3e-4]
#    λ_aki = 0.5, 2.0, 3.0  (with λ_assoc=1.0 fixed)
#
#  Block E (28-29): Edge method baselines  [GCN, best reg from A]
#    spearman, mi — matched config so LLM vs data-derived is fair
#
#  Total: 30 jobs
#  Submit:  sbatch slurm/gae_v42_sweep.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

# ── Shared defaults ──
EDGE="llm"
LAYERS=2
HEADS=4
LA_ASSOC=1.0
LA_AKI=1.0
AKI_MODE="pooled"
PATIENCE=30
EPOCHS=500
BATCH=128
GRAD_CLIP=5.0

IDX=$SLURM_ARRAY_TASK_ID

LATENTS=(8 12 16)
DROPOUTS=(0.15 0.2 0.3)
LRS=(3e-4 5e-4)

if [ $IDX -lt 18 ]; then
    # ── Block A: Regularisation grid  (3 × 3 × 2 = 18) ──
    n_drop=${#DROPOUTS[@]}
    n_lr=${#LRS[@]}
    lat_idx=$(( IDX / (n_drop * n_lr) ))
    remainder=$(( IDX % (n_drop * n_lr) ))
    drop_idx=$(( remainder / n_lr ))
    lr_idx=$(( remainder % n_lr ))

    LATENT=${LATENTS[$lat_idx]}
    DROPOUT=${DROPOUTS[$drop_idx]}
    LR=${LRS[$lr_idx]}
    HIDDEN=64
    CONV="gcn"
    WD=1e-4
    NAME="v4.2_gcn_lat${LATENT}_d${DROPOUT}_lr${LR}"
    EXTRA=""

elif [ $IDX -lt 22 ]; then
    # ── Block B: Hidden dim + weight decay (2 × 2 = 4) ──
    b_idx=$(( IDX - 18 ))
    HIDDENS=(32 48)
    WDS=(5e-4 1e-3)
    h_idx=$(( b_idx / 2 ))
    w_idx=$(( b_idx % 2 ))

    LATENT=12
    DROPOUT=0.2
    LR=3e-4
    HIDDEN=${HIDDENS[$h_idx]}
    WD=${WDS[$w_idx]}
    CONV="gcn"
    NAME="v4.2_gcn_h${HIDDEN}_wd${WD}"
    EXTRA=""

elif [ $IDX -lt 25 ]; then
    # ── Block C: Architecture comparison (3) ──
    c_idx=$(( IDX - 22 ))
    CONVS=("gatv2" "gcn" "graph_transformer")

    LATENT=12
    DROPOUT=0.2
    LR=3e-4
    HIDDEN=64
    WD=1e-4
    CONV=${CONVS[$c_idx]}
    NAME="v4.2_${CONV}_lat12_bestReg"
    # GCN with edge weights is worth testing separately
    if [ "$CONV" = "gcn" ]; then
        EXTRA="--use_edge_weights"
        NAME="v4.2_gcn_ew_lat12_bestReg"
    else
        EXTRA=""
    fi

elif [ $IDX -lt 28 ]; then
    # ── Block D: λ_aki reweight (3) ──
    d_idx=$(( IDX - 25 ))
    LAKIS=(0.5 2.0 3.0)
    LA_AKI=${LAKIS[$d_idx]}

    LATENT=12
    DROPOUT=0.2
    LR=3e-4
    HIDDEN=64
    WD=1e-4
    CONV="gcn"
    NAME="v4.2_gcn_laAki${LA_AKI}"
    EXTRA=""

else
    # ── Block E: Edge method baselines (2) ──
    # Matched config to best LLM settings for fair comparison in paper
    e_idx=$(( IDX - 28 ))
    EDGES=("spearman" "mi")
    EDGE=${EDGES[$e_idx]}

    LATENT=12
    DROPOUT=0.2
    LR=3e-4
    HIDDEN=64
    WD=1e-4
    CONV="gcn"
    NAME="v4.2_${EDGE}_gcn_lat12_bestReg"
    EXTRA=""
fi

echo "═══════════════════════════════════════════════"
echo "  Job ${IDX}: ${NAME}"
echo "  conv=${CONV} lat=${LATENT} h=${HIDDEN} drop=${DROPOUT}"
echo "  lr=${LR} wd=${WD} λ_assoc=${LA_ASSOC} λ_aki=${LA_AKI}"
echo "  $(date)"
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
