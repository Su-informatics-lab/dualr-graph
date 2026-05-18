#!/bin/bash -l
#SBATCH --job-name=gae_ablation
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-15

# ──────────────────────────────────────────────────────────────
#  AKI mode ablation  (16 jobs)
#
#  4 modes × 4 λ_rec, all on LLM edge-weighted, gatv2, L=3
#
#  Mode R:  recon       — shared Linear (original, failed)
#  Mode A:  recon_split — separate Linear_aki(dK)
#  Mode B:  dual        — MLP(zK, z_pool) from encoder
#  Mode C:  reweight    — recon + α·ℒ_AKI  (α=58)
#
#  Submit:  sbatch slurm/gae_ablation.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

AKI_MODES=("recon" "recon_split" "dual" "reweight")
LAMBDA_RECS=(0 0.1 0.5 1.0)

IDX=$SLURM_ARRAY_TASK_ID
m_idx=$(( IDX / 4 ))
l_idx=$(( IDX % 4 ))

MODE=${AKI_MODES[$m_idx]}
LAM=${LAMBDA_RECS[$l_idx]}
NAME="abl_${MODE}_lam${LAM}"

# For reweight mode, set aki_weight
EXTRA=""
if [ "$MODE" = "reweight" ]; then
    EXTRA="--aki_weight 58"
fi

echo "═══════════════════════════════════════════════"
echo "  Job ${IDX}: ${NAME}"
echo "  aki_mode=${MODE} λ_rec=${LAM} ${EXTRA}"
echo "  $(date)"
echo "═══════════════════════════════════════════════"

python train.py \
    --edge_method llm \
    --use_edge_weights \
    --conv_type gatv2 \
    --k 8 \
    --lambda_rec "$LAM" \
    --layers 3 \
    --aki_mode "$MODE" \
    --hidden 64 \
    --latent 32 \
    --heads 4 \
    --epochs 500 \
    --patience 40 \
    --wandb \
    --run_name "$NAME" \
    $EXTRA

echo "Done: ${NAME} $(date)"
