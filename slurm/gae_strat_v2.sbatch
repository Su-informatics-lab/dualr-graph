#!/bin/bash -l
#SBATCH --job-name=gae_v6
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-3

# ──────────────────────────────────────────────────────────────
#  v6 sweep — DirGNNConv direction-aware encoder (Rossi et al. LoG 2023)
#
#  v5 diagnosis: GCNConv symmetrises the adjacency → encoder can't
#  produce direction-aware embeddings → directed decoder memorises
#  the fixed LLM adj instead of learning useful representations.
#
#  Fix: DirGNNConv wraps GCNConv, splitting message-passing into
#  incoming (j→i) and outgoing (i→j) with learnable α balance.
#
#  0: DirGNNConv + directed decoder  (THE key test)
#  1: DirGNNConv + directed decoder + FGSAM
#  2: DirGNNConv + symmetric decoder (ablation: does DirConv help encoder alone?)
#  3: DirGNNConv + directed decoder + VGAE
#
#  Total: 4 jobs
#  Submit:  sbatch slurm/gae_v6_sweep.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

# ── v4.3 winning base + DirGNNConv ──
BASE=(
    --edge_method llm --k 8
    --encoder_type gcn --layers 2 --hidden 64 --latent 16 --heads 4
    --dropout 0.1 --lr 1e-3 --weight_decay 1e-4
    --lambda_assoc 1.0 --lambda_aki 1.0
    --batch_size 128 --epochs 500 --patience 40 --eval_every 5
    --grad_clip 5.0 --verbose --wandb
    --use_dir_conv
)
EW="--use_edge_weights"

IDX=$SLURM_ARRAY_TASK_ID

case $IDX in
  0) NAME="v6_dirconv+dir";          EXTRA="$EW --decoder_type directed" ;;
  1) NAME="v6_dirconv+dir+fgsam";    EXTRA="$EW --decoder_type directed --use_fgsam" ;;
  2) NAME="v6_dirconv+sym";          EXTRA="$EW --decoder_type symmetric" ;;
  3) NAME="v6_dirconv+dir+vgae";     EXTRA="$EW --decoder_type directed --use_vgae --lambda_kl 0.001" ;;
esac

echo "═══════════════════════════════════════════════"
echo "  Job ${IDX}: ${NAME}"
echo "  $(date)"
echo "═══════════════════════════════════════════════"

python train_v2.py "${BASE[@]}" --run_name "$NAME" $EXTRA

echo "Done: ${NAME} $(date)"
