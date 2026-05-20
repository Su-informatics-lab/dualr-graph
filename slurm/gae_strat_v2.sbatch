#!/bin/bash -l
#SBATCH --job-name=gae_v5
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-19

# ──────────────────────────────────────────────────────────────
#  v5 sweep — directed decoder + ICML/NeurIPS enhancements
#
#  All runs use the v4.3 winning base: GCN, LLM+ew, lat=16,
#  drop=0.1, lr=1e-3, λ_assoc=1.0, λ_aki=1.0
#
#  Block A (0-2):  Decoder comparison
#    0: directed (new default)
#    1: crosscorr (S2GAE-inspired)
#    2: symmetric (v4.3 baseline for fair comparison)
#
#  Block B (3-6):  Individual enhancements (one at a time)
#    3: +FGSAM  (Luo et al., NeurIPS 2024)
#    4: +PCGrad (Yu et al., NeurIPS 2020)
#    5: +FLAG   (Kong et al., CVPR 2022)
#    6: +SWA    (Izmailov et al., UAI 2018)
#
#  Block C (7-10): Enhancement combos
#    7:  directed + FGSAM + SWA
#    8:  directed + PCGrad + SWA
#    9:  directed + FGSAM + PCGrad
#    10: directed + FGSAM + PCGrad + SWA (kitchen sink)
#
#  Block D (11-13): VGAE variants
#    11: directed + VGAE (λ_kl=0.001)
#    12: directed + VGAE (λ_kl=0.01)
#    13: directed + VGAE + FGSAM
#
#  Block E (14-15): Laplacian PE
#    14: directed + LapPE
#    15: directed + LapPE + FGSAM
#
#  Block F (16-17): Architecture with directed decoder
#    16: GATv2 + directed
#    17: graph_transformer + directed
#
#  Block G (18-19): Edge baselines with directed decoder
#    18: spearman + symmetric decoder (data-derived baseline)
#    19: mi + symmetric decoder (data-derived baseline)
#
#  Total: 20 jobs
#  Submit:  sbatch slurm/gae_v5_sweep.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

# ── v4.3 winning base config (no --use_edge_weights; added per-run) ──
BASE=(
    --edge_method llm --k 8
    --encoder_type gcn --layers 2 --hidden 64 --latent 16 --heads 4
    --dropout 0.1 --lr 1e-3 --weight_decay 1e-4
    --lambda_assoc 1.0 --lambda_aki 1.0
    --batch_size 128 --epochs 500 --patience 40 --eval_every 5
    --grad_clip 5.0 --verbose --wandb
)
EW="--use_edge_weights"  # added for LLM runs

IDX=$SLURM_ARRAY_TASK_ID

case $IDX in
  # ── Block A: Decoder comparison ──
  0)  NAME="v5_directed";       EXTRA="$EW --decoder_type directed" ;;
  1)  NAME="v5_crosscorr";      EXTRA="$EW --decoder_type crosscorr" ;;
  2)  NAME="v5_symmetric";      EXTRA="$EW --decoder_type symmetric" ;;

  # ── Block B: Individual enhancements ──
  3)  NAME="v5_dir+fgsam";      EXTRA="$EW --decoder_type directed --use_fgsam" ;;
  4)  NAME="v5_dir+pcgrad";     EXTRA="$EW --decoder_type directed --use_pcgrad" ;;
  5)  NAME="v5_dir+flag";       EXTRA="$EW --decoder_type directed --use_flag" ;;
  6)  NAME="v5_dir+swa";        EXTRA="$EW --decoder_type directed --use_swa" ;;

  # ── Block C: Combos ──
  7)  NAME="v5_dir+fgsam+swa";        EXTRA="$EW --decoder_type directed --use_fgsam --use_swa" ;;
  8)  NAME="v5_dir+pcgrad+swa";       EXTRA="$EW --decoder_type directed --use_pcgrad --use_swa" ;;
  9)  NAME="v5_dir+fgsam+pcgrad";     EXTRA="$EW --decoder_type directed --use_fgsam --use_pcgrad" ;;
  10) NAME="v5_dir+fgsam+pcgrad+swa"; EXTRA="$EW --decoder_type directed --use_fgsam --use_pcgrad --use_swa" ;;

  # ── Block D: VGAE ──
  11) NAME="v5_dir+vgae_kl1e-3";      EXTRA="$EW --decoder_type directed --use_vgae --lambda_kl 0.001" ;;
  12) NAME="v5_dir+vgae_kl1e-2";      EXTRA="$EW --decoder_type directed --use_vgae --lambda_kl 0.01" ;;
  13) NAME="v5_dir+vgae+fgsam";       EXTRA="$EW --decoder_type directed --use_vgae --lambda_kl 0.001 --use_fgsam" ;;

  # ── Block E: Laplacian PE ──
  14) NAME="v5_dir+lappe";            EXTRA="$EW --decoder_type directed --use_laplacian_pe" ;;
  15) NAME="v5_dir+lappe+fgsam";      EXTRA="$EW --decoder_type directed --use_laplacian_pe --use_fgsam" ;;

  # ── Block F: Architecture ──
  16) NAME="v5_gatv2+dir";            EXTRA="$EW --decoder_type directed --encoder_type gatv2" ;;
  17) NAME="v5_graphtransformer+dir";  EXTRA="$EW --decoder_type directed --encoder_type graph_transformer" ;;

  # ── Block G: Edge baselines (no edge weights, symmetric decoder) ──
  18) NAME="v5_spearman_sym"; EXTRA="--decoder_type symmetric --edge_method spearman" ;;
  19) NAME="v5_mi_sym";       EXTRA="--decoder_type symmetric --edge_method mi" ;;
esac

echo "═══════════════════════════════════════════════"
echo "  Job ${IDX}: ${NAME}"
echo "  $(date)"
echo "═══════════════════════════════════════════════"

python train_v2.py "${BASE[@]}" --run_name "$NAME" $EXTRA

echo "Done: ${NAME} $(date)"
