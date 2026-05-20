#!/bin/bash -l
#SBATCH --job-name=gae_v8
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-25

# ──────────────────────────────────────────────────────────────
#  v8 sweep — fixed feature-graph AE
#
#  Base model:
#    GCN, LLM topology, k=8, hidden=64, latent=16, dropout=0.1, lr=1e-3
#
#  Main fixes being tested:
#    ① AKI node readout: z_aki / concat(z_aki, non-AKI context)
#    ② Decoupled topology from association target
#    ③ Static association loss on feature identity embeddings
#    ④ Weighted directed GCN that preserves edge weights
#    ⑤ Functional masked feature reconstruction pretext
#
#  Blocks:
#    0-8   Core readout / association / edge-weight ablations
#    9-14  Reconstruction target and decoder tests
#    15-21 Optimizer / regularization / directed-conv tests
#    22-25 Edge-method and encoder baselines
#
#  Submit:
#    sbatch slurm/gae_v8_sweep.sbatch
#
#  Optional throttle, e.g. run at most 6 tasks at once:
#    sbatch --array=0-25%6 slurm/gae_v8_sweep.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

BASE=(
    --edge_method llm --k 8
    --encoder_type gcn --layers 2 --hidden 64 --latent 16 --heads 4
    --dropout 0.1 --lr 1e-3 --weight_decay 1e-4
    --lambda_assoc 1.0 --lambda_aki 1.0 --lambda_kl 0.001
    --batch_size 128 --epochs 500 --patience 40 --eval_every 5
    --grad_clip 5.0 --verbose --wandb
    --decoder_type symmetric
    --assoc_target pearson --assoc_mode static
    --readout aki_concat
)

IDX=${SLURM_ARRAY_TASK_ID}
NAME=""
EXTRA=()

case "${IDX}" in
  # ── Block 1: Core readout / association / edge-weight ablations ──
  0)  NAME="v8_anchor_akiConcat_static_wEW"
      EXTRA=(--use_edge_weights) ;;

  1)  NAME="v8_akiConcat_noAssoc_wEW"
      EXTRA=(--use_edge_weights --assoc_target none --lambda_assoc 0) ;;

  2)  NAME="v8_akiOnly_static_wEW"
      EXTRA=(--use_edge_weights --readout aki_only) ;;

  3)  NAME="v8_nonAkiPool_static_wEW"
      EXTRA=(--use_edge_weights --readout non_aki_pool) ;;

  4)  NAME="v8_globalPool_static_wEW"
      EXTRA=(--use_edge_weights --readout global_pool) ;;

  5)  NAME="v8_akiConcat_patient_wEW"
      EXTRA=(--use_edge_weights --assoc_mode patient) ;;

  6)  NAME="v8_akiConcat_patient_decay_wEW"
      EXTRA=(--use_edge_weights --assoc_mode patient --assoc_decay) ;;

  7)  NAME="v8_akiConcat_static_decay_wEW"
      EXTRA=(--use_edge_weights --assoc_decay) ;;

  8)  NAME="v8_akiConcat_static_noEW"
      EXTRA=() ;;

  # ── Block 2: Masked feature reconstruction and decoder / target tests ──
  9)  NAME="v8_maskRecon_r015_lam005_wEW"
      EXTRA=(--use_edge_weights --use_masked_recon --mask_recon_rate 0.15 --lambda_recon 0.05) ;;

  10) NAME="v8_maskRecon_r020_lam010_wEW"
      EXTRA=(--use_edge_weights --use_masked_recon --mask_recon_rate 0.20 --lambda_recon 0.10) ;;

  11) NAME="v8_maskRecon_r030_lam020_wEW"
      EXTRA=(--use_edge_weights --use_masked_recon --mask_recon_rate 0.30 --lambda_recon 0.20) ;;

  12) NAME="v8_maskRecon_noAssoc_wEW"
      EXTRA=(--use_edge_weights --assoc_target none --lambda_assoc 0 --use_masked_recon --mask_recon_rate 0.20 --lambda_recon 0.10) ;;

  13) NAME="v8_llmTarget_directed_patient_wEW"
      EXTRA=(--use_edge_weights --assoc_target llm --assoc_mode patient --decoder_type directed) ;;

  14) NAME="v8_crosscorr_staticPearson_wEW"
      EXTRA=(--use_edge_weights --decoder_type crosscorr) ;;

  # ── Block 3: Optimizer / regularization / directed-conv tests ──
  15) NAME="v8_static_fgsam_wEW"
      EXTRA=(--use_edge_weights --use_fgsam) ;;

  16) NAME="v8_static_pcgrad_wEW"
      EXTRA=(--use_edge_weights --use_pcgrad) ;;

  17) NAME="v8_static_flag_wEW"
      EXTRA=(--use_edge_weights --use_flag --flag_eps 0.01 --flag_steps 3) ;;

  18) NAME="v8_static_swa_wEW"
      EXTRA=(--use_edge_weights --use_swa) ;;

  19) NAME="v8_static_vgae_wEW"
      EXTRA=(--use_edge_weights --use_vgae) ;;

  20) NAME="v8_static_lappe_wEW"
      EXTRA=(--use_edge_weights --use_laplacian_pe --pe_dim 16) ;;

  21) NAME="v8_static_weightedDirGCN_wEW"
      EXTRA=(--use_edge_weights --use_dir_conv) ;;

  # ── Block 4: Edge-method and encoder baselines ──
  22) NAME="v8_static_spearmanEdges_wEW"
      EXTRA=(--use_edge_weights --edge_method spearman) ;;

  23) NAME="v8_static_miEdges_wEW"
      EXTRA=(--use_edge_weights --edge_method mi) ;;

  24) NAME="v8_static_gatv2_wEW"
      EXTRA=(--use_edge_weights --encoder_type gatv2) ;;

  25) NAME="v8_static_graphTransformer_wEW"
      EXTRA=(--use_edge_weights --encoder_type graph_transformer) ;;

  *)  echo "Invalid SLURM_ARRAY_TASK_ID=${IDX}" >&2
      exit 1 ;;
esac

echo "═══════════════════════════════════════════════"
echo "  v8 Job ${IDX}: ${NAME}"
echo "  Host: $(hostname)"
echo "  Start: $(date)"
echo "═══════════════════════════════════════════════"

echo "Command: python train_v2.py ${BASE[*]} --run_name ${NAME} ${EXTRA[*]}"
python train_v2.py "${BASE[@]}" --run_name "${NAME}" "${EXTRA[@]}"

echo "Done: ${NAME} $(date)"
