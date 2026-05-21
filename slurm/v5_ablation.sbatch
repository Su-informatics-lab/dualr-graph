#!/bin/bash -l
#SBATCH --job-name=v5abl
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v5abl_%a_%j.out
#SBATCH --error=logs/v5abl_%a_%j.err
#SBATCH --array=0-11

# ── v5 ablation: start from v5 winner, change ONE thing at a time ──
# Uses existing models_v2.py + train_v2.py. No new code.

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

# v5 winner base config
BASE=(
  --edge_method llm --use_edge_weights --k 8
  --encoder_type gcn --layers 2 --hidden 64 --latent 16
  --decoder_type symmetric --assoc_target pearson --assoc_mode patient
  --lambda_assoc 1.0 --lambda_aki 1.0
  --lr 1e-3 --weight_decay 1e-4
  --batch_size 128 --epochs 500 --patience 40 --eval_every 5
  --grad_clip 5.0 --verbose --wandb
)

IDX=${SLURM_ARRAY_TASK_ID}
EXTRA=()

case $IDX in
  # ── Controls ──
  0)  NAME="v11_base_do0.0";     EXTRA+=(--dropout 0.0 --readout global_pool) ;;
  1)  NAME="v11_base_do0.1";     EXTRA+=(--dropout 0.1 --readout global_pool) ;;
  # ── Readout ablation ──
  2)  NAME="v11_aki_concat";      EXTRA+=(--dropout 0.0 --readout aki_concat) ;;
  3)  NAME="v11_non_aki_pool";    EXTRA+=(--dropout 0.0 --readout non_aki_pool) ;;
  # ── DirectedConv (v2's WeightedDirGCNConv) ──
  4)  NAME="v11_dirconv_gpool";   EXTRA+=(--dropout 0.0 --readout global_pool --use_dir_conv) ;;
  5)  NAME="v11_dirconv_concat";  EXTRA+=(--dropout 0.0 --readout aki_concat --use_dir_conv) ;;
  # ── Masked feature reconstruction auxiliary ──
  6)  NAME="v11_maskrecon";       EXTRA+=(--dropout 0.0 --readout global_pool --use_masked_recon --lambda_recon 0.1) ;;
  7)  NAME="v11_maskrecon_dir";   EXTRA+=(--dropout 0.0 --readout global_pool --use_masked_recon --lambda_recon 0.1 --use_dir_conv) ;;
  # ── Loss weight tuning ──
  8)  NAME="v11_assoc2_aki1";     EXTRA+=(--dropout 0.0 --readout global_pool --lambda_assoc 2.0) ;;
  9)  NAME="v11_assoc1_aki2";     EXTRA+=(--dropout 0.0 --readout global_pool --lambda_aki 2.0) ;;
  # ── GATv2 encoder ──
  10) NAME="v11_gatv2_gpool";     EXTRA+=(--dropout 0.0 --readout global_pool --encoder_type gatv2) ;;
  11) NAME="v11_gatv2_concat";    EXTRA+=(--dropout 0.0 --readout aki_concat --encoder_type gatv2) ;;
  *)  echo "unknown: $IDX"; exit 1 ;;
esac

echo "═══════════════════════════════════════════════"
echo "  v5 ablation ${IDX}: ${NAME}  $(date)"
echo "═══════════════════════════════════════════════"

python train_v2.py "${BASE[@]}" --run_name "$NAME" "${EXTRA[@]}"

echo "Done: ${NAME} $(date)"
