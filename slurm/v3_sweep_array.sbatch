#!/bin/bash -l
#SBATCH --job-name=v10.5
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v10.5_%a_%j.out
#SBATCH --error=logs/v10.5_%a_%j.err
#SBATCH --array=0-15

set -euo pipefail
cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

BASE=(
  --edge_method llm --use_edge_weights --k 8
  --encoder_type directed --layers 2 --hidden 64 --latent 16
  --dropout 0.0 --lr 1e-3 --weight_decay 1e-4
  --batch_size 128 --epochs 500 --patience 40 --eval_every 5
  --grad_clip 5.0 --verbose --wandb
  --jk_mode concat --edge_norm target_sum --aki_readout attention
  --lambda_assoc 1.0 --lambda_rec 0.1 --lambda_aki 1.0
)

IDX=${SLURM_ARRAY_TASK_ID}
EXTRA=()

case $IDX in
  # Core hybrid
  0)  NAME="v10.5_hybrid_L2" ;;
  1)  NAME="v10.5_hybrid_L3"; EXTRA+=(--layers 3) ;;
  2)  NAME="v10.5_concatReadout_L2"; EXTRA+=(--aki_readout concat) ;;
  3)  NAME="v10.5_concatReadout_L3"; EXTRA+=(--aki_readout concat --layers 3) ;;
  # v10.1 insight: λ_aki=2.0
  4)  NAME="v10.5_hybrid_L2_aki2"; EXTRA+=(--lambda_aki 2.0) ;;
  5)  NAME="v10.5_hybrid_L3_aki2"; EXTRA+=(--lambda_aki 2.0 --layers 3) ;;
  # Ablations
  6)  NAME="v10.5_noRec_assocOnly"; EXTRA+=(--lambda_rec 0.0) ;;
  7)  NAME="v10.5_noAssoc_recOnly"; EXTRA+=(--lambda_assoc 0.0 --lambda_rec 0.1) ;;
  8)  NAME="v10.5_legacy_reconstruct"; EXTRA+=(--lambda_assoc 0.0 --lambda_rec 1.0 --lambda_aki 2.0 --include_aki_in_rec --aki_readout reconstruct --layers 3) ;;
  # Anti-oversmoothing
  9)  NAME="v10.5_edgeNormNone"; EXTRA+=(--edge_norm none) ;;
  10) NAME="v10.5_dropEdge005"; EXTRA+=(--edge_dropout 0.05) ;;
  11) NAME="v10.5_pairnormScale"; EXTRA+=(--pairnorm scale) ;;
  # Encoder variants
  12) NAME="v10.5_gatv2_L2"; EXTRA+=(--encoder_type gatv2) ;;
  13) NAME="v10.5_gatv2_L3"; EXTRA+=(--encoder_type gatv2 --layers 3) ;;
  # JK and dropout
  14) NAME="v10.5_jkMax_L2"; EXTRA+=(--jk_mode max) ;;
  15) NAME="v10.5_hybrid_L2_do005"; EXTRA+=(--dropout 0.05) ;;
  *)  echo "unknown: $IDX"; exit 1 ;;
esac

echo "═══════════════════════════════════════════════"
echo "  Job ${IDX}: ${NAME}  $(date)"
echo "═══════════════════════════════════════════════"

python train_v3.py "${BASE[@]}" --run_name "$NAME" "${EXTRA[@]}"

echo "Done: ${NAME} $(date)"
