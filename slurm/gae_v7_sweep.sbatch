#!/bin/bash -l
#SBATCH --job-name=gae_v7
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-17

# ──────────────────────────────────────────────────────────────
#  v7 sweep — z_aki readout + target decoupling + static assoc
#
#  Base: GCN, LLM+ew, lat=16, drop=0.1, lr=1e-3
#
#  Block 1 (0-5): Readout ablation (ChatGPT Block 1)
#    0: aki_concat + patient assoc (pearson)    ← expected best
#    1: aki_only + patient assoc (pearson)
#    2: non_aki_pool + patient assoc (pearson)
#    3: global_pool + patient assoc (pearson)   ← v5 baseline
#    4: aki_concat + NO assoc loss
#    5: aki_only + NO assoc loss
#
#  Block 2 (6-9): Association mode
#    6: aki_concat + static assoc (pearson on feature embeddings)
#    7: aki_concat + static assoc + assoc_decay
#    8: aki_concat + patient assoc + assoc_decay
#    9: aki_concat + patient assoc (llm target)  ← test directed
#
#  Block 3 (10-13): Best readout + enhancements
#    10: aki_concat + static + FGSAM
#    11: aki_concat + static + DirGCN
#    12: aki_concat + static + VGAE
#    13: aki_concat + static + LapPE
#
#  Block 4 (14-17): Edge baselines + architecture
#    14: aki_concat + static + spearman edges
#    15: aki_concat + static + MI edges
#    16: aki_concat + static + GATv2
#    17: aki_concat + static + graph_transformer
#
#  Total: 18 jobs
#  Submit:  sbatch slurm/gae_v7_sweep.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

BASE=(
    --edge_method llm --use_edge_weights --k 8
    --encoder_type gcn --layers 2 --hidden 64 --latent 16 --heads 4
    --dropout 0.1 --lr 1e-3 --weight_decay 1e-4
    --lambda_assoc 1.0 --lambda_aki 1.0
    --batch_size 128 --epochs 500 --patience 40 --eval_every 5
    --grad_clip 5.0 --verbose --wandb
    --decoder_type symmetric --assoc_target pearson --assoc_mode patient
)

IDX=$SLURM_ARRAY_TASK_ID

case $IDX in
  # ── Block 1: Readout ablation ──
  0)  NAME="v7_akiConcat_patient";    EXTRA="--readout aki_concat" ;;
  1)  NAME="v7_akiOnly_patient";      EXTRA="--readout aki_only" ;;
  2)  NAME="v7_nonAkiPool_patient";   EXTRA="--readout non_aki_pool" ;;
  3)  NAME="v7_globalPool_patient";   EXTRA="--readout global_pool" ;;
  4)  NAME="v7_akiConcat_noAssoc";    EXTRA="--readout aki_concat --assoc_target none --lambda_assoc 0" ;;
  5)  NAME="v7_akiOnly_noAssoc";      EXTRA="--readout aki_only --assoc_target none --lambda_assoc 0" ;;

  # ── Block 2: Association mode ──
  6)  NAME="v7_akiConcat_static";     EXTRA="--readout aki_concat --assoc_mode static" ;;
  7)  NAME="v7_akiConcat_static_decay"; EXTRA="--readout aki_concat --assoc_mode static --assoc_decay" ;;
  8)  NAME="v7_akiConcat_patient_decay"; EXTRA="--readout aki_concat --assoc_decay" ;;
  9)  NAME="v7_akiConcat_llmTarget";  EXTRA="--readout aki_concat --assoc_target llm --decoder_type directed" ;;

  # ── Block 3: Best readout + enhancements ──
  10) NAME="v7_akiConcat_static_fgsam";  EXTRA="--readout aki_concat --assoc_mode static --use_fgsam" ;;
  11) NAME="v7_akiConcat_static_dirGCN"; EXTRA="--readout aki_concat --assoc_mode static --use_dir_conv" ;;
  12) NAME="v7_akiConcat_static_vgae";   EXTRA="--readout aki_concat --assoc_mode static --use_vgae" ;;
  13) NAME="v7_akiConcat_static_lappe";  EXTRA="--readout aki_concat --assoc_mode static --use_laplacian_pe" ;;

  # ── Block 4: Edge baselines + architecture ──
  14) NAME="v7_akiConcat_spearman";
      EXTRA="--readout aki_concat --assoc_mode static --edge_method spearman" ;;
  15) NAME="v7_akiConcat_mi";
      EXTRA="--readout aki_concat --assoc_mode static --edge_method mi" ;;
  16) NAME="v7_akiConcat_gatv2";     EXTRA="--readout aki_concat --assoc_mode static --encoder_type gatv2" ;;
  17) NAME="v7_akiConcat_graphTF";   EXTRA="--readout aki_concat --assoc_mode static --encoder_type graph_transformer" ;;
esac

echo "═══════════════════════════════════════════════"
echo "  Job ${IDX}: ${NAME}"
echo "  $(date)"
echo "═══════════════════════════════════════════════"

python train_v2.py "${BASE[@]}" --run_name "$NAME" $EXTRA

echo "Done: ${NAME} $(date)"
