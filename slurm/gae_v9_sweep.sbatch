#!/bin/bash -l
#SBATCH --job-name=gae_v9
#SBATCH --account=r01109
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-38

# ──────────────────────────────────────────────────────────────
#  v9 sweep — combine winning signals that were never tested together
#
#  Missing from v4-v8:
#    - dropout=0 was best in v4.3, but ALL v7/v8 runs used 0.1
#    - 3 layers never tested with v7/v8 fixes
#    - GATv2 + global_pool (best readout) never tested
#    - masked_recon + assoc_loss together never tested
#    - GATv2 + dropout=0 + 3 layers completely unexplored
#
#  Block A (0-11): Core grid — encoder × layers × dropout
#    {GCN, GATv2} × {2, 3} × {0.0, 0.05, 0.1}
#    Fixed: global_pool, patient assoc (pearson), LLM+ew
#
#  Block B (12-19): Auxiliary objective grid
#    {patient_assoc, masked_recon, both, none} × {GCN, GATv2}
#    Fixed: 2 layers, dropout=0.0, global_pool
#
#  Block C (20-25): Best combos with 3 layers
#    {patient_assoc, masked_recon, both} × {GCN, GATv2}
#    Fixed: 3 layers, dropout=0.0, global_pool
#
#  Block D (26-31): Latent dim sweep
#    {8, 12, 16, 24, 32, 48} at GATv2, 3L, dropout=0, global_pool
#
#  Block E (32-35): Training tricks on best architecture
#    FGSAM, SWA, assoc_decay, FGSAM+SWA
#    Fixed: GATv2, 3L, dropout=0, global_pool, patient_assoc
#
#  Block F (36-38): Readout sanity at best architecture
#    {global_pool, non_aki_pool, aki_concat} at GATv2, 3L, dropout=0
#
#  Total: 39 jobs
#  Submit:  sbatch slurm/gae_v9_sweep.sbatch
# ──────────────────────────────────────────────────────────────

set -euo pipefail

cd /N/project/depot/hw56/dualreasoning/dualr-graph
source .venv_cpu/bin/activate
mkdir -p logs results

# ── Shared base (v8 scripts, override defaults to winners) ──
BASE=(
    --edge_method llm --use_edge_weights --k 8
    --hidden 64 --latent 16 --heads 4
    --lr 1e-3 --weight_decay 1e-4
    --lambda_aki 1.0
    --batch_size 128 --epochs 500 --patience 40 --eval_every 5
    --grad_clip 5.0 --verbose --wandb
)

IDX=$SLURM_ARRAY_TASK_ID

ENCODERS=("gcn" "gatv2")
LAYERS_OPTS=(2 3)
DROPOUTS=(0.0 0.05 0.1)

if [ $IDX -lt 12 ]; then
    # ── Block A: encoder × layers × dropout (2×2×3=12) ──
    n_lay=${#LAYERS_OPTS[@]}; n_drop=${#DROPOUTS[@]}
    enc_idx=$(( IDX / (n_lay * n_drop) ))
    rem=$(( IDX % (n_lay * n_drop) ))
    lay_idx=$(( rem / n_drop ))
    drop_idx=$(( rem % n_drop ))

    ENC=${ENCODERS[$enc_idx]}
    LAY=${LAYERS_OPTS[$lay_idx]}
    DROP=${DROPOUTS[$drop_idx]}
    NAME="v9_${ENC}_L${LAY}_d${DROP}"
    EXTRA="--encoder_type $ENC --layers $LAY --dropout $DROP --readout global_pool --assoc_target pearson --assoc_mode patient --lambda_assoc 1.0"

elif [ $IDX -lt 20 ]; then
    # ── Block B: auxiliary × encoder at L=2, d=0.0 (4×2=8) ──
    b_idx=$(( IDX - 12 ))
    enc_idx=$(( b_idx / 4 )); aux_idx=$(( b_idx % 4 ))
    ENC=${ENCODERS[$enc_idx]}

    case $aux_idx in
      0) AUX="--assoc_target pearson --assoc_mode patient --lambda_assoc 1.0"
         AUX_TAG="patientAssoc" ;;
      1) AUX="--assoc_target none --lambda_assoc 0 --use_masked_recon --mask_recon_rate 0.20 --lambda_recon 0.1"
         AUX_TAG="maskRecon" ;;
      2) AUX="--assoc_target pearson --assoc_mode patient --lambda_assoc 1.0 --use_masked_recon --mask_recon_rate 0.20 --lambda_recon 0.1"
         AUX_TAG="assoc+maskRecon" ;;
      3) AUX="--assoc_target none --lambda_assoc 0"
         AUX_TAG="noAux" ;;
    esac
    NAME="v9_${ENC}_L2_d0.0_${AUX_TAG}"
    EXTRA="--encoder_type $ENC --layers 2 --dropout 0.0 --readout global_pool $AUX"

elif [ $IDX -lt 26 ]; then
    # ── Block C: auxiliary × encoder at L=3, d=0.0 (3×2=6) ──
    c_idx=$(( IDX - 20 ))
    enc_idx=$(( c_idx / 3 )); aux_idx=$(( c_idx % 3 ))
    ENC=${ENCODERS[$enc_idx]}

    case $aux_idx in
      0) AUX="--assoc_target pearson --assoc_mode patient --lambda_assoc 1.0"
         AUX_TAG="patientAssoc" ;;
      1) AUX="--assoc_target none --lambda_assoc 0 --use_masked_recon --mask_recon_rate 0.20 --lambda_recon 0.1"
         AUX_TAG="maskRecon" ;;
      2) AUX="--assoc_target pearson --assoc_mode patient --lambda_assoc 1.0 --use_masked_recon --mask_recon_rate 0.20 --lambda_recon 0.1"
         AUX_TAG="assoc+maskRecon" ;;
    esac
    NAME="v9_${ENC}_L3_d0.0_${AUX_TAG}"
    EXTRA="--encoder_type $ENC --layers 3 --dropout 0.0 --readout global_pool $AUX"

elif [ $IDX -lt 32 ]; then
    # ── Block D: latent dim at GATv2, L=3, d=0 (6) ──
    d_idx=$(( IDX - 26 ))
    LATENTS=(8 12 16 24 32 48)
    LAT=${LATENTS[$d_idx]}
    NAME="v9_gatv2_L3_d0.0_lat${LAT}"
    EXTRA="--encoder_type gatv2 --layers 3 --dropout 0.0 --latent $LAT --readout global_pool --assoc_target pearson --assoc_mode patient --lambda_assoc 1.0"

elif [ $IDX -lt 36 ]; then
    # ── Block E: training tricks at GATv2, L=3, d=0 (4) ──
    e_idx=$(( IDX - 32 ))
    case $e_idx in
      0) TRICK="--use_fgsam";            TRICK_TAG="fgsam" ;;
      1) TRICK="--use_swa";              TRICK_TAG="swa" ;;
      2) TRICK="--assoc_decay";          TRICK_TAG="assocDecay" ;;
      3) TRICK="--use_fgsam --use_swa";  TRICK_TAG="fgsam+swa" ;;
    esac
    NAME="v9_gatv2_L3_d0.0_${TRICK_TAG}"
    EXTRA="--encoder_type gatv2 --layers 3 --dropout 0.0 --readout global_pool --assoc_target pearson --assoc_mode patient --lambda_assoc 1.0 $TRICK"

else
    # ── Block F: readout sanity at GATv2, L=3, d=0 (3) ──
    f_idx=$(( IDX - 36 ))
    READOUTS=("global_pool" "non_aki_pool" "aki_concat")
    RO=${READOUTS[$f_idx]}
    NAME="v9_gatv2_L3_d0.0_${RO}"
    EXTRA="--encoder_type gatv2 --layers 3 --dropout 0.0 --readout $RO --assoc_target pearson --assoc_mode patient --lambda_assoc 1.0"
fi

echo "═══════════════════════════════════════════════"
echo "  Job ${IDX}: ${NAME}"
echo "  $(date)"
echo "═══════════════════════════════════════════════"

python train_v2.py "${BASE[@]}" --run_name "$NAME" $EXTRA

echo "Done: ${NAME} $(date)"
