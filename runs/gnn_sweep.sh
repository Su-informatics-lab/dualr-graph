#!/usr/bin/env bash
# runs/gnn_sweep.sh — Systematic GNN performance sweep
#
# ── BACKGROUND ──────────────────────────────────────────────
#
# Our graph has 38 NODES (clinical features). Each node's "embedding"
# is all 2292 patient values for that feature: shape [38, 2292].
#
# EDGES connect features that are associated. Three methods:
#   spearman  — |Spearman ρ| between feature vectors across patients
#               e.g., ρ(CHF, Renal_Disease) = 0.45 → strong edge
#   mi        — Normalized mutual information (captures nonlinear links)
#   llm       — LLM-estimated P(a|b), directed (no patient data needed)
#
# k-NN SPARSIFICATION: the full 38×38 matrix has ~700 edges. Most are
# noise. k=8 means each node keeps its 8 strongest neighbors → ~400
# edges. Smaller k = sparser graph (less smoothing, sharper signal).
# Larger k = denser (more information flow, risk of oversmoothing).
#
# λ (lambda_ce): balances reconstruction loss (37 non-AKI nodes) vs
# AKI prediction loss (1 node). λ=1 → AKI is ~2% of gradient.
# Need λ >> 1 so the model actually learns to predict AKI.
#
# ── PHASES ──────────────────────────────────────────────────
#
#   1. Lambda sweep          — fix the gradient dilution (8 runs, ~15 min)
#   2. Model size sweep      — right capacity for 38-node graph (~20 runs)
#   3. Arch × edge × k grid — find best combination (~24 runs)
#   4. LR + dropout tuning   — polish best config (~10 runs)
#   5. LLM edge experiments  — after llm_adj.npy is ready (~9 runs)
#
# Usage:
#   cd ~/dualr-graph && source .venv_cpu/bin/activate
#   bash runs/gnn_sweep.sh           # run all phases
#   bash runs/gnn_sweep.sh lambda    # run phase 1 only
#   bash runs/gnn_sweep.sh grid      # run phase 3 only

set -euo pipefail
LOGDIR="logs"; mkdir -p "$LOGDIR" results
TS=$(date +%Y%m%d_%H%M%S)
PHASE="${1:-all}"

echo "════════════════════════════════════════════════════════════════"
echo "  GNN SWEEP — $(date)"
echo "  Phase: ${PHASE}"
echo "════════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════
# PHASE 1: Lambda sweep (priority — fixes the gradient dilution)
# ══════════════════════════════════════════════════════════════
run_lambda() {
    echo ""
    echo "── PHASE 1: Lambda sweep (GATv2 × Spearman, k=8) ──"
    echo "   Why: recon loss has 37 nodes, CE has 1 node."
    echo "   λ=1 → AKI signal is ~2% of gradient. Need λ >> 1."
    echo ""

    for lam in 1 5 10 25 50 100 200 500; do
        local rn="sweep_lam${lam}"
        echo "  → λ=${lam}: ${rn}"
        python train.py \
            --arch gatv2 --edge_method spearman --k 8 \
            --lambda_ce ${lam} \
            --hidden 64 --latent 16 --heads 4 \
            --lr 1e-3 --epochs 300 --patience 30 \
            --run_name "${rn}" \
            2>&1 | tail -3
    done
}

# ══════════════════════════════════════════════════════════════
# PHASE 2: Model size sweep (at best λ from phase 1)
# ══════════════════════════════════════════════════════════════
run_size() {
    echo ""
    echo "── PHASE 2: Model size sweep ──"
    echo "   38 nodes can't support huge hidden dims."
    echo "   Using λ=50 (adjust if phase 1 found better)."
    echo ""

    local LAM=50

    # hidden × latent × heads
    for h in 16 32 64 128; do
        for lat in 4 8 16; do
            for heads in 1 2 4; do
                # Skip nonsensical combos
                if [ $h -lt $lat ]; then continue; fi
                if [ $h -eq 16 ] && [ $heads -eq 4 ]; then continue; fi

                local rn="sweep_h${h}_l${lat}_hd${heads}"
                echo "  → hidden=${h} latent=${lat} heads=${heads}: ${rn}"
                python train.py \
                    --arch gatv2 --edge_method spearman --k 8 \
                    --lambda_ce ${LAM} \
                    --hidden ${h} --latent ${lat} --heads ${heads} \
                    --lr 1e-3 --epochs 300 --patience 30 \
                    --run_name "${rn}" \
                    2>&1 | tail -3
            done
        done
    done
}

# ══════════════════════════════════════════════════════════════
# PHASE 3: Architecture × Edge method × k grid
# ══════════════════════════════════════════════════════════════
run_grid() {
    echo ""
    echo "── PHASE 3: Arch × Edge × k grid ──"
    echo "   Using λ=50 (adjust from phase 1 if needed)."
    echo "   Sweeps: 3 archs × 2 edge methods × 4 k values = 24 runs"
    echo ""

    local LAM=50

    for edge in spearman mi; do
        for k in 3 5 8 12; do
            # Build graph at this k (rebuilds each time)
            echo "  Building ${edge} graph, k=${k}..."
            python graph.py --method ${edge} --k ${k} 2>/dev/null

            for arch in gatv2 gcn transformer; do
                local rn="sweep_${arch}_${edge}_k${k}"
                echo "  → ${arch} × ${edge} × k=${k}: ${rn}"
                python train.py \
                    --arch ${arch} --edge_method ${edge} --k ${k} \
                    --lambda_ce ${LAM} \
                    --hidden 64 --latent 16 --heads 4 \
                    --lr 1e-3 --epochs 300 --patience 30 \
                    --run_name "${rn}" \
                    2>&1 | tail -3
            done
        done
    done

    # Restore default k=8 graphs
    python graph.py --method spearman --k 8 2>/dev/null
    python graph.py --method mi --k 8 2>/dev/null

    # LLM edges (only if llm_adj.npy exists)
    if [ -f "data/llm_adj.npy" ]; then
        echo ""
        echo "  ── LLM edges (directed) ──"

        for k in 5 8 12; do
            python graph.py --method llm --k ${k} --llm_prior data/llm_adj.npy 2>/dev/null

            for arch in gatv2 transformer; do
                # Without edge weights
                local rn="sweep_${arch}_llm_k${k}"
                echo "  → ${arch} × llm × k=${k}: ${rn}"
                python train.py \
                    --arch ${arch} --edge_method llm --k ${k} \
                    --lambda_ce ${LAM} \
                    --run_name "${rn}" \
                    2>&1 | tail -3

                # With edge weights (LLM P(a|b) as attention bias)
                rn="sweep_${arch}_llm_k${k}_ew"
                echo "  → ${arch} × llm × k=${k} + edge_weight: ${rn}"
                python train.py \
                    --arch ${arch} --edge_method llm --k ${k} \
                    --lambda_ce ${LAM} \
                    --use_edge_weights \
                    --run_name "${rn}" \
                    2>&1 | tail -3
            done
        done
    else
        echo "  ⚠ data/llm_adj.npy not found — skipping LLM edges"
        echo "    Run llm_edges.py on Tempest first, then SCP llm_adj.npy here"
    fi
}

# ══════════════════════════════════════════════════════════════
# PHASE 4: Fine-tuning (LR + dropout on best config)
# ══════════════════════════════════════════════════════════════
run_finetune() {
    echo ""
    echo "── PHASE 4: LR + dropout fine-tuning ──"
    echo "   Using best arch/edge/k/λ from phases 1-3."
    echo "   Defaulting to GATv2 × Spearman k=8, λ=50 (adjust if needed)."
    echo ""

    local LAM=50

    # LR sweep
    for lr in 2e-4 5e-4 1e-3 2e-3 5e-3; do
        local rn="sweep_lr${lr}"
        echo "  → lr=${lr}: ${rn}"
        python train.py \
            --arch gatv2 --edge_method spearman --k 8 \
            --lambda_ce ${LAM} --lr ${lr} \
            --run_name "${rn}" \
            2>&1 | tail -3
    done

    # Dropout sweep
    for drop in 0.0 0.1 0.2 0.3 0.5; do
        local rn="sweep_drop${drop}"
        echo "  → dropout=${drop}: ${rn}"
        python train.py \
            --arch gatv2 --edge_method spearman --k 8 \
            --lambda_ce ${LAM} --dropout ${drop} \
            --run_name "${rn}" \
            2>&1 | tail -3
    done
}

# ══════════════════════════════════════════════════════════════
# PHASE 5: LLM edge weight experiments
# ══════════════════════════════════════════════════════════════
run_llm_ew() {
    if [ ! -f "data/llm_adj.npy" ]; then
        echo "  ⚠ data/llm_adj.npy not found — skipping phase 5"
        return
    fi

    echo ""
    echo "── PHASE 5: LLM edge weight experiments ──"
    echo "   Compare: no weights vs edge weights vs uncertainty cutoff"
    echo ""

    local LAM=50

    # CV cutoff sweep (uncertainty-based pruning)
    for cv in 0.1 0.2 0.3 0.5; do
        python graph.py --method llm --k 8 \
            --llm_prior data/llm_adj.npy --cv_cutoff ${cv} 2>/dev/null
        local rn="sweep_llm_cv${cv}"
        echo "  → LLM CV cutoff=${cv}: ${rn}"
        python train.py \
            --arch gatv2 --edge_method llm --k 8 \
            --lambda_ce ${LAM} --use_edge_weights \
            --cv_cutoff ${cv} \
            --run_name "${rn}" \
            2>&1 | tail -3
    done

    # k sweep on LLM edges
    for k in 3 5 8 12 20; do
        python graph.py --method llm --k ${k} \
            --llm_prior data/llm_adj.npy 2>/dev/null
        local rn="sweep_llm_k${k}_ew"
        echo "  → LLM k=${k} + edge_weight: ${rn}"
        python train.py \
            --arch gatv2 --edge_method llm --k ${k} \
            --lambda_ce ${LAM} --use_edge_weights \
            --run_name "${rn}" \
            2>&1 | tail -3
    done
}

# ══════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════
case "$PHASE" in
    lambda)   run_lambda ;;
    size)     run_size ;;
    grid)     run_grid ;;
    finetune) run_finetune ;;
    llm_ew)   run_llm_ew ;;
    all)
        run_lambda
        run_size
        run_grid
        run_finetune
        run_llm_ew
        ;;
    *)
        echo "Usage: $0 {lambda|size|grid|finetune|llm_ew|all}"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SWEEP DONE — $(date)"
echo "  Results: results/*.json"
echo "  Logs: ${LOGDIR}/"
echo "════════════════════════════════════════════════════════════════"
