#!/usr/bin/env bash
# runs/gnn.sh — GNN experiments on Quartz CPU
#
# Sections: graphs → grid → sweep
# Each section can be run standalone with an argument.
#
# Usage:
#   cd ~/dualr-graph && source .venv_cpu/bin/activate
#   bash runs/gnn.sh graphs      # rebuild spearman/mi/llm graphs
#   bash runs/gnn.sh grid        # 3×3 arch × edge grid (defaults)
#   bash runs/gnn.sh sweep       # lambda/lr/hidden/dropout/k sweeps
#   bash runs/gnn.sh all         # graphs → grid → sweep

set -euo pipefail
LOGDIR="logs"; mkdir -p "$LOGDIR" results
TS=$(date +%Y%m%d_%H%M%S)

# ── LLM adjacency: llm_adj.npy is the raw LLM output (from Tempest).
#    graph.py reads it via --llm_prior and writes adj_llm.npy,
#    which train.py then loads as adj_{edge_method}.npy.
LLM_RAW="data/llm_adj.npy"
HAS_LLM=false; [ -f "$LLM_RAW" ] && HAS_LLM=true

edges_list() {
    local e=(spearman mi)
    $HAS_LLM && e+=(llm)
    echo "${e[@]}"
}

best_edge() { $HAS_LLM && echo "llm" || echo "mi"; }

# ═══════════════════════════════════════════════════════════════
do_graphs() {
    echo "════ BUILD GRAPHS ════"
    for m in spearman mi; do
        echo "  → $m k=8"
        python graph.py --method "$m" --k 8 \
            2>&1 | tee "$LOGDIR/graph_${m}_${TS}.log"
    done
    if $HAS_LLM; then
        echo "  → llm k=8"
        python graph.py --method llm --k 8 --llm_prior "$LLM_RAW" \
            2>&1 | tee "$LOGDIR/graph_llm_${TS}.log"
    else
        echo "  ⚠ $LLM_RAW not found"
        echo "    scp tempest:~/dualr-graph/data/llm_adj.npy data/"
    fi
}

# ═══════════════════════════════════════════════════════════════
do_grid() {
    echo "════ 3×3 GNN GRID (defaults: λ=1, lr=1e-3, h64 l16) ════"
    local edges; read -ra edges <<< "$(edges_list)"

    for arch in gatv2 gcn transformer; do
        for edge in "${edges[@]}"; do
            local rn="${arch}_${edge}_k8"
            echo "──── $rn ────"
            python train.py \
                --arch "$arch" \
                --edge_method "$edge" \
                --k 8 \
                --run_name "$rn" \
                --wandb \
                2>&1 | tee "$LOGDIR/${rn}_${TS}.log"
            echo ""
        done
    done
}

# ═══════════════════════════════════════════════════════════════
do_sweep() {
    echo "════ HYPERPARAMETER SWEEPS (GATv2) ════"
    local be; be=$(best_edge)
    local edges; read -ra edges <<< "$(edges_list)"

    # ── Lambda × LR (all edges) ──────────────────────────────
    echo "  ── Lambda × LR ──"
    for edge in "${edges[@]}"; do
        for lam in 0.5 1 2 5 10 20 50 100; do
            for lr in 1e-3 5e-4; do
                local rn="gatv2_${edge}_k8_lam${lam}_lr${lr}"
                echo "  → $rn"
                python train.py \
                    --arch gatv2 --edge_method "$edge" --k 8 \
                    --lambda_ce "$lam" --lr "$lr" \
                    --run_name "$rn" --wandb \
                    2>&1 | tee "$LOGDIR/${rn}_${TS}.log"
            done
        done
    done

    # ── Hidden × latent (best edge) ──────────────────────────
    echo ""; echo "  ── Hidden × latent (GATv2 × $be) ──"
    for h in 32 64 128 256; do
        for l in 8 16 32 64; do
            [ "$l" -gt "$h" ] && continue
            local rn="gatv2_${be}_k8_h${h}_l${l}"
            echo "  → $rn"
            python train.py \
                --arch gatv2 --edge_method "$be" --k 8 \
                --hidden "$h" --latent "$l" \
                --run_name "$rn" --wandb \
                2>&1 | tee "$LOGDIR/${rn}_${TS}.log"
        done
    done

    # ── LR (best edge) ───────────────────────────────────────
    echo ""; echo "  ── LR (GATv2 × $be) ──"
    for lr in 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5; do
        local rn="gatv2_${be}_k8_lr${lr}"
        echo "  → $rn"
        python train.py \
            --arch gatv2 --edge_method "$be" --k 8 \
            --lr "$lr" --run_name "$rn" --wandb \
            2>&1 | tee "$LOGDIR/${rn}_${TS}.log"
    done

    # ── Dropout (best edge) ──────────────────────────────────
    echo ""; echo "  ── Dropout (GATv2 × $be) ──"
    for d in 0.0 0.1 0.2 0.3 0.4 0.5; do
        local rn="gatv2_${be}_k8_drop${d}"
        echo "  → $rn"
        python train.py \
            --arch gatv2 --edge_method "$be" --k 8 \
            --dropout "$d" --run_name "$rn" --wandb \
            2>&1 | tee "$LOGDIR/${rn}_${TS}.log"
    done

    # ── k (rebuild graph each time) ──────────────────────────
    echo ""; echo "  ── k (GATv2 × $be) ──"
    for k in 3 5 8 10 12 15 20; do
        # Rebuild graph at this k
        if [ "$be" = "llm" ]; then
            python graph.py --method llm --k "$k" --llm_prior "$LLM_RAW" 2>/dev/null
        else
            python graph.py --method "$be" --k "$k" 2>/dev/null
        fi
        local rn="gatv2_${be}_k${k}"
        echo "  → $rn"
        python train.py \
            --arch gatv2 --edge_method "$be" --k "$k" \
            --run_name "$rn" --wandb \
            2>&1 | tee "$LOGDIR/${rn}_${TS}.log"
    done
    # Restore k=8
    if [ "$be" = "llm" ]; then
        python graph.py --method llm --k 8 --llm_prior "$LLM_RAW" 2>/dev/null
    else
        python graph.py --method "$be" --k 8 2>/dev/null
    fi
}

# ═══════════════════════════════════════════════════════════════
CMD="${1:-all}"
case "$CMD" in
    graphs) do_graphs ;;
    grid)   do_grid ;;
    sweep)  do_sweep ;;
    all)    do_graphs; do_grid; do_sweep ;;
    *) echo "Usage: $0 {graphs|grid|sweep|all}"; exit 1 ;;
esac

echo ""; echo "════ DONE $TS — logs: $LOGDIR/ ════"
echo "  W&B: https://wandb.ai/hainingwang/graphai-ici-aki"
