#!/usr/bin/env bash
# run_all.sh — Comprehensive experiment runner for Quartz CPU
#
# Uses actual argparse from 05_train.py / 10_baselines.py / 06_tune.py.
#
# Usage:
#   cd ~/dualr-graph && source .venv_cpu/bin/activate
#   bash run_all.sh baselines    # baselines + Cox + full LogReg (fast, do first)
#   bash run_all.sh graphs       # rebuild graphs from new feature matrix
#   bash run_all.sh grid         # 3×3 GNN grid
#   bash run_all.sh sweep        # lambda/lr/hidden/k sweeps (~150 runs)
#   bash run_all.sh tune         # Optuna Phase 1+2
#   bash run_all.sh all          # graphs → baselines → grid → sweep
#   bash run_all.sh etl          # re-run ETL only

set -euo pipefail
LOGDIR="logs"; mkdir -p "$LOGDIR" results figures
TS=$(date +%Y%m%d_%H%M%S)

# Detect script names (numbered vs unnumbered)
TRAIN="05_train.py";     [ -f "$TRAIN" ]     || TRAIN="train.py"
GRAPH="_02_graph.py";    [ -f "$GRAPH" ]     || GRAPH="graph.py"
BASELINES="10_baselines.py"; [ -f "$BASELINES" ] || BASELINES="baselines.py"
TUNE="06_tune.py";      [ -f "$TUNE" ]      || TUNE="tune.py"
ETL="01_etl_quartz.py"
echo "  train=$TRAIN  graph=$GRAPH  baselines=$BASELINES  tune=$TUNE"

# ═══════════════════════════════════════════════════════════════
do_etl() {
    echo "════ ETL ════"
    python "$ETL" 2>&1 | tee "$LOGDIR/etl_${TS}.log"
}

# ═══════════════════════════════════════════════════════════════
do_graphs() {
    echo "════ BUILD GRAPHS ════"
    for m in spearman mi; do
        echo "  → $m k=8"
        python "$GRAPH" --method "$m" --k 8 2>&1 | tee "$LOGDIR/graph_${m}_${TS}.log"
    done
    if [ -f data/adj_llm.npy ]; then
        echo "  → llm k=8"
        python "$GRAPH" --method llm --k 8 --llm_prior data/adj_llm.npy \
            2>&1 | tee "$LOGDIR/graph_llm_${TS}.log"
    else
        echo "  ⚠ data/adj_llm.npy missing — scp from Tempest"
    fi
}

# ═══════════════════════════════════════════════════════════════
do_baselines() {
    # ── Standard baselines ────────────────────────────────────
    echo "════ BASELINES (LogReg / RF / XGBoost) ════"
    python "$BASELINES" --cv_folds 5 2>&1 | tee "$LOGDIR/baselines_${TS}.log"

    # ── Full LogReg table (Dr. Su) ────────────────────────────
    echo ""; echo "════ FULL LogReg (coefs / ORs / p-values) ════"
    python - <<'PYEOF' 2>&1 | tee "$LOGDIR/logreg_full_${TS}.log"
import torch, json, numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

X_full = torch.load('data/feature_matrix.pt', weights_only=True).numpy()
names = json.load(open('data/feature_names.json'))
aki_idx = names.index('aki_event')
y = X_full[aki_idx, :]
feat_mask = [i for i in range(len(names)) if i != aki_idx]
X = X_full[feat_mask, :].T
feat_names = [names[i] for i in feat_mask]

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
clf = LogisticRegressionCV(cv=5, max_iter=5000, scoring='roc_auc', random_state=42, penalty='l2')
clf.fit(X_sc, y)
coefs = clf.coef_[0]; ORs = np.exp(coefs)
probs = clf.predict_proba(X_sc)[:, 1]
V = probs * (1 - probs)
XtVX = (X_sc.T * V) @ X_sc + np.eye(X_sc.shape[1]) / clf.C_[0]
se = np.sqrt(np.diag(np.linalg.inv(XtVX)))
z = coefs / se; pvals = 2 * norm.sf(np.abs(z))

print(f"\n{'Feature':<40} {'Coef':>8} {'OR':>8} {'95% CI':>16} {'p':>10}")
print('-' * 90)
for i in np.argsort(pvals):
    ci_lo, ci_hi = np.exp(coefs[i] - 1.96*se[i]), np.exp(coefs[i] + 1.96*se[i])
    star = '*' if pvals[i] < 0.05 else ''
    print(f'{feat_names[i]:<40} {coefs[i]:>8.4f} {ORs[i]:>8.3f} ({ci_lo:.3f}-{ci_hi:.3f}) {pvals[i]:>10.4g} {star}')
print(f'\nIntercept={clf.intercept_[0]:.4f}  C={clf.C_[0]:.4f}  N={len(y)}  events={int(y.sum())}  rate={y.mean():.3f}')
PYEOF

    # ── Cox regression (c-index target >0.75) ─────────────────
    echo ""; echo "════ COX REGRESSION (c-index target >0.75) ════"
    python - <<'PYEOF' 2>&1 | tee "$LOGDIR/cox_${TS}.log"
import torch, json, numpy as np, pandas as pd, warnings, sys, subprocess
warnings.filterwarnings('ignore')
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lifelines', '-q'])
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

X_full = torch.load('data/feature_matrix.pt', weights_only=True).numpy()
names = json.load(open('data/feature_names.json'))
meta = pd.read_csv('data/cohort_meta.csv')
aki_idx = names.index('aki_event')
feat_mask = [i for i in range(len(names)) if i != aki_idx]
feat_names = [names[i] for i in feat_mask]
X = X_full[feat_mask, :].T
df = pd.DataFrame(X, columns=feat_names)
df['T'] = meta['surv_days'].values; df['E'] = meta['aki_event'].values

print('  ── Full Cox model ──')
cph = CoxPHFitter(penalizer=0.01)
cph.fit(df, duration_col='T', event_col='E')
cph.print_summary(columns=['coef', 'exp(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%'])
c_full = concordance_index(df['T'], -cph.predict_partial_hazard(df), df['E'])
print(f'\n  Full c-index: {c_full:.4f}')

print('\n  ── 5-fold CV ──')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cis = []
for fold, (tr, te) in enumerate(skf.split(X, df['E'].values)):
    c = CoxPHFitter(penalizer=0.01)
    c.fit(df.iloc[tr], duration_col='T', event_col='E')
    ci = concordance_index(df.iloc[te]['T'], -c.predict_partial_hazard(df.iloc[te]), df.iloc[te]['E'])
    cis.append(ci); print(f'    Fold {fold+1}: {ci:.4f}')
m = np.mean(cis)
print(f'\n  CV c-index: {m:.4f} +/- {np.std(cis):.4f}')
print(f'  Target >=0.75: {"PASS" if m >= 0.75 else "BELOW — review data/features"}')
PYEOF
}

# ═══════════════════════════════════════════════════════════════
run_one() {
    # run_one arch edge k lambda lr [extra_args...]
    local arch=$1 edge=$2 k=$3 lam=$4 lr=$5; shift 5
    local n="${arch}_${edge}_k${k}_lam${lam}_lr${lr}"
    echo "  → $n"
    python "$TRAIN" \
        --arch "$arch" --edge_method "$edge" --k "$k" \
        --lambda_ce "$lam" --lr "$lr" \
        --hidden "${HIDDEN:-64}" --latent "${LATENT:-16}" \
        --heads "${HEADS:-4}" --dropout "${DROPOUT:-0.2}" \
        --epochs 300 --cv_folds 5 --patience 30 \
        --wandb "$@" \
        2>&1 | tee "$LOGDIR/${n}_${TS}.log"
}

get_edges() {
    local edges=(spearman mi)
    [ -f data/graph_llm_k8.pt ] && edges+=(llm)
    echo "${edges[@]}"
}

best_edge() {
    [ -f data/graph_llm_k8.pt ] && echo "llm" || echo "mi"
}

# ═══════════════════════════════════════════════════════════════
do_grid() {
    echo "════ 3×3 GNN GRID ════"
    local edges; read -ra edges <<< "$(get_edges)"
    for arch in gatv2 gcn transformer; do
        for edge in "${edges[@]}"; do
            run_one "$arch" "$edge" 8 1.0 1e-3
        done
    done
}

# ═══════════════════════════════════════════════════════════════
do_sweep() {
    echo "════ COMPREHENSIVE SWEEPS ════"
    local be; be=$(best_edge)
    local edges; read -ra edges <<< "$(get_edges)"

    # Lambda × LR × edge (GATv2)
    echo "  ── Lambda × LR (GATv2, all edges) ──"
    for edge in "${edges[@]}"; do
        for lam in 0.5 1 2 5 10 20 50 100; do
            for lr in 1e-3 5e-4; do
                run_one gatv2 "$edge" 8 "$lam" "$lr"
            done
        done
    done

    # Hidden / latent (GATv2 × best edge)
    echo "  ── Hidden × latent (GATv2 × $be) ──"
    for h in 32 64 128 256; do
        for l in 8 16 32 64; do
            [ "$l" -gt "$h" ] && continue
            HIDDEN=$h LATENT=$l run_one gatv2 "$be" 8 1.0 1e-3
        done
    done

    # LR (GATv2 × best edge)
    echo "  ── LR (GATv2 × $be) ──"
    for lr in 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5; do
        run_one gatv2 "$be" 8 1.0 "$lr"
    done

    # Dropout
    echo "  ── Dropout (GATv2 × $be) ──"
    for d in 0.0 0.1 0.2 0.3 0.4 0.5; do
        DROPOUT=$d run_one gatv2 "$be" 8 1.0 1e-3
    done

    # k (rebuild graph each time)
    echo "  ── k (GATv2 × $be) ──"
    for k in 3 5 8 10 12 15 20; do
        if [ "$be" = "llm" ]; then
            python "$GRAPH" --method llm --k "$k" --llm_prior data/adj_llm.npy 2>/dev/null
        else
            python "$GRAPH" --method "$be" --k "$k" 2>/dev/null
        fi
        run_one gatv2 "$be" "$k" 1.0 1e-3
    done
    # restore k=8
    if [ "$be" = "llm" ]; then
        python "$GRAPH" --method llm --k 8 --llm_prior data/adj_llm.npy 2>/dev/null
    else
        python "$GRAPH" --method "$be" --k 8 2>/dev/null
    fi
}

# ═══════════════════════════════════════════════════════════════
do_tune() {
    echo "════ OPTUNA ════"
    local be; be=$(best_edge)
    echo "  Phase 1: GATv2 × $be (50 trials, 3-fold)"
    python "$TUNE" --phase 1 --arch gatv2 --edge_method "$be" --n_trials 50 --wandb \
        2>&1 | tee "$LOGDIR/tune_p1_${TS}.log"
    echo "  Phase 2: GATv2 × $be (100 trials, 5-fold)"
    python "$TUNE" --phase 2 --arch gatv2 --edge_method "$be" --n_trials 100 --wandb \
        2>&1 | tee "$LOGDIR/tune_p2_${TS}.log"
}

# ═══════════════════════════════════════════════════════════════
CMD="${1:-all}"
case "$CMD" in
    etl)       do_etl ;;
    graphs)    do_graphs ;;
    baselines) do_baselines ;;
    grid)      do_grid ;;
    sweep)     do_sweep ;;
    tune)      do_tune ;;
    all)       do_graphs; do_baselines; do_grid; do_sweep ;;
    *) echo "Usage: $0 {etl|graphs|baselines|grid|sweep|tune|all}"; exit 1 ;;
esac

echo ""; echo "════ DONE $TS — logs: $LOGDIR/ ════"
echo "  W&B: https://wandb.ai/hainingwang/graphai-ici-aki"
