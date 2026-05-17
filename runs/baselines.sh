#!/usr/bin/env bash
# runs/baselines.sh — Non-graph baselines + Cox regression
#
# Usage:
#   cd ~/dualr-graph && source .venv_cpu/bin/activate
#   bash runs/baselines.sh

set -euo pipefail
LOGDIR="logs"; mkdir -p "$LOGDIR" results
TS=$(date +%Y%m%d_%H%M%S)

# ── 1. Standard baselines (LogReg / RF / XGBoost) ────────────
echo "════ BASELINES ════"
python baselines.py --cv_folds 5 --wandb \
    2>&1 | tee "$LOGDIR/baselines_${TS}.log"

# ── 2. Full LogReg + Cox + feature trimming ───────────────────
echo ""; echo "════ LogReg + COX (full vs lean) ════"
python - <<'EOF' 2>&1 | tee "$LOGDIR/logreg_cox_${TS}.log"
import torch, json, numpy as np, pandas as pd, warnings, sys, subprocess
warnings.filterwarnings('ignore')

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lifelines', '-q'])
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm


# ── Load data ─────────────────────────────────────────────────
X_full = torch.load('data/feature_matrix.pt', weights_only=True).numpy()
names = json.load(open('data/feature_names.json'))
meta = pd.read_csv('data/cohort_meta.csv')

aki_idx = names.index('aki_event')
all_feat_idx = [i for i in range(len(names)) if i != aki_idx]
all_feat_names = [names[i] for i in all_feat_idx]
X_all = X_full[all_feat_idx, :].T
y = X_full[aki_idx, :]
y_time = meta['surv_days'].values
y_event = meta['aki_event'].values


# ── Feature sets ──────────────────────────────────────────────
# nci_cci_score is a weighted sum of the individual NCI-CCI flags.
# Including both → multicollinearity. Drop the score, keep flags.
DROP_LEAN = {'nci_cci_score'}

lean_mask = [i for i, n in enumerate(all_feat_names) if n not in DROP_LEAN]
lean_names = [all_feat_names[i] for i in lean_mask]
X_lean = X_all[:, lean_mask]

feature_sets = {
    'full (24 feat)': (X_all, all_feat_names),
    'lean (no cci_score)': (X_lean, lean_names),
}


def sig_stars(p):
    if p < 0.001:  return '***'
    if p < 0.01:   return '**'
    if p < 0.05:   return '*'
    if p < 0.1:    return '.'
    return ''


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
summary_rows = []

for label, (X_feat, feat_names) in feature_sets.items():

    # ══════════════════════════════════════════════════════════
    # LOGISTIC REGRESSION
    # ══════════════════════════════════════════════════════════
    print(f'\n{"="*70}')
    print(f'  LOGISTIC REGRESSION — {label}')
    print(f'{"="*70}')

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_feat)
    clf = LogisticRegressionCV(cv=5, max_iter=5000, scoring='roc_auc',
                               random_state=42)
    clf.fit(X_sc, y)
    coefs = clf.coef_[0]; ORs = np.exp(coefs)
    probs = clf.predict_proba(X_sc)[:, 1]; V = probs * (1 - probs)
    XtVX = (X_sc.T * V) @ X_sc + np.eye(X_sc.shape[1]) / clf.C_[0]
    se = np.sqrt(np.diag(np.linalg.inv(XtVX)))
    z = coefs / se; pvals = 2 * norm.sf(np.abs(z))

    print(f'\n  {"Feature":<35} {"Coef":>8} {"OR":>8} {"95% CI":>16} {"p":>10}')
    print(f'  {"-"*85}')
    for i in np.argsort(pvals):
        ci_lo = np.exp(coefs[i] - 1.96*se[i])
        ci_hi = np.exp(coefs[i] + 1.96*se[i])
        stars = sig_stars(pvals[i])
        print(f'  {feat_names[i]:<35} {coefs[i]:>8.4f} {ORs[i]:>8.3f} '
              f'({ci_lo:.3f}-{ci_hi:.3f}) {pvals[i]:>10.4g} {stars}')

    # CV AUROC
    aurocs_lr = []
    for tr, te in skf.split(X_feat, y):
        X_tr = scaler.fit_transform(X_feat[tr])
        X_te = scaler.transform(X_feat[te])
        lr = LogisticRegression(C=clf.C_[0], max_iter=5000)
        lr.fit(X_tr, y[tr])
        aurocs_lr.append(roc_auc_score(y[te], lr.predict_proba(X_te)[:, 1]))
    lr_mean = np.mean(aurocs_lr)
    lr_std = np.std(aurocs_lr)
    print(f'\n  CV AUROC: {lr_mean:.4f} ± {lr_std:.4f}')

    # ══════════════════════════════════════════════════════════
    # COX REGRESSION
    # ══════════════════════════════════════════════════════════
    print(f'\n{"="*70}')
    print(f'  COX REGRESSION — {label}')
    print(f'{"="*70}')

    df = pd.DataFrame(X_feat, columns=feat_names)
    df['T'] = y_time; df['E'] = y_event

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col='T', event_col='E')

    # Format output with significance symbols
    s = cph.summary.copy()
    print(f'\n  {"Covariate":<35} {"HR":>7} {"95% CI":>16} {"p":>10}')
    print(f'  {"-"*75}')
    for name, row in s.sort_values('p').iterrows():
        hr = row['exp(coef)']
        lo = row['exp(coef) lower 95%']
        hi = row['exp(coef) upper 95%']
        p = row['p']
        stars = sig_stars(p)
        print(f'  {name:<35} {hr:>7.3f} ({lo:.3f}-{hi:.3f}) {p:>10.4g} {stars}')

    c_full = concordance_index(df['T'], -cph.predict_partial_hazard(df), df['E'])
    print(f'\n  Full c-index: {c_full:.4f}')

    # 5-fold CV
    cis = []
    for fold, (tr, te) in enumerate(skf.split(X_feat, y_event)):
        c = CoxPHFitter(penalizer=0.01)
        c.fit(df.iloc[tr], duration_col='T', event_col='E')
        ci = concordance_index(df.iloc[te]['T'],
                               -c.predict_partial_hazard(df.iloc[te]),
                               df.iloc[te]['E'])
        cis.append(ci)
        print(f'    Fold {fold+1}: {ci:.4f}')

    cox_mean = np.mean(cis)
    cox_std = np.std(cis)
    print(f'\n  CV c-index: {cox_mean:.4f} ± {cox_std:.4f}')
    print(f'  Target >=0.75: {"PASS" if cox_mean >= 0.75 else "BELOW"}')

    summary_rows.append({
        'features': label,
        'n_feat': len(feat_names),
        'lr_auroc': f'{lr_mean:.4f}±{lr_std:.4f}',
        'cox_ci': f'{cox_mean:.4f}±{cox_std:.4f}',
    })


# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'  COMPARISON')
print(f'{"="*70}')
print(f'  {"Feature set":<25} {"#":>3} {"LogReg AUROC":>15} {"Cox c-index":>15}')
print(f'  {"-"*62}')
for r in summary_rows:
    print(f'  {r["features"]:<25} {r["n_feat"]:>3} {r["lr_auroc"]:>15} {r["cox_ci"]:>15}')
print()
EOF

echo ""; echo "════ DONE — logs: $LOGDIR/*_${TS}.log ════"
