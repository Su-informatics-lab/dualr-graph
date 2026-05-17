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

# ── 2. Full LogReg output (Dr. Su: coefs / ORs / p-values) ───
echo ""; echo "════ FULL LogReg ════"
python - <<'EOF' 2>&1 | tee "$LOGDIR/logreg_full_${TS}.log"
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
clf = LogisticRegressionCV(cv=5, max_iter=5000, scoring='roc_auc',
                           random_state=42, penalty='l2')
clf.fit(X_sc, y)
coefs = clf.coef_[0]; ORs = np.exp(coefs)
probs = clf.predict_proba(X_sc)[:, 1]; V = probs * (1 - probs)
XtVX = (X_sc.T * V) @ X_sc + np.eye(X_sc.shape[1]) / clf.C_[0]
se = np.sqrt(np.diag(np.linalg.inv(XtVX)))
z = coefs / se; pvals = 2 * norm.sf(np.abs(z))

print(f"\n{'Feature':<40} {'Coef':>8} {'OR':>8} {'95% CI':>16} {'p':>10}")
print('-' * 90)
for i in np.argsort(pvals):
    ci_lo, ci_hi = np.exp(coefs[i] - 1.96*se[i]), np.exp(coefs[i] + 1.96*se[i])
    star = '*' if pvals[i] < 0.05 else ''
    print(f'{feat_names[i]:<40} {coefs[i]:>8.4f} {ORs[i]:>8.3f} '
          f'({ci_lo:.3f}-{ci_hi:.3f}) {pvals[i]:>10.4g} {star}')
print(f'\nIntercept={clf.intercept_[0]:.4f}  C={clf.C_[0]:.4f}  '
      f'N={len(y)}  events={int(y.sum())}  rate={y.mean():.3f}')
EOF

# ── 3. Cox regression (c-index target ≥0.75 per Dr. Su) ──────
echo ""; echo "════ COX REGRESSION ════"
python - <<'EOF' 2>&1 | tee "$LOGDIR/cox_${TS}.log"
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
cph.print_summary(columns=['coef', 'exp(coef)', 'p',
                            'exp(coef) lower 95%', 'exp(coef) upper 95%'])
c_full = concordance_index(df['T'], -cph.predict_partial_hazard(df), df['E'])
print(f'\n  Full c-index: {c_full:.4f}')

print('\n  ── 5-fold CV ──')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cis = []
for fold, (tr, te) in enumerate(skf.split(X, df['E'].values)):
    c = CoxPHFitter(penalizer=0.01)
    c.fit(df.iloc[tr], duration_col='T', event_col='E')
    ci = concordance_index(df.iloc[te]['T'],
                           -c.predict_partial_hazard(df.iloc[te]),
                           df.iloc[te]['E'])
    cis.append(ci); print(f'    Fold {fold+1}: {ci:.4f}')
m = np.mean(cis)
print(f'\n  CV c-index: {m:.4f} +/- {np.std(cis):.4f}')
print(f'  Target >=0.75: {"PASS" if m >= 0.75 else "BELOW — review data/features"}')
EOF

echo ""; echo "════ DONE — logs: $LOGDIR/*_${TS}.log ════"
