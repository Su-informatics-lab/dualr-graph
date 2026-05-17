#!/usr/bin/env python3
"""
analysis.py — Cox guide + 6-month case-control + subgroup breakdown.

Usage:
  python analysis.py | tee logs/analysis.log
"""

import json
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lifelines", "-q"])
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


# ── Load data ─────────────────────────────────────────────────
X_full = torch.load("data/feature_matrix.pt", weights_only=True).numpy()
names = json.load(open("data/feature_names.json"))
meta = pd.read_csv("data/cohort_meta.csv")

aki_idx = names.index("aki_event")
feat_mask = [i for i in range(len(names)) if i != aki_idx]
feat_names = [names[i] for i in feat_mask]
X = X_full[feat_mask, :].T
y_event = meta["aki_event"].values
y_time = meta["surv_days"].values
N = len(meta)

print("=" * 70)
print("ANALYSIS: Cox guide + 6-month case-control + subgroup breakdown")
print(f"  N={N}, features={len(feat_names)}, AKI rate={y_event.mean():.3f}")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ═══════════════════════════════════════════════════════════════
# 1. COX ON FULL SURVIVAL DATA (reference)
# ═══════════════════════════════════════════════════════════════
print("\n── 1. COX (full survival, N=%d) ──" % N)
df_cox = pd.DataFrame(X, columns=feat_names)
df_cox["T"] = y_time
df_cox["E"] = y_event

cph = CoxPHFitter(penalizer=0.01)
cph.fit(df_cox, duration_col="T", event_col="E")
s = cph.summary.copy()
c_full = concordance_index(
    df_cox["T"], -cph.predict_partial_hazard(df_cox), df_cox["E"]
)

print(f"\n  {'Covariate':<30s} {'HR':>7} {'95% CI':>16} {'p':>10}")
print(f"  {'-'*70}")
for name, row in s.sort_values("p").iterrows():
    hr = row["exp(coef)"]
    lo = row["exp(coef) lower 95%"]
    hi = row["exp(coef) upper 95%"]
    print(
        f"  {name:<30s} {hr:>7.3f} ({lo:.3f}-{hi:.3f}) {row['p']:>10.4g} {sig(row['p'])}"
    )

cis_cox = []
for tr, te in skf.split(X, y_event):
    c = CoxPHFitter(penalizer=0.01)
    c.fit(df_cox.iloc[tr], duration_col="T", event_col="E")
    cis_cox.append(
        concordance_index(
            df_cox.iloc[te]["T"],
            -c.predict_partial_hazard(df_cox.iloc[te]),
            df_cox.iloc[te]["E"],
        )
    )
print(f"\n  Full c-index: {c_full:.4f}")
print(f"  CV c-index:   {np.mean(cis_cox):.4f} ± {np.std(cis_cox):.4f}")

# ═══════════════════════════════════════════════════════════════
# 2. 6-MONTH CASE-CONTROL CONVERSION
# ═══════════════════════════════════════════════════════════════
print("\n── 2. 6-MONTH LANDMARK CASE-CONTROL ──")

# Inclusion: patients with follow-up ≥ 180 days OR event before 180 days
# Case: AKI event within 180 days
# Control: no AKI event AND follow-up ≥ 180 days
landmark = 180

has_event_before = (y_event == 1) & (y_time <= landmark)
has_fu_past = y_time >= landmark
eligible_6m = has_event_before | has_fu_past

X_6m = X[eligible_6m]
y_6m = has_event_before[eligible_6m].astype(int)
meta_6m = meta[eligible_6m].reset_index(drop=True)

n_6m = len(y_6m)
n_case = y_6m.sum()
n_ctrl = n_6m - n_case
print(f"  Full cohort: {N}")
print(f"  Excluded (follow-up < {landmark}d, no event): {N - n_6m}")
print(
    f"  6-month eligible: {n_6m} (cases={n_case}, controls={n_ctrl}, rate={y_6m.mean():.3f})"
)

# ═══════════════════════════════════════════════════════════════
# 3. BASELINES ON 6-MONTH CASE-CONTROL
# ═══════════════════════════════════════════════════════════════
print("\n── 3. BASELINES (6-month case-control) ──")

skf6 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {"LogReg": lambda: LogisticRegression(C=1.0, max_iter=5000)}
if HAS_XGB:
    models["XGBoost"] = lambda: xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        scale_pos_weight=(y_6m == 0).sum() / max((y_6m == 1).sum(), 1),
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
        random_state=42,
    )

results_6m = {}
for mname, mfn in models.items():
    aurocs = []
    for tr, te in skf6.split(X_6m, y_6m):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_6m[tr])
        Xte = scaler.transform(X_6m[te])
        clf = mfn()
        clf.fit(Xtr, y_6m[tr])
        aurocs.append(roc_auc_score(y_6m[te], clf.predict_proba(Xte)[:, 1]))
    m = np.mean(aurocs)
    s_ = np.std(aurocs)
    results_6m[mname] = (m, s_)
    print(f"  {mname:10s}  AUROC = {m:.4f} ± {s_:.4f}")

# Also run Cox on 6-month data
df_6m = pd.DataFrame(X_6m, columns=feat_names)
df_6m["T"] = meta_6m["surv_days"].clip(upper=landmark).values
df_6m["E"] = y_6m
cis_6m = []
for tr, te in skf6.split(X_6m, y_6m):
    c = CoxPHFitter(penalizer=0.01)
    c.fit(df_6m.iloc[tr], duration_col="T", event_col="E")
    cis_6m.append(
        concordance_index(
            df_6m.iloc[te]["T"],
            -c.predict_partial_hazard(df_6m.iloc[te]),
            df_6m.iloc[te]["E"],
        )
    )
print(f"  {'Cox (6m)':10s}  c-idx = {np.mean(cis_6m):.4f} ± {np.std(cis_6m):.4f}")

# ═══════════════════════════════════════════════════════════════
# 4. SUBGROUP PERFORMANCE BREAKDOWN
# ═══════════════════════════════════════════════════════════════
print("\n── 4. SUBGROUP PERFORMANCE BREAKDOWN ──")
print("  (Cox c-index per subgroup on full survival data)")
print("  Subgroups with poor performance → exclusion candidates")


# Reconstruct subgroup labels from feature matrix
# Binary features: find column index
def get_col(name):
    try:
        return feat_names.index(name)
    except ValueError:
        return None


subgroups = {}

# Cancer type (from feature matrix flags)
i_mel = get_col("cancer_Melanoma")
i_rc = get_col("cancer_Renal_Cell")
i_oth = get_col("cancer_Other")
i_mc = get_col("cancer_multi_cancer")
if i_mel is not None and i_rc is not None and i_oth is not None:
    cancer_labels = np.array(["Lung"] * N)
    cancer_labels[X[:, i_mel] == 1] = "Melanoma"
    cancer_labels[X[:, i_rc] == 1] = "Renal_Cell"
    cancer_labels[X[:, i_oth] == 1] = "Other"
    if i_mc is not None:
        cancer_labels[X[:, i_mc] == 1] = "Multi_cancer"
    subgroups["Cancer type"] = cancer_labels

# ICI regimen
i_pdl1 = get_col("ici_pdl1_mono")
i_combo = get_col("ici_ctla4")
if i_combo is None:
    i_combo = get_col("ici_combo")  # backward compat
if i_pdl1 is not None and i_combo is not None:
    ici_labels = np.array(["PD1_mono"] * N)
    ici_labels[X[:, i_pdl1] == 1] = "PDL1_mono"
    ici_labels[X[:, i_combo] == 1] = "CTLA4"
    subgroups["ICI regimen"] = ici_labels

# Race
i_black = get_col("race_black")
if i_black is not None:
    race_labels = np.where(X[:, i_black] == 1, "Black", "Non-Black")
    subgroups["Race"] = race_labels

# Sex
i_sex = get_col("sex_male")
if i_sex is not None:
    sex_labels = np.where(X[:, i_sex] == 1, "Male", "Female")
    subgroups["Sex"] = sex_labels

# Age (tertiles from standardized)
i_age = get_col("age")
if i_age is not None:
    age_vals = X[:, i_age]
    age_terts = np.percentile(age_vals, [33, 67])
    age_labels = np.where(
        age_vals <= age_terts[0],
        "Young",
        np.where(age_vals <= age_terts[1], "Middle", "Old"),
    )
    subgroups["Age tertile"] = age_labels

# Lab availability — use missing_mask.pt if available
try:
    miss_mask = torch.load("data/missing_mask.pt", weights_only=True).numpy()  # [F, N]
    # Any lab missing → "missing_labs"
    lab_feat_idx = [
        feat_names.index(f"baseline_{l}")
        for l in ["bun", "hgb", "alb", "k"]
        if f"baseline_{l}" in feat_names
    ]
    if lab_feat_idx:
        # missing_mask has F+1 rows (includes aki_event), feat_mask remapped
        any_lab_missing = np.zeros(N, dtype=bool)
        for fi in lab_feat_idx:
            # feat_mask[fi] is the original row in feature_matrix
            orig_row = feat_mask[fi]
            any_lab_missing |= miss_mask[orig_row, :]
        lab_labels = np.where(any_lab_missing, "missing_labs", "has_labs")
        subgroups["Labs available"] = lab_labels
except FileNotFoundError:
    pass  # no missing mask → skip this subgroup

# Nephrotoxin subgroups
i_ppi = get_col("nephro_ppi")
if i_ppi is not None:
    ppi_labels = np.where(X[:, i_ppi] == 1, "PPI_yes", "PPI_no")
    subgroups["PPI exposure"] = ppi_labels

# Run per-subgroup Cox c-index
print(
    f"\n  {'Subgroup':<20s} {'Level':<15s} {'N':>6} {'Events':>7} {'Rate':>7} {'c-index':>8}"
)
print(f"  {'-'*70}")

for sg_name, sg_labels in subgroups.items():
    for level in sorted(set(sg_labels)):
        mask = sg_labels == level
        n_sub = mask.sum()
        if n_sub < 50:
            continue  # skip tiny groups
        e_sub = y_event[mask].sum()
        rate = y_event[mask].mean()

        # Cox on subgroup — drop constant columns (zero variance)
        df_sub = df_cox[mask].copy()
        n_dropped = 0
        if len(df_sub) < 50 or e_sub < 10:
            ci = float("nan")
        else:
            try:
                # Drop features that are constant in this subgroup
                feat_cols = [c for c in df_sub.columns if c not in ("T", "E")]
                const_cols = [c for c in feat_cols if df_sub[c].nunique() <= 1]
                n_dropped = len(const_cols)
                df_fit = df_sub.drop(columns=const_cols)

                if len([c for c in df_fit.columns if c not in ("T", "E")]) == 0:
                    ci = float("nan")
                else:
                    c_sub = CoxPHFitter(penalizer=0.1)
                    c_sub.fit(df_fit, duration_col="T", event_col="E")
                    ci = concordance_index(
                        df_fit["T"], -c_sub.predict_partial_hazard(df_fit), df_fit["E"]
                    )
            except Exception:
                ci = float("nan")

        dropped = f" (dropped {n_dropped})" if n_dropped > 0 else ""
        flag = " ← LOW" if ci < 0.60 else ""
        print(
            f"  {sg_name:<20s} {level:<15s} {n_sub:>6} {int(e_sub):>7} {rate:>7.3f} {ci:>8.4f}{flag}{dropped}"
        )

# ═══════════════════════════════════════════════════════════════
# 5. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  {'Model':<15s} {'Data':<20s} {'Metric':>10} {'Value':>15}")
print(f"  {'-'*65}")
print(
    f"  {'Cox':<15s} {'full survival':<20s} {'c-index':>10} {np.mean(cis_cox):.4f}±{np.std(cis_cox):.4f}"
)
print(
    f"  {'Cox':<15s} {'6-month landmark':<20s} {'c-index':>10} {np.mean(cis_6m):.4f}±{np.std(cis_6m):.4f}"
)
for mname, (m, s_) in results_6m.items():
    print(f"  {mname:<15s} {'6-month case-ctrl':<20s} {'AUROC':>10} {m:.4f}±{s_:.4f}")
print(f"\n  Target: GNN on 6-month case-control → AUROC ≥ 0.75")
print(f"  Subgroups with c-index < 0.60 are exclusion candidates")
