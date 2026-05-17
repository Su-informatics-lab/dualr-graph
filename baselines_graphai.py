#!/usr/bin/env python3
"""
baselines.py — Non-graph baselines on same CV folds.

XGBoost, Logistic Regression (with ORs), Random Forest, and
CoxPH (c-index ceiling test, per Su: >0.75 → data OK).

Usage:
    python baselines.py
    python baselines.py --wandb
    python baselines.py --cox    # also run CoxPH (needs surv_days+evt)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


def eval_metrics(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


def run_baselines(data_dir="data", cv_folds=5, wandb_run=None, run_cox=False):
    X = torch.load(
        os.path.join(data_dir, "feature_matrix.pt"), weights_only=True
    ).numpy()  # [F, N]
    with open(os.path.join(data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    aki_idx = feature_names.index("aki_event")

    # Features = all nodes except AKI (the target)
    feat_idx = [i for i in range(X.shape[0]) if i != aki_idx]
    feat_names = [feature_names[i] for i in feat_idx]
    X_feat = X[feat_idx, :].T  # [N, F-1]
    y = X[aki_idx, :]  # [N]

    print(f"  Features: {X_feat.shape[1]}, Patients: {X_feat.shape[0]}")
    print(f"  AKI prevalence: {y.mean():.3f} ({int(y.sum())}/{len(y)})")

    # Load survival data for Cox
    meta_path = os.path.join(data_dir, "cohort_meta.csv")
    has_surv = os.path.exists(meta_path) and run_cox and HAS_LIFELINES
    if has_surv:
        meta = pd.read_csv(meta_path)
        surv_days = meta["surv_days"].values
        surv_evt = meta["aki_event"].values
        print(f"  Survival data: {len(meta)} patients, {int(surv_evt.sum())} events")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # ── Classification baselines ──────────────────────────────────
    models = {}
    if HAS_XGB:
        models["xgboost"] = lambda: xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=3,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
    models["logreg_l2"] = lambda: LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
    )
    models["random_forest"] = lambda: RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
    )

    all_results = {}
    logreg_coefs_all = []

    for name, model_fn in models.items():
        print(f"\n  ── {name} ──")
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
            # Scale for LogReg
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_feat[train_idx])
            X_te = scaler.transform(X_feat[test_idx])

            clf = model_fn()
            clf.fit(X_tr, y[train_idx])
            probs = clf.predict_proba(X_te)[:, 1]
            metrics = eval_metrics(y[test_idx], probs)
            fold_metrics.append(metrics)
            print(
                f"    Fold {fold+1}: AUROC={metrics['auroc']:.4f} "
                f"AUPRC={metrics['auprc']:.4f}"
            )

            # Collect LogReg coefficients
            if name == "logreg_l2" and hasattr(clf, "coef_"):
                logreg_coefs_all.append(clf.coef_[0])

        aurocs = [m["auroc"] for m in fold_metrics]
        auprcs = [m["auprc"] for m in fold_metrics]
        summary = {
            "auroc_mean": np.mean(aurocs),
            "auroc_std": np.std(aurocs),
            "auprc_mean": np.mean(auprcs),
            "auprc_std": np.std(auprcs),
        }
        print(
            f"    Mean: AUROC={summary['auroc_mean']:.4f}±{summary['auroc_std']:.4f} "
            f"AUPRC={summary['auprc_mean']:.4f}±{summary['auprc_std']:.4f}"
        )
        all_results[name] = summary

        if wandb_run is not None:
            for k, v in summary.items():
                wandb_run.summary[f"baseline_{name}_{k}"] = v

    # ── LogReg coefficients / ORs (averaged across folds) ─────────
    if logreg_coefs_all:
        avg_coef = np.mean(logreg_coefs_all, axis=0)
        ors = np.exp(avg_coef)
        coef_df = pd.DataFrame(
            {
                "feature": feat_names,
                "coef_mean": avg_coef,
                "OR": ors,
                "abs_coef": np.abs(avg_coef),
            }
        ).sort_values("abs_coef", ascending=False)
        coef_df.to_csv("results/logreg_coefficients.csv", index=False)
        print(f"\n  ── LogReg Coefficients (top 15) ──")
        for _, r in coef_df.head(15).iterrows():
            direction = "↑" if r["coef_mean"] > 0 else "↓"
            print(f"    {r['feature']:40s} OR={r['OR']:.3f} {direction}")

    # ── CoxPH baseline (c-index ceiling) ──────────────────────────
    if has_surv:
        print(f"\n  ── CoxPH (c-index ceiling test) ──")
        cox_cindices = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_feat[train_idx])
            X_te = scaler.transform(X_feat[test_idx])

            train_df = pd.DataFrame(X_tr, columns=feat_names)
            train_df["T"] = surv_days[train_idx]
            train_df["E"] = surv_evt[train_idx]

            test_df = pd.DataFrame(X_te, columns=feat_names)
            test_df["T"] = surv_days[test_idx]
            test_df["E"] = surv_evt[test_idx]

            cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
            try:
                cph.fit(train_df, duration_col="T", event_col="E", show_progress=False)
                partial_haz = cph.predict_partial_hazard(test_df)
                ci = concordance_index(
                    test_df["T"], -partial_haz.values.ravel(), test_df["E"]
                )
                cox_cindices.append(ci)
                print(f"    Fold {fold+1}: c-index={ci:.4f}")
            except Exception as e:
                print(f"    Fold {fold+1}: FAILED — {e}")

        if cox_cindices:
            ci_mean = np.mean(cox_cindices)
            ci_std = np.std(cox_cindices)
            print(f"    Mean: c-index={ci_mean:.4f}±{ci_std:.4f}")
            all_results["coxph"] = {"c_index_mean": ci_mean, "c_index_std": ci_std}
            if ci_mean >= 0.75:
                print(f"    ✅ c-index ≥0.75 — data quality OK for downstream GNN")
            else:
                print(f"    ⚠ c-index <0.75 — data may lack key predictive features")

            if wandb_run is not None:
                wandb_run.summary["baseline_coxph_cindex_mean"] = ci_mean
                wandb_run.summary["baseline_coxph_cindex_std"] = ci_std

            # Save Cox coefficients from full-data fit
            full_df = pd.DataFrame(
                StandardScaler().fit_transform(X_feat), columns=feat_names
            )
            full_df["T"] = surv_days
            full_df["E"] = surv_evt
            cph_full = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
            try:
                cph_full.fit(
                    full_df, duration_col="T", event_col="E", show_progress=False
                )
                cox_summary = cph_full.summary[["coef", "exp(coef)", "p"]].copy()
                cox_summary.columns = ["coef", "HR", "p"]
                cox_summary = cox_summary.sort_values("p")
                cox_summary.to_csv("results/cox_coefficients.csv")
                print(f"\n  ── CoxPH significant features (p<0.05) ──")
                sig = cox_summary[cox_summary.p < 0.05]
                for feat, r in sig.head(15).iterrows():
                    print(f"    {feat:40s} HR={r['HR']:.3f} p={r['p']:.4f}")
            except Exception as e:
                print(f"    Full-data CoxPH failed: {e}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument(
        "--cox", action="store_true", help="Run CoxPH c-index ceiling test"
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="graphai-ici-aki")
    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
            group="baselines",
            tags=["npj-pilot", "baseline"],
        )

    print("=" * 70)
    print("GRAPH AI — Non-Graph Baselines")
    print("=" * 70)

    results = run_baselines(args.data_dir, args.cv_folds, wandb_run, args.cox)

    os.makedirs("results", exist_ok=True)
    with open("results/baselines.json", "w") as f:
        json.dump(results, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
