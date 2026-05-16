#!/usr/bin/env python3
"""
10_baselines.py — Non-graph baselines on same CV folds.

XGBoost, Logistic Regression, Random Forest on identical features and
stratified folds for fair comparison with graph autoencoder.

Usage:
    python 10_baselines.py
    python 10_baselines.py --wandb
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def eval_metrics(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


def run_baselines(data_dir="data", cv_folds=5, wandb_run=None):
    X = torch.load(
        os.path.join(data_dir, "feature_matrix.pt"), weights_only=True
    ).numpy()  # [F, N]
    with open(os.path.join(data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    aki_idx = feature_names.index("aki_event")

    # Features = all nodes except AKI (the target)
    feat_idx = [i for i in range(X.shape[0]) if i != aki_idx]
    X_feat = X[feat_idx, :].T  # [N, F-1]
    y = X[aki_idx, :]  # [N]

    print(f"  Features: {X_feat.shape[1]}, Patients: {X_feat.shape[0]}")
    print(f"  AKI prevalence: {y.mean():.3f}")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

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

    for name, model_fn in models.items():
        print(f"\n  ── {name} ──")
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
            clf = model_fn()
            clf.fit(X_feat[train_idx], y[train_idx])
            probs = clf.predict_proba(X_feat[test_idx])[:, 1]
            metrics = eval_metrics(y[test_idx], probs)
            fold_metrics.append(metrics)
            print(
                f"    Fold {fold+1}: AUROC={metrics['auroc']:.4f} "
                f"AUPRC={metrics['auprc']:.4f}"
            )

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

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cv_folds", type=int, default=5)
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

    results = run_baselines(args.data_dir, args.cv_folds, wandb_run)

    os.makedirs("results", exist_ok=True)
    with open("results/baselines.json", "w") as f:
        json.dump(results, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
