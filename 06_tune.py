#!/usr/bin/env python3
"""
06_tune.py — Hyperparameter tuning with Optuna + W&B.

Phase 1: RandomSampler, 3-fold CV, 100 trials (broad sweep)
Phase 2: TPESampler, 5-fold CV, 200 trials (focused, best arch)

Usage:
    python 06_tune.py --phase 1 --arch transformer --edge_method spearman
    python 06_tune.py --phase 2 --arch transformer --edge_method spearman
"""

import argparse
import json
import os

import optuna

# Import will be resolved at runtime
# from 05_train import run_cv


def create_objective(args):
    """Factory: returns an Optuna objective function."""

    cv_folds = 3 if args.phase == 1 else 5

    def objective(trial):
        config = {
            "arch": args.arch,
            "edge_method": args.edge_method,
            "k": trial.suggest_categorical("k", [5, 8, 10, 15]),
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
            "latent": trial.suggest_categorical("latent", [8, 16, 32, 64]),
            "heads": trial.suggest_categorical("heads", [1, 2, 4, 8]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lambda_ce": trial.suggest_float("lambda_ce", 0.1, 10.0, log=True),
            "epochs": trial.suggest_categorical("epochs", [200, 300, 500]),
            "cv_folds": cv_folds,
            "patience": 30,
            "eval_every": 10,
            "grad_clip": 5.0,
        }

        # W&B run per trial (optional)
        wandb_run = None
        if args.wandb:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                config=config,
                group=f"tune_p{args.phase}_{args.arch}_{args.edge_method}",
                tags=["npj-pilot", f"phase{args.phase}", args.arch],
                reinit=True,
            )

        try:
            summary = run_cv(config, data_dir=args.data_dir, wandb_run=wandb_run)
            auroc = summary["auroc_mean"]
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            auroc = 0.0
        finally:
            if wandb_run is not None:
                wandb_run.finish(quiet=True)

        # Optuna pruning: report per fold if available
        if "folds" in summary:
            for i, fm in enumerate(summary["folds"]):
                trial.report(fm.get("auroc", 0.0), i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return auroc

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument(
        "--arch",
        type=str,
        default="transformer",
        choices=["transformer", "gatv2", "gcn"],
    )
    parser.add_argument(
        "--edge_method", type=str, default="spearman", choices=["spearman", "mi", "llm"]
    )
    parser.add_argument(
        "--n_trials", type=int, default=None, help="Override default trial count"
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/tune")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="graphai-ici-aki")
    args = parser.parse_args()

    n_trials = args.n_trials or (100 if args.phase == 1 else 200)

    print("=" * 70)
    print(f"GRAPH AI — Optuna Phase {args.phase}")
    print(f"  Arch: {args.arch}, Edges: {args.edge_method}")
    print(f"  Trials: {n_trials}")
    print("=" * 70)

    if args.phase == 1:
        sampler = optuna.samplers.RandomSampler(seed=42)
        pruner = optuna.pruners.NopPruner()
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"graphai_p{args.phase}_{args.arch}_{args.edge_method}",
    )

    objective = create_objective(args)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir, f"phase{args.phase}_{args.arch}_{args.edge_method}.json"
    )
    results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "phase": args.phase,
        "arch": args.arch,
        "edge_method": args.edge_method,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Best AUROC: {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(study.best_params, indent=4)}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    # Runtime import resolution for numbered modules
    train_mod = __import__("05_train")
    run_cv = train_mod.run_cv
    main()
