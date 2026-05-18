#!/usr/bin/env python3
"""
results_table.py — Read all results/*.json and print a formatted table.

Usage:
    python results_table.py
"""

import glob
import json
import os


def main():
    results_dir = "results"
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))

    if not files:
        print("No results found in results/")
        return

    rows = []
    for fpath in files:
        name = os.path.basename(fpath).replace(".json", "")
        with open(fpath) as f:
            data = json.load(f)
        auroc = data.get("auroc_mean", float("nan"))
        auroc_std = data.get("auroc_std", float("nan"))
        auprc = data.get("auprc_mean", float("nan"))
        auprc_std = data.get("auprc_std", float("nan"))
        rows.append((name, auroc, auroc_std, auprc, auprc_std))

    # Sort by AUROC descending
    rows.sort(key=lambda r: -r[1] if r[1] == r[1] else 0)

    print()
    print(f"{'Run':<35s}  {'AUROC':>12s}  {'AUPRC':>12s}")
    print("─" * 65)
    for name, auroc, auroc_std, auprc, auprc_std in rows:
        print(f"{name:<35s}  {auroc:.4f}±{auroc_std:.4f}  {auprc:.4f}±{auprc_std:.4f}")
    print()


if __name__ == "__main__":
    main()
