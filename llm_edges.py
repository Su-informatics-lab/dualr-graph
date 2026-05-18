#!/usr/bin/env python3
"""
llm_edges.py — LLM-estimated directed feature graph (v7).

Dr. Su's design:
  - Directed graph: edge weight from B→A = P(A | B)
  - Feature→Feature edges: P(feature_a | feature_b) for all directed pairs
  - Feature→AKI edges: P(AKI | feature_a) for each feature
  - AKI is a SINK node: receives edges, sends none
  - Uncertainty from n_reps → optional confidence-based cutoff

Output:
  data/llm_adj.npy        — [F, F] directed adjacency, adj[i,j] = P(node_i | node_j)
  data/llm_adj_std.npy    — [F, F] std across reps (for uncertainty-based cutoff)
  data/llm_edges_raw.json — all raw estimates for debugging

Usage:
    python llm_edges.py --dry_run                    # verify prompts
    python llm_edges.py --n_reps 10 --temperature 0.6  # full run
    python llm_edges.py --resume                     # resume from checkpoint
"""

import argparse
import json
import os
import re
import time
from typing import Dict

import numpy as np

# ══════════════════════════════════════════════════════════════════
# Feature descriptions
# ══════════════════════════════════════════════════════════════════
# Binary: presence of condition. Continuous: clinically abnormal direction.

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "age": "older age (≥65 years)",
    "sex_male": "male sex",
    "race_black": "Black race",
    "ethnicity_hispanic": "Hispanic ethnicity",
    "Acute_MI": "acute myocardial infarction",
    "History_MI": "history of myocardial infarction",
    "CHF": "congestive heart failure",
    "PVD": "peripheral vascular disease",
    "CVD": "cerebrovascular disease",
    "COPD": "chronic obstructive pulmonary disease",
    "Dementia": "dementia",
    "Paralysis": "hemiplegia or paraplegia",
    "Diabetes": "diabetes mellitus without complications",
    "Diabetes_Complicated": "diabetes with end-organ complications",
    "Renal_Disease": "pre-existing chronic kidney disease",
    "Liver_Disease_Mild": "mild liver disease",
    "Liver_Disease_Moderate_Severe": "moderate-to-severe liver disease",
    "Peptic_Ulcer_Disease": "peptic ulcer disease",
    "Rheumatic_Disease": "rheumatic or autoimmune disease",
    "AIDS": "HIV/AIDS",
    "ici_pdl1_mono": "anti-PD-L1 monotherapy (atezolizumab, durvalumab)",
    "ici_ctla4": "CTLA-4 combination immunotherapy (ipilimumab + anti-PD-1)",
    "cancer_Melanoma": "melanoma",
    "cancer_Renal_Cell": "renal cell carcinoma",
    "cancer_multi_cancer": "multiple primary cancer types",
    "cancer_Other": "cancer type other than lung, melanoma, or renal cell",
    "metastatic": "metastatic disease",
    "nephro_ppi": "proton pump inhibitor use",
    "nephro_nsaid": "NSAID use",
    "nephro_vanc": "vancomycin use",
    "nephro_acei_arb": "ACE inhibitor or ARB use",
    "nephro_diuretic": "loop diuretic use",
    "baseline_bun": "elevated blood urea nitrogen",
    "baseline_hgb": "low hemoglobin (anemia)",
    "baseline_alb": "low serum albumin",
    "baseline_k": "abnormal serum potassium",
    "bun_cr_ratio": "elevated BUN-to-creatinine ratio",
    "baseline_egfr": "low estimated GFR (impaired kidney function)",
    "aki_event": "acute kidney injury",
    # Legacy
    "nephrotoxin": "nephrotoxic drug exposure",
    "nci_cci_score": "high comorbidity burden",
    "ici_pd1": "anti-PD-1 monotherapy",
    "ici_combo": "CTLA-4 combination immunotherapy",
}


# ══════════════════════════════════════════════════════════════════
# Prompt templates
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a clinical epidemiologist. For cancer patients treated with "
    "immune checkpoint inhibitors, estimate conditional probabilities "
    "between clinical features based on published evidence and clinical "
    "knowledge. Respond with a single number between 0.0 and 1.0 in "
    "[ESTIMATION]...[/ESTIMATION] tags."
)

# P(feature_a | feature_b) — directed edge from b → a
TEMPLATE_FEAT_FEAT = (
    "Among cancer patients receiving immune checkpoint inhibitor therapy:\n\n"
    "Given that a patient has: {condition_b}\n\n"
    "What is the probability that this patient ALSO has: {condition_a}?\n\n"
    "Estimate P({short_a} | {short_b}) as a single number in "
    "[ESTIMATION]...[/ESTIMATION] tags."
)

# P(AKI | feature_a) — directed edge from a → AKI
TEMPLATE_FEAT_AKI = (
    "Among cancer patients receiving immune checkpoint inhibitor therapy:\n\n"
    "Given that a patient has: {condition_a}\n\n"
    "What is the probability that this patient develops acute kidney injury "
    "(serum creatinine ≥1.5× baseline) within 6 months of starting therapy?\n\n"
    "Estimate P(AKI | {short_a}) as a single number in "
    "[ESTIMATION]...[/ESTIMATION] tags."
)


def get_desc(name):
    return FEATURE_DESCRIPTIONS.get(name, name.replace("_", " "))


def parse_estimation(text):
    m = re.search(r"\[ESTIMATION\]\s*([\d.]+)\s*\[/ESTIMATION\]", text)
    if m:
        val = float(m.group(1))
        if 0.0 <= val <= 1.0:
            return val
    for f in re.findall(r"\b(0\.\d+|1\.0|0\.0)\b", text):
        val = float(f)
        if 0.0 <= val <= 1.0:
            return val
    return None


def build_prompt(qtype, feat_a, feat_b=None):
    if qtype == "feat_aki":
        return TEMPLATE_FEAT_AKI.format(
            condition_a=get_desc(feat_a),
            short_a=feat_a,
        )
    return TEMPLATE_FEAT_FEAT.format(
        condition_a=get_desc(feat_a),
        condition_b=get_desc(feat_b),
        short_a=feat_a,
        short_b=feat_b,
    )


def query_llm_vllm(prompts, model_name, temperature=0.6, max_tokens=256):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, dtype="auto", trust_remote_code=True)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    convos = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p},
        ]
        for p in prompts
    ]
    outputs = llm.chat(convos, sampling_params=params)
    return [o.outputs[0].text for o in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_reps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    F = len(feature_names)
    aki_idx = feature_names.index("aki_event")
    non_aki = [i for i in range(F) if i != aki_idx]
    n_feat = len(non_aki)

    n_ff = n_feat * (n_feat - 1)  # directed: P(a|b) ≠ P(b|a)
    n_fa = n_feat  # P(AKI|a) for each feature
    n_total = n_ff + n_fa

    print("=" * 70)
    print(f"LLM Directed Edge Estimation — {F} nodes ({n_feat} features + AKI)")
    print(f"  P(a|b) feature→feature: {n_ff:,} directed queries")
    print(f"  P(AKI|a) feature→AKI:   {n_fa} queries")
    print(f"  Total unique:            {n_total:,}")
    print(f"  × {args.n_reps} reps =   {n_total * args.n_reps:,} LLM calls")
    print(f"  Model: {args.model}")
    print("=" * 70)

    # ── Build query list ──────────────────────────────────────
    queries = []
    for i in non_aki:
        for j in non_aki:
            if i == j:
                continue
            queries.append(
                (
                    "ff",
                    i,
                    j,
                    build_prompt("feat_feat", feature_names[i], feature_names[j]),
                )
            )
    for i in non_aki:
        queries.append(("fa", aki_idx, i, build_prompt("feat_aki", feature_names[i])))

    assert len(queries) == n_total

    # ── Dry run ───────────────────────────────────────────────
    if args.dry_run:
        print("\n── FEATURE DESCRIPTIONS ──\n")
        for name in feature_names:
            print(f"  {name:35s} → {get_desc(name)}")

        print(f"\n── SAMPLE PROMPTS ──\n")
        ff = [q for q in queries if q[0] == "ff"]
        fa = [q for q in queries if q[0] == "fa"]
        for q in ff[:3] + fa[:3]:
            src, tgt = feature_names[q[2]], feature_names[q[1]]
            print(f"  P({tgt} | {src}):")
            print(f"    {q[3][:200]}...")
            print()
        return

    # ── Checkpoint ────────────────────────────────────────────
    ckpt_path = os.path.join(args.data_dir, "llm_edges_checkpoint.json")
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results = json.load(f)
        print(f"  Resumed: {sum(len(v) for v in results.values())} total estimates")
    else:
        results = {}

    # ── Query loop ────────────────────────────────────────────
    for rep in range(args.n_reps):
        print(f"\n  ── Rep {rep+1}/{args.n_reps} ──")
        batch_p, batch_k = [], []
        for qtype, ti, si, prompt in queries:
            key = f"{qtype}:{ti}:{si}"
            if len(results.get(key, [])) > rep:
                continue
            batch_p.append(prompt)
            batch_k.append(key)

        if not batch_p:
            print(f"    All done for rep {rep+1}")
            continue

        print(f"    {len(batch_p)} queries remaining")
        for bs in range(0, len(batch_p), args.batch_size):
            be = min(bs + args.batch_size, len(batch_p))
            t0 = time.time()
            responses = query_llm_vllm(
                batch_p[bs:be], args.model, args.temperature, args.max_tokens
            )
            dt = time.time() - t0
            n_ok = 0
            for key, resp in zip(batch_k[bs:be], responses):
                val = parse_estimation(resp)
                results.setdefault(key, []).append(val)
                if val is not None:
                    n_ok += 1
            print(
                f"    Batch {bs//args.batch_size+1}: {n_ok}/{be-bs} parsed, {dt:.1f}s"
            )
            with open(ckpt_path, "w") as f:
                json.dump(results, f)

    # ── Build directed adjacency ──────────────────────────────
    print("\n── Building directed adjacency ──")
    adj_mean = np.zeros((F, F), dtype=np.float64)
    adj_std = np.zeros((F, F), dtype=np.float64)

    for key, estimates in results.items():
        parts = key.split(":")
        ti, si = int(parts[1]), int(parts[2])
        valid = [v for v in estimates if v is not None]
        if valid:
            adj_mean[ti, si] = np.mean(valid)
            adj_std[ti, si] = np.std(valid) if len(valid) > 1 else 0.0

    # AKI is sink: zero outgoing
    adj_mean[:, aki_idx] = 0.0
    adj_std[:, aki_idx] = 0.0
    np.fill_diagonal(adj_mean, 0.0)
    np.fill_diagonal(adj_std, 0.0)

    n_edges = np.count_nonzero(adj_mean)
    print(f"  Directed edges: {n_edges}")
    nz = adj_mean[adj_mean > 0]
    print(f"  Weight: mean={nz.mean():.4f}, range=[{nz.min():.4f}, {nz.max():.4f}]")
    print(f"  Uncertainty: mean std={adj_std[adj_std > 0].mean():.4f}")

    # Uncertainty-based cutoff analysis
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(adj_mean > 0, adj_std / adj_mean, 0.0)
    print(f"\n  Confidence-based cutoff (CV = std/mean):")
    for t in [0.1, 0.2, 0.3, 0.5]:
        n_keep = int(np.sum((adj_mean > 0) & (cv < t)))
        print(f"    CV < {t}: {n_keep} edges ({n_keep/max(n_edges,1)*100:.0f}%)")

    # ── Save ──────────────────────────────────────────────────
    np.save(os.path.join(args.data_dir, "llm_adj.npy"), adj_mean)
    np.save(os.path.join(args.data_dir, "llm_adj_std.npy"), adj_std)
    print(f"\n  Saved: llm_adj.npy [{adj_mean.shape}] (directed, asymmetric)")
    print(f"  Saved: llm_adj_std.npy (uncertainty)")

    raw_path = os.path.join(args.data_dir, "llm_edges_raw.json")
    raw = {}
    for key, vals in results.items():
        parts = key.split(":")
        ti, si = int(parts[1]), int(parts[2])
        valid = [v for v in vals if v is not None]
        label = f"P({feature_names[ti]}|{feature_names[si]})"
        raw[label] = {
            "type": parts[0],
            "estimates": vals,
            "mean": float(np.mean(valid)) if valid else None,
            "std": float(np.std(valid)) if len(valid) > 1 else None,
            "n_valid": len(valid),
        }
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"  Saved: {raw_path}")

    print("\n" + "=" * 70)
    print("DONE — next: update graph.py for directed edges, then train")
    print("=" * 70)


if __name__ == "__main__":
    main()
