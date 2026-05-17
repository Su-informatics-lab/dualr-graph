#!/usr/bin/env python3
"""
llm_edges.py — LLM-estimated pairwise feature associations (v6).

The DualR bridge: instead of learning edges from patient data (Spearman/MI),
query an LLM for P(AKI | feature_i AND feature_j) for all feature pairs.
This produces an adjacency matrix *without seeing any patient data*.

v6 changes:
  - 38 features (was 25): individual nephrotoxins, baseline labs, eGFR
  - Continuous features get direction-aware clinical descriptions
    (e.g., "low baseline albumin" not just "baseline albumin")
  - Reads feature_names.json dynamically — no hardcoded feature list

Output: data/llm_adj.npy — [F, F] association matrix for graph.py

Usage:
    # On Tempest with GPU:
    python llm_edges.py --model meta-llama/Llama-3.1-8B-Instruct \
        --n_reps 10 --temperature 0.6

    # Dry run (prints prompts, no LLM):
    python llm_edges.py --dry_run

    # Resume from checkpoint:
    python llm_edges.py --resume
"""

import argparse
import itertools
import json
import os
import re
import time
from typing import Dict, List, Optional

import numpy as np

# ══════════════════════════════════════════════════════════════════
# Feature descriptions for LLM prompts
# ══════════════════════════════════════════════════════════════════
# Binary features: describe the PRESENCE of the condition.
# Continuous features: describe the clinically ABNORMAL direction,
#   since we're asking about AKI risk factors. The LLM needs to know
#   that "baseline_alb" means "low albumin = bad", not just "has albumin".
#
# Convention:
#   - Binary: "congestive heart failure" → "patient HAS CHF"
#   - Continuous: "low baseline albumin (malnutrition/inflammation)"
#     → "patient has LOW ALBUMIN"
#
# The LLM estimates P(AKI | feature_a AND feature_b). For continuous
# features, this means P(AKI | abnormal_value_a AND feature_b).

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # ── Demographics ──────────────────────────────────────────
    "age": "older age (≥65 years)",
    "sex_male": "male sex",
    "race_black": "Black race",
    "ethnicity_hispanic": "Hispanic ethnicity",
    # ── NCI-CCI comorbidities ─────────────────────────────────
    "Acute_MI": "acute myocardial infarction",
    "History_MI": "history of myocardial infarction",
    "CHF": "congestive heart failure",
    "PVD": "peripheral vascular disease",
    "CVD": "cerebrovascular disease",
    "COPD": "chronic obstructive pulmonary disease",
    "Dementia": "dementia",
    "Paralysis": "hemiplegia or paraplegia",
    "Diabetes": "diabetes mellitus without complications",
    "Diabetes_Complicated": "diabetes with end-organ complications (nephropathy, retinopathy)",
    "Renal_Disease": "pre-existing chronic kidney disease",
    "Liver_Disease_Mild": "mild liver disease (chronic hepatitis, fatty liver)",
    "Liver_Disease_Moderate_Severe": "moderate-to-severe liver disease (cirrhosis, portal hypertension)",
    "Peptic_Ulcer_Disease": "peptic ulcer disease",
    "Rheumatic_Disease": "rheumatic or autoimmune disease (rheumatoid arthritis, lupus)",
    "AIDS": "HIV/AIDS",
    # ── ICI regimen (ref = PD-1 mono) ────────────────────────
    "ici_pdl1_mono": "anti-PD-L1 monotherapy (atezolizumab, durvalumab, avelumab)",
    "ici_ctla4": "CTLA-4 combination immunotherapy (ipilimumab + PD-1 inhibitor)",
    # ── Cancer type (ref = Lung) ──────────────────────────────
    "cancer_Melanoma": "melanoma (primary cancer type)",
    "cancer_Renal_Cell": "renal cell carcinoma (primary cancer type)",
    "cancer_multi_cancer": "multiple primary cancer types",
    "cancer_Other": "cancer type other than lung, melanoma, or renal cell",
    "metastatic": "metastatic disease (distant spread)",
    # ── Individual nephrotoxin classes (±30d of ICI) ──────────
    "nephro_ppi": "concomitant proton pump inhibitor use (omeprazole, pantoprazole)",
    "nephro_nsaid": "concomitant NSAID use (ibuprofen, naproxen, celecoxib)",
    "nephro_vanc": "concomitant vancomycin use",
    "nephro_acei_arb": "concomitant ACE inhibitor or ARB use (lisinopril, losartan)",
    "nephro_diuretic": "concomitant loop diuretic use (furosemide, bumetanide)",
    # ── Baseline labs (continuous — ABNORMAL direction) ───────
    "baseline_bun": "elevated baseline blood urea nitrogen (suggesting dehydration or early renal dysfunction)",
    "baseline_hgb": "low baseline hemoglobin (anemia)",
    "baseline_alb": "low baseline serum albumin (malnutrition, inflammation, or nephrotic syndrome)",
    "baseline_k": "abnormal baseline serum potassium (hyperkalemia or hypokalemia)",
    "bun_cr_ratio": "elevated BUN-to-creatinine ratio (suggesting prerenal azotemia or dehydration)",
    "baseline_egfr": "low baseline estimated GFR (impaired kidney function, CKD-EPI 2021)",
    # ── Outcome ───────────────────────────────────────────────
    "aki_event": "acute kidney injury (serum creatinine ≥1.5× baseline)",
    # ── Legacy (backward compat, ignored if not in feature_names) ──
    "nephrotoxin": "concomitant nephrotoxic drug exposure",
    "nci_cci_score": "high NCI comorbidity index score",
    "ici_pd1": "anti-PD-1 monotherapy (nivolumab, pembrolizumab)",
    "ici_pdl1": "anti-PD-L1 monotherapy (atezolizumab, durvalumab)",
    "ici_combo": "CTLA-4 combination immunotherapy (ipilimumab + PD-1 inhibitor)",
}


SYSTEM_PROMPT = (
    "You are a clinical pharmacology expert estimating the probability of "
    "acute kidney injury (AKI) in cancer patients receiving immune checkpoint "
    "inhibitor (ICI) therapy. For each pair of clinical features, estimate "
    "the probability that a patient with BOTH features develops AKI within "
    "12 months of starting ICI treatment. Provide your estimate as a single "
    "number between 0.0 and 1.0, enclosed within [ESTIMATION] and "
    "[/ESTIMATION] tags. Do not provide a range."
)

USER_TEMPLATE = (
    "A cancer patient is about to start immune checkpoint inhibitor therapy. "
    "This patient has the following TWO clinical characteristics:\n"
    "  1. {feature_a}\n"
    "  2. {feature_b}\n\n"
    "Based on published clinical evidence and pharmacological mechanisms, "
    "estimate the probability that this patient develops acute kidney injury "
    "(AKI) within 12 months of starting ICI therapy.\n"
    "Respond with a single probability in [ESTIMATION]...[/ESTIMATION] tags."
)


# ══════════════════════════════════════════════════════════════════
# Parsing + query logic
# ══════════════════════════════════════════════════════════════════


def parse_estimation(text: str) -> Optional[float]:
    """Extract probability from [ESTIMATION]...[/ESTIMATION] tags."""
    m = re.search(r"\[ESTIMATION\]\s*([\d.]+)\s*\[/ESTIMATION\]", text)
    if m:
        val = float(m.group(1))
        if 0.0 <= val <= 1.0:
            return val
    # Fallback: find any standalone float between 0 and 1
    floats = re.findall(r"\b(0\.\d+|1\.0|0\.0)\b", text)
    for f in floats:
        val = float(f)
        if 0.0 <= val <= 1.0:
            return val
    return None


def get_description(feat_name: str) -> str:
    """Get clinical description for a feature, with fallback."""
    if feat_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feat_name]
    # Fallback: convert snake_case to readable
    readable = feat_name.replace("_", " ").replace("baseline ", "abnormal baseline ")
    return readable


def build_pair_prompt(feat_a: str, feat_b: str) -> str:
    """Build the user prompt for a feature pair."""
    desc_a = get_description(feat_a)
    desc_b = get_description(feat_b)
    return USER_TEMPLATE.format(feature_a=desc_a, feature_b=desc_b)


# ══════════════════════════════════════════════════════════════════
# LLM inference
# ══════════════════════════════════════════════════════════════════


def query_llm_vllm(
    prompts: List[str],
    model_name: str,
    temperature: float = 0.6,
    max_tokens: int = 256,
) -> List[str]:
    """Batch query via vLLM (Tempest GPU)."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, dtype="auto", trust_remote_code=True)
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["[/ESTIMATION]\n\n"],
    )

    conversations = []
    for prompt in prompts:
        conversations.append(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

    outputs = llm.chat(conversations, sampling_params=params)
    return [o.outputs[0].text for o in outputs]


# ══════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="LLM-prior edge estimation")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID for vLLM",
    )
    parser.add_argument(
        "--n_reps", type=int, default=10, help="Repetitions per pair for uncertainty"
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Pairs per vLLM batch (memory limit)",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print prompts without running LLM"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # ── Load feature names ────────────────────────────────────
    names_path = os.path.join(args.data_dir, "feature_names.json")
    with open(names_path) as f:
        feature_names = json.load(f)

    F = len(feature_names)
    print("=" * 70)
    print(f"LLM Edge Estimation — {F} features, {F*(F-1)//2} pairs")
    print(f"  Model: {args.model}")
    print(f"  Reps: {args.n_reps}, Temp: {args.temperature}")
    print("=" * 70)

    # Check all features have descriptions
    missing = [n for n in feature_names if n not in FEATURE_DESCRIPTIONS]
    if missing:
        print(f"\n  WARNING: {len(missing)} features without descriptions:")
        for m in missing:
            print(f"    {m} → will use fallback: '{get_description(m)}'")

    # ── Generate all pairs ────────────────────────────────────
    pairs = list(itertools.combinations(range(F), 2))
    print(f"  Total pairs: {len(pairs)}")

    # ── Checkpoint handling ───────────────────────────────────
    ckpt_path = os.path.join(args.data_dir, "llm_edges_checkpoint.json")
    raw_path = os.path.join(args.data_dir, "llm_edges_raw.json")

    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        results = ckpt.get("results", {})
        completed = set(ckpt.get("completed_pairs", []))
        print(f"  Resuming: {len(completed)} pairs already done")
    else:
        results = {}  # key: "i,j" → list of estimates
        completed = set()

    # ── Dry run ───────────────────────────────────────────────
    if args.dry_run:
        print("\n── DRY RUN: sample prompts ──\n")
        for idx, (i, j) in enumerate(pairs[:5]):
            prompt = build_pair_prompt(feature_names[i], feature_names[j])
            print(f"  Pair ({feature_names[i]}, {feature_names[j]}):")
            print(f"    {prompt[:200]}...")
            print()

        # Show all feature descriptions
        print("── All feature descriptions ──\n")
        for name in feature_names:
            desc = get_description(name)
            print(f"  {name:35s} → {desc}")
        return

    # ── Run LLM queries ───────────────────────────────────────
    remaining_pairs = [(i, j) for i, j in pairs if f"{i},{j}" not in completed]
    print(f"  Remaining pairs: {len(remaining_pairs)}")

    for rep in range(args.n_reps):
        print(f"\n  ── Rep {rep+1}/{args.n_reps} ──")

        # Build prompts for all remaining pairs
        prompts = []
        pair_keys = []
        for i, j in remaining_pairs:
            key = f"{i},{j}"
            # Only query if this rep hasn't been done for this pair
            current = results.get(key, [])
            if len(current) > rep:
                continue
            prompts.append(build_pair_prompt(feature_names[i], feature_names[j]))
            pair_keys.append(key)

        if not prompts:
            print(f"    All pairs done for rep {rep+1}")
            continue

        print(f"    Querying {len(prompts)} pairs...")

        # Batch processing
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_keys = pair_keys[batch_start:batch_end]

            t0 = time.time()
            responses = query_llm_vllm(
                batch_prompts,
                model_name=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            dt = time.time() - t0

            n_parsed = 0
            for key, resp in zip(batch_keys, responses):
                val = parse_estimation(resp)
                if val is not None:
                    results.setdefault(key, []).append(val)
                    n_parsed += 1
                else:
                    # Failed parse — store NaN
                    results.setdefault(key, []).append(None)

            print(
                f"    Batch {batch_start//args.batch_size + 1}: "
                f"{n_parsed}/{len(batch_prompts)} parsed, {dt:.1f}s"
            )

            # Checkpoint after each batch
            ckpt_data = {
                "results": results,
                "completed_pairs": list(completed),
                "feature_names": feature_names,
                "model": args.model,
                "n_reps": args.n_reps,
            }
            with open(ckpt_path, "w") as f:
                json.dump(ckpt_data, f)

    # ── Build adjacency matrix ────────────────────────────────
    print("\n── Building adjacency matrix ──")
    adj = np.zeros((F, F), dtype=np.float64)

    n_complete = 0
    n_partial = 0
    for i, j in pairs:
        key = f"{i},{j}"
        estimates = results.get(key, [])
        # Filter None values
        valid = [v for v in estimates if v is not None]
        if valid:
            mean_est = np.mean(valid)
            adj[i, j] = mean_est
            adj[j, i] = mean_est  # symmetric
            if len(valid) >= args.n_reps:
                n_complete += 1
            else:
                n_partial += 1

    print(f"  Complete pairs (≥{args.n_reps} reps): {n_complete}")
    print(f"  Partial pairs: {n_partial}")
    print(f"  Missing pairs: {len(pairs) - n_complete - n_partial}")
    print(f"  Adjacency range: [{adj.min():.4f}, {adj.max():.4f}]")
    print(f"  Adjacency mean: {adj[adj > 0].mean():.4f}")

    # ── Save ──────────────────────────────────────────────────
    out_path = os.path.join(args.data_dir, "llm_adj.npy")
    np.save(out_path, adj)
    print(f"  Saved: {out_path} [{adj.shape}]")

    # Also save raw results for debugging
    with open(raw_path, "w") as f:
        # Convert pair keys to feature names for readability
        raw_readable = {}
        for key, vals in results.items():
            i, j = map(int, key.split(","))
            readable_key = f"{feature_names[i]}__x__{feature_names[j]}"
            raw_readable[readable_key] = {
                "estimates": vals,
                "mean": (
                    np.mean([v for v in vals if v is not None])
                    if any(v is not None for v in vals)
                    else None
                ),
                "n_valid": sum(1 for v in vals if v is not None),
            }
        json.dump(raw_readable, f, indent=2)
    print(f"  Saved: {raw_path}")

    print("\n" + "=" * 70)
    print(
        "DONE — next: python graph.py --method llm --k 8 --llm_prior data/llm_adj.npy"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
