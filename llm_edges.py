#!/usr/bin/env python3
"""
12_llm_edges.py — LLM-estimated pairwise feature associations.

The DualR bridge: instead of learning edges from patient data (Spearman/MI),
query an LLM for P(AKI | feature_i AND feature_j) for all feature pairs.
This produces an adjacency matrix *without seeing any patient data* — the
same portability property as DualR's drug-disease star graph, but extended
to pairwise feature interactions.

Output: data/llm_adj.npy — [F, F] association matrix ready for 02_graph.py

The LLM query protocol mirrors dualr.py: structured prompt → probability
in [ESTIMATION] tags → parse → aggregate across n_reps repetitions.

Usage:
    # On Tempest with GPU:
    python 12_llm_edges.py --model meta-llama/Llama-3.1-8B-Instruct \
        --n_reps 10 --temperature 0.6

    # Dry run (prints prompts, no LLM):
    python 12_llm_edges.py --dry_run
"""

import argparse
import itertools
import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np

# ── Feature descriptions for LLM prompts ─────────────────────────
# Each feature gets a clinical description the LLM can reason about.

FEATURE_DESCRIPTIONS = {
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
    "Diabetes": "diabetes mellitus",
    "Diabetes_Complicated": "diabetes with end-organ complications",
    "Renal_Disease": "chronic kidney disease",
    "Liver_Disease_Mild": "mild liver disease",
    "Liver_Disease_Moderate_Severe": "moderate-to-severe liver disease",
    "Peptic_Ulcer_Disease": "peptic ulcer disease",
    "Rheumatic_Disease": "rheumatic or autoimmune disease",
    "AIDS": "HIV/AIDS",
    "ici_pd1": "anti-PD-1 immunotherapy (e.g. nivolumab, pembrolizumab)",
    "ici_pdl1": "anti-PD-L1 immunotherapy (e.g. atezolizumab, durvalumab)",
    "ici_ctla4": "anti-CTLA-4 immunotherapy (e.g. ipilimumab)",
    "nephrotoxin": "concomitant nephrotoxic drug exposure (NSAIDs, aminoglycosides)",
    "nci_cci_score": "high NCI comorbidity index score",
    "aki_event": "acute kidney injury",
}

SYSTEM_PROMPT = (
    "You are a clinical pharmacology expert estimating the probability of "
    "acute kidney injury (AKI) in cancer patients receiving immune checkpoint "
    "inhibitor (ICI) therapy. For each pair of clinical features, estimate "
    "the probability that a patient with BOTH features develops AKI after "
    "ICI treatment. Provide the probability between 0 and 1 enclosed within "
    "[ESTIMATION] and [/ESTIMATION] tags."
)

USER_TEMPLATE = (
    "A cancer patient receiving immune checkpoint inhibitor therapy has "
    "the following TWO clinical features:\n"
    "  1. {feature_a}\n"
    "  2. {feature_b}\n\n"
    "Estimate the probability that this patient develops acute kidney injury "
    "(AKI) within 12 months of starting ICI therapy. "
    "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
)


def extract_probability(text: str) -> Optional[float]:
    """Extract probability from [ESTIMATION] tags. Same as dualr.py."""
    if not text:
        return None
    match = re.search(r"\[ESTIMATION\](.*?)\[/ESTIMATION\]", text, re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        val = float(raw)
        if 0.0 <= val <= 1.0:
            return val
    except ValueError:
        perc = re.search(r"(\d+(?:\.\d+)?)%", raw)
        if perc:
            try:
                val = float(perc.group(1)) / 100.0
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                pass
    return None


def query_pair(
    llm,
    feature_a: str,
    feature_b: str,
    n_reps: int,
    temperature: float,
    max_tokens: int,
    base_seed: int,
) -> Tuple[float, List[Optional[float]]]:
    """Query LLM for P(AKI | feature_a, feature_b), averaged over n_reps."""
    from vllm import SamplingParams

    desc_a = FEATURE_DESCRIPTIONS.get(feature_a, feature_a)
    desc_b = FEATURE_DESCRIPTIONS.get(feature_b, feature_b)
    user_msg = USER_TEMPLATE.format(feature_a=desc_a, feature_b=desc_b)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    probs = []
    for rep in range(n_reps):
        params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            seed=base_seed + rep,
        )
        try:
            outputs = llm.chat(messages=[messages], sampling_params=params)
            text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            prob = extract_probability(text)
        except Exception:
            prob = None
        probs.append(prob)

    valid = [p for p in probs if p is not None]
    mean_prob = float(np.mean(valid)) if valid else float("nan")
    return mean_prob, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_reps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--dry_run", action="store_true", help="Print prompts only, no LLM inference"
    )
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    F = len(feature_names)
    pairs = list(itertools.combinations(range(F), 2))
    print(f"Features: {F}, Pairs: {len(pairs)}, Reps: {args.n_reps}")
    print(f"Total LLM calls: {len(pairs) * args.n_reps}")

    if args.dry_run:
        # Show sample prompts
        for i, j in pairs[:3]:
            desc_a = FEATURE_DESCRIPTIONS.get(feature_names[i], feature_names[i])
            desc_b = FEATURE_DESCRIPTIONS.get(feature_names[j], feature_names[j])
            print(f"\n  Pair ({feature_names[i]}, {feature_names[j]}):")
            print(
                f"    {USER_TEMPLATE.format(feature_a=desc_a, feature_b=desc_b)[:200]}..."
            )
        print(f"\n  ({len(pairs) - 3} more pairs omitted)")
        return

    # Initialize LLM
    import torch
    from vllm import LLM

    print(f"Loading {args.model}...")
    llm = LLM(
        model=args.model,
        max_model_len=args.max_tokens,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=torch.bfloat16,
    )

    adj = np.zeros((F, F))
    raw_results = {}

    from tqdm import tqdm

    for i, j in tqdm(pairs, desc="LLM pairwise queries"):
        mean_p, reps = query_pair(
            llm,
            feature_names[i],
            feature_names[j],
            n_reps=args.n_reps,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_seed=args.seed + i * F + j,
        )
        adj[i, j] = adj[j, i] = mean_p
        raw_results[f"{feature_names[i]}___{feature_names[j]}"] = {
            "mean": mean_p,
            "reps": reps,
        }

    # Normalize to [0, 1] range for edge weights
    valid_mask = ~np.isnan(adj)
    if valid_mask.any():
        adj_min = np.nanmin(adj[valid_mask])
        adj_max = np.nanmax(adj[valid_mask])
        if adj_max > adj_min:
            adj = (adj - adj_min) / (adj_max - adj_min)
    adj = np.nan_to_num(adj, nan=0.0)
    np.fill_diagonal(adj, 0)

    # Save
    np.save(os.path.join(args.data_dir, "llm_adj.npy"), adj)
    with open(os.path.join(args.data_dir, "llm_edges_raw.json"), "w") as f:
        json.dump(raw_results, f, indent=2, default=str)

    n_valid = sum(1 for v in raw_results.values() if not np.isnan(v["mean"]))
    print(f"\nSaved: data/llm_adj.npy [{F}×{F}]")
    print(f"Valid pairs: {n_valid}/{len(pairs)}")
    print(f"Refusal rate: {1 - n_valid/len(pairs):.3f}")


if __name__ == "__main__":
    main()
