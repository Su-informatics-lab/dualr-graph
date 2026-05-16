#!/usr/bin/env python3
"""
NCI Comorbidity Index — Scoring & Code Sets
============================================
Shared module for post-ICI AKI × SDoH study.
Source: NCI.comorbidity.macro.sas (2021-09-21, updated 2022-03-15)

Contains:
  - NCI 2021 ICD code sets (16 conditions, ICD-9 + ICD-10)
  - compute_charlson_score()  — Charlson integer weights, NCI hierarchy
  - compute_nci_index()       — NCI Cox-model continuous weights (Stedman)
  - build_nci_flags()         — Build binary comorbidity flags from dx codes

CRITICAL FIXES (v2, 2026-05-16):
  1. MI scoring: 1*(acute_mi OR history_mi) = max 1pt (was additive 1+1=2)
  2. Paralysis/CVD: scored INDEPENDENTLY (was incorrectly hierarchical)
  3. Hierarchy is formula-level (AND NOT / OR), NOT pre-processing
  4. NCI weights: 5 decimal places matching SAS (was 3 decimal)

Usage:
  from nci_cci_scoring import (NCI_CODESETS, NCI_CCI_CONDITIONS,
                                compute_charlson_score, compute_nci_index,
                                build_nci_flags)
"""

import pandas as pd

# ══════════════════════════════════════════════════════════════════
# NCI 2021 ICD CODE SETS
# Source: NCI SEER-Medicare, modified Quan et al. 2005
# All codes dot-stripped, uppercase. Use with prefix matching.
# ══════════════════════════════════════════════════════════════════

nci_acute_mi = {
    "9": ["410"],
    "10": ["I21", "I22"],
}

nci_history_mi = {
    "9": ["412"],
    "10": ["I252"],
}

nci_chf = {
    "9": [
        "39891",
        "40201",
        "40211",
        "40291",
        "40401",
        "40403",
        "40411",
        "40413",
        "40491",
        "40493",
        "4254",
        "4255",
        "4256",
        "4257",
        "4258",
        "4259",
        "428",
    ],
    "10": [
        "I099",
        "I110",
        "I130",
        "I132",
        "I255",
        "I420",
        "I425",
        "I426",
        "I427",
        "I428",
        "I429",
        "I43",
        "I50",
        "P290",
    ],
}

nci_pvd = {
    "9": [
        "0930",
        "440",
        "441",
        "4431",
        "4432",
        "4433",
        "4434",
        "4435",
        "4436",
        "4437",
        "4438",
        "4439",
        "4471",
        "5571",
        "5579",
        "V434",
    ],
    "10": [
        "I70",
        "I71",
        "I731",
        "I738",
        "I739",
        "I771",
        "I790",
        "I792",
        "K551",
        "K558",
        "K559",
        "Z958",
        "Z959",
    ],
}

nci_cvd = {
    "9": ["36234", "430", "431", "432", "433", "434", "435", "436", "437", "438"],
    "10": [
        "G45",
        "G46",
        "H340",
        "I60",
        "I61",
        "I62",
        "I63",
        "I64",
        "I65",
        "I66",
        "I67",
        "I68",
        "I69",
    ],
}

nci_copd = {
    "9": [
        "4168",
        "4169",
        "490",
        "491",
        "492",
        "493",
        "494",
        "495",
        "496",
        "497",
        "498",
        "499",
        "500",
        "501",
        "502",
        "503",
        "504",
        "505",
        "5064",
        "5081",
        "5088",
    ],
    "10": [
        "I278",
        "I279",
        "J40",
        "J41",
        "J42",
        "J43",
        "J44",
        "J45",
        "J46",
        "J47",
        "J60",
        "J61",
        "J62",
        "J63",
        "J64",
        "J65",
        "J66",
        "J67",
        "J684",
        "J701",
        "J703",
    ],
}

nci_dementia = {
    "9": ["290", "2941", "3312"],
    "10": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"],
}

nci_paralysis = {
    "9": [
        "3341",
        "342",
        "343",
        "3440",
        "3441",
        "3442",
        "3443",
        "3444",
        "3445",
        "3446",
        "3449",
    ],
    "10": [
        "G041",
        "G114",
        "G801",
        "G802",
        "G81",
        "G82",
        "G830",
        "G831",
        "G832",
        "G833",
        "G834",
        "G839",
    ],
}

nci_diabetes = {
    "9": ["2500", "2501", "2502", "2503", "2508", "2509"],
    "10": [
        "E100",
        "E101",
        "E106",
        "E108",
        "E109",
        "E110",
        "E111",
        "E116",
        "E118",
        "E119",
        "E130",
        "E131",
        "E136",
        "E138",
        "E139",
    ],
}

nci_diabetes_complicated = {
    "9": ["2504", "2505", "2506", "2507"],
    "10": [
        "E102",
        "E103",
        "E104",
        "E105",
        "E107",
        "E112",
        "E113",
        "E114",
        "E115",
        "E117",
        "E132",
        "E133",
        "E134",
        "E135",
        "E137",
    ],
}

nci_renal = {
    "9": [
        "40301",
        "40311",
        "40391",
        "40402",
        "40403",
        "40412",
        "40413",
        "40492",
        "40493",
        "582",
        "5830",
        "5831",
        "5832",
        "5833",
        "5834",
        "5835",
        "5836",
        "5837",
        "585",
        "586",
        "5880",
        "V420",
        "V451",
        "V56",
    ],
    "10": [
        "I120",
        "I131",
        "N032",
        "N033",
        "N034",
        "N035",
        "N036",
        "N037",
        "N052",
        "N053",
        "N054",
        "N055",
        "N056",
        "N057",
        "N18",
        "N19",
        "N250",
        "Z490",
        "Z491",
        "Z492",
        "Z940",
        "Z992",
    ],
}

nci_liver_mild = {
    "9": [
        "07022",
        "07023",
        "07032",
        "07033",
        "07044",
        "07054",
        "0706",
        "0709",
        "570",
        "571",
        "5733",
        "5734",
        "5738",
        "5739",
        "V427",
    ],
    "10": [
        "B18",
        "K700",
        "K701",
        "K702",
        "K703",
        "K709",
        "K713",
        "K714",
        "K715",
        "K717",
        "K73",
        "K74",
        "K760",
        "K762",
        "K763",
        "K764",
        "K768",
        "K769",
        "Z944",
    ],
}

nci_liver_mod_severe = {
    "9": [
        "4560",
        "4561",
        "4562",
        "5722",
        "5723",
        "5724",
        "5725",
        "5726",
        "5727",
        "5728",
    ],
    "10": [
        "I850",
        "I859",
        "I864",
        "I982",
        "K704",
        "K711",
        "K721",
        "K729",
        "K765",
        "K766",
        "K767",
    ],
}

nci_pud = {
    "9": ["531", "532", "533", "534"],
    "10": ["K25", "K26", "K27", "K28"],
}

nci_rheumatic = {
    "9": [
        "4465",
        "7100",
        "7101",
        "7102",
        "7103",
        "7104",
        "7140",
        "7141",
        "7142",
        "7148",
        "725",
    ],
    "10": ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"],
}

nci_aids = {
    "9": ["042", "043", "044"],
    "10": ["B20", "B21", "B22", "B24"],
}

# ── Master code set dictionary ────────────────────────────────────
NCI_CODESETS = {
    "Acute_MI": nci_acute_mi,
    "History_MI": nci_history_mi,
    "Congestive_Heart_Failure": nci_chf,
    "Peripheral_Vascular_Disease": nci_pvd,
    "Cerebrovascular_Disease": nci_cvd,
    "Chronic_Pulmonary_Disease": nci_copd,
    "Dementia": nci_dementia,
    "Paralysis": nci_paralysis,
    "Diabetes": nci_diabetes,
    "Diabetes_Complicated": nci_diabetes_complicated,
    "Renal_Disease": nci_renal,
    "Liver_Disease_Mild": nci_liver_mild,
    "Liver_Disease_Moderate_Severe": nci_liver_mod_severe,
    "Peptic_Ulcer_Disease": nci_pud,
    "Rheumatic_Disease": nci_rheumatic,
    "AIDS": nci_aids,
}

# ── Column list for NCI-CCI EMR parquet ───────────────────────────
NCI_CCI_CONDITIONS = list(NCI_CODESETS.keys())


# ══════════════════════════════════════════════════════════════════
# NCI WEIGHTS (Stedman technical report, 5 decimal places)
# ══════════════════════════════════════════════════════════════════

NCI_WEIGHTS_MAIN = {
    "Acute_MI": 0.12624,
    "History_MI": 0.07999,
    "Congestive_Heart_Failure": 0.64441,
    "Peripheral_Vascular_Disease": 0.26232,
    "Cerebrovascular_Disease": 0.27868,
    "Chronic_Pulmonary_Disease": 0.52487,
    "Dementia": 0.72219,
    "Paralysis": 0.39882,
    "Any_Diabetes": 0.29408,
    "Renal_Disease": 0.47010,
    "Any_Liver_Disease": 0.73803,
    "Peptic_Ulcer_Disease": 0.07506,
    "Rheumatic_Disease": 0.21905,
    "AIDS": 0.58362,
}


# ══════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS — matched to NCI SAS macro exactly
# ══════════════════════════════════════════════════════════════════


def compute_charlson_score(como_df):
    """
    Compute Charlson Comorbidity Score, exactly matching NCI SAS macro.

    Hierarchy is handled IN the formula (AND NOT / OR), NOT by zeroing
    out raw flags as pre-processing.

    Key rules:
      - MI: acute_mi OR history_mi = single 1-pt condition (not 1+1)
      - Diabetes: uncomplicated scores 1 only if NO complicated present
      - Liver: mild scores 1 only if NO mod/severe present
      - Paralysis and CVD: scored INDEPENDENTLY (no hierarchy)

    Parameters
    ----------
    como_df : DataFrame with raw binary flag columns

    Returns
    -------
    Series of int Charlson scores
    """
    df = como_df
    score = (
        # MI: OR logic — max 1 pt total
        1 * ((df["Acute_MI"] == 1) | (df["History_MI"] == 1)).astype(int)
        + 1 * df["Congestive_Heart_Failure"]
        + 1 * df["Peripheral_Vascular_Disease"]
        # CVD scores independently — NO Paralysis hierarchy
        + 1 * df["Cerebrovascular_Disease"]
        + 1 * df["Chronic_Pulmonary_Disease"]
        + 1 * df["Dementia"]
        # Paralysis scores independently — NO CVD hierarchy
        + 2 * df["Paralysis"]
        # Diabetes: AND NOT hierarchy in formula
        + 1 * ((df["Diabetes"] == 1) & (df["Diabetes_Complicated"] == 0)).astype(int)
        + 2 * df["Diabetes_Complicated"]
        # Renal: single merged category
        + 2 * df["Renal_Disease"]
        # Liver: AND NOT hierarchy in formula
        + 1
        * (
            (df["Liver_Disease_Mild"] == 1) & (df["Liver_Disease_Moderate_Severe"] == 0)
        ).astype(int)
        + 3 * df["Liver_Disease_Moderate_Severe"]
        + 1 * df["Peptic_Ulcer_Disease"]
        + 1 * df["Rheumatic_Disease"]
        + 6 * df["AIDS"]
    )
    return score


def compute_nci_index(como_df, weights=None):
    """
    Compute NCI Comorbidity Index, exactly matching NCI SAS macro.

    Key differences from Charlson scoring:
      - MI: acute and history scored SEPARATELY (additive, not OR)
      - Diabetes: OR logic (any diabetes = single weight)
      - Liver: OR logic (any liver = single weight)
      - Paralysis and CVD: scored INDEPENDENTLY (no hierarchy)

    Parameters
    ----------
    como_df : DataFrame with raw binary flag columns
    weights : dict or None. If None, uses NCI_WEIGHTS_MAIN.

    Returns
    -------
    Series of float NCI index scores
    """
    if weights is None:
        weights = NCI_WEIGHTS_MAIN

    df = como_df

    # Build merged flags using OR (matching SAS: diabetes OR diabetes_comp)
    any_diabetes = ((df["Diabetes"] == 1) | (df["Diabetes_Complicated"] == 1)).astype(
        int
    )
    any_liver = (
        (df["Liver_Disease_Mild"] == 1) | (df["Liver_Disease_Moderate_Severe"] == 1)
    ).astype(int)

    score = pd.Series(0.0, index=df.index)

    # Score individual conditions (NOT merged ones)
    for condition in [
        "Acute_MI",
        "History_MI",
        "Congestive_Heart_Failure",
        "Peripheral_Vascular_Disease",
        "Cerebrovascular_Disease",
        "Chronic_Pulmonary_Disease",
        "Dementia",
        "Paralysis",
        "Renal_Disease",
        "Peptic_Ulcer_Disease",
        "Rheumatic_Disease",
        "AIDS",
    ]:
        w = weights.get(condition)
        if w is not None and condition in df.columns:
            score += df[condition].astype(float) * w

    # Score merged conditions
    w_diab = weights.get("Any_Diabetes")
    if w_diab is not None:
        score += any_diabetes.astype(float) * w_diab

    w_liver = weights.get("Any_Liver_Disease")
    if w_liver is not None:
        score += any_liver.astype(float) * w_liver

    return score


def build_nci_flags(eligible_person_ids, condition_df, icd_col="icd_code"):
    """
    Build NCI-CCI binary flags from a condition DataFrame.

    Parameters
    ----------
    eligible_person_ids : array-like of person_id values
    condition_df : DataFrame with columns [person_id, icd_col]
        where icd_col contains dot-stripped, uppercase ICD codes
    icd_col : column name containing ICD codes

    Returns
    -------
    DataFrame with person_id + 16 binary flag columns + charlson_score + nci_index
    """
    # Initialize flags
    charlson = pd.DataFrame({"person_id": eligible_person_ids})
    for condition in NCI_CCI_CONDITIONS:
        charlson[condition] = 0

    # Build each flag via prefix matching
    for condition, codes in NCI_CODESETS.items():
        all_prefixes = []
        for ver_codes in codes.values():
            all_prefixes.extend(ver_codes)

        mask = condition_df[icd_col].apply(
            lambda x: (
                any(str(x).startswith(p) for p in all_prefixes)
                if pd.notna(x)
                else False
            )
        )
        flagged = condition_df[mask].person_id.unique()
        charlson[condition] = charlson.person_id.isin(flagged).astype("int8")

    # ── DO NOT apply hierarchy as pre-processing ──────────────────
    # ❌ WRONG (Glasheen pattern):
    #   charlson.loc[charlson.Diabetes_Complicated==1, "Diabetes"] = 0
    #   charlson.loc[charlson.Liver_Disease_Moderate_Severe==1, "Liver_Disease_Mild"] = 0
    #   charlson.loc[charlson.Paralysis==1, "Cerebrovascular_Disease"] = 0
    # ✅ CORRECT: Raw flags stay untouched. Hierarchy is handled in
    #   compute_charlson_score() and compute_nci_index() formulas.

    # Compute both scores
    charlson["charlson_score"] = compute_charlson_score(charlson)
    charlson["nci_index"] = compute_nci_index(charlson)

    # QC assertions
    assert charlson["charlson_score"].min() >= 0, "Charlson scores must be non-negative"
    assert charlson["nci_index"].min() >= 0, "NCI index must be non-negative"

    # QC report
    both_diab = (
        (charlson["Diabetes"] == 1) & (charlson["Diabetes_Complicated"] == 1)
    ).sum()
    both_mi = ((charlson["Acute_MI"] == 1) & (charlson["History_MI"] == 1)).sum()
    both_para_cvd = (
        (charlson["Paralysis"] == 1) & (charlson["Cerebrovascular_Disease"] == 1)
    ).sum()
    print(f"  QC: {both_diab} pts w/ both diabetes flags (raw, not zeroed)")
    print(f"  QC: {both_mi} pts w/ both MI types → Charlson MI = 1pt each (OR, not 2)")
    print(f"  QC: {both_para_cvd} pts w/ paralysis+CVD → both score independently")
    print(
        f"  Charlson: median {charlson.charlson_score.median():.0f}, "
        f"IQR {charlson.charlson_score.quantile(0.25):.0f}–"
        f"{charlson.charlson_score.quantile(0.75):.0f}, "
        f"max {charlson.charlson_score.max():.0f}"
    )
    print(
        f"  NCI index: median {charlson.nci_index.median():.3f}, "
        f"IQR {charlson.nci_index.quantile(0.25):.3f}–"
        f"{charlson.nci_index.quantile(0.75):.3f}, "
        f"max {charlson.nci_index.max():.3f}"
    )
    for c in NCI_CCI_CONDITIONS:
        print(f"    {c:40s} {charlson[c].sum():>6,}  ({charlson[c].mean()*100:.1f}%)")

    return charlson


def make_like_clause(codes_dict):
    """Build SQL OR clause for prefix matching on dot-stripped dx codes."""
    prefixes = []
    for codes in codes_dict.values():
        for c in codes:
            prefixes.append(str(c).replace(".", "").upper())
    prefixes = sorted(set(prefixes))
    return " OR ".join([f"dx_code_nodot LIKE '{p}%'" for p in prefixes])


def build_nci_flags_sql(cdr_prefix):
    """
    Generate BigQuery SQL for AoU NCI-CCI flag building.

    Parameters
    ----------
    cdr_prefix : str, e.g. 'fc-aou-cdr-prod-ct.C2024Q3R9'

    Returns
    -------
    str : SQL query that produces person_id + 16 binary columns
    """
    case_clauses = []
    for condition, codes in NCI_CODESETS.items():
        all_prefixes = []
        for ver_codes in codes.values():
            all_prefixes.extend(str(c).replace(".", "").upper() for c in ver_codes)
        all_prefixes = sorted(set(all_prefixes))

        like_parts = " OR ".join([f"dx_clean LIKE '{p}%'" for p in all_prefixes])
        case_clauses.append(
            f"  MAX(CASE WHEN ({like_parts}) THEN 1 ELSE 0 END) AS {condition}"
        )

    cases_sql = ",\n".join(case_clauses)

    sql = f"""
WITH dx AS (
  SELECT
    co.person_id,
    UPPER(REPLACE(c.concept_code, '.', '')) AS dx_clean
  FROM `{cdr_prefix}.condition_occurrence` co
  JOIN `{cdr_prefix}.concept` c
    ON c.concept_id = co.condition_source_concept_id
  WHERE c.vocabulary_id IN ('ICD9CM', 'ICD10CM')
)
SELECT
  person_id,
{cases_sql}
FROM dx
GROUP BY person_id
"""
    return sql
