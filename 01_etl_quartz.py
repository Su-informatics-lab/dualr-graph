#!/usr/bin/env python3
"""
Post-ICI AKI × SDoH — AoU ETL (v2 — NCI-CCI scoring fix)
Runs on AoU Researcher Workbench (Controlled Tier).

Adapted from aou_covid/01_aou_etl.py (Wang et al.)
Design: study_design_postICI_AKI_SDoH_v2.md

Steps:
  1. ICI-treated cancer patient cohort + AKI phenotyping
  2. Demographics
  3. NCI Charlson Comorbidity Index (14 conditions) ← FIXED v2
  4. SDoH (Basics Survey)
  5. Cancer type + ICI class + concomitant nephrotoxins
  6. Matching variables
  7. Regression base assembly

CCI FIX LOG (v2, 2026-05-16):
  - REMOVED hierarchy pre-processing (zeroing raw flags)
  - FIXED MI scoring: OR logic (max 1pt), was additive (2pt)
  - FIXED Paralysis/CVD: score independently (was hierarchical)
  - ADDED NCI continuous index alongside Charlson integer score
  - Uses shared nci_cci_scoring.py module

Usage: python 01_ici_aki_etl.py
Output: results/ici_aki/*.csv (01–07)
Next:   Rscript 01b_psm.R ici_aki
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*read_gbq is deprecated.*")

import numpy as np
import pandas as pd

# Import corrected NCI-CCI scoring
from nci_cci_scoring import (
    NCI_CCI_CONDITIONS,
    NCI_CODESETS,
    compute_charlson_score,
    compute_nci_index,
)

# ── Environment ───────────────────────────────────────────────────
CDR = os.environ.get("WORKSPACE_CDR", "")
if not CDR:
    print("ERROR: WORKSPACE_CDR not set. Run on AoU Workbench.")
    sys.exit(1)

RESULTS = "results/ici_aki"
os.makedirs(RESULTS, exist_ok=True)

print("=" * 70)
print("POST-ICI AKI × SDoH — AoU ETL (v2 — NCI-CCI fix)")
print("=" * 70)
print(f"  CDR: {CDR}")
print(f"  Output: {RESULTS}/")


def save(df, filename):
    path = os.path.join(RESULTS, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df):,} rows, {df.shape[1]} cols)")


def q(sql):
    return pd.read_gbq(sql, dialect="standard")


def parse_date(s):
    return pd.to_datetime(s, errors="coerce")


# ═══════════════════════════════════════════════════════════════════
# STEP 1: ICI COHORT + AKI PHENOTYPING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: ICI Cohort + AKI Phenotyping")
print("=" * 70)

# ── 1a. Find ICI drug concept IDs via name matching ───────────────
ICI_AGENTS = [
    "nivolumab",
    "pembrolizumab",
    "atezolizumab",
    "durvalumab",
    "avelumab",
    "cemiplimab",
    "ipilimumab",
    "tremelimumab",
    "dostarlimab",
    "relatlimab",
]

agent_likes = " OR ".join([f"LOWER(c.concept_name) LIKE '%{a}%'" for a in ICI_AGENTS])
ici_concepts_sql = f"""
SELECT DISTINCT c.concept_id, c.concept_name
FROM `{CDR}.concept` c
WHERE ({agent_likes})
  AND c.domain_id = 'Drug'
  AND c.standard_concept = 'S'
"""
ici_concepts = q(ici_concepts_sql)
ici_concept_ids = tuple(ici_concepts.concept_id.tolist())
print(f"  Found {len(ici_concept_ids)} ICI drug concept IDs")

if len(ici_concept_ids) == 0:
    print("FATAL: No ICI concepts found. Check CDR version.")
    sys.exit(1)

# ── 1b. ICI-treated patients ─────────────────────────────────────
ici_sql = f"""
SELECT
  de.person_id,
  MIN(de.drug_exposure_start_date) AS ici_index_date,
  ARRAY_AGG(DISTINCT LOWER(c.concept_name) ORDER BY LOWER(c.concept_name)) AS ici_drugs
FROM `{CDR}.drug_exposure` de
JOIN `{CDR}.concept` c ON c.concept_id = de.drug_concept_id
WHERE de.drug_concept_id IN ({','.join(str(x) for x in ici_concept_ids)})
GROUP BY de.person_id
"""
ici_patients = q(ici_sql)
ici_patients["ici_index_date"] = parse_date(ici_patients["ici_index_date"])
print(f"  ICI-treated patients: {len(ici_patients):,}")

# ── 1c. Cancer diagnosis filter ──────────────────────────────────
cancer_sql = f"""
SELECT DISTINCT co.person_id
FROM `{CDR}.condition_occurrence` co
JOIN `{CDR}.concept` c ON c.concept_id = co.condition_source_concept_id
WHERE c.vocabulary_id = 'ICD10CM'
  AND (c.concept_code LIKE 'C%' OR c.concept_code LIKE 'D0%'
       OR c.concept_code LIKE 'D1%' OR c.concept_code LIKE 'D2%'
       OR c.concept_code LIKE 'D3%' OR c.concept_code LIKE 'D4%')
"""
cancer_pts = q(cancer_sql)
ici_cancer = ici_patients[ici_patients.person_id.isin(cancer_pts.person_id)]
print(f"  ICI + cancer: {len(ici_cancer):,}")

# ── 1d. Basics Survey filter ─────────────────────────────────────
basics_sql = f"""
SELECT DISTINCT person_id
FROM `{CDR}.observation`
WHERE observation_source_concept_id = 1585845
"""
basics_pts = q(basics_sql)
ici_cancer_basics = ici_cancer[ici_cancer.person_id.isin(basics_pts.person_id)]
print(f"  ICI + cancer + Basics Survey: {len(ici_cancer_basics):,}")

# ── 1e. Creatinine extraction ────────────────────────────────────
cr_concept = 3016723  # LOINC 2160-0
cr_sql = f"""
SELECT
  m.person_id,
  m.measurement_date,
  m.value_as_number,
  m.unit_concept_id
FROM `{CDR}.measurement` m
WHERE m.measurement_concept_id = {cr_concept}
  AND m.value_as_number IS NOT NULL
  AND m.value_as_number > 0
  AND m.value_as_number < 30
"""
cr_all = q(cr_sql)
cr_all["measurement_date"] = parse_date(cr_all["measurement_date"])
cr_all = cr_all[cr_all.person_id.isin(ici_cancer_basics.person_id)]
print(f"  Creatinine measurements (ICI patients): {len(cr_all):,}")

# ── 1f. Baseline + follow-up Cr ──────────────────────────────────
cr_merged = cr_all.merge(
    ici_cancer_basics[["person_id", "ici_index_date"]], on="person_id"
)
cr_merged["days_from_ici"] = (
    cr_merged.measurement_date - cr_merged.ici_index_date
).dt.days

# Baseline: median Cr in [-365, -7]
baseline_cr = cr_merged[
    (cr_merged.days_from_ici >= -365) & (cr_merged.days_from_ici <= -7)
]
baseline = (
    baseline_cr.groupby("person_id")
    .agg(
        baseline_cr=("value_as_number", "median"),
        n_baseline=("value_as_number", "count"),
    )
    .reset_index()
)
print(f"  Patients with baseline Cr: {len(baseline):,}")

# Follow-up: Cr in [1, 365]
followup_cr = cr_merged[
    (cr_merged.days_from_ici >= 1) & (cr_merged.days_from_ici <= 365)
]
followup = (
    followup_cr.groupby("person_id")
    .agg(
        max_followup_cr=("value_as_number", "max"),
        n_followup=("value_as_number", "count"),
    )
    .reset_index()
)
print(f"  Patients with follow-up Cr: {len(followup):,}")

# Merge baseline + follow-up
eligible = ici_cancer_basics.merge(baseline, on="person_id").merge(
    followup, on="person_id"
)

# ── 1g. ESKD / transplant exclusion ──────────────────────────────
eskd_sql = f"""
SELECT DISTINCT co.person_id
FROM `{CDR}.condition_occurrence` co
JOIN `{CDR}.concept` c ON c.concept_id = co.condition_source_concept_id
WHERE c.vocabulary_id = 'ICD10CM'
  AND (c.concept_code IN ('N186')
       OR c.concept_code LIKE 'Z992%'
       OR c.concept_code LIKE 'Z490%' OR c.concept_code LIKE 'Z491%' OR c.concept_code LIKE 'Z492%'
       OR c.concept_code LIKE 'Z940%')
"""
eskd_pts = q(eskd_sql)
pre_eskd = len(eligible)
eligible = eligible[~eligible.person_id.isin(eskd_pts.person_id)]
eligible = eligible[eligible.baseline_cr < 4.0]
print(f"  Excluded ESKD/transplant/baseline≥4: {pre_eskd - len(eligible)}")
print(f"  Eligible cohort: {len(eligible):,}")

# ── 1h. AKI phenotyping ──────────────────────────────────────────
eligible["max_cr_ratio"] = eligible.max_followup_cr / eligible.baseline_cr
eligible["max_delta_cr"] = eligible.max_followup_cr - eligible.baseline_cr

# Primary: Cr ≥1.5× baseline (KDIGO Stage 1)
eligible["severity"] = (eligible.max_cr_ratio >= 1.5).astype(int)

# Sensitivity thresholds
eligible["aki_delta03"] = (eligible.max_delta_cr >= 0.3).astype(int)
eligible["aki_kdigo2"] = (eligible.max_cr_ratio >= 2.0).astype(int)
eligible["aki_kdigo3"] = (eligible.max_cr_ratio >= 3.0).astype(int)

# 180-day window
followup_180 = cr_merged[
    (cr_merged.days_from_ici >= 1) & (cr_merged.days_from_ici <= 180)
]
max_180 = followup_180.groupby("person_id").value_as_number.max().reset_index()
max_180.columns = ["person_id", "max_cr_180"]
eligible = eligible.merge(max_180, on="person_id", how="left")
eligible["aki_180d"] = (
    ((eligible.max_cr_180 / eligible.baseline_cr) >= 1.5).astype(int).fillna(0)
)

cases = eligible.severity.sum()
controls = (eligible.severity == 0).sum()
print(f"  Cases (Cr ≥1.5×): {cases:,} ({cases/len(eligible)*100:.1f}%)")
print(f"  Controls:          {controls:,}")

save(eligible, "01_eligible_cohort.csv")


# ═══════════════════════════════════════════════════════════════════
# STEP 2: DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Demographics")
print("=" * 70)

demo_sql = f"""
SELECT
  p.person_id,
  p.year_of_birth,
  p.gender_concept_id,
  p.race_concept_id,
  p.ethnicity_concept_id,
  p.sex_at_birth_concept_id
FROM `{CDR}.person` p
WHERE p.person_id IN ({','.join(str(x) for x in eligible.person_id.tolist())})
"""
demo = q(demo_sql)

# Sex at birth
sex_map = {45880669: "Male", 45878463: "Female"}
demo["sex_at_birth"] = demo.sex_at_birth_concept_id.map(sex_map).fillna("Other")

# Race
race_sql = f"""
SELECT person_id, answer.concept_name AS race_name
FROM `{CDR}.ds_survey` WHERE survey = 'The Basics'
  AND question_concept_id = 1586140
  AND person_id IN ({','.join(str(x) for x in eligible.person_id.tolist())})
"""
try:
    race_df = q(race_sql)
except:
    race_concept_map = {
        8516: "Black",
        8515: "Asian",
        8527: "White",
        8557: "Native_Hawaiian_PI",
        8657: "AIAN",
    }
    demo["race"] = demo.race_concept_id.map(race_concept_map).fillna("Other")
    race_df = None

if race_df is not None and len(race_df) > 0:

    def classify_race(name):
        if pd.isna(name):
            return "Other"
        n = str(name).lower()
        if "black" in n or "african" in n:
            return "Black"
        if "white" in n:
            return "White"
        if "asian" in n:
            return "Asian"
        if "hawaiian" in n or "pacific" in n:
            return "Native_Hawaiian_PI"
        if "american indian" in n or "alaska" in n:
            return "AIAN"
        if "hispanic" in n or "latino" in n:
            return "Hispanic"
        return "Other"

    race_df["race"] = race_df.race_name.apply(classify_race)
    race_final = race_df.groupby("person_id").race.first().reset_index()
    demo = demo.merge(race_final, on="person_id", how="left")
    if "race" not in demo.columns or demo.race.isna().all():
        demo["race"] = "Other"
else:
    if "race" not in demo.columns:
        demo["race"] = "Other"

demo.race = demo.race.fillna("Other")

# Ethnicity
eth_map = {38003563: "Hispanic", 38003564: "Not_Hispanic"}
demo["ethnicity"] = demo.ethnicity_concept_id.map(eth_map).fillna("Unknown")

# Age at ICI
demo = demo.merge(eligible[["person_id", "ici_index_date"]], on="person_id")
demo["age_at_ici"] = demo.ici_index_date.dt.year - demo.year_of_birth


def age_group(age):
    if age < 45:
        return "18-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    if age < 75:
        return "65-74"
    return "75+"


demo["age_group"] = demo.age_at_ici.apply(age_group)
print(f"  Demographics: {len(demo):,}")
for col in ["sex_at_birth", "race", "ethnicity", "age_group"]:
    print(f"    {col}:")
    for val, cnt in demo[col].value_counts().items():
        print(f"      {val:30s} {cnt:>5,}  ({cnt/len(demo)*100:.1f}%)")

save(
    demo[
        [
            "person_id",
            "sex_at_birth",
            "race",
            "ethnicity",
            "age_group",
            "age_at_ici",
            "year_of_birth",
        ]
    ],
    "02_demographics.csv",
)


# ═══════════════════════════════════════════════════════════════════
# STEP 3: NCI CHARLSON COMORBIDITY INDEX (14 conditions)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: NCI Charlson Comorbidity Index (v2 — corrected scoring)")
print("=" * 70)

# Get all diagnoses for eligible patients
dx_sql = f"""
SELECT
  co.person_id,
  UPPER(REPLACE(c.concept_code, '.', '')) AS icd_code
FROM `{CDR}.condition_occurrence` co
JOIN `{CDR}.concept` c
  ON c.concept_id = co.condition_source_concept_id
WHERE c.vocabulary_id IN ('ICD9CM', 'ICD10CM')
  AND co.person_id IN ({','.join(str(x) for x in eligible.person_id.tolist())})
"""
dx_all = q(dx_sql)
print(f"  Diagnosis records: {len(dx_all):,}")

# Build NCI-CCI flags using corrected module
charlson = pd.DataFrame({"person_id": eligible.person_id.values})
for condition in NCI_CCI_CONDITIONS:
    charlson[condition] = 0

for condition, codes in NCI_CODESETS.items():
    all_prefixes = []
    for ver_codes in codes.values():
        all_prefixes.extend(ver_codes)
    mask = dx_all.icd_code.apply(
        lambda x: (
            any(str(x).startswith(p) for p in all_prefixes) if pd.notna(x) else False
        )
    )
    flagged = dx_all[mask].person_id.unique()
    charlson.loc[charlson.person_id.isin(flagged), condition] = 1

# Cast to int8
for c in NCI_CCI_CONDITIONS:
    charlson[c] = charlson[c].astype("int8")

# ── SCORING (v2 fix) ─────────────────────────────────────────────
# CRITICAL: Do NOT zero out flags as pre-processing.
# ❌ OLD (v1, WRONG):
#   charlson.loc[charlson.Liver_Disease_Moderate_Severe==1, "Liver_Disease_Mild"] = 0
#   charlson.loc[charlson.Diabetes_Complicated==1, "Diabetes"] = 0
# ✅ NEW (v2, CORRECT): hierarchy handled inside scoring functions

charlson["charlson_score"] = compute_charlson_score(charlson)
charlson["nci_index"] = compute_nci_index(charlson)

# ── BACKWARD COMPAT: keep old column name for R scripts ──────────
# The R scripts reference 'nci_cci_score'. We use Charlson integer
# score (same as before, but now correctly computed).
charlson["nci_cci_score"] = charlson["charlson_score"]

# QC
both_diab = (
    (charlson["Diabetes"] == 1) & (charlson["Diabetes_Complicated"] == 1)
).sum()
both_mi = ((charlson["Acute_MI"] == 1) & (charlson["History_MI"] == 1)).sum()
both_para_cvd = (
    (charlson["Paralysis"] == 1) & (charlson["Cerebrovascular_Disease"] == 1)
).sum()
print(f"  QC: {both_diab} pts w/ both diabetes flags (raw, NOT zeroed)")
print(f"  QC: {both_mi} pts w/ both MI types → Charlson MI = 1pt (OR, not 2)")
print(f"  QC: {both_para_cvd} pts w/ paralysis+CVD → both score independently")
print(
    f"  Charlson score: median {charlson.charlson_score.median():.0f}, "
    f"IQR {charlson.charlson_score.quantile(0.25):.0f}–"
    f"{charlson.charlson_score.quantile(0.75):.0f}, "
    f"max {charlson.charlson_score.max():.0f}"
)
print(
    f"  NCI index: median {charlson.nci_index.median():.3f}, "
    f"max {charlson.nci_index.max():.3f}"
)

for c in NCI_CCI_CONDITIONS:
    print(f"    {c:40s} {charlson[c].sum():>6,}  ({charlson[c].mean()*100:.1f}%)")

save(charlson, "03_nci_charlson.csv")


# ═══════════════════════════════════════════════════════════════════
# STEP 4: SDoH (Basics Survey)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: SDoH (Basics Survey)")
print("=" * 70)

# SDoH concept IDs (CDR C2024Q3R9)
SDOH_CONCEPTS = {
    "insurance_type": 43528428,
    "income": 1585375,
    "education": 1585940,
    "employment": 1585952,
    "housing": 1585370,
    "housing_stability": 1585886,
}

sdoh_concept_ids = list(SDOH_CONCEPTS.values())
sdoh_sql = f"""
SELECT
  o.person_id,
  o.observation_source_concept_id,
  o.value_source_concept_id,
  vc.concept_name AS value_name
FROM `{CDR}.observation` o
LEFT JOIN `{CDR}.concept` vc ON vc.concept_id = o.value_source_concept_id
WHERE o.observation_source_concept_id IN ({','.join(str(x) for x in sdoh_concept_ids)})
  AND o.person_id IN ({','.join(str(x) for x in eligible.person_id.tolist())})
"""
sdoh_raw = q(sdoh_sql)
print(f"  SDoH observations: {len(sdoh_raw):,}")

# ── Insurance ─────────────────────────────────────────────────────
ins = sdoh_raw[sdoh_raw.observation_source_concept_id == 43528428].copy()


def classify_insurance(name):
    if pd.isna(name):
        return "Unknown"
    n = str(name).lower()
    if "medicaid" in n:
        return "Medicaid"
    if "medicare" in n:
        return "Medicare"
    if "employer" in n or "union" in n or "private" in n:
        return "Private"
    if "purchased" in n or "exchange" in n or "marketplace" in n:
        return "Private"
    if "military" in n or "va " in n or "tricare" in n or "champva" in n:
        return "VA_Military"
    if "indian" in n or "ihs" in n:
        return "IHS"
    if "no" in n or "uninsured" in n or "none" in n:
        return "Uninsured"
    return "Other"


ins["insurance_type"] = ins.value_name.apply(classify_insurance)
ins_final = ins.groupby("person_id").insurance_type.first().reset_index()

# ── Income ────────────────────────────────────────────────────────
inc = sdoh_raw[sdoh_raw.observation_source_concept_id == 1585375].copy()


def classify_income(name):
    # CDR C2024Q3R9 values: "Annual Income: less 10k", "Annual Income: 10k 25k", etc.
    if pd.isna(name):
        return "Unknown"
    n = str(name).lower()
    if "less 10k" in n:
        return "lt10k"
    if "10k 25k" in n:
        return "10k_25k"
    if "25k 35k" in n or "35k 50k" in n:
        return "25k_50k"
    if "50k 75k" in n:
        return "50k_75k"
    if "75k 100k" in n:
        return "75k_100k"
    if "100k 150k" in n or "150k 200k" in n or "more 200k" in n:
        return "gt100k"
    if "skip" in n or "prefer not" in n:
        return "Unknown"
    return "Unknown"


inc["income"] = inc.value_name.apply(classify_income)
inc_final = inc.groupby("person_id").income.first().reset_index()

# ── Education ─────────────────────────────────────────────────────
edu = sdoh_raw[sdoh_raw.observation_source_concept_id == 1585940].copy()


def classify_education(name):
    # CDR C2024Q3R9: "Highest Grade: Never Attended", "One Through Four",
    # "Five Through Eight", "Nine Through Eleven", "Twelve Or GED",
    # "College One to Three", "College Graduate", "Advanced Degree"
    if pd.isna(name):
        return "Unknown"
    n = str(name).lower()
    if (
        "never" in n
        or "one through four" in n
        or "five through eight" in n
        or "nine through eleven" in n
    ):
        return "lt_HS"
    if "twelve" in n or "ged" in n:
        return "HS_GED"
    if "college one" in n or "one to three" in n:
        return "Some_College"
    if "college graduate" in n:
        return "College"
    if "advanced" in n:
        return "Graduate"
    if "skip" in n or "prefer not" in n:
        return "Unknown"
    return "Unknown"


edu["education"] = edu.value_name.apply(classify_education)
edu_final = edu.groupby("person_id").education.first().reset_index()

# ── Employment ────────────────────────────────────────────────────
emp = sdoh_raw[sdoh_raw.observation_source_concept_id == 1585952].copy()


def classify_employment(name):
    if pd.isna(name):
        return "Unknown"
    n = str(name).lower()
    if "employed" in n and "not" not in n and "self" not in n:
        return "Employed"
    if "self" in n:
        return "Self_Employed"
    if "unemployed" in n or "out of work" in n or "looking" in n:
        return "Unemployed"
    if "retired" in n:
        return "Retired"
    if "unable" in n or "disabled" in n or "disability" in n:
        return "Unable_to_Work"
    if "student" in n:
        return "Student"
    if "homemaker" in n:
        return "Homemaker"
    if "skip" in n or "prefer not" in n:
        return "Unknown"
    return "Other"


emp["employment"] = emp.value_name.apply(classify_employment)
emp_final = emp.groupby("person_id").employment.first().reset_index()

# ── Housing ───────────────────────────────────────────────────────
hou = sdoh_raw[sdoh_raw.observation_source_concept_id == 1585370].copy()


def classify_housing(name):
    # CDR C2024Q3R9: "Current Home Own: Own", "Current Home Own: Rent",
    # "Current Home Own: Other Arrangement"
    # BUG FIX: "rent" is substring of "cuRRENT" and "own" is in prefix.
    # Must match the value AFTER the colon.
    if pd.isna(name):
        return "Unknown"
    n = str(name).lower()
    if ": rent" in n:
        return "Rent"
    if ": other" in n or "arrangement" in n:
        return "Other_Arrangement"
    if ": own" in n:
        return "Own"
    if "skip" in n or "prefer not" in n or "dont know" in n:
        return "Unknown"
    return "Unknown"


hou["housing"] = hou.value_name.apply(classify_housing)
hou_final = hou.groupby("person_id").housing.first().reset_index()

# ── Housing stability ─────────────────────────────────────────────
stab = sdoh_raw[sdoh_raw.observation_source_concept_id == 1585886].copy()


def classify_stability(name):
    # CDR C2024Q3R9: "Stable House Concern: No", "Stable House Concern: Yes"
    if pd.isna(name):
        return "Unknown"
    n = str(name).lower()
    if "concern: no" in n:
        return "Stable"
    if "concern: yes" in n:
        return "Unstable"
    if "skip" in n or "prefer not" in n:
        return "Unknown"
    return "Unknown"


stab["housing_stability"] = stab.value_name.apply(classify_stability)
stab_final = stab.groupby("person_id").housing_stability.first().reset_index()

# Assemble SDoH
sdoh = pd.DataFrame({"person_id": eligible.person_id.values})
for df_merge, col in [
    (ins_final, "insurance_type"),
    (inc_final, "income"),
    (edu_final, "education"),
    (emp_final, "employment"),
    (hou_final, "housing"),
    (stab_final, "housing_stability"),
]:
    sdoh = sdoh.merge(df_merge[["person_id", col]], on="person_id", how="left")

# Fill missing
for col in SDOH_CONCEPTS.keys():
    sdoh[col] = sdoh[col].astype(object).fillna("Unknown")

print(f"  SDoH assembled: {len(sdoh):,}")
for col in SDOH_CONCEPTS.keys():
    non_unk = (sdoh[col] != "Unknown").sum()
    print(f"    {col:25s} known: {non_unk:>5,} ({non_unk/len(sdoh)*100:.1f}%)")

save(sdoh, "04_sdoh.csv")


# ═══════════════════════════════════════════════════════════════════
# STEP 5: CANCER TYPE + ICI CLASS + NEPHROTOXINS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Cancer Type + ICI Class + Nephrotoxins")
print("=" * 70)

# ── Cancer type ───────────────────────────────────────────────────
cancer_dx_sql = f"""
SELECT
  co.person_id,
  c.concept_code,
  co.condition_start_date
FROM `{CDR}.condition_occurrence` co
JOIN `{CDR}.concept` c ON c.concept_id = co.condition_source_concept_id
WHERE c.vocabulary_id = 'ICD10CM'
  AND c.concept_code LIKE 'C%'
  AND co.person_id IN ({','.join(str(x) for x in eligible.person_id.tolist())})
"""
cancer_dx = q(cancer_dx_sql)


def classify_cancer(code):
    if pd.isna(code):
        return "Other_Solid"
    code = str(code).upper().replace(".", "")
    if code.startswith("C34"):
        return "Lung"
    if code.startswith("C43"):
        return "Melanoma"
    if code.startswith("C64") or code.startswith("C65"):
        return "Renal_Cell"
    if code.startswith("C67"):
        return "Urothelial"
    if code.startswith("C50"):
        return "Breast"
    if (
        code.startswith("C10")
        or code.startswith("C11")
        or code.startswith("C12")
        or code.startswith("C13")
        or code.startswith("C14")
        or code.startswith("C32")
    ):
        return "Head_Neck"
    if code.startswith("C22"):
        return "Hepatocellular"
    if code.startswith("C18") or code.startswith("C19") or code.startswith("C20"):
        return "Colorectal"
    if (
        code.startswith("C81")
        or code.startswith("C82")
        or code.startswith("C83")
        or code.startswith("C84")
        or code.startswith("C85")
        or code.startswith("C91")
        or code.startswith("C92")
    ):
        return "Hematologic"
    return "Other_Solid"


cancer_dx["cancer_type"] = cancer_dx.concept_code.apply(classify_cancer)
cancer_primary = (
    cancer_dx.groupby("person_id")
    .cancer_type.agg(lambda x: x.value_counts().index[0])
    .reset_index()
)


# ── ICI regimen classification ────────────────────────────────────
def classify_ici(drugs_list):
    if not isinstance(drugs_list, (list, np.ndarray)):
        return "unknown"
    drugs_str = " ".join([str(d).lower() for d in drugs_list])
    has_pd1 = any(
        a in drugs_str
        for a in ["nivolumab", "pembrolizumab", "cemiplimab", "dostarlimab"]
    )
    has_pdl1 = any(a in drugs_str for a in ["atezolizumab", "durvalumab", "avelumab"])
    has_ctla4 = any(a in drugs_str for a in ["ipilimumab", "tremelimumab"])
    if has_ctla4 and (has_pd1 or has_pdl1):
        return "combo_pd1_ctla4"
    if has_pd1:
        return "anti_pd1"
    if has_pdl1:
        return "anti_pdl1"
    if has_ctla4:
        return "anti_ctla4"
    return "other_ici"


ici_regimen = ici_cancer_basics[["person_id", "ici_drugs"]].copy()
ici_regimen["ici_regimen"] = ici_regimen.ici_drugs.apply(classify_ici)

# Merge cancer + ICI
covariates = eligible[["person_id"]].merge(cancer_primary, on="person_id", how="left")
covariates = covariates.merge(
    ici_regimen[["person_id", "ici_regimen"]], on="person_id", how="left"
)
covariates.cancer_type = covariates.cancer_type.fillna("Unknown")
covariates.ici_regimen = covariates.ici_regimen.fillna("unknown")

# ── Nephrotoxin flags ─────────────────────────────────────────────
NEPHROTOXIN_CLASSES = {
    "ppi": [
        "omeprazole",
        "pantoprazole",
        "lansoprazole",
        "esomeprazole",
        "rabeprazole",
        "dexlansoprazole",
    ],
    "nsaid": [
        "ibuprofen",
        "naproxen",
        "diclofenac",
        "meloxicam",
        "celecoxib",
        "indomethacin",
        "ketorolac",
        "piroxicam",
    ],
    "acei_arb": [
        "lisinopril",
        "enalapril",
        "ramipril",
        "benazepril",
        "captopril",
        "losartan",
        "valsartan",
        "irbesartan",
        "olmesartan",
        "candesartan",
        "telmisartan",
    ],
    "diuretic": [
        "furosemide",
        "hydrochlorothiazide",
        "chlorthalidone",
        "bumetanide",
        "torsemide",
        "spironolactone",
        "amiloride",
    ],
}

for drug_class, agents in NEPHROTOXIN_CLASSES.items():
    agent_likes = " OR ".join([f"LOWER(c.concept_name) LIKE '%{a}%'" for a in agents])
    neph_sql = f"""
    SELECT DISTINCT de.person_id
    FROM `{CDR}.drug_exposure` de
    JOIN `{CDR}.concept` c ON c.concept_id = de.drug_concept_id
    WHERE ({agent_likes})
      AND de.person_id IN ({','.join(str(x) for x in eligible.person_id.tolist())})
    """
    neph_pts = q(neph_sql)
    covariates[drug_class] = covariates.person_id.isin(neph_pts.person_id).astype(int)
    print(
        f"  {drug_class}: {covariates[drug_class].sum():,} ({covariates[drug_class].mean()*100:.1f}%)"
    )

# Collapsed factors
covariates["cancer_type_collapsed"] = covariates.cancer_type.apply(
    lambda x: x if x in ["Lung", "Melanoma"] else "Other"
)
covariates["ici_collapsed"] = covariates.ici_regimen.apply(
    lambda x: "anti_pd1" if x == "anti_pd1" else "other_combo"
)

print(f"  Cancer types: {covariates.cancer_type.value_counts().to_dict()}")
print(f"  ICI regimens: {covariates.ici_regimen.value_counts().to_dict()}")

save(covariates, "05_covariates.csv")


# ═══════════════════════════════════════════════════════════════════
# STEP 6: MATCHING VARIABLES
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Matching Variables")
print("=" * 70)

match_dx_sql = f"""
SELECT
  person_id,
  COUNT(DISTINCT condition_concept_id) AS n_diagnoses,
  MIN(condition_start_date) AS first_dx,
  MAX(condition_start_date) AS last_dx
FROM `{CDR}.condition_occurrence`
WHERE person_id IN ({','.join(str(x) for x in eligible.person_id.tolist())})
GROUP BY person_id
"""
match_vars = q(match_dx_sql)
match_vars["first_dx"] = parse_date(match_vars["first_dx"])
match_vars["last_dx"] = parse_date(match_vars["last_dx"])
match_vars["ehr_length_days"] = (match_vars.last_dx - match_vars.first_dx).dt.days
ref_date = match_vars.first_dx.min()
match_vars["enrollment_days"] = (match_vars.first_dx - ref_date).dt.days

print(f"  Matching vars: {len(match_vars):,}")
save(
    match_vars[["person_id", "n_diagnoses", "ehr_length_days", "enrollment_days"]],
    "06_matching_variables.csv",
)


# ═══════════════════════════════════════════════════════════════════
# STEP 7: REGRESSION BASE ASSEMBLY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Regression Base Assembly")
print("=" * 70)

reg = eligible[
    [
        "person_id",
        "severity",
        "ici_index_date",
        "baseline_cr",
        "max_cr_ratio",
        "max_delta_cr",
        "aki_delta03",
        "aki_kdigo2",
        "aki_kdigo3",
        "aki_180d",
    ]
].copy()
reg = reg.merge(
    demo[["person_id", "sex_at_birth", "race", "ethnicity", "age_group"]],
    on="person_id",
)
reg = reg.merge(charlson, on="person_id")
reg = reg.merge(covariates, on="person_id")
reg = reg.merge(sdoh, on="person_id")
reg = reg.merge(
    match_vars[["person_id", "enrollment_days", "n_diagnoses", "ehr_length_days"]],
    on="person_id",
)

print(f"  Regression base: {len(reg):,} rows, {reg.shape[1]} cols")
print(f"  Cases: {reg.severity.sum():,}  Controls: {(reg.severity==0).sum():,}")
print(
    f"  Charlson score: median {reg.charlson_score.median():.0f}, "
    f"IQR {reg.charlson_score.quantile(0.25):.0f}–{reg.charlson_score.quantile(0.75):.0f}"
)
print(f"  NCI index: median {reg.nci_index.median():.3f}")

save(reg, "07_pre_matching_base.csv")


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("AoU ETL COMPLETE (v2 — NCI-CCI scoring fix)")
print("=" * 70)
print(f"  Output: {RESULTS}/")
print(f"  Next:   Rscript 01b_psm.R ici_aki")
print(f"          Rscript 02_models.R ici_aki")
print(f"\n  NCI-CCI FIX SUMMARY:")
print(f"    ✅ MI scoring: OR logic (max 1pt)")
print(f"    ✅ Paralysis/CVD: independent scoring")
print(f"    ✅ No hierarchy pre-processing (raw flags preserved)")
print(f"    ✅ Both Charlson (integer) and NCI Index (continuous) computed")
