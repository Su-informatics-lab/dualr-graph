#!/usr/bin/env python3
"""
01_etl_quartz.py — Self-contained ETL for Graph AI.

Runs on Quartz. Reads raw INPC OMOP CSV dump, produces feature matrix
for SCP to Tempest. Zero dependencies on post_ici_aki_disparity.

Data source: /N/project/depot/hw56/irAKI_data/structured_data/
  r6335_person.csv
  r6335_drug_exposure.csv
  r6335_condition_occurrence.csv
  r6335_measurement.csv
  r6335_concept.csv  (encoding: cp1252)

Output (→ SCP to Tempest):
  data/feature_matrix.pt    [F, N] float32
  data/feature_names.json   ordered node names
  data/binary_mask.pt       [F] bool
  data/cohort_meta.csv      person_id + AKI labels (3m/6m/12m)

Usage (on Quartz):
  module load r/4.5.1
  python 01_etl_quartz.py
  scp -r data/ tempest:~/graphai_ici_aki/data/
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
DATA = "/N/project/depot/hw56/irAKI_data/structured_data"
OUT = "data"
os.makedirs(OUT, exist_ok=True)

MIN_PREVALENCE = 0.01  # drop NCI-CCI flags below this

# AKI windows
AKI_WINDOWS = {
    "aki_3m": 90,
    "aki_6m": 180,
    "aki_12m": 365,
}
PRIMARY_WINDOW = "aki_12m"
CR_RATIO_THRESHOLD = 2.0  # Cr ≥ 2.0× baseline

# ═══════════════════════════════════════════════════════════════════
# ICI DRUG MAPPING
# ═══════════════════════════════════════════════════════════════════
ICI_KEYWORDS = {
    "anti_pd1": [
        "nivolumab",
        "pembrolizumab",
        "cemiplimab",
        "dostarlimab",
        "retifanlimab",
        "toripalimab",
        "tislelizumab",
    ],
    "anti_pdl1": ["atezolizumab", "durvalumab", "avelumab"],
    "anti_ctla4": ["ipilimumab", "tremelimumab"],
    "anti_lag3": ["relatlimab"],
}


def classify_ici(drug_name: str) -> str | None:
    """Classify a drug name string into ICI class."""
    if not isinstance(drug_name, str):
        return None
    low = drug_name.lower()
    for cls, keywords in ICI_KEYWORDS.items():
        for kw in keywords:
            if kw in low:
                return cls
    return None


# ═══════════════════════════════════════════════════════════════════
# NCI-CCI 14-CONDITION CODE SETS
# (Identical to nci_cci_scoring.py — reproduced here for independence)
# ═══════════════════════════════════════════════════════════════════
NCI_CCI = {
    "Acute_MI": {
        "9": ["410"],
        "10": ["I21", "I22"],
    },
    "History_MI": {
        "9": ["412"],
        "10": ["I252"],
    },
    "CHF": {
        "9": [
            "39891",
            "4280",
            "4281",
            "42820",
            "42821",
            "42822",
            "42823",
            "42830",
            "42831",
            "42832",
            "42833",
            "42840",
            "42841",
            "42842",
            "42843",
            "4289",
        ],
        "10": [
            "I0981",
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
    },
    "PVD": {
        "9": [
            "0930",
            "4373",
            "440",
            "441",
            "4431",
            "4432",
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
    },
    "CVD": {
        "9": ["36234", "430", "431", "432", "433", "434", "435", "436", "437", "438"],
        "10": [
            "G45",
            "G46",
            "H340",
            "H341",
            "H342",
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
    },
    "COPD": {
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
    },
    "Dementia": {
        "9": ["290", "2941", "3312"],
        "10": ["F01", "F02", "F03", "F051", "G30", "G311", "G312", "G3101", "G3109"],
    },
    "Paralysis": {
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
    },
    "Diabetes": {
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
            "E120",
            "E121",
            "E126",
            "E128",
            "E129",
            "E130",
            "E131",
            "E136",
            "E138",
            "E139",
            "E140",
            "E141",
            "E146",
            "E148",
            "E149",
        ],
    },
    "Diabetes_Complicated": {
        "9": ["2504", "2505", "2506", "2507"],
        "10": [
            "E102",
            "E103",
            "E104",
            "E105",
            "E112",
            "E113",
            "E114",
            "E115",
            "E122",
            "E123",
            "E124",
            "E125",
            "E132",
            "E133",
            "E134",
            "E135",
            "E142",
            "E143",
            "E144",
            "E145",
        ],
    },
    "Renal_Disease": {
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
            "5834",
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
    },
    "Liver_Disease_Mild": {
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
    },
    "Liver_Disease_Moderate_Severe": {
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
    },
    "Peptic_Ulcer_Disease": {
        "9": ["531", "532", "533", "534"],
        "10": ["K25", "K26", "K27", "K28"],
    },
    "Rheumatic_Disease": {
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
    },
    "AIDS": {
        "9": ["042"],
        "10": ["B20"],
    },
}

NCI_WEIGHTS = {
    "Acute_MI": 1,
    "History_MI": 1,
    "CHF": 1,
    "PVD": 1,
    "CVD": 1,
    "COPD": 1,
    "Dementia": 1,
    "Paralysis": 1,
    "Diabetes": 1,
    "Diabetes_Complicated": 1,
    "Renal_Disease": 1,
    "Liver_Disease_Mild": 1,
    "Liver_Disease_Moderate_Severe": 3,
    "Peptic_Ulcer_Disease": 1,
    "Rheumatic_Disease": 1,
    "AIDS": 6,
}

# Nephrotoxin concept names (substring match)
NEPHROTOXIN_KEYWORDS = [
    "ibuprofen",
    "naproxen",
    "diclofenac",
    "indomethacin",
    "meloxicam",
    "celecoxib",
    "ketorolac",
    "piroxicam",  # NSAIDs
    "gentamicin",
    "tobramycin",
    "amikacin",
    "vancomycin",  # aminoglycosides
    "amphotericin",
    "cisplatin",
    "carboplatin",  # other nephrotoxins
]

# Cancer ICD codes (for cohort confirmation)
CANCER_ICD10_PREFIXES = ["C", "D0"]

# ESKD exclusion
ESKD_ICD9 = ["5856", "V420", "V451", "V56"]
ESKD_ICD10 = ["N186", "Z940", "Z490", "Z491", "Z492", "Z992", "T861"]


def parse_date(s):
    return pd.to_datetime(s, format="mixed", dayfirst=False, errors="coerce")


def prefix_match(code: str, prefixes: list) -> bool:
    """Check if ICD code starts with any prefix (after removing dots)."""
    c = str(code).upper().replace(".", "")
    return any(c.startswith(p.upper().replace(".", "")) for p in prefixes)


# ═══════════════════════════════════════════════════════════════════
# MAIN ETL
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("GRAPH AI — INPC ETL (Quartz)")
    print("=" * 70)
    print(f"  Data: {DATA}")
    print(f"  Output: {OUT}/")

    # ── Load tables ───────────────────────────────────────────────
    print("\n  Loading tables...")
    person = pd.read_csv(f"{DATA}/r6335_person.csv", low_memory=False)
    print(f"    person: {len(person):,}")

    drug = pd.read_csv(
        f"{DATA}/r6335_drug_exposure.csv",
        low_memory=False,
        usecols=[
            "person_id",
            "drug_concept_id",
            "drug_exposure_start_date",
            "drug_source_value",
        ],
    )
    print(f"    drug_exposure: {len(drug):,}")

    concept = pd.read_csv(
        f"{DATA}/r6335_concept.csv",
        encoding="cp1252",
        low_memory=False,
        usecols=["concept_id", "concept_name", "vocabulary_id", "concept_class_id"],
    )
    print(f"    concept: {len(concept):,}")

    cond = pd.read_csv(
        f"{DATA}/r6335_condition_occurrence.csv",
        low_memory=False,
        usecols=[
            "person_id",
            "condition_concept_id",
            "condition_start_date",
            "condition_source_value",
        ],
    )
    print(f"    condition_occurrence: {len(cond):,}")

    meas = pd.read_csv(f"{DATA}/r6335_measurement.csv", low_memory=False)
    # Uppercase column names (INPC dump uses mixed case)
    meas.columns = [c.lower() for c in meas.columns]
    print(f"    measurement: {len(meas):,}")

    # ══════════════════════════════════════════════════════════════
    # STEP 1: ICI COHORT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: ICI Cohort Identification")
    print("=" * 70)

    # Map drug_concept_id → concept_name
    drug = drug.merge(
        concept[["concept_id", "concept_name"]].rename(
            columns={"concept_id": "drug_concept_id"}
        ),
        on="drug_concept_id",
        how="left",
    )
    drug["ici_class"] = drug["concept_name"].apply(classify_ici)
    # Also check source value
    drug.loc[drug.ici_class.isna(), "ici_class"] = drug.loc[
        drug.ici_class.isna(), "drug_source_value"
    ].apply(classify_ici)

    ici = drug[drug.ici_class.notna()].copy()
    ici["drug_exposure_start_date"] = parse_date(ici["drug_exposure_start_date"])
    print(f"  ICI exposures: {len(ici):,}")
    print(f"  Unique ICI patients: {ici.person_id.nunique():,}")

    # ICI index date = first ICI exposure
    ici_index = (
        ici.groupby("person_id")
        .agg(
            ici_index_date=("drug_exposure_start_date", "min"),
            ici_classes=("ici_class", lambda x: set(x)),
        )
        .reset_index()
    )

    # ICI regimen flags (graph nodes: 3 binary)
    for cls in ["anti_pd1", "anti_pdl1", "anti_ctla4"]:
        ici_index[f"ici_{cls.replace('anti_', '')}"] = ici_index["ici_classes"].apply(
            lambda s: int(cls in s)
        )
    # anti_lag3 → fold into pd1 (almost always combination)
    has_lag3 = ici_index.ici_classes.apply(lambda s: "anti_lag3" in s)
    ici_index.loc[has_lag3, "ici_pd1"] = 1

    print(f"  ICI regimen distribution:")
    for cls in ["ici_pd1", "ici_pdl1", "ici_ctla4"]:
        print(f"    {cls}: {ici_index[cls].sum():,}")

    # ── Cancer diagnosis confirmation ─────────────────────────────
    cond["condition_source_value"] = cond["condition_source_value"].astype(str)
    cancer_pts = cond[
        cond.condition_source_value.apply(
            lambda x: any(
                str(x).upper().replace(".", "").startswith(p)
                for p in CANCER_ICD10_PREFIXES
            )
        )
    ].person_id.unique()
    ici_index = ici_index[ici_index.person_id.isin(cancer_pts)].copy()
    print(f"  ICI + cancer dx: {len(ici_index):,}")

    # ── ESKD exclusion ────────────────────────────────────────────
    eskd_pts = cond[
        cond.condition_source_value.apply(
            lambda x: prefix_match(x, ESKD_ICD9 + ESKD_ICD10)
        )
    ].person_id.unique()
    n_eskd = ici_index.person_id.isin(eskd_pts).sum()
    ici_index = ici_index[~ici_index.person_id.isin(eskd_pts)].copy()
    print(f"  Excluded ESKD: {n_eskd:,}")
    print(f"  After ESKD exclusion: {len(ici_index):,}")

    # ══════════════════════════════════════════════════════════════
    # STEP 2: SERUM CREATININE → AKI PHENOTYPING
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: Creatinine → AKI Phenotyping")
    print("=" * 70)

    # Find creatinine measurements
    cr_concepts = concept[
        concept.concept_name.str.contains("creatinine", case=False, na=False)
        & concept.concept_name.str.contains("serum|blood|plasma", case=False, na=False)
    ].concept_id.tolist()

    # Also try by LOINC 2160-0 if available
    loinc_cr = concept[
        (concept.vocabulary_id == "LOINC")
        & (concept.concept_name.str.contains("Creatinine", case=False, na=False))
    ].concept_id.tolist()
    cr_concepts = list(set(cr_concepts + loinc_cr))

    cr = meas[meas.measurement_concept_id.isin(cr_concepts)].copy()
    cr = cr[cr.person_id.isin(ici_index.person_id)].copy()
    cr["measurement_date"] = parse_date(cr["measurement_date"])
    cr["value_as_number"] = pd.to_numeric(cr["value_as_number"], errors="coerce")
    cr = cr.dropna(subset=["value_as_number", "measurement_date"])
    cr = cr[(cr.value_as_number > 0.1) & (cr.value_as_number < 30)]
    print(f"  Creatinine measurements: {len(cr):,}")
    print(f"  Patients with Cr: {cr.person_id.nunique():,}")

    # Merge with ICI index date
    cr = cr.merge(ici_index[["person_id", "ici_index_date"]], on="person_id")
    cr["days_from_index"] = (cr.measurement_date - cr.ici_index_date).dt.days

    # Baseline Cr: median outpatient in [-365d, -7d]
    baseline = cr[(cr.days_from_index >= -365) & (cr.days_from_index <= -7)]
    baseline_cr = (
        baseline.groupby("person_id")["value_as_number"].median().rename("baseline_cr")
    )

    # Follow-up Cr: [1d, 365d]
    followup = cr[(cr.days_from_index >= 1) & (cr.days_from_index <= 365)]

    # Compute max Cr ratio per window
    cohort = ici_index.merge(baseline_cr, on="person_id", how="inner")
    n_no_baseline = len(ici_index) - len(cohort)
    print(f"  Excluded (no baseline Cr): {n_no_baseline:,}")

    # For each window, find max Cr ratio
    followup_with_base = followup.merge(
        cohort[["person_id", "baseline_cr", "ici_index_date"]],
        on="person_id",
    )
    followup_with_base["cr_ratio"] = (
        followup_with_base.value_as_number / followup_with_base.baseline_cr
    )

    for label, max_days in AKI_WINDOWS.items():
        window = followup_with_base[followup_with_base.days_from_index <= max_days]
        max_ratio = (
            window.groupby("person_id")["cr_ratio"]
            .max()
            .rename(f"max_cr_ratio_{label}")
        )
        cohort = cohort.merge(max_ratio, on="person_id", how="left")
        cohort[label] = (cohort[f"max_cr_ratio_{label}"] >= CR_RATIO_THRESHOLD).astype(
            int
        )
        cohort.loc[cohort[f"max_cr_ratio_{label}"].isna(), label] = np.nan

    # Drop patients with no follow-up Cr at all
    has_followup = followup.person_id.unique()
    n_no_followup = (~cohort.person_id.isin(has_followup)).sum()
    cohort = cohort[cohort.person_id.isin(has_followup)].copy()
    print(f"  Excluded (no follow-up Cr): {n_no_followup:,}")

    # Drop baseline Cr >= 4.0 (pre-existing severe CKD)
    n_high_base = (cohort.baseline_cr >= 4.0).sum()
    cohort = cohort[cohort.baseline_cr < 4.0].copy()
    print(f"  Excluded (baseline Cr ≥ 4.0): {n_high_base:,}")

    print(f"\n  Final cohort: {len(cohort):,}")
    for label in AKI_WINDOWS:
        valid = cohort[label].notna()
        n_events = int(cohort.loc[valid, label].sum())
        print(f"    {label}: {n_events:,} events " f"({n_events/valid.sum()*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: DEMOGRAPHICS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: Demographics")
    print("=" * 70)

    demo = person[person.person_id.isin(cohort.person_id)].copy()

    # Age
    cohort = cohort.merge(
        demo[["person_id", "year_of_birth"]],
        on="person_id",
        how="left",
    )
    cohort["age"] = cohort.ici_index_date.dt.year - cohort.year_of_birth

    # Sex (concept-based)
    sex_map = {8507: "Male", 8532: "Female"}
    demo["sex"] = demo.gender_concept_id.map(sex_map).fillna("Other")
    cohort = cohort.merge(demo[["person_id", "sex"]], on="person_id", how="left")

    # Race
    race_map = {8527: "White", 8516: "Black", 8515: "Asian"}
    demo["race"] = demo.race_concept_id.map(race_map).fillna("Other")
    cohort = cohort.merge(demo[["person_id", "race"]], on="person_id", how="left")

    # Ethnicity
    eth_map = {38003563: "Hispanic", 38003564: "Not Hispanic"}
    demo["ethnicity"] = demo.ethnicity_concept_id.map(eth_map).fillna("Other")
    cohort = cohort.merge(
        demo[["person_id", "ethnicity"]],
        on="person_id",
        how="left",
    )

    print(f"  Sex: {cohort.sex.value_counts().to_dict()}")
    print(f"  Race: {cohort.race.value_counts().to_dict()}")
    print(f"  Age: mean={cohort.age.mean():.1f}, " f"median={cohort.age.median():.0f}")

    # ══════════════════════════════════════════════════════════════
    # STEP 4: NCI-CCI COMORBIDITIES
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4: NCI-CCI (14 conditions)")
    print("=" * 70)

    cond_cohort = cond[cond.person_id.isin(cohort.person_id)].copy()
    cond_cohort["code"] = (
        cond_cohort.condition_source_value.astype(str)
        .str.upper()
        .str.replace(".", "", regex=False)
    )

    for condition, codes in NCI_CCI.items():
        all_prefixes = codes.get("9", []) + codes.get("10", [])
        mask = cond_cohort.code.apply(
            lambda c: any(
                c.startswith(p.upper().replace(".", "")) for p in all_prefixes
            )
        )
        pts_with = set(cond_cohort.loc[mask, "person_id"])
        cohort[condition] = cohort.person_id.isin(pts_with).astype(int)
        prev = cohort[condition].mean()
        print(f"    {condition:40s} {cohort[condition].sum():5,} " f"({prev*100:.1f}%)")

    # NCI-CCI score
    cohort["nci_cci_score"] = sum(
        cohort[cond].astype(int) * wt
        for cond, wt in NCI_WEIGHTS.items()
        if cond in cohort.columns
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 5: NEPHROTOXIN FLAG
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 5: Concomitant Nephrotoxins")
    print("=" * 70)

    drug_cohort = drug[drug.person_id.isin(cohort.person_id)].copy()
    drug_cohort["drug_exposure_start_date"] = parse_date(
        drug_cohort["drug_exposure_start_date"]
    )
    drug_cohort = drug_cohort.merge(
        cohort[["person_id", "ici_index_date"]],
        on="person_id",
    )
    # Concomitant = within [-30d, +30d] of ICI index
    drug_cohort["days"] = (
        drug_cohort.drug_exposure_start_date - drug_cohort.ici_index_date
    ).dt.days
    concomitant = drug_cohort[(drug_cohort.days >= -30) & (drug_cohort.days <= 30)]

    def has_nephrotoxin(names):
        if not isinstance(names, str):
            return False
        low = names.lower()
        return any(kw in low for kw in NEPHROTOXIN_KEYWORDS)

    nephro_check = concomitant.copy()
    nephro_check["is_nephrotoxin"] = nephro_check["concept_name"].apply(
        has_nephrotoxin
    ) | nephro_check["drug_source_value"].apply(has_nephrotoxin)
    nephro_pts = set(nephro_check[nephro_check.is_nephrotoxin].person_id)
    cohort["nephrotoxin"] = cohort.person_id.isin(nephro_pts).astype(int)
    print(
        f"  Nephrotoxin exposure: {cohort.nephrotoxin.sum():,} "
        f"({cohort.nephrotoxin.mean()*100:.1f}%)"
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 6: BUILD FEATURE MATRIX [F, N]
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 6: Build Feature Matrix")
    print("=" * 70)

    feature_names = []
    feature_cols = []
    is_binary = []

    # Demographics (4 nodes)
    # Age (standardized)
    age_raw = cohort["age"].values.astype(np.float64)
    age_std = ((age_raw - np.nanmean(age_raw)) / (np.nanstd(age_raw) + 1e-8)).astype(
        np.float32
    )
    feature_names.append("age")
    feature_cols.append(age_std)
    is_binary.append(False)

    # Sex (binary: male=1)
    feature_names.append("sex_male")
    feature_cols.append((cohort.sex == "Male").values.astype(np.float32))
    is_binary.append(True)

    # Race (binary: Black=1)
    feature_names.append("race_black")
    feature_cols.append((cohort.race == "Black").values.astype(np.float32))
    is_binary.append(True)

    # Ethnicity (binary: Hispanic=1)
    feature_names.append("ethnicity_hispanic")
    feature_cols.append((cohort.ethnicity == "Hispanic").values.astype(np.float32))
    is_binary.append(True)

    # NCI-CCI flags (drop low prevalence)
    dropped = []
    for condition in NCI_CCI:
        prev = cohort[condition].mean()
        if prev < MIN_PREVALENCE:
            dropped.append((condition, prev))
            continue
        feature_names.append(condition)
        feature_cols.append(cohort[condition].values.astype(np.float32))
        is_binary.append(True)

    if dropped:
        print(
            f"  Dropped {len(dropped)} NCI-CCI flags "
            f"(prevalence < {MIN_PREVALENCE}):"
        )
        for c, p in dropped:
            print(f"    {c}: {p*100:.2f}%")

    # ICI class (3 binary nodes)
    for cls in ["ici_pd1", "ici_pdl1", "ici_ctla4"]:
        feature_names.append(cls)
        feature_cols.append(cohort[cls].values.astype(np.float32))
        is_binary.append(True)

    # Nephrotoxin (1 binary node)
    feature_names.append("nephrotoxin")
    feature_cols.append(cohort["nephrotoxin"].values.astype(np.float32))
    is_binary.append(True)

    # NCI-CCI score (1 continuous node, standardized)
    score_raw = cohort["nci_cci_score"].values.astype(np.float64)
    score_std = (
        (score_raw - np.nanmean(score_raw)) / (np.nanstd(score_raw) + 1e-8)
    ).astype(np.float32)
    feature_names.append("nci_cci_score")
    feature_cols.append(score_std)
    is_binary.append(False)

    # AKI outcome node (using primary window)
    feature_names.append("aki_event")
    aki_col = cohort[PRIMARY_WINDOW].fillna(0).values.astype(np.float32)
    feature_cols.append(aki_col)
    is_binary.append(True)

    # Assemble
    X = np.stack(feature_cols, axis=0)  # [F, N]
    X = np.nan_to_num(X, nan=0.0)
    feature_matrix = torch.tensor(X, dtype=torch.float32)
    binary_mask = torch.tensor(is_binary, dtype=torch.bool)

    F_nodes, N = feature_matrix.shape
    aki_idx = feature_names.index("aki_event")

    print(f"\n  Feature matrix: [{F_nodes}, {N}]")
    print(
        f"  Nodes: {F_nodes} "
        f"({sum(is_binary)} binary, {F_nodes - sum(is_binary)} continuous)"
    )
    print(f"  AKI node index: {aki_idx}")

    for i, (name, b) in enumerate(zip(feature_names, is_binary)):
        tag = "bin" if b else "con"
        vals = feature_cols[i]
        if b:
            print(f"    [{i:2d}] {name:40s} ({tag})  " f"prev={vals.mean():.3f}")
        else:
            print(
                f"    [{i:2d}] {name:40s} ({tag})  "
                f"mean={vals.mean():.3f} std={vals.std():.3f}"
            )

    # ── Save ──────────────────────────────────────────────────────
    torch.save(feature_matrix, os.path.join(OUT, "feature_matrix.pt"))
    torch.save(binary_mask, os.path.join(OUT, "binary_mask.pt"))
    with open(os.path.join(OUT, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    # Cohort metadata
    meta_cols = ["person_id"] + list(AKI_WINDOWS.keys())
    meta = cohort[meta_cols].copy()
    meta.to_csv(os.path.join(OUT, "cohort_meta.csv"), index=False)

    print(f"\n  Saved to {OUT}/:")
    print(f"    feature_matrix.pt   {list(feature_matrix.shape)}")
    print(f"    feature_names.json  {F_nodes} names")
    print(f"    binary_mask.pt      {list(binary_mask.shape)}")
    print(f"    cohort_meta.csv     {len(meta):,} rows")
    print()
    print("  Next: scp -r data/ tempest:~/graphai_ici_aki/data/")
    print("=" * 70)


if __name__ == "__main__":
    main()
