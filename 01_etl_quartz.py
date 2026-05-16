#!/usr/bin/env python3
"""
01_etl_quartz.py — Self-contained ETL for Graph AI (memory-safe).

Runs on Quartz (~70 GB RAM, 7 threads). Reads the purpose-built INPC
OMOP CSV dump for post-ICI patients, produces feature tensors for SCP
to Tempest.

Strategy to stay under memory limit:
  - Chunk-read all large tables (drug, condition, measurement)
  - Filter to ICI patient IDs ASAP (drug_source_value keyword match,
    no full concept join)
  - Never hold more than one chunk of a large table at once
  - Explicit gc.collect() after each large table pass

Data source: /N/project/depot/hw56/irAKI_data/structured_data/

Output (→ SCP to Tempest):
  data/feature_matrix.pt    [F, N] float32
  data/feature_names.json   ordered node names
  data/binary_mask.pt       [F] bool
  data/cohort_meta.csv      person_id + AKI labels (3m/6m/12m)

Usage:
  python 01_etl_quartz.py
  scp -r data/ tempest:~/dualr-graph/data/
"""

import gc
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "7"
os.environ["OPENBLAS_NUM_THREADS"] = "7"
os.environ["MKL_NUM_THREADS"] = "7"

DATA = "/N/project/depot/hw56/irAKI_data/structured_data"
OUT = "data"
os.makedirs(OUT, exist_ok=True)

CHUNK = 500_000  # rows per chunk for large CSVs
MIN_PREVALENCE = 0.01
CR_RATIO_THRESHOLD = 2.0  # Cr ≥ 2.0× baseline

AKI_WINDOWS = {"aki_3m": 90, "aki_6m": 180, "aki_12m": 365}
PRIMARY_WINDOW = "aki_12m"

# ── ICI keyword matching (on drug_source_value, no concept join) ──
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
ALL_ICI_KW = [kw for v in ICI_KEYWORDS.values() for kw in v]


def classify_ici(text):
    if not isinstance(text, str):
        return None
    low = text.lower()
    for cls, kws in ICI_KEYWORDS.items():
        for kw in kws:
            if kw in low:
                return cls
    return None


def parse_date(s):
    return pd.to_datetime(s, format="mixed", dayfirst=False, errors="coerce")


def mem_mb():
    """Resident memory in MB (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0


# ── NCI-CCI code sets (14 conditions, excludes Malignancy/Metastatic) ──
NCI_CCI = {
    "Acute_MI": {"9": ["410"], "10": ["I21", "I22"]},
    "History_MI": {"9": ["412"], "10": ["I252"]},
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
    "AIDS": {"9": ["042"], "10": ["B20"]},
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

CANCER_PREFIXES_10 = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "D0"]
ESKD_PREFIXES = [
    "5856",
    "N186",
    "Z940",
    "V420",
    "V451",
    "V56",
    "Z490",
    "Z491",
    "Z492",
    "Z992",
    "T861",
]

NEPHROTOXIN_KW = [
    "ibuprofen",
    "naproxen",
    "diclofenac",
    "indomethacin",
    "meloxicam",
    "celecoxib",
    "ketorolac",
    "piroxicam",
    "gentamicin",
    "tobramycin",
    "amikacin",
    "vancomycin",
    "amphotericin",
    "cisplatin",
    "carboplatin",
]


def code_matches(code_str, prefixes):
    c = str(code_str).upper().replace(".", "")
    return any(c.startswith(p.upper().replace(".", "")) for p in prefixes)


# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("GRAPH AI — INPC ETL (Quartz, memory-safe)")
    print("=" * 70)
    print(f"  Data:   {DATA}")
    print(f"  Output: {OUT}/")
    print(f"  Mem:    {mem_mb():.0f} MB")

    # ══════════════════════════════════════════════════════════════
    # STEP 1: FIND ICI PATIENTS (chunked drug_exposure scan)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: ICI Cohort (chunked drug_exposure scan)")
    print("=" * 70)

    # Collect ICI exposures: person_id, date, class
    ici_rows = []
    n_drug_total = 0
    for chunk in pd.read_csv(
        f"{DATA}/r6335_drug_exposure.csv",
        low_memory=False,
        chunksize=CHUNK,
        usecols=["PERSON_ID", "DRUG_EXPOSURE_START_DATE", "DRUG_SOURCE_VALUE"],
    ):
        n_drug_total += len(chunk)
        chunk.columns = [c.lower() for c in chunk.columns]
        # Fast keyword filter on drug_source_value
        mask = (
            chunk["drug_source_value"]
            .astype(str)
            .str.lower()
            .apply(lambda x: any(kw in x for kw in ALL_ICI_KW))
        )
        if mask.any():
            hits = chunk[mask].copy()
            hits["ici_class"] = hits["drug_source_value"].apply(classify_ici)
            ici_rows.append(hits[hits.ici_class.notna()])

    ici = pd.concat(ici_rows, ignore_index=True) if ici_rows else pd.DataFrame()
    del ici_rows
    gc.collect()
    print(f"  Scanned {n_drug_total:,} drug rows")
    print(f"  ICI exposures found: {len(ici):,}")
    print(f"  Unique ICI patients: {ici.person_id.nunique():,}")
    print(f"  Mem: {mem_mb():.0f} MB")

    if len(ici) == 0:
        raise RuntimeError("No ICI exposures found — check drug_source_value keywords")

    # ICI index date + regimen flags
    ici["drug_exposure_start_date"] = parse_date(ici["drug_exposure_start_date"])
    ici_index = (
        ici.groupby("person_id")
        .agg(
            ici_index_date=("drug_exposure_start_date", "min"),
            ici_classes=("ici_class", lambda x: set(x)),
        )
        .reset_index()
    )
    for cls in ["anti_pd1", "anti_pdl1", "anti_ctla4"]:
        ici_index[f"ici_{cls.replace('anti_', '')}"] = ici_index["ici_classes"].apply(
            lambda s: int(cls in s)
        )
    # anti_lag3 → fold into pd1
    has_lag3 = ici_index.ici_classes.apply(lambda s: "anti_lag3" in s)
    ici_index.loc[has_lag3, "ici_pd1"] = 1
    ici_pids = set(ici_index.person_id)

    for c in ["ici_pd1", "ici_pdl1", "ici_ctla4"]:
        print(f"    {c}: {ici_index[c].sum():,}")

    # Free raw ICI exposures (keep ici_index only)
    # But we need drug data later for nephrotoxins — save ICI dates
    ici_dates = ici_index[["person_id", "ici_index_date"]].copy()
    del ici
    gc.collect()

    # ══════════════════════════════════════════════════════════════
    # STEP 2: CANCER DX + ESKD EXCLUSION (chunked condition scan)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: Cancer + ESKD filter (chunked condition scan)")
    print("=" * 70)

    cancer_pids = set()
    eskd_pids = set()
    # Also collect ICD codes per ICI patient for NCI-CCI (step 4)
    patient_codes = {}  # pid → set of normalized ICD codes

    n_cond_total = 0
    for chunk in pd.read_csv(
        f"{DATA}/r6335_condition_occurrence.csv",
        low_memory=False,
        chunksize=CHUNK,
        usecols=["PERSON_ID", "CONDITION_SOURCE_VALUE"],
    ):
        n_cond_total += len(chunk)
        chunk.columns = [c.lower() for c in chunk.columns]
        # Filter to ICI patients only
        chunk = chunk[chunk.person_id.isin(ici_pids)]
        if chunk.empty:
            continue

        chunk["code"] = (
            chunk.condition_source_value.astype(str)
            .str.upper()
            .str.replace(".", "", regex=False)
        )

        for _, row in chunk.iterrows():
            pid = row.person_id
            code = row.code

            # Cancer check
            if any(code.startswith(p) for p in CANCER_PREFIXES_10):
                cancer_pids.add(pid)
            # ESKD check
            if any(code.startswith(p.upper().replace(".", "")) for p in ESKD_PREFIXES):
                eskd_pids.add(pid)
            # Store codes for NCI-CCI
            if pid not in patient_codes:
                patient_codes[pid] = set()
            patient_codes[pid].add(code)

    print(f"  Scanned {n_cond_total:,} condition rows")
    print(f"  ICI + cancer dx: {len(cancer_pids & ici_pids):,}")
    print(f"  ESKD patients: {len(eskd_pids & ici_pids):,}")
    print(f"  Mem: {mem_mb():.0f} MB")

    # Apply filters
    valid_pids = (ici_pids & cancer_pids) - eskd_pids
    ici_index = ici_index[ici_index.person_id.isin(valid_pids)].copy()
    print(f"  After cancer + ESKD filter: {len(ici_index):,}")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: CREATININE → AKI PHENOTYPING (chunked measurement)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: Creatinine → AKI (chunked measurement scan)")
    print("=" * 70)

    # Find creatinine concept IDs from concept table (small filtered read)
    concept_cr = pd.read_csv(
        f"{DATA}/r6335_concept.csv",
        encoding="cp1252",
        low_memory=False,
        usecols=["CONCEPT_ID", "CONCEPT_NAME", "VOCABULARY_ID"],
    )
    concept_cr.columns = [c.lower() for c in concept_cr.columns]
    cr_mask = concept_cr.concept_name.str.contains(
        "reatinine", case=False, na=False
    ) & concept_cr.concept_name.str.contains(
        "serum|blood|plasma|Creatinine in S", case=False, na=False
    )
    cr_concepts = set(concept_cr.loc[cr_mask, "concept_id"].tolist())
    # Add known LOINC concept for serum creatinine
    cr_concepts.add(3016723)  # AoU/OMOP standard for LOINC 2160-0
    del concept_cr
    gc.collect()
    print(f"  Creatinine concept IDs: {len(cr_concepts)}")

    valid_pid_set = set(ici_index.person_id)
    cr_rows = []
    n_meas_total = 0

    for chunk in pd.read_csv(
        f"{DATA}/r6335_measurement.csv",
        low_memory=False,
        chunksize=CHUNK,
        usecols=[
            "PERSON_ID",
            "MEASUREMENT_CONCEPT_ID",
            "MEASUREMENT_DATE",
            "VALUE_AS_NUMBER",
        ],
    ):
        n_meas_total += len(chunk)
        chunk.columns = [c.lower() for c in chunk.columns]
        # Filter: ICI patients + creatinine concepts
        chunk = chunk[
            chunk.person_id.isin(valid_pid_set)
            & chunk.measurement_concept_id.isin(cr_concepts)
        ]
        if not chunk.empty:
            cr_rows.append(chunk)

    cr = pd.concat(cr_rows, ignore_index=True) if cr_rows else pd.DataFrame()
    del cr_rows
    gc.collect()
    print(f"  Scanned {n_meas_total:,} measurement rows")
    print(f"  Creatinine records: {len(cr):,}")
    print(f"  Patients with Cr: {cr.person_id.nunique():,}")
    print(f"  Mem: {mem_mb():.0f} MB")

    if len(cr) == 0:
        raise RuntimeError("No creatinine measurements found — check concept IDs")

    cr["measurement_date"] = parse_date(cr["measurement_date"])
    cr["value_as_number"] = pd.to_numeric(cr["value_as_number"], errors="coerce")
    cr = cr.dropna(subset=["value_as_number", "measurement_date"])
    cr = cr[(cr.value_as_number > 0.1) & (cr.value_as_number < 30)]

    # Merge with ICI index date
    cr = cr.merge(ici_dates, on="person_id")
    cr["days"] = (cr.measurement_date - cr.ici_index_date).dt.days

    # Baseline: median Cr in [-365d, -7d]
    baseline = cr[(cr.days >= -365) & (cr.days <= -7)]
    baseline_cr = (
        baseline.groupby("person_id")["value_as_number"].median().rename("baseline_cr")
    )

    cohort = ici_index.merge(baseline_cr, on="person_id", how="inner")
    print(f"  With baseline Cr: {len(cohort):,}")

    # Follow-up Cr per window
    followup = cr[(cr.days >= 1) & (cr.days <= 365)]
    fu_with_base = followup.merge(
        cohort[["person_id", "baseline_cr"]],
        on="person_id",
    )
    fu_with_base["cr_ratio"] = fu_with_base.value_as_number / fu_with_base.baseline_cr

    for label, max_days in AKI_WINDOWS.items():
        window = fu_with_base[fu_with_base.days <= max_days]
        max_ratio = (
            window.groupby("person_id")["cr_ratio"].max().rename(f"max_ratio_{label}")
        )
        cohort = cohort.merge(max_ratio, on="person_id", how="left")
        cohort[label] = (cohort[f"max_ratio_{label}"] >= CR_RATIO_THRESHOLD).astype(int)
        cohort.loc[cohort[f"max_ratio_{label}"].isna(), label] = 0

    # Require at least one follow-up Cr
    has_fu = set(followup.person_id)
    cohort = cohort[cohort.person_id.isin(has_fu)].copy()
    # Exclude baseline Cr >= 4.0
    cohort = cohort[cohort.baseline_cr < 4.0].copy()

    del cr, followup, fu_with_base
    gc.collect()

    print(f"  Final cohort: {len(cohort):,}")
    for label in AKI_WINDOWS:
        n = int(cohort[label].sum())
        print(f"    {label}: {n:,} events ({n/len(cohort)*100:.1f}%)")
    print(f"  Mem: {mem_mb():.0f} MB")

    # ══════════════════════════════════════════════════════════════
    # STEP 4: DEMOGRAPHICS + NCI-CCI + NEPHROTOXINS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4: Demographics + NCI-CCI + Nephrotoxins")
    print("=" * 70)

    # Demographics from person table
    person = pd.read_csv(
        f"{DATA}/r6335_person.csv",
        low_memory=False,
        usecols=[
            "PERSON_ID",
            "YEAR_OF_BIRTH",
            "GENDER_CONCEPT_ID",
            "RACE_CONCEPT_ID",
            "ETHNICITY_CONCEPT_ID",
        ],
    )
    person.columns = [c.lower() for c in person.columns]
    person = person[person.person_id.isin(cohort.person_id)]

    cohort = cohort.merge(person, on="person_id", how="left")
    cohort["age"] = cohort.ici_index_date.dt.year - cohort.year_of_birth

    sex_map = {8507: "Male", 8532: "Female"}
    cohort["sex"] = cohort.gender_concept_id.map(sex_map).fillna("Other")

    race_map = {8527: "White", 8516: "Black", 8515: "Asian"}
    cohort["race"] = cohort.race_concept_id.map(race_map).fillna("Other")

    eth_map = {38003563: "Hispanic", 38003564: "Not Hispanic"}
    cohort["ethnicity"] = cohort.ethnicity_concept_id.map(eth_map).fillna("Other")

    print(f"  Age: mean={cohort.age.mean():.1f}, median={cohort.age.median():.0f}")
    print(f"  Sex: {cohort.sex.value_counts().to_dict()}")
    print(f"  Race: {cohort.race.value_counts().to_dict()}")

    # NCI-CCI (from pre-collected patient_codes dict)
    cohort_pids = set(cohort.person_id)
    for condition, codes in NCI_CCI.items():
        all_pfx = codes.get("9", []) + codes.get("10", [])
        all_pfx_norm = [p.upper().replace(".", "") for p in all_pfx]
        pts_with = set()
        for pid in cohort_pids:
            if pid in patient_codes:
                for c in patient_codes[pid]:
                    if any(c.startswith(p) for p in all_pfx_norm):
                        pts_with.add(pid)
                        break
        cohort[condition] = cohort.person_id.isin(pts_with).astype(int)
        print(
            f"    {condition:40s} {cohort[condition].sum():5,} "
            f"({cohort[condition].mean()*100:.1f}%)"
        )

    cohort["nci_cci_score"] = sum(
        cohort[c].astype(int) * w for c, w in NCI_WEIGHTS.items() if c in cohort.columns
    )

    del patient_codes
    gc.collect()

    # Nephrotoxins (chunked drug scan, concomitant ±30d)
    nephro_pids = set()
    for chunk in pd.read_csv(
        f"{DATA}/r6335_drug_exposure.csv",
        low_memory=False,
        chunksize=CHUNK,
        usecols=["PERSON_ID", "DRUG_EXPOSURE_START_DATE", "DRUG_SOURCE_VALUE"],
    ):
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk = chunk[chunk.person_id.isin(cohort_pids)]
        if chunk.empty:
            continue
        chunk = chunk.merge(ici_dates, on="person_id")
        chunk["drug_exposure_start_date"] = parse_date(
            chunk["drug_exposure_start_date"]
        )
        chunk["days"] = (chunk.drug_exposure_start_date - chunk.ici_index_date).dt.days
        conco = chunk[(chunk.days >= -30) & (chunk.days <= 30)]
        if conco.empty:
            continue
        is_nephro = (
            conco.drug_source_value.astype(str)
            .str.lower()
            .apply(lambda x: any(kw in x for kw in NEPHROTOXIN_KW))
        )
        nephro_pids.update(conco.loc[is_nephro, "person_id"])

    cohort["nephrotoxin"] = cohort.person_id.isin(nephro_pids).astype(int)
    print(
        f"  Nephrotoxin: {cohort.nephrotoxin.sum():,} "
        f"({cohort.nephrotoxin.mean()*100:.1f}%)"
    )
    print(f"  Mem: {mem_mb():.0f} MB")

    # ══════════════════════════════════════════════════════════════
    # STEP 5: BUILD FEATURE MATRIX [F, N]
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 5: Build Feature Matrix")
    print("=" * 70)

    feature_names = []
    feature_cols = []
    is_binary = []

    # Age (standardized)
    age = cohort.age.values.astype(np.float64)
    age = ((age - np.nanmean(age)) / (np.nanstd(age) + 1e-8)).astype(np.float32)
    feature_names.append("age")
    feature_cols.append(age)
    is_binary.append(False)

    # Sex (male=1)
    feature_names.append("sex_male")
    feature_cols.append((cohort.sex == "Male").values.astype(np.float32))
    is_binary.append(True)

    # Race (Black=1)
    feature_names.append("race_black")
    feature_cols.append((cohort.race == "Black").values.astype(np.float32))
    is_binary.append(True)

    # Ethnicity (Hispanic=1)
    feature_names.append("ethnicity_hispanic")
    feature_cols.append((cohort.ethnicity == "Hispanic").values.astype(np.float32))
    is_binary.append(True)

    # NCI-CCI flags (drop low prevalence)
    dropped = []
    for cond in NCI_CCI:
        prev = cohort[cond].mean()
        if prev < MIN_PREVALENCE:
            dropped.append((cond, prev))
            continue
        feature_names.append(cond)
        feature_cols.append(cohort[cond].values.astype(np.float32))
        is_binary.append(True)
    if dropped:
        print(f"  Dropped {len(dropped)} flags (prev < {MIN_PREVALENCE}):")
        for c, p in dropped:
            print(f"    {c}: {p*100:.2f}%")

    # ICI class (3 binary)
    for c in ["ici_pd1", "ici_pdl1", "ici_ctla4"]:
        feature_names.append(c)
        feature_cols.append(cohort[c].values.astype(np.float32))
        is_binary.append(True)

    # Nephrotoxin
    feature_names.append("nephrotoxin")
    feature_cols.append(cohort.nephrotoxin.values.astype(np.float32))
    is_binary.append(True)

    # NCI-CCI score (standardized)
    score = cohort.nci_cci_score.values.astype(np.float64)
    score = ((score - np.nanmean(score)) / (np.nanstd(score) + 1e-8)).astype(np.float32)
    feature_names.append("nci_cci_score")
    feature_cols.append(score)
    is_binary.append(False)

    # AKI outcome node (last)
    feature_names.append("aki_event")
    feature_cols.append(cohort[PRIMARY_WINDOW].values.astype(np.float32))
    is_binary.append(True)

    # Assemble
    X = np.stack(feature_cols, axis=0)
    X = np.nan_to_num(X, nan=0.0)
    feature_matrix = torch.tensor(X, dtype=torch.float32)
    binary_mask = torch.tensor(is_binary, dtype=torch.bool)
    F, N = feature_matrix.shape

    print(f"\n  Feature matrix: [{F}, {N}]")
    print(f"  Nodes: {F} ({sum(is_binary)} binary, {F-sum(is_binary)} continuous)")
    for i, (name, b) in enumerate(zip(feature_names, is_binary)):
        v = feature_cols[i]
        tag = "bin" if b else "con"
        if b:
            print(f"    [{i:2d}] {name:40s} ({tag})  prev={v.mean():.3f}")
        else:
            print(f"    [{i:2d}] {name:40s} ({tag})  μ={v.mean():.3f} σ={v.std():.3f}")

    # Save
    torch.save(feature_matrix, os.path.join(OUT, "feature_matrix.pt"))
    torch.save(binary_mask, os.path.join(OUT, "binary_mask.pt"))
    with open(os.path.join(OUT, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    meta = cohort[["person_id"] + list(AKI_WINDOWS.keys())].copy()
    meta.to_csv(os.path.join(OUT, "cohort_meta.csv"), index=False)

    print(f"\n  Saved to {OUT}/:")
    print(f"    feature_matrix.pt   [{F}, {N}]")
    print(f"    feature_names.json  {F} names")
    print(f"    binary_mask.pt      [{F}]")
    print(f"    cohort_meta.csv     {len(meta):,} rows")
    print(f"  Mem: {mem_mb():.0f} MB")
    print(f"\n  Next: scp -r {OUT}/ tempest:~/dualr-graph/{OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
