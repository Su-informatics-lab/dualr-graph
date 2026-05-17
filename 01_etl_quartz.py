#!/usr/bin/env python3
"""
01_etl_quartz.py — Self-contained ETL for Graph AI (memory-safe).

Runs on Quartz (~70 GB RAM, 7 threads). Reads the purpose-built INPC
OMOP CSV dump for post-ICI patients, produces feature tensors for SCP
to Tempest.

Strategy to stay under memory limit:
  - Chunk-read all large tables (drug, condition, measurement)
  - Filter to ICI patient IDs ASAP
  - Use both drug_source_value keyword matching and OMOP concept_id matching
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
import re
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

CHUNK = 500_000  # rows per chunk for large csvs
MIN_PREVALENCE = 0.01
CR_RATIO_THRESHOLD = 2.0  # Cr ≥ 2.0× baseline

AKI_WINDOWS = {"aki_3m": 90, "aki_6m": 180, "aki_12m": 365}
PRIMARY_WINDOW = "aki_12m"

# ── ICI keyword/concept matching ──
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
ICI_NAME_TO_CLASS = {kw: cls for cls, kws in ICI_KEYWORDS.items() for kw in kws}
ICI_REGEX = "|".join(re.escape(kw) for kw in ALL_ICI_KW)


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
NEPHROTOXIN_REGEX = "|".join(re.escape(kw) for kw in NEPHROTOXIN_KW)


def code_matches(code_str, prefixes):
    c = str(code_str).upper().replace(".", "")
    return any(c.startswith(p.upper().replace(".", "")) for p in prefixes)


def resolve_cols(csv_path, wanted, encoding=None):
    """Peek at CSV header, return usecols list in the file's actual case."""
    header = pd.read_csv(csv_path, nrows=0, encoding=encoding).columns.tolist()
    lower_map = {c.lower(): c for c in header}
    resolved = []
    for w in wanted:
        actual = lower_map.get(w.lower())
        if actual is None:
            raise ValueError(
                f"Column '{w}' not found in {csv_path}. " f"Available: {header[:20]}"
            )
        resolved.append(actual)
    return resolved


def resolve_optional_cols(csv_path, wanted, encoding=None):
    """Return existing optional columns in the file's actual case."""
    header = pd.read_csv(csv_path, nrows=0, encoding=encoding).columns.tolist()
    lower_map = {c.lower(): c for c in header}
    return [lower_map[w.lower()] for w in wanted if w.lower() in lower_map]


def normalize_pid_column(df, col="person_id"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col]).copy()
    df[col] = df[col].astype(int)
    return df


def load_ici_concept_map(concept_path):
    """Map OMOP concept_id/source concept_id to ICI class by concept_name."""
    concept_cols = resolve_cols(
        concept_path, ["concept_id", "concept_name"], encoding="cp1252"
    )
    concept_to_class = {}

    for chunk in pd.read_csv(
        concept_path,
        encoding="cp1252",
        low_memory=False,
        chunksize=CHUNK,
        usecols=concept_cols,
    ):
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk["concept_id"] = pd.to_numeric(chunk["concept_id"], errors="coerce")
        chunk = chunk.dropna(subset=["concept_id"])
        if chunk.empty:
            continue
        chunk["concept_id"] = chunk["concept_id"].astype(int)
        names = chunk["concept_name"].astype(str).str.lower()
        for drug_name, ici_class in ICI_NAME_TO_CLASS.items():
            mask = names.str.contains(drug_name, regex=False, na=False)
            if mask.any():
                for cid in chunk.loc[mask, "concept_id"]:
                    concept_to_class[int(cid)] = ici_class

    return concept_to_class


def load_creatinine_concepts(concept_path):
    """Find creatinine measurement concept IDs without holding all concept rows."""
    concept_cols = resolve_cols(
        concept_path, ["concept_id", "concept_name"], encoding="cp1252"
    )
    cr_concepts = set()

    for chunk in pd.read_csv(
        concept_path,
        encoding="cp1252",
        low_memory=False,
        chunksize=CHUNK,
        usecols=concept_cols,
    ):
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk["concept_id"] = pd.to_numeric(chunk["concept_id"], errors="coerce")
        chunk = chunk.dropna(subset=["concept_id"])
        if chunk.empty:
            continue
        chunk["concept_id"] = chunk["concept_id"].astype(int)
        names = chunk["concept_name"].astype(str)
        cr_mask = names.str.contains(
            "reatinine", case=False, na=False
        ) & names.str.contains(
            "serum|blood|plasma|Creatinine in S", case=False, na=False, regex=True
        )
        if cr_mask.any():
            cr_concepts.update(int(c) for c in chunk.loc[cr_mask, "concept_id"])

    cr_concepts.add(3016723)  # LOINC 2160-0, serum/plasma creatinine
    return cr_concepts


def add_numeric_concept_columns(chunk, concept_cols):
    for c in concept_cols:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
    return chunk


def concept_id_series(chunk, col):
    return chunk[col].fillna(-1).astype(np.int64)


# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("GRAPH AI — INPC ETL (Quartz, memory-safe)")
    print("=" * 70)
    print(f"  Data:   {DATA}")
    print(f"  Output: {OUT}/")
    print(f"  Mem:    {mem_mb():.0f} MB")

    drug_path = f"{DATA}/r6335_drug_exposure.csv"
    cond_path = f"{DATA}/r6335_condition_occurrence.csv"
    meas_path = f"{DATA}/r6335_measurement.csv"
    concept_path = f"{DATA}/r6335_concept.csv"
    person_path = f"{DATA}/r6335_person.csv"

    # ══════════════════════════════════════════════════════════════
    # STEP 1: FIND ICI PATIENTS (chunked drug_exposure scan)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: ICI Cohort (chunked drug_exposure scan)")
    print("=" * 70)

    print("  Loading ICI concept IDs from concept table...")
    ici_concept_to_class = load_ici_concept_map(concept_path)
    ici_concept_ids = set(ici_concept_to_class)
    print(f"  ICI concept IDs: {len(ici_concept_ids):,}")
    if not ici_concept_ids:
        raise RuntimeError("No ICI concept IDs found from concept table")

    # use drug_concept_id and, when present, drug_source_concept_id for coded rows
    drug_base_cols = resolve_cols(
        drug_path, ["person_id", "drug_exposure_start_date", "drug_source_value"]
    )
    drug_concept_cols_original = resolve_optional_cols(
        drug_path, ["drug_concept_id", "drug_source_concept_id"]
    )
    if not drug_concept_cols_original:
        raise ValueError(
            "Neither drug_concept_id nor drug_source_concept_id found in drug_exposure. "
            "Concept-based ICI detection cannot run."
        )
    drug_cols = drug_base_cols + drug_concept_cols_original
    drug_concept_cols = [c.lower() for c in drug_concept_cols_original]
    print(f"  ICI concept matching columns: {drug_concept_cols}")

    ici_rows = []
    n_drug_total = 0
    n_keyword_hits = 0
    n_concept_hits = 0

    for chunk in pd.read_csv(
        drug_path,
        low_memory=False,
        chunksize=CHUNK,
        usecols=drug_cols,
    ):
        n_drug_total += len(chunk)
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk = normalize_pid_column(chunk, "person_id")
        chunk = add_numeric_concept_columns(chunk, drug_concept_cols)

        source_text = chunk["drug_source_value"].astype(str)
        kw_mask = source_text.str.contains(ICI_REGEX, case=False, regex=True, na=False)

        concept_mask = pd.Series(False, index=chunk.index)
        for cid_col in drug_concept_cols:
            concept_mask = concept_mask | concept_id_series(chunk, cid_col).isin(
                ici_concept_ids
            )

        n_keyword_hits += int(kw_mask.sum())
        n_concept_hits += int(concept_mask.sum())

        mask = kw_mask | concept_mask
        if not mask.any():
            continue

        hits = chunk.loc[mask].copy()
        classes = hits["drug_source_value"].apply(classify_ici)
        for cid_col in drug_concept_cols:
            missing = classes.isna()
            if not missing.any():
                break
            mapped = concept_id_series(hits.loc[missing], cid_col).map(
                ici_concept_to_class
            )
            classes.loc[missing] = mapped

        hits["ici_class"] = classes
        hits = hits[hits.ici_class.notna()].copy()
        if not hits.empty:
            keep_cols = [
                "person_id",
                "drug_exposure_start_date",
                "drug_source_value",
                *drug_concept_cols,
                "ici_class",
            ]
            ici_rows.append(hits[keep_cols])

    ici = pd.concat(ici_rows, ignore_index=True) if ici_rows else pd.DataFrame()
    del ici_rows
    gc.collect()
    print(f"  Scanned {n_drug_total:,} drug rows")
    print(f"  ICI keyword hits: {n_keyword_hits:,}")
    print(f"  ICI concept hits: {n_concept_hits:,}")
    print(f"  ICI exposures found: {len(ici):,}")

    if len(ici) == 0:
        raise RuntimeError(
            "No ICI exposures found. Check drug_source_value keywords and concept_id mapping."
        )

    print(f"  Unique ICI patients: {ici.person_id.nunique():,}")
    print(f"  Mem: {mem_mb():.0f} MB")

    ici["drug_exposure_start_date"] = parse_date(ici["drug_exposure_start_date"])
    ici = ici.dropna(subset=["drug_exposure_start_date"])

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

    # anti_lag3 is folded into pd1 for feature construction
    has_lag3 = ici_index.ici_classes.apply(lambda s: "anti_lag3" in s)
    ici_index.loc[has_lag3, "ici_pd1"] = 1

    ici_pids = set(int(p) for p in ici_index.person_id)

    for c in ["ici_pd1", "ici_pdl1", "ici_ctla4"]:
        print(f"    {c}: {ici_index[c].sum():,}")

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
    patient_codes = {}  # pid → set of normalized ICD codes

    cancer_norm = [p.upper().replace(".", "") for p in CANCER_PREFIXES_10]
    eskd_norm = [p.upper().replace(".", "") for p in ESKD_PREFIXES]

    n_cond_total = 0
    n_ici_cond_rows = 0
    cond_cols = resolve_cols(cond_path, ["person_id", "condition_source_value"])
    first_chunk = True

    for chunk in pd.read_csv(
        cond_path,
        low_memory=False,
        chunksize=CHUNK,
        usecols=cond_cols,
    ):
        n_cond_total += len(chunk)
        chunk.columns = [c.lower() for c in chunk.columns]

        if first_chunk:
            print(
                f"  [diag] cond person_id dtype={chunk.person_id.dtype}, "
                f"sample={chunk.person_id.head(3).tolist()}"
            )
            print(
                f"  [diag] ici_pids type={type(next(iter(ici_pids)))}, "
                f"sample={list(ici_pids)[:3]}"
            )
            print(
                f"  [diag] cond_source_value sample="
                f"{chunk.condition_source_value.head(3).tolist()}"
            )
            first_chunk = False

        chunk = normalize_pid_column(chunk, "person_id")
        chunk = chunk[chunk.person_id.isin(ici_pids)]
        n_ici_cond_rows += len(chunk)
        if chunk.empty:
            continue

        raw = chunk.condition_source_value.astype(str)
        chunk["code"] = (
            raw.str.split("^^", regex=False)
            .str[-1]
            .str.upper()
            .str.replace(".", "", regex=False)
        )

        is_cancer = chunk.code.apply(
            lambda c: any(c.startswith(p) for p in cancer_norm)
        )
        cancer_pids.update(int(p) for p in chunk.loc[is_cancer, "person_id"])

        is_eskd = chunk.code.apply(lambda c: any(c.startswith(p) for p in eskd_norm))
        eskd_pids.update(int(p) for p in chunk.loc[is_eskd, "person_id"])

        for pid, code in zip(chunk.person_id, chunk.code):
            patient_codes.setdefault(int(pid), set()).add(code)

    print(f"  Scanned {n_cond_total:,} condition rows")
    print(f"  ICI patient condition rows: {n_ici_cond_rows:,}")
    print(f"  ICI + cancer dx: {len(cancer_pids & ici_pids):,}")
    print(f"  ESKD patients: {len(eskd_pids & ici_pids):,}")
    print(f"  Mem: {mem_mb():.0f} MB")

    valid_pids = (ici_pids & cancer_pids) - eskd_pids
    ici_index = ici_index[ici_index.person_id.isin(valid_pids)].copy()
    print(f"  After cancer + ESKD filter: {len(ici_index):,}")
    if len(ici_index) == 0:
        raise RuntimeError("No patients left after cancer + ESKD filter")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: CREATININE → AKI PHENOTYPING (chunked measurement)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: Creatinine → AKI (chunked measurement scan)")
    print("=" * 70)

    print("  Loading creatinine concept IDs from concept table...")
    cr_concepts = load_creatinine_concepts(concept_path)
    gc.collect()
    print(f"  Creatinine concept IDs: {len(cr_concepts):,}")

    valid_pid_set = set(int(p) for p in ici_index.person_id)
    cr_rows = []
    n_meas_total = 0
    meas_cols = resolve_cols(
        meas_path,
        ["person_id", "measurement_concept_id", "measurement_date", "value_as_number"],
    )

    for chunk in pd.read_csv(
        meas_path,
        low_memory=False,
        chunksize=CHUNK,
        usecols=meas_cols,
    ):
        n_meas_total += len(chunk)
        chunk.columns = [c.lower() for c in chunk.columns]

        if n_meas_total == len(chunk):
            print(
                f"  [diag] meas person_id dtype={chunk.person_id.dtype}, "
                f"sample={chunk.person_id.head(3).tolist()}"
            )
            print(
                f"  [diag] meas concept_id dtype="
                f"{chunk.measurement_concept_id.dtype}"
            )

        chunk["person_id"] = pd.to_numeric(chunk["person_id"], errors="coerce")
        chunk["measurement_concept_id"] = pd.to_numeric(
            chunk["measurement_concept_id"], errors="coerce"
        )
        chunk = chunk.dropna(subset=["person_id", "measurement_concept_id"])
        if chunk.empty:
            continue
        chunk["person_id"] = chunk["person_id"].astype(int)
        chunk["measurement_concept_id"] = chunk["measurement_concept_id"].astype(int)

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

    if len(cr) == 0:
        raise RuntimeError(
            f"No creatinine measurements found.\n"
            f"  valid_pid_set size: {len(valid_pid_set)}\n"
            f"  cr_concepts size: {len(cr_concepts)}\n"
            f"  Check: Step 2 produced 0 patients → no Cr to find."
        )

    print(f"  Patients with Cr: {cr.person_id.nunique():,}")
    print(f"  Mem: {mem_mb():.0f} MB")

    cr["measurement_date"] = parse_date(cr["measurement_date"])
    cr["value_as_number"] = pd.to_numeric(cr["value_as_number"], errors="coerce")
    cr = cr.dropna(subset=["value_as_number", "measurement_date"])
    cr = cr[(cr.value_as_number > 0.1) & (cr.value_as_number < 30)]

    cr = cr.merge(ici_dates, on="person_id")
    cr["days"] = (cr.measurement_date - cr.ici_index_date).dt.days

    # baseline: median in [-365, -7]; fallback to most recent in [-365, -1]
    baseline_main = cr[(cr.days >= -365) & (cr.days <= -7)]
    baseline_main_cr = (
        baseline_main.groupby("person_id")["value_as_number"]
        .median()
        .rename("baseline_cr")
    )

    baseline_fallback = cr[(cr.days >= -365) & (cr.days <= -1)].sort_values(
        ["person_id", "days"]
    )
    baseline_fallback_cr = (
        baseline_fallback.groupby("person_id")
        .tail(1)
        .set_index("person_id")["value_as_number"]
        .rename("baseline_cr")
    )

    baseline_cr = baseline_main_cr.combine_first(baseline_fallback_cr).rename(
        "baseline_cr"
    )
    fallback_only = baseline_cr.index.difference(baseline_main_cr.index)
    print(
        f"  Baseline Cr: main={len(baseline_main_cr):,}, "
        f"fallback_only={len(fallback_only):,}, total={len(baseline_cr):,}"
    )

    cohort = ici_index.merge(baseline_cr, on="person_id", how="inner")
    print(f"  With baseline Cr: {len(cohort):,}")

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

    has_fu = set(int(p) for p in followup.person_id)
    cohort = cohort[cohort.person_id.isin(has_fu)].copy()
    cohort = cohort[cohort.baseline_cr < 4.0].copy()

    del cr, followup, fu_with_base
    gc.collect()

    print(f"  Final cohort: {len(cohort):,}")
    if len(cohort) == 0:
        raise RuntimeError(
            "Final cohort is empty after follow-up and baseline Cr filters"
        )
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

    cohort_pids = set(int(p) for p in cohort.person_id)

    person_cols = resolve_cols(
        person_path,
        [
            "person_id",
            "year_of_birth",
            "gender_concept_id",
            "race_concept_id",
            "ethnicity_concept_id",
        ],
    )
    person = pd.read_csv(
        person_path,
        low_memory=False,
        usecols=person_cols,
    )
    person.columns = [c.lower() for c in person.columns]
    person = normalize_pid_column(person, "person_id")
    person = person[person.person_id.isin(cohort_pids)]

    cohort = cohort.merge(person, on="person_id", how="left")
    cohort["year_of_birth"] = pd.to_numeric(cohort["year_of_birth"], errors="coerce")
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

    for condition, codes in NCI_CCI.items():
        all_pfx = codes.get("9", []) + codes.get("10", [])
        all_pfx_norm = [p.upper().replace(".", "") for p in all_pfx]
        pts_with = set()
        for pid in cohort_pids:
            if pid not in patient_codes:
                continue
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

    # nephrotoxins: concomitant +/-30d by drug_source_value keyword
    nephro_pids = set()
    for chunk in pd.read_csv(
        drug_path,
        low_memory=False,
        chunksize=CHUNK,
        usecols=drug_cols,
    ):
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk = normalize_pid_column(chunk, "person_id")
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
        is_nephro = conco.drug_source_value.astype(str).str.contains(
            NEPHROTOXIN_REGEX, case=False, regex=True, na=False
        )
        nephro_pids.update(int(p) for p in conco.loc[is_nephro, "person_id"])

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

    age = cohort.age.values.astype(np.float64)
    age = ((age - np.nanmean(age)) / (np.nanstd(age) + 1e-8)).astype(np.float32)
    feature_names.append("age")
    feature_cols.append(age)
    is_binary.append(False)

    feature_names.append("sex_male")
    feature_cols.append((cohort.sex == "Male").values.astype(np.float32))
    is_binary.append(True)

    feature_names.append("race_black")
    feature_cols.append((cohort.race == "Black").values.astype(np.float32))
    is_binary.append(True)

    feature_names.append("ethnicity_hispanic")
    feature_cols.append((cohort.ethnicity == "Hispanic").values.astype(np.float32))
    is_binary.append(True)

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

    for c in ["ici_pd1", "ici_pdl1", "ici_ctla4"]:
        feature_names.append(c)
        feature_cols.append(cohort[c].values.astype(np.float32))
        is_binary.append(True)

    feature_names.append("nephrotoxin")
    feature_cols.append(cohort.nephrotoxin.values.astype(np.float32))
    is_binary.append(True)

    score = cohort.nci_cci_score.values.astype(np.float64)
    score = ((score - np.nanmean(score)) / (np.nanstd(score) + 1e-8)).astype(np.float32)
    feature_names.append("nci_cci_score")
    feature_cols.append(score)
    is_binary.append(False)

    feature_names.append("aki_event")
    feature_cols.append(cohort[PRIMARY_WINDOW].values.astype(np.float32))
    is_binary.append(True)

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
