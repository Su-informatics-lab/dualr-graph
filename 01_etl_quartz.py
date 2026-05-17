#!/usr/bin/env python3
"""
01_etl_quartz.py — Self-contained ETL for Graph AI (memory-safe).

AKI definition (v3, per Dr. Su 2026-05-17):
  Cr ≥ 1.5× baseline within 365d of first ICI
  OR  AKI ICD (N17.x / 584.x) during severe visit or incident (no pre-index AKI ICD)

ICI regimen (30-day window per Dr. Su):
  PD-1 mono [ref], PD-L1 mono, combo (any CTLA-4 in first 30d)

Features: demographics + NCI-CCI flags (no score) + ICI regimen + cancer type + nephrotoxin

Data source: /N/project/depot/hw56/irAKI_data/structured_data/
"""

import gc
import json
import os
import re
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "7"
os.environ["OPENBLAS_NUM_THREADS"] = "7"
os.environ["MKL_NUM_THREADS"] = "7"

DATA = "/N/project/depot/hw56/irAKI_data/structured_data"
OUT = "data"
FIG = "figures"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

CHUNK = 500_000
MIN_PREVALENCE = 0.01
CR_RATIO_THRESHOLD = 1.5  # any AKI (KDIGO stage 1+)

AKI_WINDOWS = {"aki_3m": 90, "aki_6m": 180, "aki_12m": 365}
PRIMARY_WINDOW = "aki_12m"

AKI_ICD_PREFIXES = {"9": ["584"], "10": ["N17"]}
AKI_ICD_NORM = [
    p.upper().replace(".", "") for codes in AKI_ICD_PREFIXES.values() for p in codes
]

SEVERE_VISIT_CONCEPTS = {9201, 9203, 262, 8717, 8782}

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


def classify_cancer(dx_code):
    """Cancer type from ICD-10 code (dots stripped, uppercase).
    Ported from AoU ETL (Chenxi's classification)."""
    code = str(dx_code).upper().replace(".", "")
    if code.startswith("C34"):
        return "Lung"
    if code.startswith("C43"):
        return "Melanoma"
    if code.startswith("C64") or code.startswith("C65"):
        return "Renal_Cell"
    if code.startswith("C67"):
        return "Urothelial"
    if any(
        code.startswith(p)
        for p in [
            "C00",
            "C01",
            "C02",
            "C03",
            "C04",
            "C05",
            "C06",
            "C07",
            "C08",
            "C09",
            "C10",
            "C11",
            "C12",
            "C13",
            "C14",
            "C30",
            "C31",
            "C32",
        ]
    ):
        return "Head_Neck"
    if code.startswith("C50"):
        return "Breast"
    if code.startswith("C22"):
        return "Hepatocellular"
    if any(code.startswith(p) for p in ["C18", "C19", "C20"]):
        return "Colorectal"
    if any(
        code.startswith(p)
        for p in [
            "C81",
            "C82",
            "C83",
            "C84",
            "C85",
            "C86",
            "C88",
            "C90",
            "C91",
            "C92",
            "C93",
            "C94",
            "C95",
            "C96",
        ]
    ):
        return "Hematologic"
    return "Other_Solid"


def parse_date(s):
    return pd.to_datetime(s, format="mixed", dayfirst=False, errors="coerce")


def mem_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0
    return 0


def apply_nature_style():
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.0,
            "legend.frameon": False,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.constrained_layout.use": True,
        }
    )


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


def resolve_cols(csv_path, wanted, encoding=None):
    header = pd.read_csv(csv_path, nrows=0, encoding=encoding).columns.tolist()
    lower_map = {c.lower(): c for c in header}
    resolved = []
    for w in wanted:
        actual = lower_map.get(w.lower())
        if actual is None:
            raise ValueError(
                f"Column '{w}' not in {csv_path}. Available: {header[:20]}"
            )
        resolved.append(actual)
    return resolved


def resolve_optional_cols(csv_path, wanted, encoding=None):
    header = pd.read_csv(csv_path, nrows=0, encoding=encoding).columns.tolist()
    lower_map = {c.lower(): c for c in header}
    return [lower_map[w.lower()] for w in wanted if w.lower() in lower_map]


def normalize_pid_column(df, col="person_id"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col]).copy()
    df[col] = df[col].astype(int)
    return df


def load_ici_concept_map(concept_path):
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
    cr_concepts.add(3016723)
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
    print("GRAPH AI — INPC ETL v4 (Quartz)")
    print("  AKI: Cr ≥1.5× OR ICD+severe/incident")
    print("  ICI: PD-1 mono [ref], PD-L1 mono, combo (CTLA-4 30d)")
    print("  Cancer: Lung [ref], Melanoma, Renal_Cell, Other")
    print("=" * 70)
    print(f"  Data:   {DATA}")
    print(f"  Output: {OUT}/")
    print(f"  Mem:    {mem_mb():.0f} MB")

    drug_path = f"{DATA}/r6335_drug_exposure.csv"
    cond_path = f"{DATA}/r6335_condition_occurrence.csv"
    meas_path = f"{DATA}/r6335_measurement.csv"
    concept_path = f"{DATA}/r6335_concept.csv"
    person_path = f"{DATA}/r6335_person.csv"
    visit_path = f"{DATA}/r6335_visit_occurrence.csv"

    # ══════════════════════════════════════════════════════════════
    # STEP 1: ICI PATIENTS + REGIMEN CLASSIFICATION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: ICI Cohort + regimen (30-day window)")
    print("=" * 70)

    print("  Loading ICI concept IDs...")
    ici_concept_to_class = load_ici_concept_map(concept_path)
    ici_concept_ids = set(ici_concept_to_class)
    print(f"  ICI concept IDs: {len(ici_concept_ids):,}")
    if not ici_concept_ids:
        raise RuntimeError("No ICI concept IDs found")

    drug_base_cols = resolve_cols(
        drug_path, ["person_id", "drug_exposure_start_date", "drug_source_value"]
    )
    drug_concept_cols_original = resolve_optional_cols(
        drug_path, ["drug_concept_id", "drug_source_concept_id"]
    )
    if not drug_concept_cols_original:
        raise ValueError("No drug concept columns found")
    drug_cols = drug_base_cols + drug_concept_cols_original
    drug_concept_cols = [c.lower() for c in drug_concept_cols_original]

    ici_rows = []
    n_drug_total = 0
    for chunk in pd.read_csv(
        drug_path, low_memory=False, chunksize=CHUNK, usecols=drug_cols
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
        mask = kw_mask | concept_mask
        if not mask.any():
            continue
        hits = chunk.loc[mask].copy()
        classes = hits["drug_source_value"].apply(classify_ici)
        for cid_col in drug_concept_cols:
            missing = classes.isna()
            if not missing.any():
                break
            classes.loc[missing] = concept_id_series(hits.loc[missing], cid_col).map(
                ici_concept_to_class
            )
        hits["ici_class"] = classes
        hits = hits[hits.ici_class.notna()].copy()
        if not hits.empty:
            ici_rows.append(
                hits[
                    [
                        "person_id",
                        "drug_exposure_start_date",
                        "drug_source_value",
                        *drug_concept_cols,
                        "ici_class",
                    ]
                ]
            )

    ici = pd.concat(ici_rows, ignore_index=True) if ici_rows else pd.DataFrame()
    del ici_rows
    gc.collect()
    print(f"  Scanned {n_drug_total:,} drug rows")
    print(f"  ICI exposures: {len(ici):,}, patients: {ici.person_id.nunique():,}")
    if len(ici) == 0:
        raise RuntimeError("No ICI exposures found")

    ici["drug_exposure_start_date"] = parse_date(ici["drug_exposure_start_date"])
    ici = ici.dropna(subset=["drug_exposure_start_date"])

    # Index date = first ICI
    ici_index = (
        ici.groupby("person_id")
        .agg(
            ici_index_date=("drug_exposure_start_date", "min"),
        )
        .reset_index()
    )

    # Regimen from first 30 days (Dr. Su: CTLA-4 in first month → combo)
    ici_30d = ici.merge(ici_index[["person_id", "ici_index_date"]], on="person_id")
    ici_30d["days"] = (
        ici_30d.drug_exposure_start_date - ici_30d.ici_index_date
    ).dt.days
    ici_30d = ici_30d[(ici_30d.days >= 0) & (ici_30d.days <= 30)]
    classes_30d = (
        ici_30d.groupby("person_id")["ici_class"].apply(set).rename("ici_classes_30d")
    )
    ici_index = ici_index.merge(classes_30d, on="person_id", how="left")
    ici_index["ici_classes_30d"] = ici_index["ici_classes_30d"].apply(
        lambda x: x if isinstance(x, set) else set()
    )

    def _regimen(classes):
        if "anti_ctla4" in classes:
            return "combo"
        if "anti_pdl1" in classes:
            return "pdl1_mono"
        return "pd1_mono"

    ici_index["ici_regimen"] = ici_index["ici_classes_30d"].apply(_regimen)
    ici_pids = set(int(p) for p in ici_index.person_id)
    print(f"  ICI regimen (30d): {ici_index.ici_regimen.value_counts().to_dict()}")

    ici_dates = ici_index[["person_id", "ici_index_date"]].copy()
    del ici, ici_30d
    gc.collect()

    # ══════════════════════════════════════════════════════════════
    # STEP 2: CANCER + ESKD + AKI ICD + CANCER DX COLLECTION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: Cancer + ESKD + AKI ICD + cancer dx codes")
    print("=" * 70)

    cancer_pids = set()
    eskd_pids = set()
    patient_codes = {}
    aki_icd_rows = []
    cancer_dx_rows = []  # NEW: for cancer type classification

    cancer_norm = [p.upper().replace(".", "") for p in CANCER_PREFIXES_10]
    eskd_norm = [p.upper().replace(".", "") for p in ESKD_PREFIXES]

    n_cond_total = 0
    cond_cols = resolve_cols(
        cond_path,
        [
            "person_id",
            "condition_source_value",
            "condition_start_date",
            "visit_occurrence_id",
        ],
    )

    for chunk in pd.read_csv(
        cond_path, low_memory=False, chunksize=CHUNK, usecols=cond_cols
    ):
        n_cond_total += len(chunk)
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk = normalize_pid_column(chunk, "person_id")
        chunk = chunk[chunk.person_id.isin(ici_pids)]
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

        # Cancer dx with dates (for type classification in Step 4)
        if is_cancer.any():
            cancer_dx_rows.append(
                chunk.loc[
                    is_cancer, ["person_id", "condition_start_date", "code"]
                ].copy()
            )

        is_eskd = chunk.code.apply(lambda c: any(c.startswith(p) for p in eskd_norm))
        eskd_pids.update(int(p) for p in chunk.loc[is_eskd, "person_id"])

        for pid, code in zip(chunk.person_id, chunk.code):
            patient_codes.setdefault(int(pid), set()).add(code)

        is_aki_icd = chunk.code.apply(
            lambda c: any(c.startswith(p) for p in AKI_ICD_NORM)
        )
        if is_aki_icd.any():
            aki_icd_rows.append(
                chunk.loc[
                    is_aki_icd,
                    ["person_id", "condition_start_date", "visit_occurrence_id"],
                ].copy()
            )

    print(f"  Scanned {n_cond_total:,} condition rows")
    print(f"  ICI + cancer: {len(cancer_pids & ici_pids):,}")
    print(f"  ESKD: {len(eskd_pids & ici_pids):,}")

    aki_icd_all = (
        pd.concat(aki_icd_rows, ignore_index=True)
        if aki_icd_rows
        else pd.DataFrame(
            columns=["person_id", "condition_start_date", "visit_occurrence_id"]
        )
    )
    cancer_dx_all = (
        pd.concat(cancer_dx_rows, ignore_index=True)
        if cancer_dx_rows
        else pd.DataFrame(columns=["person_id", "condition_start_date", "code"])
    )
    del aki_icd_rows, cancer_dx_rows
    gc.collect()
    print(f"  AKI ICD records: {len(aki_icd_all):,}")
    print(f"  Cancer dx records: {len(cancer_dx_all):,}")

    valid_pids = (ici_pids & cancer_pids) - eskd_pids
    ici_index = ici_index[ici_index.person_id.isin(valid_pids)].copy()
    print(f"  After cancer + ESKD filter: {len(ici_index):,}")
    if len(ici_index) == 0:
        raise RuntimeError("No patients after filter")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: CREATININE → AKI PHENOTYPING
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: Creatinine → AKI")
    print("=" * 70)

    cr_concepts = load_creatinine_concepts(concept_path)
    gc.collect()
    print(f"  Cr concept IDs: {len(cr_concepts):,}")

    valid_pid_set = set(int(p) for p in ici_index.person_id)
    cr_rows = []
    n_meas_total = 0
    meas_cols = resolve_cols(
        meas_path,
        ["person_id", "measurement_concept_id", "measurement_date", "value_as_number"],
    )

    for chunk in pd.read_csv(
        meas_path, low_memory=False, chunksize=CHUNK, usecols=meas_cols
    ):
        n_meas_total += len(chunk)
        chunk.columns = [c.lower() for c in chunk.columns]
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
    print(f"  Scanned {n_meas_total:,} meas rows, Cr records: {len(cr):,}")
    if len(cr) == 0:
        raise RuntimeError("No creatinine measurements found")

    cr["measurement_date"] = parse_date(cr["measurement_date"])
    cr["value_as_number"] = pd.to_numeric(cr["value_as_number"], errors="coerce")
    cr = cr.dropna(subset=["value_as_number", "measurement_date"])
    cr = cr[(cr.value_as_number > 0.1) & (cr.value_as_number < 30)]
    cr = cr.merge(ici_dates, on="person_id")
    cr["days"] = (cr.measurement_date - cr.ici_index_date).dt.days

    # Baseline Cr
    baseline_main = cr[(cr.days >= -365) & (cr.days <= -7)]
    baseline_main_cr = (
        baseline_main.groupby("person_id")["value_as_number"]
        .median()
        .rename("baseline_cr")
    )
    baseline_fb = cr[(cr.days >= -365) & (cr.days <= -1)].sort_values(
        ["person_id", "days"]
    )
    baseline_fb_cr = (
        baseline_fb.groupby("person_id")
        .tail(1)
        .set_index("person_id")["value_as_number"]
        .rename("baseline_cr")
    )
    baseline_cr = baseline_main_cr.combine_first(baseline_fb_cr).rename("baseline_cr")

    cohort = ici_index.merge(baseline_cr, on="person_id", how="inner")
    print(f"  With baseline Cr: {len(cohort):,}")

    followup = cr[(cr.days >= 1) & (cr.days <= 365)]
    fu = followup.merge(cohort[["person_id", "baseline_cr"]], on="person_id")
    fu["cr_ratio"] = fu.value_as_number / fu.baseline_cr

    for label, max_days in AKI_WINDOWS.items():
        w = fu[fu.days <= max_days]
        mr = w.groupby("person_id")["cr_ratio"].max().rename(f"max_ratio_{label}")
        cohort = cohort.merge(mr, on="person_id", how="left")
        cohort[label] = (cohort[f"max_ratio_{label}"] >= CR_RATIO_THRESHOLD).astype(int)
        cohort.loc[cohort[f"max_ratio_{label}"].isna(), label] = 0

    # Earliest Cr event date
    cr_events = fu[fu.cr_ratio >= CR_RATIO_THRESHOLD].copy()
    cr_first_evt = (
        (
            cr_events.sort_values("measurement_date")
            .groupby("person_id")
            .first()
            .reset_index()[["person_id", "measurement_date"]]
            .rename(columns={"measurement_date": "cr_evt_date"})
        )
        if len(cr_events) > 0
        else pd.DataFrame(columns=["person_id", "cr_evt_date"])
    )
    print(f"  Cr AKI events (≥{CR_RATIO_THRESHOLD}×): {len(cr_first_evt):,}")

    # Last obs for censoring
    last_obs = cr.groupby("person_id")["measurement_date"].max().rename("last_obs_date")
    cohort = cohort.merge(last_obs, on="person_id", how="left")

    has_fu = set(int(p) for p in followup.person_id)
    cohort = cohort[cohort.person_id.isin(has_fu) & (cohort.baseline_cr < 4.0)].copy()

    del cr, followup, fu, cr_events
    gc.collect()
    print(f"  Final cohort: {len(cohort):,}")
    for label in AKI_WINDOWS:
        n = int(cohort[label].sum())
        print(f"    {label} (Cr only): {n:,} ({n/len(cohort)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════
    # STEP 3b: ICD-BASED AKI (FIXED: single visit_occurrence read)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3b: ICD-based AKI (severe visit OR incident)")
    print("=" * 70)

    cohort_pids = set(int(p) for p in cohort.person_id)
    aki_icd = aki_icd_all[aki_icd_all.person_id.isin(cohort_pids)].copy()
    del aki_icd_all
    gc.collect()

    if len(aki_icd) > 0:
        aki_icd["condition_date"] = parse_date(aki_icd.condition_start_date)
        aki_icd = aki_icd.merge(cohort[["person_id", "ici_index_date"]], on="person_id")
        aki_icd["days"] = (aki_icd.condition_date - aki_icd.ici_index_date).dt.days

        # Filter A: severe visit (single read)
        if os.path.exists(visit_path):
            visit_cols = resolve_cols(
                visit_path, ["visit_occurrence_id", "person_id", "visit_concept_id"]
            )
            severe_visit_ids = set()
            for chunk in pd.read_csv(
                visit_path, low_memory=False, chunksize=CHUNK, usecols=visit_cols
            ):
                chunk.columns = [c.lower() for c in chunk.columns]
                chunk = normalize_pid_column(chunk, "person_id")
                chunk = chunk[
                    chunk.person_id.isin(cohort_pids)
                    & chunk.visit_concept_id.isin(SEVERE_VISIT_CONCEPTS)
                ]
                if not chunk.empty:
                    severe_visit_ids.update(chunk.visit_occurrence_id)
            print(f"  Severe visit IDs: {len(severe_visit_ids):,}")
            aki_icd["visit_occurrence_id"] = pd.to_numeric(
                aki_icd["visit_occurrence_id"], errors="coerce"
            )
            is_severe = aki_icd.visit_occurrence_id.isin(severe_visit_ids)
        else:
            is_severe = pd.Series(True, index=aki_icd.index)

        # Filter B: incident (no AKI ICD in [-90d, 0d])
        pre_idx = aki_icd[(aki_icd.days >= -90) & (aki_icd.days <= 0)]
        pts_pre = set(pre_idx.person_id)
        is_incident = ~aki_icd.person_id.isin(pts_pre)
        print(f"  Pre-index AKI ICD patients: {len(pts_pre):,}")

        aki_icd = aki_icd[is_severe | is_incident].copy()
        aki_icd = aki_icd[(aki_icd.days >= 1) & (aki_icd.days <= 365)]

        icd_first_evt = (
            aki_icd.sort_values("condition_date")
            .groupby("person_id")
            .first()
            .reset_index()[["person_id", "condition_date"]]
            .rename(columns={"condition_date": "icd_evt_date"})
        )
    else:
        icd_first_evt = pd.DataFrame(columns=["person_id", "icd_evt_date"])

    print(f"  ICD AKI events: {len(icd_first_evt):,}")
    del aki_icd
    gc.collect()

    # ══════════════════════════════════════════════════════════════
    # STEP 3c: UNION → SURVIVAL
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3c: Union → survival time")
    print("=" * 70)

    cohort = cohort.merge(cr_first_evt, on="person_id", how="left")
    cohort = cohort.merge(icd_first_evt, on="person_id", how="left")

    cr_ok = cohort.cr_evt_date.notna()
    icd_ok = cohort.icd_evt_date.notna()
    cr_first = cr_ok & (~icd_ok | (cohort.cr_evt_date <= cohort.icd_evt_date))
    icd_first = icd_ok & (~cr_ok | (cohort.icd_evt_date < cohort.cr_evt_date))

    cohort["evt_date"] = pd.NaT
    cohort.loc[cr_first, "evt_date"] = cohort.loc[cr_first, "cr_evt_date"]
    cohort.loc[icd_first, "evt_date"] = cohort.loc[icd_first, "icd_evt_date"]
    cohort["evt_source"] = "none"
    cohort.loc[cr_first, "evt_source"] = "cr"
    cohort.loc[icd_first, "evt_source"] = "icd"
    cohort["aki_event"] = (cohort.evt_source != "none").astype(int)

    has_evt = cohort.aki_event == 1
    cohort["max_fu_date"] = cohort.ici_index_date + pd.Timedelta(days=365)
    cohort.loc[has_evt, "surv_days"] = (
        cohort.loc[has_evt, "evt_date"] - cohort.loc[has_evt, "ici_index_date"]
    ).dt.days
    cohort.loc[~has_evt, "surv_days"] = (
        cohort.loc[~has_evt, ["last_obs_date", "max_fu_date"]].min(axis=1)
        - cohort.loc[~has_evt, "ici_index_date"]
    ).dt.days
    cohort["surv_days"] = cohort["surv_days"].clip(lower=1, upper=365).astype(int)

    n_total = len(cohort)
    n_evt = int(cohort.aki_event.sum())
    n_cr = int((cohort.evt_source == "cr").sum())
    n_icd = int((cohort.evt_source == "icd").sum())
    print(f"  Total: {n_total:,}, Events: {n_evt} ({n_evt/n_total*100:.1f}%)")
    print(f"    Cr-first: {n_cr}, ICD-first: {n_icd}")
    del cr_first_evt, icd_first_evt
    gc.collect()

    # ══════════════════════════════════════════════════════════════
    # STEP 4: DEMOGRAPHICS + NCI-CCI + CANCER TYPE + NEPHROTOXINS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4: Demographics + NCI-CCI + cancer type + nephrotoxins")
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
    person = pd.read_csv(person_path, low_memory=False, usecols=person_cols)
    person.columns = [c.lower() for c in person.columns]
    person = normalize_pid_column(person, "person_id")
    person = person[person.person_id.isin(cohort_pids)]

    cohort = cohort.merge(person, on="person_id", how="left")
    cohort["year_of_birth"] = pd.to_numeric(cohort["year_of_birth"], errors="coerce")
    cohort["age"] = cohort.ici_index_date.dt.year - cohort.year_of_birth
    cohort["sex"] = cohort.gender_concept_id.map({8507: "Male", 8532: "Female"}).fillna(
        "Other"
    )
    cohort["race"] = cohort.race_concept_id.map(
        {8527: "White", 8516: "Black", 8515: "Asian"}
    ).fillna("Other")
    cohort["ethnicity"] = cohort.ethnicity_concept_id.map(
        {38003563: "Hispanic", 38003564: "Not Hispanic"}
    ).fillna("Other")

    print(f"  Age: mean={cohort.age.mean():.1f}, median={cohort.age.median():.0f}")
    print(f"  Sex: {cohort.sex.value_counts().to_dict()}")
    print(f"  Race: {cohort.race.value_counts().to_dict()}")

    # ── Cancer type classification ────────────────────────────
    cancer_dx = cancer_dx_all[cancer_dx_all.person_id.isin(cohort_pids)].copy()
    del cancer_dx_all
    if len(cancer_dx) > 0:
        cancer_dx["cancer_type"] = cancer_dx["code"].apply(classify_cancer)
        cancer_dx["condition_start_date"] = parse_date(cancer_dx.condition_start_date)
        cancer_dx = cancer_dx.merge(
            cohort[["person_id", "ici_index_date"]], on="person_id"
        )
        cancer_dx["days_to_ici"] = abs(
            (cancer_dx.condition_start_date - cancer_dx.ici_index_date).dt.days
        )
        cancer_primary = (
            cancer_dx.sort_values("days_to_ici")
            .groupby("person_id")
            .first()
            .reset_index()[["person_id", "cancer_type"]]
        )
        cohort = cohort.merge(cancer_primary, on="person_id", how="left")
        cohort["cancer_type"] = cohort["cancer_type"].fillna("Unknown")
    else:
        cohort["cancer_type"] = "Unknown"

    # Collapse to 4 groups for power
    def _collapse(ct):
        if ct == "Lung":
            return "Lung"
        if ct == "Melanoma":
            return "Melanoma"
        if ct == "Renal_Cell":
            return "Renal_Cell"
        return "Other"

    cohort["cancer_collapsed"] = cohort["cancer_type"].apply(_collapse)
    print(f"  Cancer raw: {cohort.cancer_type.value_counts().to_dict()}")
    print(f"  Cancer collapsed: {cohort.cancer_collapsed.value_counts().to_dict()}")

    # ── NCI-CCI flags ─────────────────────────────────────────
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
            f"    {condition:40s} {cohort[condition].sum():5,} ({cohort[condition].mean()*100:.1f}%)"
        )

    del patient_codes
    gc.collect()

    # ── Nephrotoxins ──────────────────────────────────────────
    nephro_pids = set()
    for chunk in pd.read_csv(
        drug_path, low_memory=False, chunksize=CHUNK, usecols=drug_cols
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
        f"  Nephrotoxin: {cohort.nephrotoxin.sum():,} ({cohort.nephrotoxin.mean()*100:.1f}%)"
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 5: FEATURE MATRIX
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

    # Demographics
    feature_names.append("sex_male")
    feature_cols.append((cohort.sex == "Male").values.astype(np.float32))
    is_binary.append(True)
    feature_names.append("race_black")
    feature_cols.append((cohort.race == "Black").values.astype(np.float32))
    is_binary.append(True)
    feature_names.append("ethnicity_hispanic")
    feature_cols.append((cohort.ethnicity == "Hispanic").values.astype(np.float32))
    is_binary.append(True)

    # NCI-CCI flags (individual, NO score — score was collinear)
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
        print(
            f"  Dropped {len(dropped)} flags (prev < {MIN_PREVALENCE}): {[c for c,_ in dropped]}"
        )

    # ICI regimen: 2 binary flags (ref = pd1_mono)
    feature_names.append("ici_pdl1_mono")
    feature_cols.append((cohort.ici_regimen == "pdl1_mono").values.astype(np.float32))
    is_binary.append(True)
    feature_names.append("ici_combo")
    feature_cols.append((cohort.ici_regimen == "combo").values.astype(np.float32))
    is_binary.append(True)

    # Cancer type: 3 binary flags (ref = Lung)
    for ct in ["Melanoma", "Renal_Cell", "Other"]:
        feature_names.append(f"cancer_{ct}")
        feature_cols.append((cohort.cancer_collapsed == ct).values.astype(np.float32))
        is_binary.append(True)

    # Nephrotoxin
    feature_names.append("nephrotoxin")
    feature_cols.append(cohort.nephrotoxin.values.astype(np.float32))
    is_binary.append(True)

    # AKI event (last node)
    feature_names.append("aki_event")
    feature_cols.append(cohort["aki_event"].values.astype(np.float32))
    is_binary.append(True)

    X = np.stack(feature_cols, axis=0)
    X = np.nan_to_num(X, nan=0.0)
    feature_matrix = torch.tensor(X, dtype=torch.float32)
    binary_mask = torch.tensor(is_binary, dtype=torch.bool)
    F, N = feature_matrix.shape

    print(f"\n  Feature matrix: [{F}, {N}]")
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

    meta_cols = (
        ["person_id"]
        + list(AKI_WINDOWS.keys())
        + ["aki_event", "surv_days", "evt_source"]
    )
    meta = cohort[[c for c in meta_cols if c in cohort.columns]].copy()
    meta.to_csv(os.path.join(OUT, "cohort_meta.csv"), index=False)

    print(
        f"\n  Saved: feature_matrix.pt [{F},{N}], cohort_meta.csv [{len(meta):,} rows]"
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 6: KM CURVE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 6: KM curve")
    print("=" * 70)
    apply_nature_style()
    time_arr = cohort["surv_days"].values
    event_arr = cohort["aki_event"].values
    order = np.argsort(time_arr)
    t_sorted = time_arr[order]
    e_sorted = event_arr[order]
    unique_times = np.unique(t_sorted[e_sorted == 1])
    if len(unique_times) > 0:
        n_risk = np.array([np.sum(t_sorted >= t) for t in unique_times])
        n_events = np.array(
            [np.sum((t_sorted == t) & (e_sorted == 1)) for t in unique_times]
        )
        surv_prob = np.cumprod(1 - n_events / n_risk)
        km_times = np.concatenate([[0], unique_times])
        km_surv = np.concatenate([[1.0], surv_prob])
    else:
        km_times = np.array([0, 365])
        km_surv = np.array([1.0, 1.0])
        surv_prob = np.array([1.0])
    risk_ticks = [0, 30, 60, 90, 120, 180, 270, 365]
    risk_counts = [int(np.sum(t_sorted >= rt)) for rt in risk_ticks]
    C_BLUE = "#0072B2"
    fig, ax = plt.subplots(figsize=(3.504, 2.8))
    ax.step(
        km_times,
        km_surv,
        where="post",
        color=C_BLUE,
        linewidth=1.2,
        label=f"Overall (n={n_total:,})",
    )
    cens_t = t_sorted[e_sorted == 0]
    if len(cens_t) > 200:
        cens_t = np.random.default_rng(42).choice(cens_t, 200, replace=False)
    for ct in cens_t:
        idx = np.searchsorted(unique_times, ct, side="right") - 1
        y = (
            1.0
            if idx < 0
            else (surv_prob[idx] if idx < len(surv_prob) else surv_prob[-1])
        )
        ax.plot(ct, y, "|", color=C_BLUE, markersize=3, markeredgewidth=0.4)
    ax.set_xlabel("Days from ICI initiation")
    ax.set_ylabel("AKI-free survival probability")
    ax.set_xlim(0, 370)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(risk_ticks)
    ax.text(
        0.5,
        -0.22,
        "No. at risk",
        transform=ax.transAxes,
        fontsize=5,
        fontweight="bold",
        ha="center",
    )
    for rt, rc in zip(risk_ticks, risk_counts):
        ax.text(
            rt / 370,
            -0.28,
            str(rc),
            transform=ax.transAxes,
            fontsize=5,
            ha="center",
            color=C_BLUE,
        )
    ax.legend(loc="lower left", fontsize=6)
    ann = f"Events: {n_evt}\n  Cr ≥{CR_RATIO_THRESHOLD}×: {n_cr}\n  ICD: {n_icd}"
    ax.text(
        0.98,
        0.95,
        ann,
        transform=ax.transAxes,
        fontsize=5,
        ha="right",
        va="top",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="lightgrey",
            alpha=0.9,
        ),
    )
    fig.savefig(os.path.join(FIG, "km_aki_check.pdf"), dpi=600)
    fig.savefig(os.path.join(FIG, "km_aki_check.png"), dpi=150)
    plt.close()
    print(f"  Saved: {FIG}/km_aki_check.pdf + .png")

    print("\n" + "=" * 70)
    print("DONE — next: rebuild graphs, run baselines + Cox")
    print("=" * 70)


if __name__ == "__main__":
    main()
