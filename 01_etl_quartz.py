#!/usr/bin/env python3
"""
01_etl_quartz.py — Graph AI ETL v5

v5 changes (Path A features):
  - Baseline labs: BUN, hemoglobin, albumin, potassium (continuous)
  - BUN/Cr ratio (dehydration marker)
  - Nephrotoxin split: PPI, NSAID, vancomycin, ACEi/ARB, loop diuretic

AKI: Cr ≥1.5× OR ICD+severe/incident
ICI: PD-1 mono [ref], PD-L1 mono, combo (CTLA-4 30d)
Cancer: Lung [ref], Melanoma, Renal_Cell, Other
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
CR_RATIO_THRESHOLD = 1.5

AKI_WINDOWS = {"aki_3m": 90, "aki_6m": 180, "aki_12m": 365}
PRIMARY_WINDOW = "aki_12m"

AKI_ICD_PREFIXES = {"9": ["584"], "10": ["N17"]}
AKI_ICD_NORM = [
    p.upper().replace(".", "") for codes in AKI_ICD_PREFIXES.values() for p in codes
]
SEVERE_VISIT_CONCEPTS = {9201, 9203, 262, 8717, 8782}

# ── Lab concept IDs (from probe_inpc.py) ──────────────────────
LAB_CONCEPTS = {
    "cr": {3016723},  # Creatinine [Mass/volume] in Serum or Plasma
    "bun": {3013682},  # Urea nitrogen [Mass/volume] in Serum or Plasma
    "hgb": {3000963},  # Hemoglobin [Mass/volume] in Blood
    "alb": {3024561},  # Albumin [Mass/volume] in Serum or Plasma
    "k": {3023103},  # Potassium [Moles/volume] in Serum or Plasma
}
ALL_LAB_CONCEPTS = set().union(*LAB_CONCEPTS.values())

# Plausible value ranges (mg/dL or g/dL or mEq/L depending on lab)
LAB_RANGES = {
    "cr": (0.1, 30),  # mg/dL
    "bun": (1, 200),  # mg/dL
    "hgb": (2, 25),  # g/dL
    "alb": (0.5, 7),  # g/dL
    "k": (1.5, 10),  # mEq/L
}

# ── ICI keywords ──────────────────────────────────────────────
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

# ── Individual nephrotoxin drug classes ───────────────────────
NEPHRO_CLASSES = {
    "nephro_ppi": [
        "omeprazole",
        "esomeprazole",
        "lansoprazole",
        "pantoprazole",
        "rabeprazole",
        "dexlansoprazole",
    ],
    "nephro_nsaid": [
        "ibuprofen",
        "naproxen",
        "diclofenac",
        "indomethacin",
        "meloxicam",
        "celecoxib",
        "ketorolac",
        "piroxicam",
    ],
    "nephro_vanc": ["vancomycin"],
    "nephro_acei_arb": [
        "lisinopril",
        "enalapril",
        "ramipril",
        "captopril",
        "benazepril",
        "fosinopril",
        "losartan",
        "valsartan",
        "irbesartan",
        "candesartan",
        "olmesartan",
        "telmisartan",
    ],
    "nephro_diuretic": ["furosemide", "bumetanide", "torsemide"],
}
NEPHRO_REGEXES = {
    cls: "|".join(re.escape(kw) for kw in kws) for cls, kws in NEPHRO_CLASSES.items()
}
# Combined for backward compat
ALL_NEPHRO_KW = [kw for kws in NEPHRO_CLASSES.values() for kw in kws]
ALL_NEPHRO_KW += [
    "gentamicin",
    "tobramycin",
    "amikacin",
    "amphotericin",
    "cisplatin",
    "carboplatin",
]
NEPHROTOXIN_REGEX = "|".join(re.escape(kw) for kw in ALL_NEPHRO_KW)


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
    except:
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


def resolve_cols(csv_path, wanted, encoding=None):
    header = pd.read_csv(csv_path, nrows=0, encoding=encoding).columns.tolist()
    lower_map = {c.lower(): c for c in header}
    resolved = []
    for w in wanted:
        actual = lower_map.get(w.lower())
        if actual is None:
            raise ValueError(f"Column '{w}' not in {csv_path}")
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
    m = {}
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
        chunk["concept_id"] = chunk["concept_id"].astype(int)
        names = chunk["concept_name"].astype(str).str.lower()
        for dn, ic in ICI_NAME_TO_CLASS.items():
            mask = names.str.contains(dn, regex=False, na=False)
            if mask.any():
                for cid in chunk.loc[mask, "concept_id"]:
                    m[int(cid)] = ic
    return m


def add_numeric_concept_columns(chunk, cols):
    for c in cols:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
    return chunk


def concept_id_series(chunk, col):
    return chunk[col].fillna(-1).astype(np.int64)


# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("GRAPH AI — INPC ETL v5")
    print("  AKI: Cr≥1.5× OR ICD+severe/incident")
    print("  NEW: baseline BUN/Hgb/Alb/K + individual nephrotoxin classes")
    print("=" * 70)
    print(f"  Data: {DATA}\n  Output: {OUT}/\n  Mem: {mem_mb():.0f} MB")

    drug_path = f"{DATA}/r6335_drug_exposure.csv"
    cond_path = f"{DATA}/r6335_condition_occurrence.csv"
    meas_path = f"{DATA}/r6335_measurement.csv"
    concept_path = f"{DATA}/r6335_concept.csv"
    person_path = f"{DATA}/r6335_person.csv"
    visit_path = f"{DATA}/r6335_visit_occurrence.csv"

    # ══════════════════ STEP 1: ICI COHORT + REGIMEN ══════════
    print("\n" + "=" * 70 + "\nSTEP 1: ICI Cohort + regimen (30d)\n" + "=" * 70)
    ici_cmap = load_ici_concept_map(concept_path)
    ici_cids = set(ici_cmap)
    print(f"  ICI concept IDs: {len(ici_cids):,}")

    drug_base = resolve_cols(
        drug_path, ["person_id", "drug_exposure_start_date", "drug_source_value"]
    )
    drug_concept_orig = resolve_optional_cols(
        drug_path, ["drug_concept_id", "drug_source_concept_id"]
    )
    drug_cols = drug_base + drug_concept_orig
    dcols = [c.lower() for c in drug_concept_orig]

    ici_rows = []
    n_drug = 0
    for chunk in pd.read_csv(
        drug_path, low_memory=False, chunksize=CHUNK, usecols=drug_cols
    ):
        n_drug += len(chunk)
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk = normalize_pid_column(chunk)
        chunk = add_numeric_concept_columns(chunk, dcols)
        kw = chunk.drug_source_value.astype(str).str.contains(
            ICI_REGEX, case=False, regex=True, na=False
        )
        cm = pd.Series(False, index=chunk.index)
        for cc in dcols:
            cm = cm | concept_id_series(chunk, cc).isin(ici_cids)
        mask = kw | cm
        if not mask.any():
            continue
        hits = chunk.loc[mask].copy()
        classes = hits.drug_source_value.apply(classify_ici)
        for cc in dcols:
            mis = classes.isna()
            if not mis.any():
                break
            classes.loc[mis] = concept_id_series(hits.loc[mis], cc).map(ici_cmap)
        hits["ici_class"] = classes
        hits = hits[hits.ici_class.notna()]
        if not hits.empty:
            ici_rows.append(
                hits[
                    [
                        "person_id",
                        "drug_exposure_start_date",
                        "drug_source_value",
                        *dcols,
                        "ici_class",
                    ]
                ]
            )

    ici = pd.concat(ici_rows, ignore_index=True) if ici_rows else pd.DataFrame()
    del ici_rows
    gc.collect()
    print(
        f"  Scanned {n_drug:,} drug rows, ICI: {len(ici):,}, patients: {ici.person_id.nunique():,}"
    )
    if len(ici) == 0:
        raise RuntimeError("No ICI found")

    ici["drug_exposure_start_date"] = parse_date(ici.drug_exposure_start_date)
    ici = ici.dropna(subset=["drug_exposure_start_date"])
    ici_index = (
        ici.groupby("person_id")
        .agg(ici_index_date=("drug_exposure_start_date", "min"))
        .reset_index()
    )

    # Regimen from first 30 days
    ici_30d = ici.merge(ici_index, on="person_id")
    ici_30d["days"] = (
        ici_30d.drug_exposure_start_date - ici_30d.ici_index_date
    ).dt.days
    ici_30d = ici_30d[(ici_30d.days >= 0) & (ici_30d.days <= 30)]
    cls30 = ici_30d.groupby("person_id")["ici_class"].apply(set).rename("cls30")
    ici_index = ici_index.merge(cls30, on="person_id", how="left")
    ici_index["cls30"] = ici_index.cls30.apply(
        lambda x: x if isinstance(x, set) else set()
    )
    ici_index["ici_regimen"] = ici_index.cls30.apply(
        lambda c: (
            "combo"
            if "anti_ctla4" in c
            else ("pdl1_mono" if "anti_pdl1" in c else "pd1_mono")
        )
    )
    ici_pids = set(int(p) for p in ici_index.person_id)
    print(f"  Regimen: {ici_index.ici_regimen.value_counts().to_dict()}")
    ici_dates = ici_index[["person_id", "ici_index_date"]].copy()
    del ici, ici_30d
    gc.collect()

    # ══════════════════ STEP 2: CANCER + ESKD + AKI ICD ═══════
    print(
        "\n" + "=" * 70 + "\nSTEP 2: Cancer + ESKD + AKI ICD + cancer dx\n" + "=" * 70
    )
    cancer_pids = set()
    eskd_pids = set()
    patient_codes = {}
    aki_icd_rows = []
    cancer_dx_rows = []
    cancer_norm = [p.upper().replace(".", "") for p in CANCER_PREFIXES_10]
    eskd_norm = [p.upper().replace(".", "") for p in ESKD_PREFIXES]
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
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk = normalize_pid_column(chunk)
        chunk = chunk[chunk.person_id.isin(ici_pids)]
        if chunk.empty:
            continue
        chunk["code"] = (
            chunk.condition_source_value.astype(str)
            .str.split("^^", regex=False)
            .str[-1]
            .str.upper()
            .str.replace(".", "", regex=False)
        )
        ic = chunk.code.apply(lambda c: any(c.startswith(p) for p in cancer_norm))
        cancer_pids.update(int(p) for p in chunk.loc[ic, "person_id"])
        if ic.any():
            cancer_dx_rows.append(
                chunk.loc[ic, ["person_id", "condition_start_date", "code"]].copy()
            )
        ie = chunk.code.apply(lambda c: any(c.startswith(p) for p in eskd_norm))
        eskd_pids.update(int(p) for p in chunk.loc[ie, "person_id"])
        for pid, code in zip(chunk.person_id, chunk.code):
            patient_codes.setdefault(int(pid), set()).add(code)
        ia = chunk.code.apply(lambda c: any(c.startswith(p) for p in AKI_ICD_NORM))
        if ia.any():
            aki_icd_rows.append(
                chunk.loc[
                    ia, ["person_id", "condition_start_date", "visit_occurrence_id"]
                ].copy()
            )

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
    valid_pids = (ici_pids & cancer_pids) - eskd_pids
    ici_index = ici_index[ici_index.person_id.isin(valid_pids)].copy()
    print(
        f"  Cancer: {len(cancer_pids&ici_pids):,}, ESKD: {len(eskd_pids&ici_pids):,}, after filter: {len(ici_index):,}"
    )

    # ══════════════════ STEP 3: LABS (Cr + BUN + Hgb + Alb + K) ═══
    print(
        "\n" + "=" * 70 + "\nSTEP 3: Labs → AKI + baseline BUN/Hgb/Alb/K\n" + "=" * 70
    )

    valid_pid_set = set(int(p) for p in ici_index.person_id)
    lab_rows = {lab: [] for lab in LAB_CONCEPTS}
    n_meas = 0
    meas_cols = resolve_cols(
        meas_path,
        ["PERSON_ID", "MEASUREMENT_CONCEPT_ID", "MEASUREMENT_DATE", "VALUE_AS_NUMBER"],
    )

    for chunk in pd.read_csv(
        meas_path, low_memory=False, chunksize=CHUNK, usecols=meas_cols
    ):
        n_meas += len(chunk)
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk.person_id = pd.to_numeric(chunk.person_id, errors="coerce")
        chunk.measurement_concept_id = pd.to_numeric(
            chunk.measurement_concept_id, errors="coerce"
        )
        chunk = chunk.dropna(subset=["person_id", "measurement_concept_id"])
        if chunk.empty:
            continue
        chunk.person_id = chunk.person_id.astype(int)
        chunk.measurement_concept_id = chunk.measurement_concept_id.astype(int)
        chunk = chunk[
            chunk.person_id.isin(valid_pid_set)
            & chunk.measurement_concept_id.isin(ALL_LAB_CONCEPTS)
        ]
        if chunk.empty:
            continue
        for lab, cids in LAB_CONCEPTS.items():
            sub = chunk[chunk.measurement_concept_id.isin(cids)]
            if not sub.empty:
                lab_rows[lab].append(sub)

    print(f"  Scanned {n_meas:,} measurement rows")
    labs = {}
    for lab in LAB_CONCEPTS:
        if lab_rows[lab]:
            df = pd.concat(lab_rows[lab], ignore_index=True)
            df["measurement_date"] = parse_date(df.measurement_date)
            df["value_as_number"] = pd.to_numeric(df.value_as_number, errors="coerce")
            lo, hi = LAB_RANGES[lab]
            df = df.dropna(subset=["value_as_number", "measurement_date"])
            df = df[(df.value_as_number >= lo) & (df.value_as_number <= hi)]
            labs[lab] = df
            print(
                f"    {lab}: {len(df):,} records, {df.person_id.nunique():,} patients"
            )
        else:
            labs[lab] = pd.DataFrame(
                columns=["person_id", "measurement_date", "value_as_number"]
            )
            print(f"    {lab}: 0 records")
    del lab_rows
    gc.collect()

    # ── Creatinine: AKI phenotyping (same as before) ──────────
    cr = labs["cr"].merge(ici_dates, on="person_id")
    cr["days"] = (cr.measurement_date - cr.ici_index_date).dt.days

    bl_main = cr[(cr.days >= -365) & (cr.days <= -7)]
    bl_cr = bl_main.groupby("person_id").value_as_number.median().rename("baseline_cr")
    bl_fb = cr[(cr.days >= -365) & (cr.days <= -1)].sort_values(["person_id", "days"])
    bl_fb_cr = (
        bl_fb.groupby("person_id")
        .tail(1)
        .set_index("person_id")
        .value_as_number.rename("baseline_cr")
    )
    baseline_cr = bl_cr.combine_first(bl_fb_cr).rename("baseline_cr")

    cohort = ici_index.merge(baseline_cr, on="person_id", how="inner")
    print(f"  With baseline Cr: {len(cohort):,}")

    fu = cr[(cr.days >= 1) & (cr.days <= 365)].merge(
        cohort[["person_id", "baseline_cr"]], on="person_id"
    )
    fu["cr_ratio"] = fu.value_as_number / fu.baseline_cr
    for label, md in AKI_WINDOWS.items():
        w = fu[fu.days <= md]
        mr = w.groupby("person_id").cr_ratio.max().rename(f"max_ratio_{label}")
        cohort = cohort.merge(mr, on="person_id", how="left")
        cohort[label] = (cohort[f"max_ratio_{label}"] >= CR_RATIO_THRESHOLD).astype(int)
        cohort.loc[cohort[f"max_ratio_{label}"].isna(), label] = 0

    cr_evt = fu[fu.cr_ratio >= CR_RATIO_THRESHOLD].copy()
    cr_first = (
        (
            cr_evt.sort_values("measurement_date")
            .groupby("person_id")
            .first()
            .reset_index()[["person_id", "measurement_date"]]
            .rename(columns={"measurement_date": "cr_evt_date"})
        )
        if len(cr_evt) > 0
        else pd.DataFrame(columns=["person_id", "cr_evt_date"])
    )
    last_obs = cr.groupby("person_id").measurement_date.max().rename("last_obs_date")
    cohort = cohort.merge(last_obs, on="person_id", how="left")
    has_fu = set(int(p) for p in cr[(cr.days >= 1) & (cr.days <= 365)].person_id)
    cohort = cohort[cohort.person_id.isin(has_fu) & (cohort.baseline_cr < 4.0)].copy()
    print(f"  Final cohort: {len(cohort):,}")

    # ── Baseline labs (BUN, Hgb, Alb, K) ──────────────────────
    cohort_pids_set = set(int(p) for p in cohort.person_id)
    for lab in ["bun", "hgb", "alb", "k"]:
        df = labs[lab]
        df = df[df.person_id.isin(cohort_pids_set)].copy()
        if len(df) == 0:
            cohort[f"baseline_{lab}"] = np.nan
            print(f"    baseline_{lab}: no data")
            continue
        df = df.merge(ici_dates, on="person_id")
        df["days"] = (df.measurement_date - df.ici_index_date).dt.days
        bl = df[(df.days >= -365) & (df.days <= -7)]
        bl_med = (
            bl.groupby("person_id").value_as_number.median().rename(f"baseline_{lab}")
        )
        # fallback
        bl_fb = df[(df.days >= -365) & (df.days <= -1)].sort_values(
            ["person_id", "days"]
        )
        bl_fb_v = (
            bl_fb.groupby("person_id")
            .tail(1)
            .set_index("person_id")
            .value_as_number.rename(f"baseline_{lab}")
        )
        combined = bl_med.combine_first(bl_fb_v)
        cohort = cohort.merge(combined, on="person_id", how="left")
        n_have = cohort[f"baseline_{lab}"].notna().sum()
        print(
            f"    baseline_{lab}: {n_have:,}/{len(cohort):,} ({n_have/len(cohort)*100:.1f}%)"
        )

    # BUN/Cr ratio
    cohort["bun_cr_ratio"] = cohort["baseline_bun"] / cohort["baseline_cr"]
    n_bcr = cohort.bun_cr_ratio.notna().sum()
    print(f"    bun_cr_ratio: {n_bcr:,}/{len(cohort):,} ({n_bcr/len(cohort)*100:.1f}%)")

    del labs, cr, fu, cr_evt
    gc.collect()

    # ══════════════════ STEP 3b: ICD AKI ══════════════════════
    print("\n" + "=" * 70 + "\nSTEP 3b: ICD-based AKI\n" + "=" * 70)
    cohort_pids = set(int(p) for p in cohort.person_id)
    aki_icd = aki_icd_all[aki_icd_all.person_id.isin(cohort_pids)].copy()
    del aki_icd_all
    gc.collect()

    if len(aki_icd) > 0:
        aki_icd["condition_date"] = parse_date(aki_icd.condition_start_date)
        aki_icd = aki_icd.merge(cohort[["person_id", "ici_index_date"]], on="person_id")
        aki_icd["days"] = (aki_icd.condition_date - aki_icd.ici_index_date).dt.days
        if os.path.exists(visit_path):
            vc = resolve_cols(
                visit_path, ["visit_occurrence_id", "person_id", "visit_concept_id"]
            )
            svi = set()
            for chunk in pd.read_csv(
                visit_path, low_memory=False, chunksize=CHUNK, usecols=vc
            ):
                chunk.columns = [c.lower() for c in chunk.columns]
                chunk = normalize_pid_column(chunk)
                chunk = chunk[
                    chunk.person_id.isin(cohort_pids)
                    & chunk.visit_concept_id.isin(SEVERE_VISIT_CONCEPTS)
                ]
                if not chunk.empty:
                    svi.update(chunk.visit_occurrence_id)
            aki_icd["visit_occurrence_id"] = pd.to_numeric(
                aki_icd.visit_occurrence_id, errors="coerce"
            )
            is_severe = aki_icd.visit_occurrence_id.isin(svi)
        else:
            is_severe = pd.Series(True, index=aki_icd.index)
        pre = aki_icd[(aki_icd.days >= -90) & (aki_icd.days <= 0)]
        is_inc = ~aki_icd.person_id.isin(set(pre.person_id))
        aki_icd = aki_icd[is_severe | is_inc].copy()
        aki_icd = aki_icd[(aki_icd.days >= 1) & (aki_icd.days <= 365)]
        icd_first = (
            aki_icd.sort_values("condition_date")
            .groupby("person_id")
            .first()
            .reset_index()[["person_id", "condition_date"]]
            .rename(columns={"condition_date": "icd_evt_date"})
        )
    else:
        icd_first = pd.DataFrame(columns=["person_id", "icd_evt_date"])
    print(f"  ICD AKI events: {len(icd_first):,}")
    del aki_icd
    gc.collect()

    # ══════════════════ STEP 3c: UNION → SURVIVAL ═════════════
    print("\n" + "=" * 70 + "\nSTEP 3c: Union → survival\n" + "=" * 70)
    cohort = cohort.merge(cr_first, on="person_id", how="left").merge(
        icd_first, on="person_id", how="left"
    )
    cok = cohort.cr_evt_date.notna()
    iok = cohort.icd_evt_date.notna()
    cf = cok & (~iok | (cohort.cr_evt_date <= cohort.icd_evt_date))
    if_ = iok & (~cok | (cohort.icd_evt_date < cohort.cr_evt_date))
    cohort["evt_date"] = pd.NaT
    cohort.loc[cf, "evt_date"] = cohort.loc[cf, "cr_evt_date"]
    cohort.loc[if_, "evt_date"] = cohort.loc[if_, "icd_evt_date"]
    cohort["evt_source"] = "none"
    cohort.loc[cf, "evt_source"] = "cr"
    cohort.loc[if_, "evt_source"] = "icd"
    cohort["aki_event"] = (cohort.evt_source != "none").astype(int)
    he = cohort.aki_event == 1
    cohort["max_fu"] = cohort.ici_index_date + pd.Timedelta(days=365)
    cohort.loc[he, "surv_days"] = (
        cohort.loc[he, "evt_date"] - cohort.loc[he, "ici_index_date"]
    ).dt.days
    cohort.loc[~he, "surv_days"] = (
        cohort.loc[~he, ["last_obs_date", "max_fu"]].min(axis=1)
        - cohort.loc[~he, "ici_index_date"]
    ).dt.days
    cohort["surv_days"] = cohort.surv_days.clip(lower=1, upper=365).astype(int)
    nt = len(cohort)
    ne = int(cohort.aki_event.sum())
    ncr = int((cohort.evt_source == "cr").sum())
    nicd = int((cohort.evt_source == "icd").sum())
    print(
        f"  Total: {nt:,}, Events: {ne} ({ne/nt*100:.1f}%), Cr-first: {ncr}, ICD-first: {nicd}"
    )
    del cr_first, icd_first
    gc.collect()

    # ══════════════════ STEP 4: DEMO + NCI-CCI + CANCER + NEPHRO ══
    print(
        "\n"
        + "=" * 70
        + "\nSTEP 4: Demographics + NCI-CCI + cancer type + drug classes\n"
        + "=" * 70
    )
    cohort_pids = set(int(p) for p in cohort.person_id)
    person = pd.read_csv(
        person_path,
        low_memory=False,
        usecols=resolve_cols(
            person_path,
            [
                "person_id",
                "year_of_birth",
                "gender_concept_id",
                "race_concept_id",
                "ethnicity_concept_id",
            ],
        ),
    )
    person.columns = [c.lower() for c in person.columns]
    person = normalize_pid_column(person)
    person = person[person.person_id.isin(cohort_pids)]
    cohort = cohort.merge(person, on="person_id", how="left")
    cohort["year_of_birth"] = pd.to_numeric(cohort.year_of_birth, errors="coerce")
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
    print(
        f"  Age: mean={cohort.age.mean():.1f}  Sex: {cohort.sex.value_counts().to_dict()}"
    )

    # Cancer type
    cdx = cancer_dx_all[cancer_dx_all.person_id.isin(cohort_pids)].copy()
    del cancer_dx_all
    if len(cdx) > 0:
        cdx["cancer_type"] = cdx.code.apply(classify_cancer)
        # Metastatic flag (C77-C80)
        cdx["is_meta"] = cdx.code.apply(
            lambda c: any(c.startswith(p) for p in ["C77", "C78", "C79", "C80"])
        )
        meta_pts = set(cdx[cdx.is_meta].person_id)
        cohort["metastatic"] = cohort.person_id.isin(meta_pts).astype(int)
        print(
            f"  Metastatic: {cohort.metastatic.sum():,} ({cohort.metastatic.mean()*100:.1f}%)"
        )

        # Multi-cancer: >1 unique PRIMARY type per patient (excl metastatic codes)
        primary = cdx[~cdx.is_meta & cdx.cancer_type.notna()].copy()
        types_per_pt = primary.groupby("person_id")["cancer_type"].apply(
            lambda x: set(x)
        )
        n_types = types_per_pt.apply(len).rename("n_cancer_types")

        # Assign: multi_cancer if >1 type, else the single type
        def _assign(pid):
            if pid not in types_per_pt.index:
                return "Unknown"
            ts = types_per_pt[pid]
            if len(ts) > 1:
                return "multi_cancer"
            return list(ts)[0]

        cohort["cancer_type"] = cohort.person_id.apply(_assign)
    else:
        cohort["cancer_type"] = "Unknown"
        cohort["metastatic"] = 0
    # Collapse: Lung [ref], Melanoma, Renal_Cell, multi_cancer, Other
    cohort["cancer_collapsed"] = cohort.cancer_type.apply(
        lambda c: (
            c if c in ("Lung", "Melanoma", "Renal_Cell", "multi_cancer") else "Other"
        )
    )
    print(f"  Cancer raw: {cohort.cancer_type.value_counts().to_dict()}")
    print(f"  Cancer collapsed: {cohort.cancer_collapsed.value_counts().to_dict()}")

    # NCI-CCI
    for cond, codes in NCI_CCI.items():
        pfx = [
            p.upper().replace(".", "") for p in codes.get("9", []) + codes.get("10", [])
        ]
        pts = {
            pid
            for pid in cohort_pids
            if pid in patient_codes
            and any(c.startswith(p) for c in patient_codes[pid] for p in pfx)
        }
        cohort[cond] = cohort.person_id.isin(pts).astype(int)
    del patient_codes
    gc.collect()

    # Individual nephrotoxin drug classes (±30d concomitant)
    nephro_flags = {cls: set() for cls in NEPHRO_CLASSES}
    nephro_any = set()
    for chunk in pd.read_csv(
        drug_path, low_memory=False, chunksize=CHUNK, usecols=drug_cols
    ):
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk = normalize_pid_column(chunk)
        chunk = chunk[chunk.person_id.isin(cohort_pids)]
        if chunk.empty:
            continue
        chunk = chunk.merge(ici_dates, on="person_id")
        chunk["drug_exposure_start_date"] = parse_date(chunk.drug_exposure_start_date)
        chunk["days"] = (chunk.drug_exposure_start_date - chunk.ici_index_date).dt.days
        conco = chunk[(chunk.days >= -30) & (chunk.days <= 30)]
        if conco.empty:
            continue
        dsv = conco.drug_source_value.astype(str).str.lower()
        for cls, regex in NEPHRO_REGEXES.items():
            m = dsv.str.contains(regex, regex=True, na=False)
            nephro_flags[cls].update(int(p) for p in conco.loc[m, "person_id"])
        m_any = dsv.str.contains(NEPHROTOXIN_REGEX, case=False, regex=True, na=False)
        nephro_any.update(int(p) for p in conco.loc[m_any, "person_id"])

    cohort["nephrotoxin"] = cohort.person_id.isin(nephro_any).astype(int)
    for cls in NEPHRO_CLASSES:
        cohort[cls] = cohort.person_id.isin(nephro_flags[cls]).astype(int)
        print(f"    {cls}: {cohort[cls].sum():,} ({cohort[cls].mean()*100:.1f}%)")
    print(f"  nephrotoxin (any): {cohort.nephrotoxin.sum():,}")

    # ══════════════════ STEP 5: FEATURE MATRIX ════════════════
    print("\n" + "=" * 70 + "\nSTEP 5: Feature Matrix\n" + "=" * 70)
    fn = []
    fc = []
    ib = []

    # Age (standardized)
    a = cohort.age.values.astype(np.float64)
    a = ((a - np.nanmean(a)) / (np.nanstd(a) + 1e-8)).astype(np.float32)
    fn.append("age")
    fc.append(a)
    ib.append(False)

    # Demographics
    fn.append("sex_male")
    fc.append((cohort.sex == "Male").values.astype(np.float32))
    ib.append(True)
    fn.append("race_black")
    fc.append((cohort.race == "Black").values.astype(np.float32))
    ib.append(True)
    fn.append("ethnicity_hispanic")
    fc.append((cohort.ethnicity == "Hispanic").values.astype(np.float32))
    ib.append(True)

    # NCI-CCI flags (no score)
    for cond in NCI_CCI:
        if cohort[cond].mean() < MIN_PREVALENCE:
            continue
        fn.append(cond)
        fc.append(cohort[cond].values.astype(np.float32))
        ib.append(True)

    # ICI regimen (ref=pd1_mono)
    fn.append("ici_pdl1_mono")
    fc.append((cohort.ici_regimen == "pdl1_mono").values.astype(np.float32))
    ib.append(True)
    fn.append("ici_combo")
    fc.append((cohort.ici_regimen == "combo").values.astype(np.float32))
    ib.append(True)

    # Cancer type (ref=Lung)
    for ct in ["Melanoma", "Renal_Cell", "multi_cancer", "Other"]:
        fn.append(f"cancer_{ct}")
        fc.append((cohort.cancer_collapsed == ct).values.astype(np.float32))
        ib.append(True)

    # Metastatic flag
    if cohort.metastatic.mean() >= MIN_PREVALENCE and cohort.metastatic.mean() <= (
        1 - MIN_PREVALENCE
    ):
        fn.append("metastatic")
        fc.append(cohort.metastatic.values.astype(np.float32))
        ib.append(True)

    # Individual nephrotoxin classes
    for cls in NEPHRO_CLASSES:
        if cohort[cls].mean() >= MIN_PREVALENCE:
            fn.append(cls)
            fc.append(cohort[cls].values.astype(np.float32))
            ib.append(True)

    # Baseline labs (standardized; NO IMPUTATION — missing_mask.pt tracks NaNs)
    missing_flags = {}
    for lab in ["bun", "hgb", "alb", "k"]:
        col = f"baseline_{lab}"
        v = cohort[col].values.astype(np.float64)
        is_miss = np.isnan(v)
        missing_flags[col] = is_miss
        v_ok = v[~is_miss]
        mu = v_ok.mean() if len(v_ok) > 0 else 0.0
        sd = v_ok.std() if len(v_ok) > 0 else 1.0
        v = ((v - mu) / (sd + 1e-8)).astype(np.float32)
        v[is_miss] = 0.0  # neutral placeholder, NOT imputation
        fn.append(col)
        fc.append(v)
        ib.append(False)
        print(f"    {col}: {is_miss.sum()} missing ({is_miss.mean()*100:.1f}%)")

    # BUN/Cr ratio
    bcr = cohort.bun_cr_ratio.values.astype(np.float64)
    is_miss_bcr = np.isnan(bcr)
    missing_flags["bun_cr_ratio"] = is_miss_bcr
    bcr_ok = bcr[~is_miss_bcr]
    mu_b = bcr_ok.mean() if len(bcr_ok) > 0 else 0.0
    sd_b = bcr_ok.std() if len(bcr_ok) > 0 else 1.0
    bcr = ((bcr - mu_b) / (sd_b + 1e-8)).astype(np.float32)
    bcr[is_miss_bcr] = 0.0
    fn.append("bun_cr_ratio")
    fc.append(bcr)
    ib.append(False)
    print(
        f"    bun_cr_ratio: {is_miss_bcr.sum()} missing ({is_miss_bcr.mean()*100:.1f}%)"
    )

    # AKI event (last)
    fn.append("aki_event")
    fc.append(cohort.aki_event.values.astype(np.float32))
    ib.append(True)

    X = np.stack(fc, axis=0)
    X = np.nan_to_num(X, nan=0.0)
    fm = torch.tensor(X, dtype=torch.float32)
    bm = torch.tensor(ib, dtype=torch.bool)
    F, N = fm.shape

    print(f"\n  Feature matrix: [{F}, {N}]")
    for i, (name, b) in enumerate(zip(fn, ib)):
        v = fc[i]
        tag = "bin" if b else "con"
        if b:
            print(f"    [{i:2d}] {name:30s} ({tag})  prev={v.mean():.3f}")
        else:
            print(f"    [{i:2d}] {name:30s} ({tag})  μ={v.mean():.3f} σ={v.std():.3f}")

    torch.save(fm, os.path.join(OUT, "feature_matrix.pt"))
    torch.save(bm, os.path.join(OUT, "binary_mask.pt"))
    with open(os.path.join(OUT, "feature_names.json"), "w") as f:
        json.dump(fn, f, indent=2)

    # Missing mask: [F, N] bool, True = value was missing (labs only)
    miss_matrix = np.zeros((F, N), dtype=bool)
    for feat_name, miss_arr in missing_flags.items():
        if feat_name in fn:
            idx = fn.index(feat_name)
            miss_matrix[idx, :] = miss_arr
    torch.save(
        torch.tensor(miss_matrix, dtype=torch.bool),
        os.path.join(OUT, "missing_mask.pt"),
    )
    n_miss_total = miss_matrix.sum()
    print(
        f"  missing_mask.pt: {n_miss_total:,} missing values across {len(missing_flags)} lab features"
    )

    meta_cols = (
        ["person_id"]
        + list(AKI_WINDOWS.keys())
        + ["aki_event", "surv_days", "evt_source"]
    )
    cohort[[c for c in meta_cols if c in cohort.columns]].to_csv(
        os.path.join(OUT, "cohort_meta.csv"), index=False
    )
    print(f"\n  Saved: [{F},{N}] feature_matrix.pt + missing_mask.pt + cohort_meta.csv")

    # ══════════════════ STEP 6: KM ════════════════════════════
    print("\n" + "=" * 70 + "\nSTEP 6: KM curve\n" + "=" * 70)
    apply_nature_style()
    ta = cohort.surv_days.values
    ea = cohort.aki_event.values
    o = np.argsort(ta)
    ts = ta[o]
    es = ea[o]
    ut = np.unique(ts[es == 1])
    if len(ut) > 0:
        nr = np.array([np.sum(ts >= t) for t in ut])
        ne_ = np.array([np.sum((ts == t) & (es == 1)) for t in ut])
        sp = np.cumprod(1 - ne_ / nr)
        kt = np.concatenate([[0], ut])
        ks = np.concatenate([[1.0], sp])
    else:
        kt = np.array([0, 365])
        ks = np.array([1.0, 1.0])
        sp = np.array([1.0])
    rt = [0, 30, 60, 90, 120, 180, 270, 365]
    rc = [int(np.sum(ts >= r)) for r in rt]
    fig, ax = plt.subplots(figsize=(3.504, 2.8))
    ax.step(kt, ks, where="post", color="#0072B2", linewidth=1.2, label=f"n={nt:,}")
    ax.set_xlabel("Days from ICI")
    ax.set_ylabel("AKI-free survival")
    ax.set_xlim(0, 370)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(rt)
    ax.text(
        0.5,
        -0.22,
        "No. at risk",
        transform=ax.transAxes,
        fontsize=5,
        fontweight="bold",
        ha="center",
    )
    for r, c in zip(rt, rc):
        ax.text(
            r / 370,
            -0.28,
            str(c),
            transform=ax.transAxes,
            fontsize=5,
            ha="center",
            color="#0072B2",
        )
    ax.legend(loc="lower left", fontsize=6)
    fig.savefig(os.path.join(FIG, "km_aki_check.pdf"), dpi=600)
    fig.savefig(os.path.join(FIG, "km_aki_check.png"), dpi=150)
    plt.close()
    print(f"  Saved KM")

    print(
        "\n" + "=" * 70 + "\nDONE — run baselines + Cox to check c-index\n" + "=" * 70
    )


if __name__ == "__main__":
    main()
