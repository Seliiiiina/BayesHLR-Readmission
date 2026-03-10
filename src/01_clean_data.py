#!/usr/bin/env python3
"""
01_clean_data.py
================
Data cleaning, specialty regrouping, and feature engineering for
Bayesian Hierarchical Logistic Regression on 30-Day Hospital Readmission.

Covers Pipeline Tasks 2-4:
  Task 2: Data cleaning & cohort definition
  Task 3: Specialty regrouping
  Task 4: Variable construction (design matrix)

Input:
  - data_raw/diabetic_data.csv       (UCI Diabetes 130-US hospitals)
  - data_raw/IDS_mapping.csv         (ID mapping file)

Output (all written to data_processed/):
  - analytic_cohort.csv              Final analytic cohort
  - X_matrix.csv                     Standardised design matrix
  - y_vector.csv                     Binary outcome vector
  - group_index.csv                  Specialty group index (0-based)
  - specialty_mapping.csv            Specialty label <-> integer mapping
  - data_dictionary.csv              Variable dictionary
  - cleaning_log.txt                 Cleaning audit trail
  - X_matrix.npy / y_vector.npy / group_index.npy   (numpy arrays for sampler)

Usage on Duke DCC:
  cd /work/zq63/Bayes/Joint-Bayesian
  python src/01_clean_data.py

Author : [Your names]
Date   : 2026-03-10
Seed   : 724
"""

# == 0. Imports ================================================================
import os
import sys
import warnings
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# == 0.1 Reproducibility & paths ==============================================
RANDOM_SEED = 724
np.random.seed(RANDOM_SEED)

# Auto-detect project root: script may live in src/ or code/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ("src", "code") else SCRIPT_DIR

DATA_RAW = PROJECT_ROOT / "data_raw"
if not DATA_RAW.exists():
    # Fallback: try 'data' directory (matches your DCC layout)
    DATA_RAW = PROJECT_ROOT / "data"

DATA_PROCESSED = PROJECT_ROOT / "data_processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for subdir in ["tables", "figures", "chains", "logs"]:
    (OUTPUT_DIR / subdir).mkdir(exist_ok=True)

# == 0.2 Cleaning log =========================================================
log_lines = []

def log(msg):
    """Append a message to the cleaning log and print to stdout."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    log_lines.append(entry)
    print(entry)


# =============================================================================
# TASK 2 -- DATA CLEANING & COHORT DEFINITION
# =============================================================================

log("=" * 70)
log("TASK 2: Data Cleaning & Cohort Definition")
log("=" * 70)

# -- 2.1 Read raw data --------------------------------------------------------
raw_path = DATA_RAW / "diabetic_data.csv"
if not raw_path.exists():
    sys.exit(f"ERROR: Cannot find raw data at {raw_path}")

df = pd.read_csv(raw_path, na_values=["?", "None", ""])
N_raw, P_raw = df.shape
log(f"Raw data loaded: N_raw = {N_raw}, P_raw = {P_raw}")

# -- 2.2 Standardise missing value encoding -----------------------------------
# The UCI dataset uses '?' for missing values; pandas na_values handles this
# above, but some fields encode missing as specific strings.
missing_map = {
    "Unknown/Invalid": np.nan,
    "NULL": np.nan,
    "Not Available": np.nan,
}
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].replace(missing_map)

# -- 2.3 Remove duplicate encounters ------------------------------------------
n_before = len(df)
df = df.drop_duplicates(subset=["encounter_id"], keep="first")
n_dup = n_before - len(df)
log(f"Removed {n_dup} duplicate encounter_id rows")

# -- 2.4 Drop high-missingness / low-utility columns --------------------------
# weight: >96% missing; payer_code: >50% missing; citoglipton & examide:
# essentially single-level (only "No").
drop_cols = ["weight", "payer_code", "citoglipton", "examide"]
actually_dropped = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=actually_dropped, errors="ignore")
log(f"Dropped columns (high missingness / no variance): {actually_dropped}")

# -- 2.5 Handle repeated patients ----------------------------------------------
# Pipeline specifies encounter-level analysis. However, keeping only the
# first encounter per patient avoids within-patient dependence that the
# model does not account for (see Limitations in manuscript).
n_before = len(df)
df = df.sort_values("encounter_id").groupby("patient_nbr").first().reset_index()
n_repeat = n_before - len(df)
log(f"Kept first encounter per patient; removed {n_repeat} repeated encounters")

# -- 2.6 Remove discharge to hospice / expired ---------------------------------
# Patients who died or were discharged to hospice cannot be readmitted.
# discharge_disposition_id: 11=Expired, 13=Hospice/home, 14=Hospice/medical,
# 19=Expired at home, 20=Expired in medical facility, 21=Expired (place unknown)
exclude_discharge = [11, 13, 14, 19, 20, 21]
n_before = len(df)
df = df[~df["discharge_disposition_id"].isin(exclude_discharge)]
n_removed_discharge = n_before - len(df)
log(f"Removed {n_removed_discharge} encounters with expired/hospice discharge")

# -- 2.7 Construct binary outcome ----------------------------------------------
# Y = 1 if readmitted == "<30", else 0
df["readmit_30"] = (df["readmitted"] == "<30").astype(int)
readmit_rate = df["readmit_30"].mean()
log(f"Outcome constructed: readmit_30. Rate = {readmit_rate:.4f} ({readmit_rate*100:.2f}%)")

# -- 2.8 Drop identifiers no longer needed ------------------------------------
id_cols = ["encounter_id", "patient_nbr", "readmitted"]
df = df.drop(columns=[c for c in id_cols if c in df.columns])

N_clean = len(df)
log(f"After cleaning: N_clean = {N_clean}")
log("")


# =============================================================================
# TASK 3 -- SPECIALTY REGROUPING
# =============================================================================

log("=" * 70)
log("TASK 3: Specialty Regrouping")
log("=" * 70)

# -- 3.1 Examine raw specialty distribution ------------------------------------
df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")

raw_specialty_counts = df["medical_specialty"].value_counts()
log(f"Number of raw specialty categories: {len(raw_specialty_counts)}")
log(f"Top 10 specialties by count:\n{raw_specialty_counts.head(10).to_string()}")

# -- 3.2 Apply regrouping threshold -------------------------------------------
# Pipeline suggests trying 100 / 250 / 500; target J in [8, 20].
# Adjust THRESHOLD below based on sensitivity results.
THRESHOLD = 250

specialty_counts = df["medical_specialty"].value_counts()
keep_specialties = specialty_counts[specialty_counts >= THRESHOLD].index.tolist()

# Remove "Unknown" from keep list; it stays as its own group regardless of count
if "Unknown" in keep_specialties:
    keep_specialties.remove("Unknown")

df["specialty_group"] = df["medical_specialty"].apply(
    lambda x: x if x in keep_specialties else ("Unknown" if x == "Unknown" else "Other")
)

J = df["specialty_group"].nunique()
log(f"Regrouping threshold = {THRESHOLD}")
log(f"Number of specialty groups after regrouping: J = {J}")
log(f"Specialty group distribution:\n{df['specialty_group'].value_counts().to_string()}")

# -- 3.3 Sensitivity check: report J under different thresholds ----------------
for thr in [100, 250, 500]:
    _keep = specialty_counts[specialty_counts >= thr].index.tolist()
    if "Unknown" in _keep:
        _keep.remove("Unknown")
    _groups = df["medical_specialty"].apply(
        lambda x, k=_keep: x if x in k else ("Unknown" if x == "Unknown" else "Other")
    )
    _J = _groups.nunique()
    log(f"  [Sensitivity] threshold={thr:>4d} -> J={_J}")

# -- 3.4 Build integer group index --------------------------------------------
specialty_labels = sorted(df["specialty_group"].unique())
specialty_to_idx = {s: i for i, s in enumerate(specialty_labels)}
df["group_idx"] = df["specialty_group"].map(specialty_to_idx)

specialty_map_df = pd.DataFrame({
    "specialty_group": specialty_labels,
    "group_idx": range(len(specialty_labels)),
    "n_obs": [int((df["specialty_group"] == s).sum()) for s in specialty_labels],
})
log(f"Specialty mapping:\n{specialty_map_df.to_string(index=False)}")
log("")


# =============================================================================
# TASK 4 -- VARIABLE CONSTRUCTION (DESIGN MATRIX)
# =============================================================================

log("=" * 70)
log("TASK 4: Variable Construction")
log("=" * 70)

# -- 4.1 Age recoding ---------------------------------------------------------
# Dataset stores age as bracketed ranges: [0-10), [10-20), ... [90-100)
# Recode to midpoints for use as continuous covariate.
age_midpoints = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}
df["age_numeric"] = df["age"].map(age_midpoints)
if df["age_numeric"].isna().any():
    log(f"WARNING: {df['age_numeric'].isna().sum()} unmapped age values")
log(f"Age recoded to midpoints. Mean age = {df['age_numeric'].mean():.1f}")

# -- 4.2 Gender recoding ------------------------------------------------------
# Binary: Female=1, Male=0. Drop rows with unknown gender (very few).
n_before = len(df)
df = df[df["gender"].isin(["Female", "Male"])]
log(f"Dropped {n_before - len(df)} rows with unknown gender")
df["female"] = (df["gender"] == "Female").astype(int)

# -- 4.3 Race encoding --------------------------------------------------------
df["race"] = df["race"].fillna("Unknown")
# Create dummies; reference category = Caucasian (most frequent)
race_dummies = pd.get_dummies(df["race"], prefix="race", drop_first=False, dtype=int)
if "race_Caucasian" in race_dummies.columns:
    race_dummies = race_dummies.drop(columns=["race_Caucasian"])
df = pd.concat([df, race_dummies], axis=1)
log(f"Race dummies created. Categories (excl. reference Caucasian): {list(race_dummies.columns)}")

# -- 4.4 Diagnosis grouping (ICD-9) -------------------------------------------
# The dataset contains THREE diagnosis fields: diag_1 (primary / admission
# reason), diag_2 and diag_3 (secondary / comorbidity diagnoses).
#
# Strategy (two layers):
#   Layer 1 — Primary diagnosis group:  Map diag_1 into broad clinical
#             categories and include as categorical dummies. This captures
#             *why* the patient was admitted.
#   Layer 2 — Comorbidity indicators:   Scan ALL of diag_1/2/3 to flag
#             whether key comorbidity categories appear *anywhere* across
#             the three fields. This is clinically important because in a
#             diabetes cohort, diabetes itself is often diag_2 or diag_3
#             rather than diag_1. We also compute a simple comorbidity
#             burden count (number of distinct diagnosis groups across
#             the three fields).

def map_icd9_to_group(code):
    """Map a single ICD-9 code to a broad clinical category."""
    if pd.isna(code):
        return "Other"
    code_str = str(code).strip()
    try:
        num = float(code_str)
    except ValueError:
        if code_str.startswith("E"):
            return "Injury"
        elif code_str.startswith("V"):
            return "Other"
        return "Other"

    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    elif 460 <= num <= 519 or num == 786:
        return "Respiratory"
    elif 520 <= num <= 579 or num == 787:
        return "Digestive"
    elif 250 <= num < 251:
        return "Diabetes"
    elif 800 <= num <= 999:
        return "Injury"
    elif 710 <= num <= 739:
        return "Musculoskeletal"
    elif 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    elif 140 <= num <= 239:
        return "Neoplasms"
    elif 240 <= num < 250 or 251 <= num <= 279:
        return "Endocrine_Other"
    elif 680 <= num <= 709 or num == 782:
        return "Skin"
    elif 1 <= num <= 139:
        return "Infectious"
    elif 290 <= num <= 319:
        return "Mental"
    else:
        return "Other"

# --- Layer 1: Primary diagnosis group from diag_1 ---
df["diagnosis_group"] = df["diag_1"].apply(map_icd9_to_group)
diag_counts = df["diagnosis_group"].value_counts()
log(f"Primary diagnosis groups (from diag_1):\n{diag_counts.to_string()}")

# Create dummies; reference = most frequent category
diag_ref = diag_counts.idxmax()
diag_dummies = pd.get_dummies(df["diagnosis_group"], prefix="diag1", drop_first=False, dtype=int)
if f"diag1_{diag_ref}" in diag_dummies.columns:
    diag_dummies = diag_dummies.drop(columns=[f"diag1_{diag_ref}"])
df = pd.concat([df, diag_dummies], axis=1)
log(f"Primary diagnosis dummies created. Reference category: {diag_ref}")

# --- Layer 2: Comorbidity indicators from diag_1 + diag_2 + diag_3 ---
diag_fields = ["diag_1", "diag_2", "diag_3"]
existing_diag_fields = [f for f in diag_fields if f in df.columns]
log(f"Building comorbidity indicators from: {existing_diag_fields}")

# Map all three fields to groups
for field in existing_diag_fields:
    df[f"{field}_group"] = df[field].apply(map_icd9_to_group)

# Key comorbidity flags: does the category appear in ANY of the three fields?
# These are clinically meaningful comorbidities for readmission prediction.
comorbidity_categories = [
    "Circulatory", "Respiratory", "Diabetes", "Digestive",
    "Genitourinary", "Neoplasms", "Infectious", "Mental",
]

comorbid_cols = []
for cat in comorbidity_categories:
    col_name = f"comorbid_{cat.lower()}"
    df[col_name] = 0
    for field in existing_diag_fields:
        df[col_name] = df[col_name] | (df[f"{field}_group"] == cat).astype(int)
    comorbid_cols.append(col_name)
    prevalence = df[col_name].mean()
    log(f"  {col_name}: prevalence = {prevalence:.4f} ({prevalence*100:.1f}%)")

# Comorbidity burden: number of distinct diagnosis groups across all fields
group_cols = [f"{f}_group" for f in existing_diag_fields]
df["n_comorbid_groups"] = df[group_cols].apply(
    lambda row: len(set(row.values) - {"Other"}), axis=1
)
log(f"  n_comorbid_groups: mean = {df['n_comorbid_groups'].mean():.2f}, "
    f"max = {df['n_comorbid_groups'].max()}")

# Clean up temporary group columns
df = df.drop(columns=[f"{f}_group" for f in existing_diag_fields], errors="ignore")

# -- 4.5 Medication variables --------------------------------------------------
# insulin: recode to binary (any use vs no)
df["insulin_use"] = (df["insulin"] != "No").astype(int) if "insulin" in df.columns else 0

# change: medication was changed
df["med_change"] = (df["change"] == "Ch").astype(int) if "change" in df.columns else 0

# diabetesMed: any diabetes medication prescribed
df["diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int) if "diabetesMed" in df.columns else 0

log("Medication variables: insulin_use, med_change, diabetes_med")

# -- 4.6 Lab result indicators (with missingness encoding) ---------------------
# A1Cresult: >7, >8, Norm, None -> ordinal + missingness indicator
if "A1Cresult" in df.columns:
    df["A1C_measured"] = (~df["A1Cresult"].isin(["None", np.nan])).astype(int)
    df.loc[df["A1Cresult"].isna(), "A1C_measured"] = 0
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
    df["A1C_level"] = df["A1Cresult"].map(a1c_map).fillna(0).astype(int)
    log("A1Cresult -> A1C_measured (binary) + A1C_level (ordinal 0-3)")

if "max_glu_serum" in df.columns:
    df["glu_measured"] = (~df["max_glu_serum"].isin(["None", np.nan])).astype(int)
    df.loc[df["max_glu_serum"].isna(), "glu_measured"] = 0
    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    df["glu_level"] = df["max_glu_serum"].map(glu_map).fillna(0).astype(int)
    log("max_glu_serum -> glu_measured (binary) + glu_level (ordinal 0-3)")

# -- 4.7 Utilisation variables (continuous) ------------------------------------
continuous_vars = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "age_numeric",
    "n_comorbid_groups",
]
# Verify all present
missing_vars = [v for v in continuous_vars if v not in df.columns]
if missing_vars:
    log(f"WARNING: Missing continuous variables: {missing_vars}")
    continuous_vars = [v for v in continuous_vars if v in df.columns]

# Fill any remaining NaN in continuous with 0 (should be rare)
for v in continuous_vars:
    n_na = df[v].isna().sum()
    if n_na > 0:
        log(f"  Filling {n_na} NaN in {v} with 0")
        df[v] = df[v].fillna(0)

# -- 4.8 Standardise continuous variables --------------------------------------
# Centre and scale (z-score) for stable MCMC
continuous_stats = {}
for v in continuous_vars:
    mu = df[v].mean()
    sd = df[v].std()
    if sd == 0:
        sd = 1.0
    continuous_stats[v] = {"mean": mu, "std": sd}
    df[f"{v}_z"] = (df[v] - mu) / sd

log("Continuous variables standardised (z-score):")
for v, s in continuous_stats.items():
    log(f"  {v}: mean={s['mean']:.3f}, std={s['std']:.3f}")

# -- 4.9 Assemble final design matrix -----------------------------------------
z_cols = [f"{v}_z" for v in continuous_vars]
race_cols = sorted([c for c in df.columns if c.startswith("race_")])
diag1_cols = sorted([c for c in df.columns if c.startswith("diag1_")])
comorbid_flag_cols = sorted([c for c in df.columns if c.startswith("comorbid_")])
med_cols = ["insulin_use", "med_change", "diabetes_med"]
lab_cols = ["A1C_measured", "A1C_level", "glu_measured", "glu_level"]
lab_cols = [c for c in lab_cols if c in df.columns]
gender_cols = ["female"]

X_cols = (z_cols + gender_cols + race_cols + diag1_cols
          + comorbid_flag_cols + med_cols + lab_cols)
X_cols = [c for c in X_cols if c in df.columns]

K = len(X_cols)
N_final = len(df)
log(f"\nFinal design matrix: N = {N_final}, K = {K}")
log(f"Covariate columns ({K} total):")
for i, col in enumerate(X_cols):
    log(f"  [{i:2d}] {col}")

# Verify no NaN in design matrix
X_df = df[X_cols].copy()
nan_count = X_df.isna().sum().sum()
assert nan_count == 0, f"Design matrix still contains {nan_count} NaN values!"
log("Design matrix verified: no NaN values.")


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

log("")
log("=" * 70)
log("Saving outputs to data_processed/")
log("=" * 70)

# (a) Full analytic cohort (for EDA and reference)
cohort_cols = (
    ["readmit_30", "specialty_group", "group_idx"]
    + continuous_vars + z_cols + gender_cols + race_cols
    + diag1_cols + comorbid_flag_cols + med_cols + lab_cols
    + ["diagnosis_group"]
)
cohort_cols = [c for c in cohort_cols if c in df.columns]
cohort_df = df[cohort_cols].copy()
cohort_df.to_csv(DATA_PROCESSED / "analytic_cohort.csv", index=False)
log(f"  analytic_cohort.csv  ->  {len(cohort_df)} rows x {len(cohort_cols)} cols")

# (b) Design matrix X
X_df.to_csv(DATA_PROCESSED / "X_matrix.csv", index=False)
log(f"  X_matrix.csv         ->  {X_df.shape}")

# (c) Outcome vector y
y_df = df[["readmit_30"]].copy()
y_df.to_csv(DATA_PROCESSED / "y_vector.csv", index=False)
log(f"  y_vector.csv         ->  {len(y_df)} rows")

# (d) Group index
group_df = df[["group_idx"]].copy()
group_df.to_csv(DATA_PROCESSED / "group_index.csv", index=False)
log(f"  group_index.csv      ->  {len(group_df)} rows, J = {J}")

# (e) Specialty mapping
specialty_map_df.to_csv(DATA_PROCESSED / "specialty_mapping.csv", index=False)
log(f"  specialty_mapping.csv")

# (f) Data dictionary
dict_rows = []
for i, col in enumerate(X_cols):
    if col.endswith("_z"):
        base = col.replace("_z", "")
        desc = (f"Standardised {base} "
                f"(mean={continuous_stats[base]['mean']:.3f}, "
                f"sd={continuous_stats[base]['std']:.3f})")
        vtype = "continuous (z-scored)"
    elif col.startswith("race_"):
        desc = f"Indicator for race = {col.replace('race_', '')} (ref: Caucasian)"
        vtype = "binary"
    elif col.startswith("diag1_"):
        desc = f"Indicator for primary diagnosis group = {col.replace('diag1_', '')} (ref: {diag_ref})"
        vtype = "binary"
    elif col.startswith("comorbid_"):
        cat = col.replace("comorbid_", "")
        desc = f"Comorbidity flag: {cat} appears in any of diag_1/2/3"
        vtype = "binary"
    elif col in med_cols:
        desc = {
            "insulin_use": "Any insulin prescribed",
            "med_change": "Diabetes medication changed during encounter",
            "diabetes_med": "Any diabetes medication prescribed",
        }.get(col, col)
        vtype = "binary"
    elif col == "female":
        desc = "Female indicator (ref: Male)"
        vtype = "binary"
    elif "measured" in col:
        desc = f"Indicator: {col.replace('_measured', '')} was measured"
        vtype = "binary"
    elif "level" in col:
        desc = (f"Ordinal level of {col.replace('_level', '')} "
                "(0=not measured, 1=normal, 2/3=elevated)")
        vtype = "ordinal"
    else:
        desc = col
        vtype = "unknown"
    dict_rows.append({
        "index": i, "variable": col, "type": vtype, "description": desc
    })

dict_df = pd.DataFrame(dict_rows)
dict_df.to_csv(DATA_PROCESSED / "data_dictionary.csv", index=False)
log(f"  data_dictionary.csv  ->  {len(dict_df)} variables")

# (g) Save numpy arrays for direct use in sampler
np.save(DATA_PROCESSED / "X_matrix.npy", X_df.values.astype(np.float64))
np.save(DATA_PROCESSED / "y_vector.npy", y_df.values.ravel().astype(np.float64))
np.save(DATA_PROCESSED / "group_index.npy", group_df.values.ravel().astype(np.int32))
log("  .npy arrays saved for sampler")

# (h) Save standardisation parameters (needed for interpreting results later)
std_params = pd.DataFrame(continuous_stats).T
std_params.index.name = "variable"
std_params.to_csv(DATA_PROCESSED / "standardisation_params.csv")
log("  standardisation_params.csv")


# =============================================================================
# SUMMARY STATISTICS (for manuscript placeholders)
# =============================================================================

log("")
log("=" * 70)
log("SUMMARY -- Manuscript Placeholder Values")
log("=" * 70)
log(f"  [N_raw]       = {N_raw}")
log(f"  [N_final]     = {N_final}")
log(f"  [P_raw]       = {P_raw}")
log(f"  [J]           = {J}")
log(f"  [K]           = {K}")
log(f"  [threshold]   = {THRESHOLD}")
log(f"  [X%] overall readmission rate = {readmit_rate*100:.2f}%")
log(f"  Mean age      = {df['age_numeric'].mean():.1f}")
log(f"  Female %      = {df['female'].mean()*100:.1f}%")
log(f"  Unit of analysis: encounter-level (first encounter per patient)")
log(f"  Random seed   = {RANDOM_SEED}")

# Readmission rate by specialty
log("\n  Readmission rate by specialty group:")
spec_readmit = df.groupby("specialty_group")["readmit_30"].agg(["mean", "count"])
spec_readmit.columns = ["readmit_rate", "n"]
spec_readmit = spec_readmit.sort_values("readmit_rate", ascending=False)
for idx, row in spec_readmit.iterrows():
    log(f"    {idx:<30s}  rate={row['readmit_rate']:.4f}  n={int(row['n'])}")

rate_min = spec_readmit["readmit_rate"].min()
rate_max = spec_readmit["readmit_rate"].max()
rate_sd = spec_readmit["readmit_rate"].std()
log(f"\n  Raw specialty readmission rate range: [{rate_min:.4f}, {rate_max:.4f}]")
log(f"  Raw specialty readmission rate SD: {rate_sd:.4f}")


# =============================================================================
# WRITE CLEANING LOG
# =============================================================================

log_path = DATA_PROCESSED / "cleaning_log.txt"
with open(log_path, "w") as f:
    f.write("\n".join(log_lines))

print(f"\n{'='*70}")
print(f"DONE. Cleaning log saved to {log_path}")
print(f"All outputs written to {DATA_PROCESSED}/")
print(f"01_clean_data.py completed successfully.")
print(f"{'='*70}")
