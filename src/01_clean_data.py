#!/usr/bin/env python3
"""
01_clean_data.py
================
Data cleaning and cohort definition for
Bayesian Hierarchical Logistic Regression on 30-Day Hospital Readmission.

Covers Pipeline Task 2:
  Task 2: Data cleaning & cohort definition

Input:
  - data_raw/diabetic_data.csv       (UCI Diabetes 130-US hospitals)
  - data_raw/IDS_mapping.csv         (ID mapping file)

Output (all written to data_processed/):
  - cleaned_intermediate.csv         Cleaned cohort before feature engineering
  - cleaning_log.txt                 Cleaning audit trail

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
# SAVE INTERMEDIATE OUTPUT
# =============================================================================

log("=" * 70)
log("Saving intermediate cleaned cohort")
log("=" * 70)

df.to_csv(DATA_PROCESSED / "cleaned_intermediate.csv", index=False)
log(f"  cleaned_intermediate.csv  ->  {len(df)} rows x {df.shape[1]} cols")

# Summary
log("")
log("=" * 70)
log("SUMMARY -- Task 2")
log("=" * 70)
log(f"  [N_raw]       = {N_raw}")
log(f"  [N_clean]     = {N_clean}")
log(f"  [P_raw]       = {P_raw}")
log(f"  [X%] overall readmission rate = {readmit_rate*100:.2f}%")
log(f"  Random seed   = {RANDOM_SEED}")

# Write cleaning log
log_path = DATA_PROCESSED / "cleaning_log.txt"
with open(log_path, "w") as f:
    f.write("\n".join(log_lines))

print(f"\n{'='*70}")
print(f"DONE. Cleaning log saved to {log_path}")
print(f"01_clean_data.py completed successfully.")
print(f"{'='*70}")
