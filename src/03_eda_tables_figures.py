#!/usr/bin/env python3
"""
03_eda_tables_figures.py
========================
Exploratory data analysis for the Bayesian hierarchical logistic
regression project.  Produces manuscript-ready Table 1 and Figures 1/3.

Pipeline Task 5 outputs:
  output/tables/table1_cohort_characteristics.csv
  output/figures/fig1_specialty_distribution.png
  output/figures/fig3_raw_readmission_by_specialty.png

Usage on Duke DCC:
  cd /work/zq63/Bayes/Joint-Bayesian
  python src/03_eda_tables_figures.py

Author : [Your names]
Date   : 2026-03-10
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # headless backend for DCC
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ("src", "code") else SCRIPT_DIR
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIG_DIR   = PROJECT_ROOT / "output" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.15,
})

# ── Read data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PROCESSED / "analytic_cohort.csv")
spec_map = pd.read_csv(DATA_PROCESSED / "specialty_mapping.csv")
print(f"Loaded analytic cohort: {df.shape}")


# =============================================================================
# TABLE 1 — Cohort Characteristics, Overall and by 30-Day Readmission
# =============================================================================

def fmt_continuous(series, name):
    """Return a row dict:  mean (SD) overall and by readmission."""
    return {
        "Variable": name,
        "Overall (N={:,})".format(len(df)): f"{series.mean():.1f} ({series.std():.1f})",
        "Readmit=0 (N={:,})".format((df['readmit_30']==0).sum()):
            f"{series[df['readmit_30']==0].mean():.1f} ({series[df['readmit_30']==0].std():.1f})",
        "Readmit=1 (N={:,})".format((df['readmit_30']==1).sum()):
            f"{series[df['readmit_30']==1].mean():.1f} ({series[df['readmit_30']==1].std():.1f})",
    }

def fmt_binary(series, name, label="Yes"):
    """Return a row dict:  n (%) overall and by readmission."""
    n_all = series.sum()
    p_all = series.mean() * 100
    n0 = series[df["readmit_30"] == 0].sum()
    p0 = series[df["readmit_30"] == 0].mean() * 100
    n1 = series[df["readmit_30"] == 1].sum()
    p1 = series[df["readmit_30"] == 1].mean() * 100
    return {
        "Variable": f"  {name} — {label}",
        "Overall (N={:,})".format(len(df)): f"{int(n_all):,} ({p_all:.1f}%)",
        "Readmit=0 (N={:,})".format((df['readmit_30']==0).sum()):
            f"{int(n0):,} ({p0:.1f}%)",
        "Readmit=1 (N={:,})".format((df['readmit_30']==1).sum()):
            f"{int(n1):,} ({p1:.1f}%)",
    }

def fmt_categorical(col_name, display_name):
    """Return a list of row dicts for a categorical variable."""
    rows = [{"Variable": display_name,
             "Overall (N={:,})".format(len(df)): "",
             "Readmit=0 (N={:,})".format((df['readmit_30']==0).sum()): "",
             "Readmit=1 (N={:,})".format((df['readmit_30']==1).sum()): ""}]
    for cat in sorted(df[col_name].unique()):
        mask = df[col_name] == cat
        n_all = mask.sum();  p_all = mask.mean() * 100
        n0 = mask[df["readmit_30"]==0].sum(); p0 = mask[df["readmit_30"]==0].mean() * 100
        n1 = mask[df["readmit_30"]==1].sum(); p1 = mask[df["readmit_30"]==1].mean() * 100
        rows.append({
            "Variable": f"  {cat}",
            "Overall (N={:,})".format(len(df)): f"{int(n_all):,} ({p_all:.1f}%)",
            "Readmit=0 (N={:,})".format((df['readmit_30']==0).sum()):
                f"{int(n0):,} ({p0:.1f}%)",
            "Readmit=1 (N={:,})".format((df['readmit_30']==1).sum()):
                f"{int(n1):,} ({p1:.1f}%)",
        })
    return rows

# --- Build rows ---
rows = []

# Continuous: use raw (un-standardised) columns
continuous_display = [
    ("age_numeric",       "Age, years"),
    ("time_in_hospital",  "Length of stay, days"),
    ("num_lab_procedures","No. lab procedures"),
    ("num_procedures",    "No. procedures"),
    ("num_medications",   "No. medications"),
    ("number_outpatient", "No. prior outpatient visits"),
    ("number_emergency",  "No. prior emergency visits"),
    ("number_inpatient",  "No. prior inpatient visits"),
    ("number_diagnoses",  "No. diagnoses on record"),
    ("n_comorbid_groups", "No. distinct comorbidity groups"),
]
for col, label in continuous_display:
    if col in df.columns:
        rows.append(fmt_continuous(df[col], label))

# Binary: gender
rows.append(fmt_binary(df["female"], "Sex", "Female"))

# Categorical: race (reconstruct from dummies)
race_cols = [c for c in df.columns if c.startswith("race_")]
if race_cols:
    # Reconstruct race label
    def get_race(row):
        for c in race_cols:
            if row[c] == 1:
                return c.replace("race_", "")
        return "Caucasian"  # reference
    df["_race"] = df.apply(get_race, axis=1)
    rows.extend(fmt_categorical("_race", "Race"))

# Categorical: primary diagnosis group
if "diagnosis_group" in df.columns:
    rows.extend(fmt_categorical("diagnosis_group", "Primary diagnosis group"))

# Binary: medication & lab
binary_display = [
    ("insulin_use",   "Insulin use",              "Yes"),
    ("med_change",    "Medication changed",        "Yes"),
    ("diabetes_med",  "Any diabetes medication",   "Yes"),
    ("A1C_measured",  "A1C measured",              "Yes"),
    ("glu_measured",  "Glucose serum measured",    "Yes"),
]
for col, label, lvl in binary_display:
    if col in df.columns:
        rows.append(fmt_binary(df[col], label, lvl))

# Comorbidity flags
comorbid_cols = sorted([c for c in df.columns if c.startswith("comorbid_")])
if comorbid_cols:
    rows.append({"Variable": "Comorbidity flags (any of diag 1/2/3)",
                 "Overall (N={:,})".format(len(df)): "",
                 "Readmit=0 (N={:,})".format((df['readmit_30']==0).sum()): "",
                 "Readmit=1 (N={:,})".format((df['readmit_30']==1).sum()): ""})
    for col in comorbid_cols:
        nice = col.replace("comorbid_", "").capitalize()
        rows.append(fmt_binary(df[col], nice, "Present"))

# --- Assemble and save ---
table1 = pd.DataFrame(rows)
out_path = TABLE_DIR / "table1_cohort_characteristics.csv"
table1.to_csv(out_path, index=False)
print(f"Table 1 saved: {out_path}  ({len(table1)} rows)")


# =============================================================================
# FIGURE 1 — Distribution of Sample Size Across Specialty Groups
# =============================================================================

spec_counts = (
    df["specialty_group"]
    .value_counts()
    .sort_values(ascending=True)
)

fig1, ax1 = plt.subplots(figsize=(8, 6.5))

colors = ["#2c7bb6" if s != "Unknown" else "#999999" for s in spec_counts.index]
bars = ax1.barh(range(len(spec_counts)), spec_counts.values, color=colors,
                edgecolor="white", linewidth=0.5, height=0.72)

ax1.set_yticks(range(len(spec_counts)))
ax1.set_yticklabels(spec_counts.index, fontsize=9.5)

# Annotate bar values
for i, (val, name) in enumerate(zip(spec_counts.values, spec_counts.index)):
    offset = max(spec_counts.values) * 0.01
    ax1.text(val + offset, i, f"{val:,}", va="center", fontsize=8.5)

ax1.set_xlabel("Number of Encounters")
ax1.set_title("Figure 1.  Distribution of Observations Across\nSpecialty Groups After Regrouping",
              fontweight="bold", fontsize=12)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax1.set_xlim(0, spec_counts.max() * 1.15)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig1_path = FIG_DIR / "fig1_specialty_distribution.png"
fig1.savefig(fig1_path)
plt.close(fig1)
print(f"Figure 1 saved: {fig1_path}")


# =============================================================================
# FIGURE 3 — Raw 30-Day Readmission Rate by Specialty Group
# =============================================================================

spec_stats = (
    df.groupby("specialty_group")["readmit_30"]
    .agg(["mean", "count", "sum"])
    .rename(columns={"mean": "rate", "count": "n", "sum": "n_readmit"})
)

# Wilson confidence interval (better than normal approx for proportions)
z = 1.96
spec_stats["wilson_center"] = (
    (spec_stats["n_readmit"] + z**2 / 2) / (spec_stats["n"] + z**2)
)
spec_stats["wilson_half"] = (
    z * np.sqrt(
        (spec_stats["n_readmit"] * (spec_stats["n"] - spec_stats["n_readmit"]) / spec_stats["n"]
         + z**2 / 4)
        / (spec_stats["n"] + z**2)
    ) / (spec_stats["n"] + z**2)
)
spec_stats["ci_lo"] = spec_stats["wilson_center"] - spec_stats["wilson_half"]
spec_stats["ci_hi"] = spec_stats["wilson_center"] + spec_stats["wilson_half"]

# Sort by rate
spec_stats = spec_stats.sort_values("rate", ascending=True)

fig3, ax3 = plt.subplots(figsize=(8, 6.5))

y_pos = np.arange(len(spec_stats))
overall_rate = df["readmit_30"].mean()

# CI error bars
xerr_lo = np.maximum(spec_stats["rate"].values - spec_stats["ci_lo"].values, 0)
xerr_hi = np.maximum(spec_stats["ci_hi"].values - spec_stats["rate"].values, 0)

# Color: highlight specialties whose CI does not overlap overall rate
colors3 = []
for _, row in spec_stats.iterrows():
    if row["ci_lo"] > overall_rate:
        colors3.append("#d7191c")     # significantly above
    elif row["ci_hi"] < overall_rate:
        colors3.append("#2c7bb6")     # significantly below
    else:
        colors3.append("#636363")     # overlaps overall

ax3.errorbar(
    spec_stats["rate"].values, y_pos,
    xerr=[xerr_lo, xerr_hi],
    fmt="o", markersize=6, color="none",
    ecolor="#888888", elinewidth=1.2, capsize=3,
)
ax3.scatter(spec_stats["rate"].values, y_pos, c=colors3, s=50, zorder=5,
            edgecolors="white", linewidths=0.5)

# Overall rate reference line
ax3.axvline(overall_rate, color="#444444", linestyle="--", linewidth=1, alpha=0.7)
ax3.text(overall_rate + 0.002, len(spec_stats) - 0.5,
         f"Overall = {overall_rate:.1%}", fontsize=8.5, color="#444444")

ax3.set_yticks(y_pos)
ax3.set_yticklabels(spec_stats.index, fontsize=9.5)

# Annotate n on right side
for i, (idx, row) in enumerate(spec_stats.iterrows()):
    ax3.text(spec_stats["ci_hi"].max() + 0.012, i,
             f"n={int(row['n']):,}", fontsize=8, color="#555555", va="center")

ax3.set_xlabel("30-Day Readmission Rate")
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax3.set_title("Figure 3.  Observed 30-Day Readmission Rates by\nSpecialty Group (with 95% Wilson CI)",
              fontweight="bold", fontsize=12)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_xlim(0, spec_stats["ci_hi"].max() + 0.06)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d7191c',
           markersize=7, label='Above overall rate'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#636363',
           markersize=7, label='Overlaps overall rate'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2c7bb6',
           markersize=7, label='Below overall rate'),
]
ax3.legend(handles=legend_elements, loc="lower right", fontsize=8.5,
           framealpha=0.9)

fig3_path = FIG_DIR / "fig3_raw_readmission_by_specialty.png"
fig3.savefig(fig3_path)
plt.close(fig3)
print(f"Figure 3 saved: {fig3_path}")


# =============================================================================
# Bonus: print key numbers for quick manuscript fill-in
# =============================================================================

print("\n" + "=" * 60)
print("Quick-reference numbers for manuscript Section 5.1 / 5.2")
print("=" * 60)
n0 = (df["readmit_30"] == 0).sum()
n1 = (df["readmit_30"] == 1).sum()
print(f"  N total     = {len(df):,}")
print(f"  N readmit=0 = {n0:,} ({n0/len(df)*100:.1f}%)")
print(f"  N readmit=1 = {n1:,} ({n1/len(df)*100:.1f}%)")
print(f"  J (groups)  = {df['specialty_group'].nunique()}")
print(f"  Overall readmission rate = {df['readmit_30'].mean():.2%}")
print(f"  Mean age    = {df['age_numeric'].mean():.1f} yrs")
print(f"  Female %    = {df['female'].mean()*100:.1f}%")
print(f"  Raw specialty rate range = [{spec_stats['rate'].min():.2%}, {spec_stats['rate'].max():.2%}]")
print(f"  Raw specialty rate SD    = {spec_stats['rate'].std():.4f}")
print(f"\nReadmit=1 vs Readmit=0 (mean comparison):")
for col, label in continuous_display:
    if col in df.columns:
        m0 = df.loc[df["readmit_30"]==0, col].mean()
        m1 = df.loc[df["readmit_30"]==1, col].mean()
        print(f"  {label:<35s}  No={m0:.2f}  Yes={m1:.2f}  diff={m1-m0:+.2f}")

print("\nDone. All outputs in output/tables/ and output/figures/")
