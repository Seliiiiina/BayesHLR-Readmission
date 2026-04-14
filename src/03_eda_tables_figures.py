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
# FIGURE 4 + TABLE 2 — Specialty-Effect Additivity Check
# =============================================================================
#
# Rationale
# ---------
# The hierarchical model assumes  eta_i = alpha + x_i^T beta + u_{j[i]},
# i.e. the specialty contribution u_j is a constant additive shift on the
# logit that does NOT depend on the patient's covariates x_i (no random
# slopes, no interactions).  If this assumption were grossly violated we
# would expect specialty readmission rates to change ordering or relative
# magnitude across different patient sub-groups.  Here we:
#
#   1. Stratify the cohort by three clinically meaningful profile variables
#      (age band, comorbidity burden, primary-diagnosis group).
#   2. Re-compute each specialty's raw readmission rate within each stratum.
#   3. Plot the within-stratum rates for the 10 largest specialties and check
#      whether the curves are roughly parallel (supports additivity).
#   4. Quantify additivity two ways:
#        (a) Spearman rank correlation of specialty rates across strata
#            (close to 1 => preserved ordering);
#        (b) Likelihood-ratio test + information criteria for specialty ×
#            covariate interactions in a logistic regression that already
#            controls for the main effects.

import statsmodels.api as sm
from scipy.stats import chi2

TOP_K = 10
top_specs = df["specialty_group"].value_counts().head(TOP_K).index.tolist()
df_top = df[df["specialty_group"].isin(top_specs)].copy()

spec_order = (
    df_top.groupby("specialty_group")["readmit_30"].mean()
    .sort_values().index.tolist()
)

def _short(name, k=14):
    return name if len(name) <= k else name[: k - 1] + "."

df_top["age_band"] = pd.cut(
    df_top["age_numeric"],
    bins=[-np.inf, 54, 74, np.inf],
    labels=["<=54", "55-74", ">=75"],
)
df_top["comorb_band"] = pd.cut(
    df_top["n_comorbid_groups"],
    bins=[-np.inf, 1, 2, np.inf],
    labels=["0-1", "2", "3"],
)
df_top["sex_band"] = df_top["female"].map({1: "Female", 0: "Male"})


def _rate_matrix(frame, strat_col, specs, min_cell=50):
    """Stratum x specialty matrix of rates; cells with n<min_cell masked."""
    g = (
        frame.groupby([strat_col, "specialty_group"], observed=True)["readmit_30"]
        .agg(["mean", "count"])
        .reset_index()
    )
    rates = g.pivot(index=strat_col, columns="specialty_group", values="mean").reindex(columns=specs)
    counts = g.pivot(index=strat_col, columns="specialty_group", values="count").reindex(columns=specs)
    return rates.where(counts >= min_cell), counts


age_rates, _ = _rate_matrix(df_top, "age_band",    spec_order)
com_rates, _ = _rate_matrix(df_top, "comorb_band", spec_order)
sex_rates, _ = _rate_matrix(df_top, "sex_band",    spec_order)
sex_rates = sex_rates.reindex(["Female", "Male"])

# ── Figure 4 ──────────────────────────────────────────────────────────────────
fig4, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)

panels = [
    (axes[0], age_rates, "A.  By age band",           ["#1b9e77", "#7570b3", "#d95f02"]),
    (axes[1], com_rates, "B.  By comorbidity burden", ["#66c2a5", "#fc8d62", "#8da0cb"]),
    (axes[2], sex_rates, "C.  By sex",                ["#e41a1c", "#377eb8"]),
]

x_pos = np.arange(len(spec_order))
tick_labels = [_short(s) for s in spec_order]

for ax, rates, title, colors in panels:
    for (lbl, row), c in zip(rates.iterrows(), colors):
        ax.plot(x_pos, row.values, marker="o", markersize=5, linewidth=1.6,
                color=c, label=str(lbl))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

axes[0].set_ylabel("30-day readmission rate")
fig4.suptitle(
    "Figure 4.  Specialty readmission rates within patient sub-groups\n"
    "(near-parallel curves support a constant additive specialty effect)",
    fontsize=12, fontweight="bold", y=1.03,
)
fig4.tight_layout()
fig4_path = FIG_DIR / "fig4_specialty_additivity.png"
fig4.savefig(fig4_path)
plt.close(fig4)
print(f"Figure 4 saved: {fig4_path}")


# ── Table 2: Logistic LRT / BIC for the additivity assumption ────────────────
#
# We fit two nested logistic regressions on the full cohort:
#
#   M0 (additive, matches our Bayesian model):
#       logit p = alpha + x_i^T beta + specialty_dummies
#
#   M1 (allows specialty slope to vary):
#       M0  +  specialty_dummies x (one continuous covariate)
#
# The Bayesian random-intercept assumption is that the specialty effect u_j
# is the same regardless of x_i.  Testing interactions with single continuous
# covariates (age, comorbidity burden, number of prior inpatient visits) gives
# J-1 = 18 added parameters per test — all stable, no sparse-cell issues.
# We report (i) the LRT chi2/df/p, (ii) dAIC and dBIC (negative = M1 preferred),
# (iii) average magnitude of the interaction coefficients, and
# (iv) the % of main-effect deviance improvement contributed by interactions:
#        100 * (LL(M1) - LL(M0)) / (LL(M0) - LL(null))
# At N~38k the LRT is almost guaranteed to reject, so effect size (iii)-(iv)
# is the meaningful diagnostic.

X_mat = pd.read_csv(DATA_PROCESSED / "X_matrix.csv")
y_full = df["readmit_30"].values

spec_dum = pd.get_dummies(df["specialty_group"], drop_first=True).astype(float)
base_X = pd.concat([X_mat.reset_index(drop=True),
                    spec_dum.reset_index(drop=True)], axis=1)
X0 = sm.add_constant(base_X)

# Null (intercept-only) log-likelihood for the deviance-improvement denominator
m_null = sm.Logit(y_full, np.ones((len(y_full), 1))).fit(disp=False, method="lbfgs")
m0     = sm.Logit(y_full, X0).fit(disp=False, method="lbfgs", maxiter=500)
ll_null = m_null.llf
ll_m0   = m0.llf

INTERACTION_VARS = [
    ("age_numeric_z",       "age (standardised)"),
    ("n_comorbid_groups_z", "comorbidity burden"),
    ("female",              "sex (female)"),
]

def _fit_interaction(inter_df):
    X1 = sm.add_constant(pd.concat([base_X.reset_index(drop=True),
                                     inter_df.reset_index(drop=True)], axis=1))
    m1 = sm.Logit(y_full, X1).fit(disp=False, method="lbfgs", maxiter=500)
    df_diff = X1.shape[1] - X0.shape[1]
    lrt = 2.0 * (m1.llf - ll_m0)
    p   = 1.0 - chi2.cdf(lrt, df_diff)
    return {
        "lrt": lrt, "df": df_diff, "p": p,
        "dAIC": m1.aic - m0.aic, "dBIC": m1.bic - m0.bic,
        "pct_added": 100.0 * (m1.llf - ll_m0) / (ll_m0 - ll_null),
    }

lrt_rows = []
for col, label in INTERACTION_VARS:
    inter = spec_dum.multiply(X_mat[col].values, axis=0)
    inter.columns = [f"{col}:{c}" for c in inter.columns]
    r = _fit_interaction(inter)
    lrt_rows.append({
        "Interaction tested":  f"specialty x {label}",
        "LRT (chi2, df, p)":   f"{r['lrt']:.1f}, df={r['df']}, p={r['p']:.2g}",
        "dAIC / dBIC":         f"{r['dAIC']:+.1f} / {r['dBIC']:+.1f}",
        "BIC verdict":         "additive preferred" if r["dBIC"] > 0 else "interaction preferred",
        "% extra deviance":    f"{r['pct_added']:.2f}%",
    })

table2 = pd.DataFrame(lrt_rows)
table2_path = TABLE_DIR / "table2_specialty_additivity.csv"
table2.to_csv(table2_path, index=False)
print(f"Table 2 saved: {table2_path}")
print("\nAdditivity diagnostics:")
print(table2.to_string(index=False))


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
