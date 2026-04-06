#!/usr/bin/env python3
"""
07_sensitivity_analysis.py
===========================
Prior sensitivity analysis (Pipeline Task 15).

Configurations tested:
  A1: s_beta = 5.0      (vs baseline 2.5)
  B1: s_tau  = 2.5, Half-Normal   (vs baseline s_tau=1.0)
  B2: s_tau  = 2.5, Half-Cauchy   (vs baseline Half-Normal)

Each config runs a shorter MCMC (10000 iter, 2 chains) and compares
key posterior summaries against the baseline (full run from Task 10).

Output:
  - output/tables/sensitivity_summary.csv
  - output/tables/sensitivity_beta_comparison.csv
  - output/figures/fig6_sensitivity_tau.png/pdf
  - output/figures/fig7_sensitivity_beta.png/pdf
  - output/sensitivity/<config>/chain_*.npz

Usage:
  cd /path/to/Joint-Bayesian
  python src/07_sensitivity_analysis.py

Author : [Your names]
Date   : 2026-03-16
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from importlib.machinery import SourceFileLoader

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent

sampler = SourceFileLoader(
    "sampler", str(SCRIPT_DIR / "05_mwg_sampler.py")).load_module()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 724
SENS_ITER = 10000
SENS_BURNIN = 3000
SENS_THIN = 5
SENS_CHAINS = 2

SENSITIVITY_CONFIGS = {
    "A1_sbeta5": {
        "label": r"$s_\beta = 5$",
        "hyperparams": {
            "sigma_alpha": 5.0, "s_beta": 5.0,
            "s_tau": 1.0, "prior_family": "half_normal",
        },
    },
    "B1_stau2.5_hn": {
        "label": r"$s_\tau = 2.5$ (Half-Normal)",
        "hyperparams": {
            "sigma_alpha": 5.0, "s_beta": 2.5,
            "s_tau": 2.5, "prior_family": "half_normal",
        },
    },
    "B2_stau2.5_hc": {
        "label": r"$s_\tau = 2.5$ (Half-Cauchy)",
        "hyperparams": {
            "sigma_alpha": 5.0, "s_beta": 2.5,
            "s_tau": 2.5, "prior_family": "half_cauchy",
        },
    },
}

# Output paths
FIG_DIR = PROJECT / "output" / "figures"
TAB_DIR = PROJECT / "output" / "tables"
SENS_DIR = PROJECT / "output" / "sensitivity"

# Plot style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_baseline_chains():
    """Load baseline chains from Task 10 full run."""
    chain_dir = PROJECT / "output" / "chains"
    with open(chain_dir / "mcmc_meta.json") as f:
        meta = json.load(f)
    chains = []
    for c in range(meta["n_chains"]):
        data = np.load(chain_dir / f"chain_{c}.npz")
        chains.append({k: data[k] for k in data.files})
    return chains, meta


def summarize(chains):
    """Extract key posterior summaries from a list of chain dicts."""
    alpha_all = np.concatenate([c["alpha"] for c in chains])
    beta_all = np.vstack([c["beta"] for c in chains])
    tau_all = np.concatenate([c["tau"] for c in chains])
    u_all = np.vstack([c["u"] for c in chains])

    return {
        "alpha_mean": float(alpha_all.mean()),
        "alpha_sd": float(alpha_all.std()),
        "tau_mean": float(tau_all.mean()),
        "tau_sd": float(tau_all.std()),
        "tau_ci_lo": float(np.percentile(tau_all, 2.5)),
        "tau_ci_hi": float(np.percentile(tau_all, 97.5)),
        "tau_samples": tau_all,
        "beta_means": beta_all.mean(axis=0),
        "beta_sds": beta_all.std(axis=0),
        "u_means": u_all.mean(axis=0),
        "u_rank": np.argsort(np.argsort(-u_all.mean(axis=0))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    DATA_DIR = str(PROJECT / "data_processed")

    # Load data
    X, y, group_idx, N, K, J = sampler.load_data(DATA_DIR)
    var_names = pd.read_csv(
        PROJECT / "data_processed" / "data_dictionary.csv")["variable"].tolist()
    spec_map = pd.read_csv(
        PROJECT / "data_processed" / "specialty_mapping.csv"
    ).sort_values("group_idx")
    spec_names = spec_map["specialty_group"].tolist()

    # Load tuned proposal SDs
    tuning_path = PROJECT / "output" / "chains" / "tuning_result.json"
    if tuning_path.exists():
        with open(tuning_path) as f:
            proposal_sd = json.load(f)["tuned_proposal_sd"]
        print("Loaded tuned proposal SDs")
    else:
        proposal_sd = sampler.DEFAULT_PROPOSAL_SD.copy()
        print("WARNING: Using default proposal SDs (no tuning file found)")

    # -----------------------------------------------------------------------
    # Baseline summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Loading baseline (Task 10 full run)")
    print("=" * 60)
    baseline_chains, baseline_meta = load_baseline_chains()
    summaries = {"baseline": summarize(baseline_chains)}
    labels = {"baseline": "Baseline"}

    # -----------------------------------------------------------------------
    # Run each sensitivity configuration
    # -----------------------------------------------------------------------
    for name, cfg in SENSITIVITY_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"Sensitivity: {name} -- {cfg['label']}")
        print(f"{'=' * 60}")

        out_dir = str(SENS_DIR / name)

        all_samples, all_diag = sampler.run_multi_chain(
            X, y, group_idx, N, K, J,
            n_iter=SENS_ITER, burnin=SENS_BURNIN, thin=SENS_THIN,
            n_chains=SENS_CHAINS,
            hyperparams=cfg["hyperparams"],
            proposal_sd=proposal_sd,
            seed=SEED, output_dir=out_dir,
        )

        summaries[name] = summarize(
            [{k: s[k] for k in s} for s in all_samples])
        labels[name] = cfg["label"]

    # -----------------------------------------------------------------------
    # Comparison table: tau
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SENSITIVITY COMPARISON")
    print("=" * 60)

    all_names = ["baseline"] + list(SENSITIVITY_CONFIGS.keys())

    rows = []
    for name in all_names:
        s = summaries[name]
        rows.append({
            "config": name,
            "label": labels[name],
            "tau_mean": round(s["tau_mean"], 4),
            "tau_sd": round(s["tau_sd"], 4),
            "tau_ci": f"[{s['tau_ci_lo']:.4f}, {s['tau_ci_hi']:.4f}]",
            "alpha_mean": round(s["alpha_mean"], 4),
            "alpha_sd": round(s["alpha_sd"], 4),
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(TAB_DIR / "sensitivity_summary.csv", index=False)
    print(f"  Saved sensitivity_summary.csv")
    print(comp_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Key beta comparison (top 5 by |baseline effect|)
    # -----------------------------------------------------------------------
    base_beta = summaries["baseline"]["beta_means"]
    top_idx = np.argsort(np.abs(base_beta))[-5:][::-1]

    beta_rows = []
    for k in top_idx:
        row = {"variable": var_names[k] if k < len(var_names) else f"beta_{k}"}
        for name in all_names:
            s = summaries[name]
            row[f"{name}_mean"] = round(float(s["beta_means"][k]), 4)
            row[f"{name}_sd"] = round(float(s["beta_sds"][k]), 4)
        beta_rows.append(row)

    beta_df = pd.DataFrame(beta_rows)
    beta_df.to_csv(TAB_DIR / "sensitivity_beta_comparison.csv", index=False)
    print(f"\n  Saved sensitivity_beta_comparison.csv")

    print(f"\n  Key beta comparison (top 5 by |effect|):")
    header = f"  {'Variable':>30s}"
    for name in all_names:
        header += f"  {name:>18s}"
    print(header)
    for k in top_idx:
        line = f"  {var_names[k]:>30s}"
        for name in all_names:
            line += f"  {summaries[name]['beta_means'][k]:>+18.4f}"
        print(line)

    # -----------------------------------------------------------------------
    # Specialty ranking stability
    # -----------------------------------------------------------------------
    print(f"\n  Specialty ranking comparison:")
    base_rank = summaries["baseline"]["u_rank"]
    for name in list(SENSITIVITY_CONFIGS.keys()):
        other_rank = summaries[name]["u_rank"]
        rank_corr = float(np.corrcoef(base_rank, other_rank)[0, 1])
        max_rank_shift = int(np.max(np.abs(
            base_rank.astype(int) - other_rank.astype(int))))
        print(f"    {name}: rank corr = {rank_corr:.4f}, "
              f"max rank shift = {max_rank_shift}")

    # -----------------------------------------------------------------------
    # Figure 6: tau posterior under different priors
    # -----------------------------------------------------------------------
    colors = {
        "baseline": "#185FA5",
        "A1_sbeta5": "#D85A30",
        "B1_stau2.5_hn": "#0F6E56",
        "B2_stau2.5_hc": "#993556",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in all_names:
        tau_s = summaries[name]["tau_samples"]
        ax.hist(tau_s, bins=50, density=True, alpha=0.4,
                color=colors.get(name, "gray"),
                label=f"{labels[name]} (mean={tau_s.mean():.3f})")
    ax.set_xlabel(r"$\tau$ (specialty-level SD)")
    ax.set_ylabel("Posterior density")
    ax.set_title(r"Prior Sensitivity: Posterior distribution of $\tau$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_sensitivity_tau.png")
    fig.savefig(FIG_DIR / "fig6_sensitivity_tau.pdf")
    plt.close(fig)
    print(f"\n  Saved fig6_sensitivity_tau.{{png,pdf}}")

    # -----------------------------------------------------------------------
    # Figure 7: key beta forest plot across configs
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    n_configs = len(all_names)
    offsets = np.linspace(-0.3, 0.3, n_configs)
    y_pos = np.arange(len(top_idx))

    for ci, name in enumerate(all_names):
        s = summaries[name]
        for i, k in enumerate(top_idx):
            m = s["beta_means"][k]
            sd = s["beta_sds"][k]
            c = colors.get(name, "gray")
            ax.plot(m, i + offsets[ci], "o", color=c, markersize=5)
            ax.plot([m - 1.96 * sd, m + 1.96 * sd],
                    [i + offsets[ci], i + offsets[ci]],
                    color=c, linewidth=1.2)

    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([var_names[k] for k in top_idx], fontsize=9)
    ax.set_xlabel("Posterior mean (logit scale)")
    ax.set_title("Prior Sensitivity: Key fixed effects")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=colors.get(n, "gray"), marker="o",
                       linestyle="-", label=labels[n]) for n in all_names]
    ax.legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_sensitivity_beta.png")
    fig.savefig(FIG_DIR / "fig7_sensitivity_beta.pdf")
    plt.close(fig)
    print(f"  Saved fig7_sensitivity_beta.{{png,pdf}}")

    # -----------------------------------------------------------------------
    # Interpretation summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INTERPRETATION NOTES")
    print("=" * 60)

    # Check tau stability
    tau_means = [summaries[n]["tau_mean"] for n in all_names]
    tau_range = max(tau_means) - min(tau_means)
    print(f"  tau posterior mean range across configs: {tau_range:.4f}")
    if tau_range < 0.05:
        print("  -> tau is STABLE across prior choices.")
    else:
        print("  -> tau shows SOME sensitivity to prior choices.")

    # Check beta stability
    print(f"\n  Beta stability (top 5 predictors):")
    for k in top_idx:
        vals = [summaries[n]["beta_means"][k] for n in all_names]
        rng = max(vals) - min(vals)
        stable = "STABLE" if rng < 0.02 else "SENSITIVE"
        vname = var_names[k] if k < len(var_names) else f"beta_{k}"
        print(f"    {vname:>30s}: range={rng:.4f} -> {stable}")

    print("\n" + "=" * 60)
    print("Sensitivity analysis complete.")
    print(f"Figures: {FIG_DIR}")
    print(f"Tables:  {TAB_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
