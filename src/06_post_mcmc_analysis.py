#!/usr/bin/env python3
"""
06_post_mcmc_analysis.py
========================
Post-MCMC analysis pipeline covering Pipeline Tasks 11-14:
  Task 11: MCMC Diagnostics (trace plots, ACF, ESS)
  Task 12: Posterior Inference (summaries, forest plots, OR)
  Task 13: Baseline Model Comparison (AUC, Brier, calibration)
  Task 14: Posterior Predictive Checks

Input:
  - output/chains/chain_0.npz ... chain_3.npz
  - output/chains/mcmc_meta.json
  - data_processed/X_matrix.npy, y_vector.npy, group_index.npy
  - data_processed/analytic_cohort.csv
  - data_processed/data_dictionary.csv
  - data_processed/specialty_mapping.csv

Output:
  - output/figures/fig2_trace_plots.pdf
  - output/figures/fig4_specialty_random_effects.pdf
  - output/figures/fig5_posterior_predictive.pdf
  - output/figures/fig_acf.pdf
  - output/figures/fig_tau_posterior.pdf
  - output/tables/table2_fixed_effects.csv
  - output/tables/table3_model_comparison.csv
  - output/tables/diagnostics_summary.csv
  - output/tables/sensitivity_ready_baseline.json

Usage:
  cd /work/zq63/Bayes/Joint-Bayesian
  python src/06_post_mcmc_analysis.py

Author : [Your names]
Date   : 2026-03-15
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.special import expit  # logistic sigmoid

# =============================================================================
# 0. PATHS & DATA LOADING
# =============================================================================

PROJECT = Path(__file__).resolve().parent.parent
CHAIN_DIR = PROJECT / "output" / "chains"
DATA_DIR = PROJECT / "data_processed"
FIG_DIR = PROJECT / "output" / "figures"
TAB_DIR = PROJECT / "output" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_chains():
    """Load all chains and metadata."""
    with open(CHAIN_DIR / "mcmc_meta.json") as f:
        meta = json.load(f)
    n_chains = meta["n_chains"]
    chains = []
    for c in range(n_chains):
        data = np.load(CHAIN_DIR / f"chain_{c}.npz")
        chains.append({k: data[k] for k in data.files})
    print(f"Loaded {n_chains} chains, {chains[0]['alpha'].shape[0]} samples each")
    return chains, meta


def load_data():
    """Load design matrix, outcome, group index."""
    X = np.load(DATA_DIR / "X_matrix.npy")
    y = np.load(DATA_DIR / "y_vector.npy")
    g = np.load(DATA_DIR / "group_index.npy")
    return X, y, g


def load_var_names():
    """Load variable names from data dictionary."""
    dd = pd.read_csv(DATA_DIR / "data_dictionary.csv")
    return dd["variable"].tolist()


def load_specialty_mapping():
    """Load specialty group labels."""
    sm = pd.read_csv(DATA_DIR / "specialty_mapping.csv")
    return sm.sort_values("group_idx")["specialty_group"].tolist()


# =============================================================================
# TASK 11: MCMC DIAGNOSTICS
# =============================================================================

def compute_ess_single(x):
    """Effective sample size via autocorrelation (Geyer's initial monotone)."""
    n = len(x)
    x = x - x.mean()
    var = np.var(x)
    if var == 0:
        return float(n)
    acf = np.correlate(x, x, mode="full")[n - 1:] / (var * n)
    # Sum consecutive pairs until they become negative
    total = 0.0
    for i in range(1, n // 2):
        pair = acf[2 * i - 1] + acf[2 * i]
        if pair < 0:
            break
        total += pair
    ess = n / (1 + 2 * total)
    return max(1.0, ess)


def run_diagnostics(chains, meta, var_names, spec_names):
    """Task 11: Trace plots, ACF, ESS, acceptance rates."""
    print("\n" + "=" * 60)
    print("TASK 11: MCMC Diagnostics")
    print("=" * 60)

    n_chains = len(chains)
    n_samples = chains[0]["alpha"].shape[0]

    # --- Select representative parameters for diagnostics ---
    # alpha, 3 key betas (first, a mid-range, last), tau, 3 representative u_j
    K = chains[0]["beta"].shape[1]
    J = chains[0]["u"].shape[1]
    beta_idx = [0, K // 2, K - 1]
    u_idx = [0, J // 2, J - 1]

    param_specs = (
        [("alpha", lambda c: c["alpha"], r"$\alpha$")]
        + [(f"beta_{i}", lambda c, i=i: c["beta"][:, i],
            f"$\\beta_{{{i}}}$ ({var_names[i][:20]})")
           for i in beta_idx]
        + [("tau", lambda c: c["tau"], r"$\tau$")]
        + [(f"u_{j}", lambda c, j=j: c["u"][:, j],
            f"$u_{{{j}}}$ ({spec_names[j][:20]})")
           for j in u_idx]
    )

    # --- Figure 2: Trace plots ---
    n_params = len(param_specs)
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 2.5 * n_params))
    if n_params == 1:
        axes = axes.reshape(1, -1)

    colors = ["#185FA5", "#D85A30", "#0F6E56", "#993556"]
    ess_records = []

    for row, (name, extractor, label) in enumerate(param_specs):
        ax_trace = axes[row, 0]
        ax_acf = axes[row, 1]

        all_ess = []
        for c_id in range(n_chains):
            vals = extractor(chains[c_id])
            ax_trace.plot(vals, alpha=0.6, linewidth=0.4, color=colors[c_id],
                          label=f"Chain {c_id}")
            ess = compute_ess_single(vals)
            all_ess.append(ess)

            # ACF for chain 0 only
            if c_id == 0:
                max_lag = min(100, n_samples // 2)
                x = vals - vals.mean()
                var = np.var(x)
                if var > 0:
                    acf = np.correlate(x, x, mode="full")[n_samples - 1:]
                    acf = acf[:max_lag + 1] / (var * n_samples)
                    ax_acf.bar(range(max_lag + 1), acf, width=1.0,
                               color="#185FA5", alpha=0.6)
                ax_acf.axhline(0, color="black", linewidth=0.5)
                ax_acf.set_xlim(0, max_lag)
                ax_acf.set_ylim(-0.2, 1.05)
                ax_acf.set_xlabel("Lag")
                ax_acf.set_ylabel("ACF")
                ax_acf.set_title(f"{label} — autocorrelation (chain 0)")

        ax_trace.set_ylabel(label)
        ax_trace.set_xlabel("Iteration (post burn-in)")
        mean_ess = np.mean(all_ess)
        ax_trace.set_title(f"{label} — trace (mean ESS={mean_ess:.0f})")
        if row == 0:
            ax_trace.legend(fontsize=8, loc="upper right")

        ess_records.append({
            "parameter": name, "label": label,
            **{f"ESS_chain_{c}": all_ess[c] for c in range(n_chains)},
            "ESS_mean": mean_ess,
        })

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_trace_plots.png")
    plt.close(fig)
    print(f"  Saved fig2_trace_plots.png")

    # --- ESS summary table ---
    ess_df = pd.DataFrame(ess_records)
    ess_df.to_csv(TAB_DIR / "diagnostics_summary.csv", index=False)
    print(f"  Saved diagnostics_summary.csv")
    print("\n  ESS Summary:")
    for _, row in ess_df.iterrows():
        print(f"    {row['parameter']:>12s}: ESS = {row['ESS_mean']:.0f}")

    # --- Load acceptance rates from diagnostics JSON ---
    print("\n  Acceptance Rates (from chain diagnostics):")
    for c_id in range(n_chains):
        with open(CHAIN_DIR / f"chain_{c_id}_diagnostics.json") as f:
            diag = json.load(f)
        print(f"    Chain {c_id}: alpha={diag['accept_rate_alpha']:.3f}, "
              f"beta={diag['accept_rate_beta_mean']:.3f}, "
              f"u={diag['accept_rate_u_mean']:.3f}, "
              f"tau={diag['accept_rate_log_tau']:.3f}")

    return ess_df


# =============================================================================
# TASK 12: POSTERIOR INFERENCE
# =============================================================================

def run_inference(chains, meta, var_names, spec_names):
    """Task 12: Posterior summaries, credible intervals, forest plots."""
    print("\n" + "=" * 60)
    print("TASK 12: Posterior Inference")
    print("=" * 60)

    # Pool all chains
    alpha_all = np.concatenate([c["alpha"] for c in chains])
    beta_all = np.vstack([c["beta"] for c in chains])
    u_all = np.vstack([c["u"] for c in chains])
    tau_all = np.concatenate([c["tau"] for c in chains])

    n_total = len(alpha_all)
    K = beta_all.shape[1]
    J = u_all.shape[1]
    print(f"  Pooled samples: {n_total} (from {len(chains)} chains)")

    # --- Table 2: Fixed effects posterior summary ---
    rows = []
    # Intercept
    rows.append({
        "parameter": "alpha (intercept)",
        "post_mean": alpha_all.mean(),
        "post_sd": alpha_all.std(),
        "ci_lower": np.percentile(alpha_all, 2.5),
        "ci_upper": np.percentile(alpha_all, 97.5),
        "OR": np.exp(alpha_all.mean()),
        "OR_ci_lower": np.exp(np.percentile(alpha_all, 2.5)),
        "OR_ci_upper": np.exp(np.percentile(alpha_all, 97.5)),
    })
    # Beta coefficients
    for k in range(K):
        b = beta_all[:, k]
        rows.append({
            "parameter": var_names[k] if k < len(var_names) else f"beta_{k}",
            "post_mean": b.mean(),
            "post_sd": b.std(),
            "ci_lower": np.percentile(b, 2.5),
            "ci_upper": np.percentile(b, 97.5),
            "OR": np.exp(b.mean()),
            "OR_ci_lower": np.exp(np.percentile(b, 2.5)),
            "OR_ci_upper": np.exp(np.percentile(b, 97.5)),
        })

    table2 = pd.DataFrame(rows)
    table2.to_csv(TAB_DIR / "table2_fixed_effects.csv", index=False)
    print(f"  Saved table2_fixed_effects.csv ({len(table2)} rows)")

    # Print key results
    print("\n  Key fixed effects (|post_mean| > 0.05):")
    for _, r in table2.iterrows():
        if abs(r["post_mean"]) > 0.05 or r["parameter"] == "alpha (intercept)":
            sig = "*" if (r["ci_lower"] > 0 or r["ci_upper"] < 0) else ""
            print(f"    {r['parameter']:>35s}: "
                  f"mean={r['post_mean']:+.4f} "
                  f"95%CI=[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}] "
                  f"OR={r['OR']:.3f} {sig}")

    # --- tau posterior ---
    print(f"\n  tau (specialty SD):")
    print(f"    mean = {tau_all.mean():.4f}")
    print(f"    sd   = {tau_all.std():.4f}")
    print(f"    95%CI = [{np.percentile(tau_all, 2.5):.4f}, "
          f"{np.percentile(tau_all, 97.5):.4f}]")
    print(f"    P(tau > 0.1) = {(tau_all > 0.1).mean():.4f}")
    print(f"    P(tau > 0.2) = {(tau_all > 0.2).mean():.4f}")

    # --- Figure: tau posterior density ---
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(tau_all, bins=60, density=True, color="#185FA5", alpha=0.6,
            edgecolor="white", linewidth=0.3)
    ax.axvline(tau_all.mean(), color="#A32D2D", linestyle="--", linewidth=1.2,
               label=f"Mean = {tau_all.mean():.3f}")
    ax.axvline(np.percentile(tau_all, 2.5), color="#888780", linestyle=":",
               linewidth=1, label="95% CrI")
    ax.axvline(np.percentile(tau_all, 97.5), color="#888780", linestyle=":",
               linewidth=1)
    ax.set_xlabel(r"$\tau$ (specialty-level SD)")
    ax.set_ylabel("Posterior density")
    ax.set_title(r"Posterior distribution of $\tau$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tau_posterior.png")
    plt.close(fig)
    print(f"  Saved fig_tau_posterior.png")

    # --- Figure 4: Specialty random effects (forest plot) ---
    u_means = u_all.mean(axis=0)
    u_lower = np.percentile(u_all, 2.5, axis=0)
    u_upper = np.percentile(u_all, 97.5, axis=0)

    # Sort by posterior mean
    order = np.argsort(u_means)
    fig, ax = plt.subplots(figsize=(7, 6))
    y_pos = np.arange(J)

    for i, j in enumerate(order):
        color = "#A32D2D" if u_lower[j] > 0 else (
                "#185FA5" if u_upper[j] < 0 else "#888780")
        ax.plot([u_lower[j], u_upper[j]], [i, i], color=color,
                linewidth=1.5, solid_capstyle="round")
        ax.plot(u_means[j], i, "o", color=color, markersize=5)

    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([spec_names[j] for j in order], fontsize=9)
    ax.set_xlabel("Posterior random intercept (logit scale)")
    ax.set_title("Specialty random effects with 95% credible intervals")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_specialty_random_effects.png")
    plt.close(fig)
    print(f"  Saved fig4_specialty_random_effects.png")

    # Specialty summary
    print("\n  Specialty random effects (sorted):")
    for i, j in enumerate(order):
        sig = "*" if (u_lower[j] > 0 or u_upper[j] < 0) else ""
        print(f"    {spec_names[j]:>35s}: "
              f"mean={u_means[j]:+.4f} "
              f"95%CI=[{u_lower[j]:+.4f}, {u_upper[j]:+.4f}] {sig}")

    return table2, tau_all, u_means, u_lower, u_upper


# =============================================================================
# TASK 13: MODEL COMPARISON
# =============================================================================

def run_model_comparison(chains, X, y, group_idx):
    """Task 13: Compare hierarchical model with baselines."""
    print("\n" + "=" * 60)
    print("TASK 13: Model Comparison")
    print("=" * 60)

    N, K = X.shape
    J = int(group_idx.max()) + 1

    # --- Hierarchical model predictions (posterior mean) ---
    alpha_all = np.concatenate([c["alpha"] for c in chains])
    beta_all = np.vstack([c["beta"] for c in chains])
    u_all = np.vstack([c["u"] for c in chains])

    alpha_hat = alpha_all.mean()
    beta_hat = beta_all.mean(axis=0)
    u_hat = u_all.mean(axis=0)

    eta_hier = alpha_hat + X @ beta_hat + u_hat[group_idx]
    p_hier = expit(eta_hier)

    # --- Baseline 1: logistic regression (no specialty) ---
    lr_no_spec = LogisticRegression(max_iter=5000, penalty=None, solver="lbfgs")
    lr_no_spec.fit(X, y)
    p_no_spec = lr_no_spec.predict_proba(X)[:, 1]

    # --- Baseline 2: logistic regression with specialty fixed effects ---
    spec_dummies = np.zeros((N, J))
    for j in range(J):
        spec_dummies[group_idx == j, j] = 1
    # Drop last column for identifiability
    X_with_spec = np.hstack([X, spec_dummies[:, :-1]])
    lr_fixed = LogisticRegression(max_iter=5000, penalty=None, solver="lbfgs")
    lr_fixed.fit(X_with_spec, y)
    p_fixed = lr_fixed.predict_proba(X_with_spec)[:, 1]

    # --- Metrics ---
    results = []
    for name, p_pred in [("No specialty (baseline)", p_no_spec),
                          ("Specialty fixed effects", p_fixed),
                          ("Bayesian hierarchical", p_hier)]:
        p_pred = np.clip(p_pred, 1e-8, 1 - 1e-8)
        auc = roc_auc_score(y, p_pred)
        brier = brier_score_loss(y, p_pred)
        log_loss = -np.mean(y * np.log(p_pred) + (1 - y) * np.log(1 - p_pred))

        # Calibration: observed vs predicted in deciles
        deciles = pd.qcut(p_pred, 10, duplicates="drop")
        cal_df = pd.DataFrame({"y": y, "p": p_pred, "decile": deciles})
        cal_grouped = cal_df.groupby("decile", observed=True).agg(
            obs_rate=("y", "mean"), pred_rate=("p", "mean")).reset_index()
        cal_slope = np.polyfit(cal_grouped["pred_rate"], cal_grouped["obs_rate"], 1)[0]

        results.append({
            "model": name,
            "AUC": round(auc, 4),
            "Brier_score": round(brier, 4),
            "log_loss": round(log_loss, 4),
            "calibration_slope": round(cal_slope, 3),
        })
        print(f"  {name:>30s}: AUC={auc:.4f}  Brier={brier:.4f}  "
              f"LogLoss={log_loss:.4f}  CalSlope={cal_slope:.3f}")

    table3 = pd.DataFrame(results)
    table3.to_csv(TAB_DIR / "table3_model_comparison.csv", index=False)
    print(f"\n  Saved table3_model_comparison.csv")

    return table3


# =============================================================================
# TASK 14: POSTERIOR PREDICTIVE CHECK
# =============================================================================

def run_ppc(chains, X, y, group_idx, spec_names):
    """Task 14: Posterior predictive checks."""
    print("\n" + "=" * 60)
    print("TASK 14: Posterior Predictive Checks")
    print("=" * 60)

    N, K = X.shape
    J = int(group_idx.max()) + 1

    # Pool chains
    alpha_all = np.concatenate([c["alpha"] for c in chains])
    beta_all = np.vstack([c["beta"] for c in chains])
    u_all = np.vstack([c["u"] for c in chains])
    n_posterior = len(alpha_all)

    # Draw subset of posterior samples for PPC (200 is enough)
    rng = np.random.RandomState(724)
    n_rep = 200
    idx = rng.choice(n_posterior, size=n_rep, replace=False)

    # --- PPC 1: Overall readmission rate ---
    rep_overall = np.zeros(n_rep)
    # --- PPC 2: SD of specialty-level readmission rates ---
    rep_spec_sd = np.zeros(n_rep)
    # --- PPC 3: Specialty-level rates (for each replicate) ---
    rep_spec_rates = np.zeros((n_rep, J))

    for s in range(n_rep):
        i = idx[s]
        eta_s = alpha_all[i] + X @ beta_all[i] + u_all[i][group_idx]
        p_s = expit(eta_s)
        y_rep = rng.binomial(1, p_s)

        rep_overall[s] = y_rep.mean()
        for j in range(J):
            mask = group_idx == j
            rep_spec_rates[s, j] = y_rep[mask].mean() if mask.sum() > 0 else 0
        rep_spec_sd[s] = rep_spec_rates[s].std()

    # Observed statistics
    obs_overall = y.mean()
    obs_spec_rates = np.array([y[group_idx == j].mean() for j in range(J)])
    obs_spec_sd = obs_spec_rates.std()

    # Bayesian p-values
    pval_overall = np.mean(rep_overall >= obs_overall)
    pval_sd = np.mean(rep_spec_sd >= obs_spec_sd)

    print(f"  Observed overall readmission rate: {obs_overall:.4f}")
    print(f"  Replicated mean: {rep_overall.mean():.4f} "
          f"(95% interval: [{np.percentile(rep_overall, 2.5):.4f}, "
          f"{np.percentile(rep_overall, 97.5):.4f}])")
    print(f"  Bayesian p-value (overall rate): {pval_overall:.3f}")
    print()
    print(f"  Observed specialty rate SD: {obs_spec_sd:.4f}")
    print(f"  Replicated mean: {rep_spec_sd.mean():.4f} "
          f"(95% interval: [{np.percentile(rep_spec_sd, 2.5):.4f}, "
          f"{np.percentile(rep_spec_sd, 97.5):.4f}])")
    print(f"  Bayesian p-value (specialty SD): {pval_sd:.3f}")

    # --- Figure 5: PPC ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Overall readmission rate
    ax = axes[0]
    ax.hist(rep_overall, bins=30, density=True, color="#185FA5", alpha=0.5,
            edgecolor="white", linewidth=0.3, label="Replicated")
    ax.axvline(obs_overall, color="#A32D2D", linewidth=2, linestyle="--",
               label=f"Observed = {obs_overall:.4f}")
    ax.set_xlabel("Overall 30-day readmission rate")
    ax.set_ylabel("Density")
    ax.set_title(f"(A) Overall rate (p-value = {pval_overall:.3f})")
    ax.legend(fontsize=9)

    # Panel B: SD of specialty rates
    ax = axes[1]
    ax.hist(rep_spec_sd, bins=30, density=True, color="#0F6E56", alpha=0.5,
            edgecolor="white", linewidth=0.3, label="Replicated")
    ax.axvline(obs_spec_sd, color="#A32D2D", linewidth=2, linestyle="--",
               label=f"Observed = {obs_spec_sd:.4f}")
    ax.set_xlabel("SD of specialty-level readmission rates")
    ax.set_ylabel("Density")
    ax.set_title(f"(B) Specialty rate variability (p-value = {pval_sd:.3f})")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_posterior_predictive.png")
    plt.close(fig)
    print(f"\n  Saved fig5_posterior_predictive.png")

    return {
        "obs_overall": obs_overall,
        "rep_overall_mean": rep_overall.mean(),
        "pval_overall": pval_overall,
        "obs_spec_sd": obs_spec_sd,
        "rep_spec_sd_mean": rep_spec_sd.mean(),
        "pval_sd": pval_sd,
    }


# =============================================================================
# MANUSCRIPT SUMMARY
# =============================================================================

def print_manuscript_summary(table2, tau_all, ppc_results, table3,
                              u_means, u_lower, u_upper, spec_names, y):
    """Print all placeholder values for direct manuscript insertion."""
    print("\n" + "=" * 60)
    print("MANUSCRIPT PLACEHOLDER VALUES")
    print("=" * 60)

    N = len(y)
    K = len(table2) - 1  # exclude intercept row
    J = len(spec_names)

    print(f"\n  [N]         = {N}")
    print(f"  [K]         = {K}")
    print(f"  [J]         = {J}")
    print(f"  [X%]        = {y.mean()*100:.2f}%")
    print(f"  [sampler name] = Metropolis-within-Gibbs")
    print(f"  [total iterations] = 20,000")
    print(f"  [burn-in]   = 5,000")
    print(f"  [thinning interval] = 5")

    # tau
    print(f"\n  [tau posterior mean]  = {tau_all.mean():.4f}")
    print(f"  [tau 95% CrI]        = [{np.percentile(tau_all, 2.5):.4f}, "
          f"{np.percentile(tau_all, 97.5):.4f}]")
    tau_evidence = "modest" if tau_all.mean() > 0.15 else "limited"
    print(f"  [evidence / limited evidence] = {tau_evidence}")

    # Top predictors (sorted by |post_mean|)
    fx = table2.iloc[1:].copy()  # exclude intercept
    fx["abs_mean"] = fx["post_mean"].abs()
    fx = fx.sort_values("abs_mean", ascending=False)
    sig_fx = fx[(fx["ci_lower"] > 0) | (fx["ci_upper"] < 0)]
    print(f"\n  Significant predictors (95% CrI excludes 0):")
    for _, r in sig_fx.head(10).iterrows():
        direction = "increased" if r["post_mean"] > 0 else "decreased"
        print(f"    {r['parameter']:>35s}: {direction} risk, "
              f"OR={r['OR']:.3f} [{r['OR_ci_lower']:.3f}, {r['OR_ci_upper']:.3f}]")

    # Specialty extremes
    order = np.argsort(u_means)
    print(f"\n  Specialty with highest adjusted readmission:")
    j = order[-1]
    print(f"    [specialty A] = {spec_names[j]} "
          f"(u = {u_means[j]:+.4f}, 95%CI=[{u_lower[j]:+.4f}, {u_upper[j]:+.4f}])")
    j = order[-2]
    print(f"    [specialty B] = {spec_names[j]} "
          f"(u = {u_means[j]:+.4f})")
    print(f"  Specialty with lowest adjusted readmission:")
    j = order[0]
    print(f"    [specialty C] = {spec_names[j]} "
          f"(u = {u_means[j]:+.4f}, 95%CI=[{u_lower[j]:+.4f}, {u_upper[j]:+.4f}])")
    j = order[1]
    print(f"    [specialty D] = {spec_names[j]} "
          f"(u = {u_means[j]:+.4f})")

    # Model comparison
    hier_row = table3[table3["model"] == "Bayesian hierarchical"].iloc[0]
    base_row = table3[table3["model"] == "No specialty (baseline)"].iloc[0]
    perf = "improved" if hier_row["AUC"] > base_row["AUC"] else "similar"
    print(f"\n  [better / similar] = {perf}")
    print(f"  [metric] = AUC")
    print(f"  Hierarchical AUC = {hier_row['AUC']}")
    print(f"  Baseline AUC     = {base_row['AUC']}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    chains, meta = load_chains()
    X, y, group_idx = load_data()
    var_names = load_var_names()
    spec_names = load_specialty_mapping()

    # Task 11
    ess_df = run_diagnostics(chains, meta, var_names, spec_names)

    # Task 12
    table2, tau_all, u_means, u_lower, u_upper = run_inference(
        chains, meta, var_names, spec_names)

    # Task 13
    table3 = run_model_comparison(chains, X, y, group_idx)

    # Task 14
    ppc_results = run_ppc(chains, X, y, group_idx, spec_names)

    # Summary for manuscript
    print_manuscript_summary(table2, tau_all, ppc_results, table3,
                              u_means, u_lower, u_upper, spec_names, y)

    print("\n" + "=" * 60)
    print("All post-MCMC analyses complete.")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Tables saved to:  {TAB_DIR}")
    print("=" * 60)
