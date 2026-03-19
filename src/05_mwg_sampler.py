#!/usr/bin/env python3
"""
05_mwg_sampler.py
=================
Metropolis-within-Gibbs sampler for the Bayesian Hierarchical Logistic
Regression model of 30-day hospital readmission.

Implementation follows the Posterior Derivation Notes Section 4 exactly:
  Block 1: Update alpha          (Section 4.2)
  Block 2: Update beta, comp-wise (Section 4.3)
  Block 3: Update u, comp-wise    (Section 4.4)
  Block 4: Update log(tau)        (Section 4.5)

Usage:
  # Demo run (subsample, quick check)
  python 05_mwg_sampler.py --mode demo

  # Full run (all data, 4 chains)
  python 05_mwg_sampler.py --mode full

  # Custom settings
  python 05_mwg_sampler.py --mode full --n-iter 30000 --burnin 10000

Author : [Your names]
Date   : 2026-03-15
Seed   : 724
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

# Import log-posterior functions from companion module.
# The filename starts with a digit, so we use importlib.
SCRIPT_DIR = Path(__file__).resolve().parent
from importlib.machinery import SourceFileLoader as _SFL

# Try multiple possible filenames in order of preference
_CANDIDATES = [
    SCRIPT_DIR / "04_logposterior_functions.py",
    SCRIPT_DIR / "logposterior_functions.py",
]
_lp_mod = None
for _cand in _CANDIDATES:
    if _cand.exists():
        _lp_mod = _SFL("logposterior_functions", str(_cand)).load_module()
        break
if _lp_mod is None:
    sys.exit("ERROR: Cannot find 04_logposterior_functions.py in " + str(SCRIPT_DIR))

compute_eta = _lp_mod.compute_eta
log_likelihood = _lp_mod.log_likelihood
log_prior_alpha = _lp_mod.log_prior_alpha
log_prior_beta = _lp_mod.log_prior_beta
log_prior_u_given_tau = _lp_mod.log_prior_u_given_tau
log_prior_tau = _lp_mod.log_prior_tau
log_posterior = _lp_mod.log_posterior
log_target_log_tau = _lp_mod.log_target_log_tau

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

RANDOM_SEED = 724

# Default hyperparameters (Derivation Notes Section 5)
DEFAULT_HYPERPARAMS = {
    "sigma_alpha": 5.0,       # Prior SD for intercept
    "s_beta": 2.5,            # Prior SD for fixed effects
    "s_tau": 1.0,             # Half-Normal scale for tau
    "prior_family": "half_normal",  # "half_normal" or "half_cauchy"
}

# Default proposal SDs (to be tuned in Task 9)
DEFAULT_PROPOSAL_SD = {
    "alpha": 0.02,
    "beta": 0.01,
    "u": 0.05,
    "log_tau": 0.15,
}

# MCMC settings (Derivation Notes Section 6)
MCMC_SETTINGS = {
    "demo": {
        "n_iter": 2000,
        "burnin": 500,
        "thin": 1,
        "n_chains": 1,
        "n_subsample": 2000,
    },
    "full": {
        "n_iter": 30000,
        "burnin": 8000,
        "thin": 5,
        "n_chains": 4,
        "n_subsample": None,  # Use all data
    },
}


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data(data_dir, n_subsample=None, seed=724):
    """Load X, y, group_idx from .npy files produced by 01_clean_data.py."""
    data_dir = Path(data_dir)

    X = np.load(data_dir / "X_matrix.npy")
    y = np.load(data_dir / "y_vector.npy")
    group_idx = np.load(data_dir / "group_index.npy")

    N, K = X.shape
    J = int(group_idx.max()) + 1

    print(f"Data loaded: N={N}, K={K}, J={J}")
    print(f"  Readmission rate: {y.mean():.4f}")

    # Subsample for demo / tuning
    if n_subsample is not None and n_subsample < N:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, size=n_subsample, replace=False)
        X = X[idx]
        y = y[idx]
        group_idx = group_idx[idx]
        N = n_subsample
        # Recompute J from subsampled data
        J = int(group_idx.max()) + 1
        print(f"  Subsampled to N={N}, J={J}")

    return X, y, group_idx, N, K, J


def precompute_group_obs(group_idx, J):
    """
    Build a dict mapping group index j -> array of observation indices.

    This is the key optimisation for Block 3 (u_j update):
    the likelihood sum for u_j only involves n_j observations,
    not all N.  See Derivation Notes Section 4.4.
    """
    group_obs = {}
    for j in range(J):
        group_obs[j] = np.where(group_idx == j)[0]
    return group_obs


# =============================================================================
# 2. METROPOLIS-HASTINGS UPDATE FUNCTIONS
# =============================================================================
# Each function follows the pattern:
#   1. Propose from symmetric normal random walk
#   2. Compute log-acceptance ratio = h(proposed) - h(current)
#   3. Accept/reject
#
# All proposals are symmetric, so the Hastings ratio = 1 and the
# acceptance probability reduces to the Metropolis ratio.
# (Derivation Notes Section 4.1)

def update_alpha(alpha, eta, y, X, group_idx, sigma_alpha, prop_sd, rng):
    """
    Block 1: Update alpha.  Derivation Notes Section 4.2.

    Proposal:  alpha* ~ N(alpha^(t), prop_sd^2)
    Log-target: h(alpha) = LL(eta) - alpha^2 / (2*sigma_alpha^2)
    """
    # Propose
    alpha_star = alpha + rng.randn() * prop_sd

    # Compute log-target at current and proposed
    h_current = log_likelihood(y, eta) + log_prior_alpha(alpha, sigma_alpha)

    # Incremental eta update: eta changes by (alpha_star - alpha) everywhere
    delta = alpha_star - alpha
    eta_star = eta + delta
    h_proposed = log_likelihood(y, eta_star) + log_prior_alpha(alpha_star, sigma_alpha)

    # Accept/reject (Eq. 21)
    log_ratio = h_proposed - h_current
    if np.log(rng.rand()) < log_ratio:
        return alpha_star, eta_star, True
    else:
        return alpha, eta, False


def update_beta_k(k, beta, eta, y, X, s_beta, prop_sd, rng):
    """
    Block 2: Update beta_k (single coordinate).  Derivation Notes Section 4.3.

    Proposal:  beta_k* ~ N(beta_k^(t), prop_sd^2)
    Log-target: h(beta_k) = LL(eta) - beta_k^2 / (2*s_beta^2)

    Computational note (from Derivation Notes):
      When updating beta_k, eta_i changes by (beta_k* - beta_k) * X_{ik}.
      This incremental update avoids recomputing the full linear predictor,
      reducing cost from O(NK) to O(N) per coordinate.
    """
    beta_k_current = beta[k]

    # Propose
    beta_k_star = beta_k_current + rng.randn() * prop_sd

    # Current log-target
    h_current = log_likelihood(y, eta) + (-0.5 * beta_k_current**2 / s_beta**2)

    # Incremental eta update: only the k-th covariate column matters
    delta = beta_k_star - beta_k_current
    eta_star = eta + delta * X[:, k]
    h_proposed = log_likelihood(y, eta_star) + (-0.5 * beta_k_star**2 / s_beta**2)

    # Accept/reject (Eq. 24)
    log_ratio = h_proposed - h_current
    if np.log(rng.rand()) < log_ratio:
        beta[k] = beta_k_star
        return eta_star, True
    else:
        return eta, False


def update_u_j(j, u, eta, y, group_obs_j, tau, prop_sd, rng):
    """
    Block 3: Update u_j (single group).  Derivation Notes Section 4.4.

    Proposal:  u_j* ~ N(u_j^(t), prop_sd^2)
    Log-target: h(u_j) = sum_{i: j[i]=j} [y_i*eta_i - log(1+exp(eta_i))]
                          - u_j^2 / (2*tau^2)

    Key computational advantage: the likelihood sum only involves n_j
    observations (group_obs_j), not all N.
    """
    u_j_current = u[j]
    idx = group_obs_j  # Indices of observations in group j

    if len(idx) == 0:
        # No observations in this group; update from prior only
        u_j_star = rng.randn() * tau
        u[j] = u_j_star
        return eta, True

    # Propose
    u_j_star = u_j_current + rng.randn() * prop_sd

    # Extract group-specific y and eta
    y_j = y[idx]
    eta_j = eta[idx]

    # Current log-target (Eq. 26)
    h_current = (np.sum(y_j * eta_j - np.logaddexp(0.0, eta_j))
                 - 0.5 * u_j_current**2 / tau**2)

    # Proposed: eta changes by (u_j* - u_j) only for group j observations
    delta = u_j_star - u_j_current
    eta_j_star = eta_j + delta
    h_proposed = (np.sum(y_j * eta_j_star - np.logaddexp(0.0, eta_j_star))
                  - 0.5 * u_j_star**2 / tau**2)

    # Accept/reject (Eq. 27)
    log_ratio = h_proposed - h_current
    if np.log(rng.rand()) < log_ratio:
        u[j] = u_j_star
        eta[idx] = eta_j_star
        return eta, True
    else:
        return eta, False


def update_alpha_u_block(alpha, u, sigma_alpha, tau, prop_sd, J, rng):
    """
    Block 5: Joint translation move for alpha and u.

    Propose delta ~ N(0, prop_sd^2), then:
      alpha* = alpha + delta
      u_j*   = u_j - delta   for all j

    Since eta_i = alpha + x_i'beta + u_{j[i]}, the +delta and -delta
    cancel exactly, leaving eta unchanged.  Therefore the likelihood
    cancels in the acceptance ratio and only priors matter:

      log R = log p(alpha*) + log p(u*|tau) - log p(alpha) - log p(u|tau)

    This move is O(J) — no likelihood evaluation — so we can run it
    multiple times per iteration to dramatically improve alpha-u mixing.
    """
    delta = rng.randn() * prop_sd

    alpha_star = alpha + delta
    u_star = u - delta   # broadcast: scalar subtracted from length-J array

    # Log-prior ratio (likelihood cancels)
    lp_current = (-0.5 * alpha**2 / sigma_alpha**2
                  - 0.5 * np.sum(u**2) / tau**2)
    lp_proposed = (-0.5 * alpha_star**2 / sigma_alpha**2
                   - 0.5 * np.sum(u_star**2) / tau**2)

    log_ratio = lp_proposed - lp_current
    if np.log(rng.rand()) < log_ratio:
        return alpha_star, u_star, True
    else:
        return alpha, u, False


# Number of block translation moves per iteration (cheap, so repeat)
N_BLOCK_REPEATS = 10


def update_log_tau(u, log_tau, s_tau, prior_family, prop_sd, rng):
    """
    Block 4: Update log(tau).  Derivation Notes Section 4.5.

    We reparametrise phi = log(tau) to work on the unconstrained space.

    Proposal:  phi* ~ N(phi^(t), prop_sd^2)
    Log-target (Eq. 29):
        h(phi) = -J*phi - (1/(2*exp(2*phi))) * sum(u_j^2)
                 - exp(2*phi)/(2*s_tau^2) + phi       [Half-Normal]

    The +phi term is the log-Jacobian |d(tau)/d(phi)| = exp(phi).

    NOTE: This does NOT involve the likelihood.  Data inform tau
    only indirectly through the posterior draws of u.
    """
    phi_current = log_tau

    # Propose
    phi_star = phi_current + rng.randn() * prop_sd

    # Log-targets (Eq. 29 or 31)
    h_current = log_target_log_tau(phi_current, u, s_tau, prior_family)
    h_proposed = log_target_log_tau(phi_star, u, s_tau, prior_family)

    # Accept/reject (Eq. 30)
    log_ratio = h_proposed - h_current
    if np.log(rng.rand()) < log_ratio:
        return phi_star, np.exp(phi_star), True
    else:
        return phi_current, np.exp(phi_current), False


# =============================================================================
# 3. MAIN SAMPLER LOOP
# =============================================================================
# One full iteration of MwG (Derivation Notes Section 4.6):
#   1. Update alpha
#   2. For k = 1, ..., K: update beta_k
#   3. For j = 1, ..., J: update u_j
#   4. Update log(tau)
# Each sub-step uses the most recently updated values of all other
# parameters (defining characteristic of Metropolis-within-Gibbs).

def run_mwg(X, y, group_idx, N, K, J,
            n_iter, burnin, thin,
            hyperparams, proposal_sd,
            init=None, chain_id=0, seed=724, verbose=True):
    """
    Run a single chain of the MwG sampler.

    Parameters
    ----------
    X         : ndarray (N, K)
    y         : ndarray (N,)
    group_idx : ndarray (N,), int
    N, K, J   : int
    n_iter    : int, total iterations (including burn-in)
    burnin    : int, iterations to discard
    thin      : int, thinning interval
    hyperparams : dict with sigma_alpha, s_beta, s_tau, prior_family
    proposal_sd : dict with alpha, beta, u, log_tau
    init      : dict or None (initial values)
    chain_id  : int, for logging
    seed      : int
    verbose   : bool

    Returns
    -------
    samples : dict of ndarrays (post-burn-in, thinned)
    diagnostics : dict (acceptance rates, runtime)
    """
    rng = np.random.RandomState(seed + chain_id * 1000)

    # --- Unpack hyperparameters ---
    sigma_alpha = hyperparams["sigma_alpha"]
    s_beta = hyperparams["s_beta"]
    s_tau = hyperparams["s_tau"]
    prior_family = hyperparams["prior_family"]

    # --- Unpack proposal SDs (support per-coordinate arrays) ---
    prop_alpha = float(proposal_sd["alpha"])

    _pb = proposal_sd["beta"]
    if isinstance(_pb, (int, float)):
        prop_beta = np.full(K, float(_pb))
    else:
        prop_beta = np.asarray(_pb, dtype=np.float64)

    _pu = proposal_sd["u"]
    if isinstance(_pu, (int, float)):
        prop_u = np.full(J, float(_pu))
    else:
        prop_u = np.asarray(_pu, dtype=np.float64)

    prop_log_tau = float(proposal_sd["log_tau"])

    # Block translation proposal SD (alpha-u joint move)
    prop_block = float(proposal_sd.get("block", prop_alpha))

    # --- Precompute group membership ---
    group_obs = precompute_group_obs(group_idx, J)

    # --- Initialisation ---
    if init is not None:
        alpha = float(init.get("alpha", 0.0))
        beta = np.array(init.get("beta", np.zeros(K)), dtype=np.float64)
        u = np.array(init.get("u", np.zeros(J)), dtype=np.float64)
        tau = float(init.get("tau", 0.5))
    else:
        # Dispersed initialisation: small random perturbations
        alpha = rng.randn() * 0.1
        beta = rng.randn(K) * 0.05
        u = rng.randn(J) * 0.1
        tau = np.exp(rng.randn() * 0.3)  # log-normal around 1
    log_tau = np.log(tau)

    # Initial eta
    eta = compute_eta(alpha, beta, u, X, group_idx)

    # --- Storage ---
    n_saved = (n_iter - burnin) // thin
    samples = {
        "alpha": np.zeros(n_saved),
        "beta": np.zeros((n_saved, K)),
        "u": np.zeros((n_saved, J)),
        "tau": np.zeros(n_saved),
        "log_posterior": np.zeros(n_saved),
    }

    # Acceptance counters
    accept_alpha = 0
    accept_beta = np.zeros(K, dtype=int)
    accept_u = np.zeros(J, dtype=int)
    accept_log_tau = 0
    accept_block = 0

    total_alpha = 0
    total_beta = np.zeros(K, dtype=int)
    total_u = np.zeros(J, dtype=int)
    total_log_tau = 0
    total_block = 0

    save_idx = 0
    t_start = time.time()

    # --- Main loop ---
    for t in range(n_iter):

        # --- Block 1: Update alpha (Section 4.2) ---
        alpha, eta, accepted = update_alpha(
            alpha, eta, y, X, group_idx, sigma_alpha, prop_alpha, rng)
        total_alpha += 1
        accept_alpha += int(accepted)

        # --- Block 2: Update beta, component-wise (Section 4.3) ---
        for k in range(K):
            eta, accepted = update_beta_k(
                k, beta, eta, y, X, s_beta, prop_beta[k], rng)
            total_beta[k] += 1
            accept_beta[k] += int(accepted)

        # --- Block 3: Update u, component-wise (Section 4.4) ---
        for j in range(J):
            eta, accepted = update_u_j(
                j, u, eta, y, group_obs[j], tau, prop_u[j], rng)
            total_u[j] += 1
            accept_u[j] += int(accepted)

        # --- Block 4: Update log(tau) (Section 4.5) ---
        log_tau, tau, accepted = update_log_tau(
            u, log_tau, s_tau, prior_family, prop_log_tau, rng)
        total_log_tau += 1
        accept_log_tau += int(accepted)

        # --- Block 5: Alpha-u translation move (repeated) ---
        for _ in range(N_BLOCK_REPEATS):
            alpha_new, u_new, accepted = update_alpha_u_block(
                alpha, u, sigma_alpha, tau, prop_block, J, rng)
            total_block += 1
            accept_block += int(accepted)
            if accepted:
                alpha = alpha_new
                u = u_new
                # Recompute eta with new alpha and u
                eta = compute_eta(alpha, beta, u, X, group_idx)

        # --- Save sample (post-burn-in, thinned) ---
        if t >= burnin and (t - burnin) % thin == 0 and save_idx < n_saved:
            samples["alpha"][save_idx] = alpha
            samples["beta"][save_idx] = beta.copy()
            samples["u"][save_idx] = u.copy()
            samples["tau"][save_idx] = tau
            samples["log_posterior"][save_idx] = log_posterior(
                alpha, beta, u, tau, y, eta,
                sigma_alpha, s_beta, s_tau, prior_family)
            save_idx += 1

        # --- Progress reporting ---
        if verbose and (t + 1) % max(1, n_iter // 10) == 0:
            elapsed = time.time() - t_start
            rate = (t + 1) / elapsed
            eta_sec = (n_iter - t - 1) / rate
            ar_alpha = accept_alpha / total_alpha if total_alpha else 0
            ar_tau = accept_log_tau / total_log_tau if total_log_tau else 0
            ar_beta_mean = (accept_beta.sum() / total_beta.sum()
                           if total_beta.sum() else 0)
            ar_u_mean = (accept_u.sum() / total_u.sum()
                         if total_u.sum() else 0)
            ar_block = accept_block / total_block if total_block else 0
            print(f"  Chain {chain_id} | iter {t+1:>6d}/{n_iter} | "
                  f"AR: alpha={ar_alpha:.3f} beta={ar_beta_mean:.3f} "
                  f"u={ar_u_mean:.3f} tau={ar_tau:.3f} blk={ar_block:.3f} | "
                  f"tau={tau:.4f} | "
                  f"{rate:.1f} it/s | ETA {eta_sec:.0f}s")

    elapsed_total = time.time() - t_start

    # --- Diagnostics ---
    diagnostics = {
        "chain_id": chain_id,
        "n_iter": n_iter,
        "burnin": burnin,
        "thin": thin,
        "n_saved": save_idx,
        "runtime_sec": elapsed_total,
        "accept_rate_alpha": accept_alpha / total_alpha,
        "accept_rate_beta_mean": float(accept_beta.sum() / total_beta.sum()),
        "accept_rate_beta_per_k": (accept_beta / total_beta).tolist(),
        "accept_rate_u_mean": float(accept_u.sum() / total_u.sum()),
        "accept_rate_u_per_j": (accept_u / total_u).tolist(),
        "accept_rate_log_tau": accept_log_tau / total_log_tau,
        "accept_rate_block": accept_block / total_block if total_block else 0,
        "hyperparams": hyperparams,
        "proposal_sd": {
            "alpha": prop_alpha,
            "beta": prop_beta.tolist(),
            "u": prop_u.tolist(),
            "log_tau": prop_log_tau,
            "block": prop_block,
        },
        "seed": seed + chain_id * 1000,
    }

    if verbose:
        print(f"\n  Chain {chain_id} completed in {elapsed_total:.1f}s "
              f"({save_idx} saved samples)")
        print(f"  Acceptance rates:")
        print(f"    alpha:   {diagnostics['accept_rate_alpha']:.4f}")
        print(f"    beta:    {diagnostics['accept_rate_beta_mean']:.4f} (mean)")
        print(f"    u:       {diagnostics['accept_rate_u_mean']:.4f} (mean)")
        print(f"    log_tau: {diagnostics['accept_rate_log_tau']:.4f}")
        print(f"    block:   {diagnostics['accept_rate_block']:.4f} "
              f"({N_BLOCK_REPEATS} repeats/iter)")

    return samples, diagnostics


# =============================================================================
# 4. MULTI-CHAIN RUNNER
# =============================================================================

def run_multi_chain(X, y, group_idx, N, K, J,
                    n_iter, burnin, thin, n_chains,
                    hyperparams, proposal_sd, seed=724,
                    output_dir=None):
    """Run multiple chains with dispersed initialisations and save results."""
    all_samples = []
    all_diagnostics = []

    for c in range(n_chains):
        print(f"\n{'='*60}")
        print(f"Starting chain {c+1}/{n_chains}")
        print(f"{'='*60}")

        samples, diag = run_mwg(
            X, y, group_idx, N, K, J,
            n_iter=n_iter, burnin=burnin, thin=thin,
            hyperparams=hyperparams, proposal_sd=proposal_sd,
            init=None,  # Dispersed initialisation
            chain_id=c, seed=seed, verbose=True,
        )
        all_samples.append(samples)
        all_diagnostics.append(diag)

        # Save chain immediately (in case of crash)
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out / f"chain_{c}.npz",
                alpha=samples["alpha"],
                beta=samples["beta"],
                u=samples["u"],
                tau=samples["tau"],
                log_posterior=samples["log_posterior"],
            )
            with open(out / f"chain_{c}_diagnostics.json", "w") as f:
                json.dump(diag, f, indent=2)
            print(f"  Saved to {out / f'chain_{c}.npz'}")

    # Save combined metadata
    if output_dir:
        meta = {
            "n_chains": n_chains,
            "n_iter": n_iter,
            "burnin": burnin,
            "thin": thin,
            "N": N, "K": K, "J": J,
            "hyperparams": hyperparams,
            "proposal_sd": {
                k: (v.tolist() if isinstance(v, np.ndarray) else
                    float(v) if isinstance(v, (int, float)) else v)
                for k, v in proposal_sd.items()
            },
            "seed": seed,
        }
        with open(Path(output_dir) / "mcmc_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return all_samples, all_diagnostics


# =============================================================================
# 5. COMMAND-LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MwG sampler for Bayesian Hierarchical Logistic Regression")
    parser.add_argument("--mode", choices=["demo", "full", "tune"],
                        default="demo",
                        help="Run mode: demo (subsample), full (all data)")
    parser.add_argument("--data-dir", type=str, default="data_processed",
                        help="Directory with X_matrix.npy etc.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for chains (default: output/chains/)")
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--burnin", type=int, default=None)
    parser.add_argument("--thin", type=int, default=None)
    parser.add_argument("--n-chains", type=int, default=None)
    parser.add_argument("--n-subsample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    # Hyperparameters (for sensitivity analysis)
    parser.add_argument("--s-beta", type=float, default=None)
    parser.add_argument("--s-tau", type=float, default=None)
    parser.add_argument("--prior-family", type=str, default=None,
                        choices=["half_normal", "half_cauchy"])
    # Proposal SDs
    parser.add_argument("--prop-alpha", type=float, default=None)
    parser.add_argument("--prop-beta", type=float, default=None)
    parser.add_argument("--prop-u", type=float, default=None)
    parser.add_argument("--prop-log-tau", type=float, default=None)
    parser.add_argument("--load-tuning", type=str, default=None,
                        help="Load proposal SDs from tuning_result.json")

    args = parser.parse_args()

    # Resolve settings from mode defaults + CLI overrides
    mode_cfg = MCMC_SETTINGS[args.mode if args.mode != "tune" else "demo"]
    n_iter = args.n_iter or mode_cfg["n_iter"]
    burnin = args.burnin or mode_cfg["burnin"]
    thin = args.thin or mode_cfg["thin"]
    n_chains = args.n_chains or mode_cfg["n_chains"]
    n_subsample = args.n_subsample or mode_cfg["n_subsample"]

    hyperparams = DEFAULT_HYPERPARAMS.copy()
    if args.s_beta is not None:
        hyperparams["s_beta"] = args.s_beta
    if args.s_tau is not None:
        hyperparams["s_tau"] = args.s_tau
    if args.prior_family is not None:
        hyperparams["prior_family"] = args.prior_family

    if args.load_tuning:
        with open(args.load_tuning) as f:
            proposal_sd = json.load(f)["tuned_proposal_sd"]
        print(f"  Loaded tuned proposal SDs from {args.load_tuning}")
    else:
        proposal_sd = DEFAULT_PROPOSAL_SD.copy()
    if args.prop_alpha is not None:
        proposal_sd["alpha"] = args.prop_alpha
    if args.prop_beta is not None:
        proposal_sd["beta"] = args.prop_beta
    if args.prop_u is not None:
        proposal_sd["u"] = args.prop_u
    if args.prop_log_tau is not None:
        proposal_sd["log_tau"] = args.prop_log_tau

    output_dir = args.output_dir or "output/chains"

    # Print configuration
    print("=" * 60)
    print("MwG Sampler Configuration")
    print("=" * 60)
    print(f"  Mode:       {args.mode}")
    print(f"  Iterations: {n_iter} (burn-in: {burnin}, thin: {thin})")
    print(f"  Chains:     {n_chains}")
    print(f"  Subsample:  {n_subsample or 'full data'}")
    print(f"  Hyperparams: {hyperparams}")
    print(f"  Proposal SD: {proposal_sd}")
    print(f"  Output:     {output_dir}")
    print(f"  Seed:       {args.seed}")
    print()

    # Load data
    X, y, group_idx, N, K, J = load_data(
        args.data_dir, n_subsample=n_subsample, seed=args.seed)

    # Run
    all_samples, all_diagnostics = run_multi_chain(
        X, y, group_idx, N, K, J,
        n_iter=n_iter, burnin=burnin, thin=thin, n_chains=n_chains,
        hyperparams=hyperparams, proposal_sd=proposal_sd,
        seed=args.seed, output_dir=output_dir,
    )

    # Summary
    print("\n" + "=" * 60)
    print("MCMC Summary")
    print("=" * 60)
    for diag in all_diagnostics:
        c = diag["chain_id"]
        print(f"  Chain {c}: runtime={diag['runtime_sec']:.1f}s, "
              f"AR(alpha)={diag['accept_rate_alpha']:.3f}, "
              f"AR(beta)={diag['accept_rate_beta_mean']:.3f}, "
              f"AR(u)={diag['accept_rate_u_mean']:.3f}, "
              f"AR(tau)={diag['accept_rate_log_tau']:.3f}, "
              f"AR(blk)={diag['accept_rate_block']:.3f}")

    # Quick posterior summary from last chain
    s = all_samples[-1]
    print(f"\n  Posterior summary (chain {len(all_samples)-1}):")
    print(f"    alpha: mean={s['alpha'].mean():.4f}, sd={s['alpha'].std():.4f}")
    print(f"    tau:   mean={s['tau'].mean():.4f}, sd={s['tau'].std():.4f}")
    print(f"    beta[0:5] means: {s['beta'][:, :5].mean(axis=0).round(4)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
