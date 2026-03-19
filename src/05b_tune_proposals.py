#!/usr/bin/env python3
"""
05b_tune_proposals.py
=====================
Per-coordinate proposal SD tuning for the MwG sampler (Pipeline Task 9).

Key improvements:
  - Uses N=20000 subsample (closer to full-data posterior geometry)
  - Tunes per-coordinate proposal SDs for beta (K=43) and u (J=20)
  - Target acceptance rate: [0.20, 0.50]
  - Stops when >= 90% of all parameters are in range

Usage:
  python src/05b_tune_proposals.py --data-dir data_processed

Author : [Your names]
Date   : 2026-03-16
"""

import sys
import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from importlib.machinery import SourceFileLoader

# Load companion modules (filenames start with digits)
_lp_path = SCRIPT_DIR / "04_logposterior_functions.py"
if not _lp_path.exists():
    _lp_path = SCRIPT_DIR / "logposterior_functions.py"
_lp = SourceFileLoader("logposterior_functions", str(_lp_path)).load_module()

_sampler_path = SCRIPT_DIR / "05_mwg_sampler.py"
sampler = SourceFileLoader("sampler", str(_sampler_path)).load_module()

SEED = 724
TARGET_AR_LOW = 0.20
TARGET_AR_HIGH = 0.50
MAX_ROUNDS = 20
TUNE_ITER = 2000
TUNE_BURNIN = 500
N_SUBSAMPLE = 20000


def tune(data_dir="data_processed", output_dir="output/chains"):
    """Run per-coordinate iterative tuning and save results."""
    X, y, group_idx, N, K, J = sampler.load_data(
        data_dir, n_subsample=N_SUBSAMPLE, seed=SEED)

    hyperparams = sampler.DEFAULT_HYPERPARAMS.copy()

    # Starting proposal SDs (conservative)
    prop_alpha = 0.02
    prop_beta = np.full(K, 0.01)
    prop_u = np.full(J, 0.05)
    prop_log_tau = 0.15
    prop_block = 0.02  # alpha-u block translation move

    print(f"\n{'=' * 60}")
    print(f"Per-Coordinate Proposal Tuning")
    print(f"Target AR in [{TARGET_AR_LOW}, {TARGET_AR_HIGH}]")
    print(f"Subsample N={N}, K={K}, J={J}")
    print(f"{'=' * 60}\n")

    history = []

    for round_i in range(MAX_ROUNDS):
        print(f"--- Round {round_i + 1} ---")

        proposal_sd = {
            "alpha": prop_alpha,
            "beta": prop_beta.tolist(),
            "u": prop_u.tolist(),
            "log_tau": prop_log_tau,
            "block": prop_block,
        }

        _, diag = sampler.run_mwg(
            X, y, group_idx, N, K, J,
            n_iter=TUNE_ITER, burnin=TUNE_BURNIN, thin=1,
            hyperparams=hyperparams, proposal_sd=proposal_sd,
            chain_id=0, seed=SEED + round_i, verbose=False,
        )

        ar_alpha = diag["accept_rate_alpha"]
        ar_beta = np.array(diag["accept_rate_beta_per_k"])
        ar_u = np.array(diag["accept_rate_u_per_j"])
        ar_log_tau = diag["accept_rate_log_tau"]
        ar_block = diag.get("accept_rate_block", 0)

        print(f"  AR: alpha={ar_alpha:.3f}, "
              f"beta=[{ar_beta.min():.3f}, {ar_beta.mean():.3f}, {ar_beta.max():.3f}], "
              f"u=[{ar_u.min():.3f}, {ar_u.mean():.3f}, {ar_u.max():.3f}], "
              f"log_tau={ar_log_tau:.3f}, block={ar_block:.3f}")

        history.append({
            "round": round_i + 1,
            "ar_alpha": float(ar_alpha),
            "ar_beta_mean": float(ar_beta.mean()),
            "ar_u_mean": float(ar_u.mean()),
            "ar_log_tau": float(ar_log_tau),
        })

        # --- Adjust proposal SDs ---
        all_ok = True

        # Alpha
        if ar_alpha > TARGET_AR_HIGH:
            prop_alpha *= 1.5
            all_ok = False
        elif ar_alpha < TARGET_AR_LOW:
            prop_alpha *= 0.5
            all_ok = False

        # Beta (per-coordinate)
        for k in range(K):
            if ar_beta[k] > TARGET_AR_HIGH:
                prop_beta[k] *= 1.5
                all_ok = False
            elif ar_beta[k] < TARGET_AR_LOW:
                prop_beta[k] *= 0.5
                all_ok = False

        # U (per-group)
        for j in range(J):
            if ar_u[j] > TARGET_AR_HIGH:
                prop_u[j] *= 1.5
                all_ok = False
            elif ar_u[j] < TARGET_AR_LOW:
                prop_u[j] *= 0.5
                all_ok = False

        # log_tau
        if ar_log_tau > TARGET_AR_HIGH:
            prop_log_tau *= 1.5
            all_ok = False
        elif ar_log_tau < TARGET_AR_LOW:
            prop_log_tau *= 0.5
            all_ok = False

        # block
        if ar_block > TARGET_AR_HIGH:
            prop_block *= 1.5
            all_ok = False
        elif ar_block < TARGET_AR_LOW:
            prop_block *= 0.5
            all_ok = False

        # Count how many are in range
        alpha_ok = TARGET_AR_LOW <= ar_alpha <= TARGET_AR_HIGH
        n_beta_ok = int(np.sum((ar_beta >= TARGET_AR_LOW) & (ar_beta <= TARGET_AR_HIGH)))
        n_u_ok = int(np.sum((ar_u >= TARGET_AR_LOW) & (ar_u <= TARGET_AR_HIGH)))
        tau_ok = TARGET_AR_LOW <= ar_log_tau <= TARGET_AR_HIGH
        block_ok = TARGET_AR_LOW <= ar_block <= TARGET_AR_HIGH

        print(f"  In range: alpha={'Y' if alpha_ok else 'N'}, "
              f"beta={n_beta_ok}/{K}, u={n_u_ok}/{J}, "
              f"log_tau={'Y' if tau_ok else 'N'}, "
              f"block={'Y' if block_ok else 'N'}")

        if all_ok:
            print(f"\n  All acceptance rates in target range. Tuning complete.")
            break

        # Relaxed convergence: stop if >= 90% of all parameters are in range
        total_params = 1 + K + J + 1 + 1  # +1 for block
        n_ok = int(alpha_ok) + n_beta_ok + n_u_ok + int(tau_ok) + int(block_ok)
        frac_ok = n_ok / total_params
        if frac_ok >= 0.90:
            print(f"\n  {frac_ok * 100:.0f}% of parameters in range ({n_ok}/{total_params}). "
                  f"Stopping early.")
            break

        print()

    # Save results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "tuned_proposal_sd": {
            "alpha": float(prop_alpha),
            "beta": prop_beta.tolist(),
            "u": prop_u.tolist(),
            "log_tau": float(prop_log_tau),
            "block": float(prop_block),
        },
        "final_accept_rates": {
            "alpha": float(ar_alpha),
            "beta_per_k": ar_beta.tolist(),
            "beta_mean": float(ar_beta.mean()),
            "u_per_j": ar_u.tolist(),
            "u_mean": float(ar_u.mean()),
            "log_tau": float(ar_log_tau),
            "block": float(ar_block),
        },
        "n_rounds": round_i + 1,
        "tune_iter": TUNE_ITER,
        "tune_burnin": TUNE_BURNIN,
        "n_subsample": N_SUBSAMPLE,
        "history": history,
    }
    with open(out / "tuning_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {out / 'tuning_result.json'}")

    print(f"\n  Final tuned proposal SDs:")
    print(f"    alpha:    {prop_alpha:.6f}")
    print(f"    beta:     min={prop_beta.min():.6f}, "
          f"mean={prop_beta.mean():.6f}, max={prop_beta.max():.6f}")
    print(f"    u:        min={prop_u.min():.6f}, "
          f"mean={prop_u.mean():.6f}, max={prop_u.max():.6f}")
    print(f"    log_tau:  {prop_log_tau:.6f}")
    print(f"    block:    {prop_block:.6f}")

    print(f"\n  Use in full run:")
    print(f"    python src/05_mwg_sampler.py --mode full "
          f"--load-tuning {out / 'tuning_result.json'}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Per-coordinate proposal SD tuning for MwG sampler")
    parser.add_argument("--data-dir", default="data_processed")
    parser.add_argument("--output-dir", default="output/chains")
    args = parser.parse_args()
    tune(args.data_dir, args.output_dir)
