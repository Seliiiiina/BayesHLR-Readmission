#!/usr/bin/env python3
"""
05b_tune_proposals.py
=====================
Automated proposal SD tuning for the MwG sampler (Pipeline Task 9).

Strategy:
  1. Use a subsample (default 5000 obs) for fast iteration
  2. Start with conservative proposal SDs
  3. Run short chains (1000 iter), check acceptance rates
  4. Adjust: if AR > 0.5, increase proposal SD by 50%
             if AR < 0.2, decrease proposal SD by 50%
  5. Repeat until all blocks are in [0.20, 0.50]
  6. Save tuned proposal SDs to JSON

Usage on DCC:
  python src/05b_tune_proposals.py --data-dir data_processed

Author : [Your names]
Date   : 2026-03-15
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
compute_eta = _lp.compute_eta
log_likelihood = _lp.log_likelihood

_sampler_path = SCRIPT_DIR / "05_mwg_sampler.py"
sampler = SourceFileLoader("sampler", str(_sampler_path)).load_module()

SEED = 724
TARGET_AR_LOW = 0.20
TARGET_AR_HIGH = 0.50
MAX_ROUNDS = 15
TUNE_ITER = 1500
TUNE_BURNIN = 500
N_SUBSAMPLE = 5000


def tune(data_dir="data_processed", output_dir="output/chains"):
    """Run iterative tuning and save results."""
    X, y, group_idx, N, K, J = sampler.load_data(
        data_dir, n_subsample=N_SUBSAMPLE, seed=SEED)

    hyperparams = sampler.DEFAULT_HYPERPARAMS.copy()

    # Starting proposal SDs (conservative)
    prop = {"alpha": 0.02, "beta": 0.01, "u": 0.05, "log_tau": 0.15}

    print(f"\n{'='*60}")
    print(f"Proposal Tuning: target AR in [{TARGET_AR_LOW}, {TARGET_AR_HIGH}]")
    print(f"Subsample N={N}, K={K}, J={J}")
    print(f"{'='*60}\n")

    history = []

    for round_i in range(MAX_ROUNDS):
        print(f"--- Round {round_i+1} ---")
        print(f"  Proposals: alpha={prop['alpha']:.4f}, beta={prop['beta']:.4f}, "
              f"u={prop['u']:.4f}, log_tau={prop['log_tau']:.4f}")

        _, diag = sampler.run_mwg(
            X, y, group_idx, N, K, J,
            n_iter=TUNE_ITER, burnin=TUNE_BURNIN, thin=1,
            hyperparams=hyperparams, proposal_sd=prop,
            chain_id=0, seed=SEED + round_i, verbose=False,
        )

        ar = {
            "alpha": diag["accept_rate_alpha"],
            "beta": diag["accept_rate_beta_mean"],
            "u": diag["accept_rate_u_mean"],
            "log_tau": diag["accept_rate_log_tau"],
        }
        print(f"  AR: alpha={ar['alpha']:.3f}, beta={ar['beta']:.3f}, "
              f"u={ar['u']:.3f}, log_tau={ar['log_tau']:.3f}")

        history.append({"round": round_i + 1, "proposal_sd": prop.copy(),
                        "accept_rates": ar.copy()})

        # Check convergence
        all_ok = True
        for block in ["alpha", "beta", "u", "log_tau"]:
            if ar[block] > TARGET_AR_HIGH:
                prop[block] *= 1.5
                all_ok = False
            elif ar[block] < TARGET_AR_LOW:
                prop[block] *= 0.5
                all_ok = False

        if all_ok:
            print(f"\n  All acceptance rates in target range. Tuning complete.")
            break
        print()

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "tuned_proposal_sd": prop,
        "final_accept_rates": ar,
        "n_rounds": round_i + 1,
        "tune_iter": TUNE_ITER,
        "n_subsample": N_SUBSAMPLE,
        "history": history,
    }
    with open(out / "tuning_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {out / 'tuning_result.json'}")

    print(f"\n  Final tuned proposal SDs:")
    for k, v in prop.items():
        print(f"    {k}: {v:.6f}")
    print(f"\n  Use these in 05_mwg_sampler.py via:")
    print(f"    python 05_mwg_sampler.py --mode full \\")
    print(f"      --prop-alpha {prop['alpha']:.6f} \\")
    print(f"      --prop-beta {prop['beta']:.6f} \\")
    print(f"      --prop-u {prop['u']:.6f} \\")
    print(f"      --prop-log-tau {prop['log_tau']:.6f}")

    return prop, ar


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_processed")
    parser.add_argument("--output-dir", default="output/chains")
    args = parser.parse_args()
    tune(args.data_dir, args.output_dir)
