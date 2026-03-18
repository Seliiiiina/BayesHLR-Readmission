#!/usr/bin/env python3
"""
04_logposterior_functions.py
============================
Log-posterior and component density functions for the Bayesian
Hierarchical Logistic Regression model.

All formulas correspond exactly to the Posterior Derivation Notes:
  - Section 2  (Model Specification)
  - Section 3  (Joint Distribution & Log-Posterior)
  - Section 4  (MwG Target Densities)

Author : [Your names]
Date   : 2026-03-15
"""

import numpy as np

# =============================================================================
# 1. LINEAR PREDICTOR
# =============================================================================

def compute_eta(alpha, beta, u, X, group_idx):
    """
    Compute the linear predictor vector.

    eta_i = alpha + x_i^T beta + u_{j[i]}

    Parameters
    ----------
    alpha : float
    beta  : ndarray, shape (K,)
    u     : ndarray, shape (J,)
    X     : ndarray, shape (N, K)
    group_idx : ndarray, shape (N,), int

    Returns
    -------
    eta : ndarray, shape (N,)

    Corresponds to Derivation Notes Eq. (2).
    """
    return alpha + X @ beta + u[group_idx]


# =============================================================================
# 2. LOG-LIKELIHOOD  (Derivation Notes Eq. 16, term I / Eq. 17)
# =============================================================================

def log_likelihood(y, eta):
    """
    Bernoulli log-likelihood using the numerically stable identity:

        log p(y | eta) = sum_i [ y_i * eta_i - log(1 + exp(eta_i)) ]

    We use np.logaddexp(0, eta) = log(1 + exp(eta)) for stability.
    """
    return np.sum(y * eta - np.logaddexp(0.0, eta))


# =============================================================================
# 3. LOG-PRIORS  (Derivation Notes Eq. 8-12, Eq. 16 terms II-V)
# =============================================================================

def log_prior_alpha(alpha, sigma_alpha=5.0):
    """
    alpha ~ N(0, sigma_alpha^2)

    Log-prior (up to constant): -alpha^2 / (2 * sigma_alpha^2)
    Derivation Notes Eq. (8), Eq. (16) term (II).
    """
    return -0.5 * alpha**2 / sigma_alpha**2


def log_prior_beta(beta, s_beta=2.5):
    """
    beta_k ~ N(0, s_beta^2), independently for k = 1, ..., K

    Log-prior (up to constant): -sum(beta_k^2) / (2 * s_beta^2)
    Derivation Notes Eq. (9), Eq. (16) term (III).
    """
    return -0.5 * np.sum(beta**2) / s_beta**2


def log_prior_u_given_tau(u, tau):
    """
    u_j | tau ~ N(0, tau^2), independently for j = 1, ..., J

    Log-prior (up to constant): -J * log(tau) - sum(u_j^2) / (2 * tau^2)
    Derivation Notes Eq. (7), Eq. (16) term (IV).
    """
    J = len(u)
    return -J * np.log(tau) - 0.5 * np.sum(u**2) / tau**2


def log_prior_tau(tau, s_tau=1.0, prior_family="half_normal"):
    """
    tau ~ Half-Normal(0, s_tau)  or  Half-Cauchy(0, s_tau)

    Derivation Notes Eq. (10-11), Eq. (16) term (V).
    For Half-Cauchy: Eq. (31).
    """
    if tau <= 0:
        return -np.inf
    if prior_family == "half_normal":
        return -0.5 * tau**2 / s_tau**2
    elif prior_family == "half_cauchy":
        return -np.log(1.0 + tau**2 / s_tau**2)
    else:
        raise ValueError(f"Unknown prior family: {prior_family}")


# =============================================================================
# 4. FULL LOG-POSTERIOR  (Derivation Notes Eq. 16)
# =============================================================================

def log_posterior(alpha, beta, u, tau, y, eta, sigma_alpha, s_beta,
                  s_tau, prior_family):
    """
    Full log-posterior (up to additive constant).

    ell = (I) + (II) + (III) + (IV) + (V)

    This is used mainly for debugging / verification.
    Individual MH updates use block-specific targets (Section 5 below).
    """
    lp = log_likelihood(y, eta)
    lp += log_prior_alpha(alpha, sigma_alpha)
    lp += log_prior_beta(beta, s_beta)
    lp += log_prior_u_given_tau(u, tau)
    lp += log_prior_tau(tau, s_tau, prior_family)
    return lp


# =============================================================================
# 5. BLOCK-SPECIFIC LOG-TARGETS  (Derivation Notes Section 4.7)
# =============================================================================

def log_target_alpha(alpha, beta, u, tau, y, X, group_idx,
                     sigma_alpha):
    """
    Log-target for alpha (Derivation Notes Eq. 20):
        h(alpha) = sum_i [y_i*eta_i - log(1+exp(eta_i))] - alpha^2/(2*sigma_alpha^2)

    In practice we pass the pre-computed eta for efficiency.
    """
    eta = compute_eta(alpha, beta, u, X, group_idx)
    return log_likelihood(y, eta) + log_prior_alpha(alpha, sigma_alpha)


def log_target_beta_k(k, beta_k, alpha, beta, u, tau, y, eta,
                       s_beta):
    """
    Log-target for beta_k (Derivation Notes Eq. 23):
        h(beta_k) = sum_i [y_i*eta_i - log(1+exp(eta_i))] - beta_k^2/(2*s_beta^2)

    NOTE: eta must already reflect the proposed beta_k value.
    """
    return log_likelihood(y, eta) + (-0.5 * beta_k**2 / s_beta**2)


def log_target_u_j(j, u_j, y_j, eta_j, tau):
    """
    Log-target for u_j (Derivation Notes Eq. 26):
        h(u_j) = sum_{i: j[i]=j} [y_i*eta_i - log(1+exp(eta_i))] - u_j^2/(2*tau^2)

    Only uses observations in group j.
    """
    ll_j = np.sum(y_j * eta_j - np.logaddexp(0.0, eta_j))
    return ll_j + (-0.5 * u_j**2 / tau**2)


def log_target_log_tau(phi, u, s_tau, prior_family="half_normal"):
    """
    Log-target for phi = log(tau) (Derivation Notes Eq. 29):
        h(phi) = -J*phi - (1/(2*exp(2*phi))) * sum(u_j^2)
                 - exp(2*phi)/(2*s_tau^2) + phi

    The +phi term is the log-Jacobian |d(tau)/d(phi)| = tau = exp(phi).

    For Half-Cauchy (Derivation Notes Eq. 31):
        replace -exp(2*phi)/(2*s_tau^2) with -log(1 + exp(2*phi)/s_c^2)

    NOTE: This does NOT depend on the likelihood (see Derivation Notes
    Section 4.5, "Why tau does not depend on the likelihood").
    """
    J = len(u)
    sum_u_sq = np.sum(u**2)
    exp_2phi = np.exp(2.0 * phi)

    h = -J * phi - 0.5 * sum_u_sq / exp_2phi + phi  # Jacobian term

    if prior_family == "half_normal":
        h -= 0.5 * exp_2phi / s_tau**2
    elif prior_family == "half_cauchy":
        h -= np.log(1.0 + exp_2phi / s_tau**2)

    return h
