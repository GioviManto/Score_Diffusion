"""Utilities for Gaussian and Laplace AR(1) diffusion benchmarks.

The main exact non-Gaussian benchmark is the K=1 Laplace model

    A_0 ~ N(mu0, sigma0_sq),
    A_1 = alpha A_0 + eps,   eps ~ Laplace(0, b),
    X_k = exp(-t) A_k + sqrt(Delta_t) Z_k.

For this model the noisy density, score, and negative log-Hessian can be
computed exactly after reducing the problem to one-dimensional Gaussian
expectations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy import integrate, special

Array = np.ndarray


# -----------------------------------------------------------------------------
# Basic scalar densities and helpers
# -----------------------------------------------------------------------------


def delta_t(t: float) -> float:
    """OU noise variance Delta_t = 1 - exp(-2 t)."""
    return 1.0 - np.exp(-2.0 * float(t))



def gaussian_pdf(x: Array | float, mean: float, var: float) -> Array:
    """Gaussian density N(mean, var) evaluated at x."""
    x_arr = np.asarray(x, dtype=float)
    return np.exp(-0.5 * (x_arr - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)



def _broadcast_pair(x0: Array | float, x1: Array | float) -> Tuple[Array, Array]:
    """Broadcast two inputs to a common shape as float arrays."""
    a0 = np.asarray(x0, dtype=float)
    a1 = np.asarray(x1, dtype=float)
    return np.broadcast_arrays(a0, a1)


@lru_cache(maxsize=None)
def _hermgauss_cached(order: int) -> Tuple[Array, Array]:
    """Cached Gauss-Hermite nodes and weights."""
    nodes, weights = hermgauss(int(order))
    return nodes.astype(float), weights.astype(float)


# -----------------------------------------------------------------------------
# Finite Gaussian AR(1) chain: covariance, precision, and OU noising
# -----------------------------------------------------------------------------


def gaussian_chain_covariance(
    n: int,
    alpha: float,
    sigma0_sq: float,
    sigma_eta_sq: float,
) -> Array:
    """Covariance of a length-n Gaussian AR(1) chain.

    Model:
        A_{k+1} = alpha A_k + eta_k,  eta_k ~ N(0, sigma_eta_sq).
    """
    n = int(n)
    variances = np.zeros(n, dtype=float)
    variances[0] = sigma0_sq
    for k in range(1, n):
        variances[k] = alpha ** 2 * variances[k - 1] + sigma_eta_sq

    sigma = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            m = min(i, j)
            sigma[i, j] = alpha ** abs(i - j) * variances[m]
    return sigma



def gaussian_chain_precision(
    n: int,
    alpha: float,
    sigma0_sq: float,
    sigma_eta_sq: float,
) -> Array:
    """Exact precision matrix of a finite Gaussian AR(1) chain."""
    n = int(n)
    q = np.zeros((n, n), dtype=float)
    q[0, 0] = 1.0 / sigma0_sq + alpha ** 2 / sigma_eta_sq
    if n >= 2:
        q[n - 1, n - 1] = 1.0 / sigma_eta_sq
    for k in range(1, n - 1):
        q[k, k] = (1.0 + alpha ** 2) / sigma_eta_sq
    for k in range(n - 1):
        q[k, k + 1] = -alpha / sigma_eta_sq
        q[k + 1, k] = -alpha / sigma_eta_sq
    return q



def gaussian_ou_covariance(sigma0: Array, t: float) -> Array:
    """Sigma_t = exp(-2 t) Sigma_0 + Delta_t I."""
    sigma0 = np.asarray(sigma0, dtype=float)
    return np.exp(-2.0 * t) * sigma0 + delta_t(t) * np.eye(sigma0.shape[0])



def gaussian_ou_precision(sigma0: Array, t: float) -> Array:
    """Q_t = Sigma_t^{-1}."""
    return np.linalg.inv(gaussian_ou_covariance(sigma0, t))



def deterministic_control_precision(
    n: int,
    alpha: float,
    sigma0_sq: float,
    t: float,
) -> Tuple[Array, Array, Array]:
    """OU covariance and precision for deterministic propagation A_k = alpha^k A_0."""
    v = alpha ** np.arange(int(n), dtype=float)
    beta_t = np.exp(-2.0 * t) * sigma0_sq
    dt = delta_t(t)
    sigma_t = dt * np.eye(int(n)) + beta_t * np.outer(v, v)
    q_t = (1.0 / dt) * np.eye(int(n))
    q_t -= beta_t * np.outer(v, v) / (dt * (dt + beta_t * np.dot(v, v)))
    return sigma_t, q_t, v


# -----------------------------------------------------------------------------
# Matched Gaussian K=1 benchmark
# -----------------------------------------------------------------------------


def gaussian_k1_mean_cov_precision(t: float, params: Dict[str, float]) -> Tuple[Array, Array, Array]:
    """Mean, covariance, precision for the matched Gaussian K=1 model.

    If params contains b but not sigma_eta_sq, then sigma_eta_sq is set to 2 b^2,
    matching the variance of Laplace(0, b).
    """
    mu0 = float(params["mu0"])
    sigma0_sq = float(params["sigma0_sq"])
    alpha = float(params["alpha"])
    if "sigma_eta_sq" in params:
        sigma_eta_sq = float(params["sigma_eta_sq"])
    else:
        sigma_eta_sq = 2.0 * float(params["b"]) ** 2

    mean_a = np.array([mu0, alpha * mu0], dtype=float)
    sigma_a = np.array(
        [
            [sigma0_sq, alpha * sigma0_sq],
            [alpha * sigma0_sq, alpha ** 2 * sigma0_sq + sigma_eta_sq],
        ],
        dtype=float,
    )
    mu_t = np.exp(-t) * mean_a
    sigma_t = np.exp(-2.0 * t) * sigma_a + delta_t(t) * np.eye(2)
    q_t = np.linalg.inv(sigma_t)
    return mu_t, sigma_t, q_t



def gaussian_k1_score_and_hessian(
    x0: Array | float,
    x1: Array | float,
    t: float,
    params: Dict[str, float],
) -> Tuple[Array, Array, Array]:
    """Score and constant negative log-Hessian for the matched Gaussian K=1 model."""
    x0_arr, x1_arr = _broadcast_pair(x0, x1)
    mu_t, _, q_t = gaussian_k1_mean_cov_precision(t, params)
    dx0 = x0_arr - mu_t[0]
    dx1 = x1_arr - mu_t[1]
    s0 = -(q_t[0, 0] * dx0 + q_t[0, 1] * dx1)
    s1 = -(q_t[1, 0] * dx0 + q_t[1, 1] * dx1)
    return s0, s1, q_t


# -----------------------------------------------------------------------------
# Normal-Laplace convolution and derivatives
# -----------------------------------------------------------------------------


def normal_laplace_density(y: Array | float, beta: float, sigma_sq: float) -> Array:
    r"""Density of L + G with L ~ Laplace(0, beta), G ~ N(0, sigma_sq).

    Exact formula:
        h(y) = exp(sigma_sq / (2 beta^2)) / (2 beta)
               [exp(-y/beta) Phi(y/sigma - sigma/beta)
                + exp(y/beta) Phi(-y/sigma - sigma/beta)].
    """
    y_arr = np.asarray(y, dtype=float)
    sigma = np.sqrt(sigma_sq)
    u = y_arr / sigma - sigma / beta
    v = -y_arr / sigma - sigma / beta
    log_term1 = -y_arr / beta + special.log_ndtr(u)
    log_term2 = y_arr / beta + special.log_ndtr(v)
    log_pref = sigma_sq / (2.0 * beta ** 2) - np.log(2.0 * beta)
    return np.exp(log_pref + np.logaddexp(log_term1, log_term2))



def normal_laplace_density_prime(y: Array | float, beta: float, sigma_sq: float) -> Array:
    r"""Derivative of the normal-Laplace density.

    Exact formula:
        h'(y) = exp(sigma_sq / (2 beta^2)) / (2 beta^2)
                [exp(y/beta) Phi(-y/sigma - sigma/beta)
                 - exp(-y/beta) Phi(y/sigma - sigma/beta)].
    """
    y_arr = np.asarray(y, dtype=float)
    sigma = np.sqrt(sigma_sq)
    u = y_arr / sigma - sigma / beta
    v = -y_arr / sigma - sigma / beta
    log_a = y_arr / beta + special.log_ndtr(v)
    log_b = -y_arr / beta + special.log_ndtr(u)
    max_log = np.maximum(log_a, log_b)
    min_log = np.minimum(log_a, log_b)
    sign = np.where(log_a >= log_b, 1.0, -1.0)
    diff = sign * np.exp(max_log) * (-np.expm1(min_log - max_log))
    pref = np.exp(sigma_sq / (2.0 * beta ** 2)) / (2.0 * beta ** 2)
    return pref * diff



def normal_laplace_density_double_prime(y: Array | float, beta: float, sigma_sq: float) -> Array:
    r"""Second derivative via (1 - beta^2 d^2/dy^2) h = phi_{sigma_sq}."""
    y_arr = np.asarray(y, dtype=float)
    return (normal_laplace_density(y_arr, beta, sigma_sq) - gaussian_pdf(y_arr, 0.0, sigma_sq)) / (beta ** 2)


# -----------------------------------------------------------------------------
# Exact Laplace K=1 benchmark
# -----------------------------------------------------------------------------


def laplace_k1_posterior_a0_given_x0(
    x0: Array | float,
    t: float,
    params: Dict[str, float],
) -> Tuple[Array, Array, float, float]:
    """Posterior mean/variance of A_0 | X_0 = x_0 and marginal variance of X_0.

    Returns
    -------
    m_t : array
        Posterior mean m_t(x_0).
    v_t : float
        Posterior variance of A_0 | X_0.
    v_x : float
        Marginal variance of X_0.
    m_prime : float
        Derivative dm_t / dx_0.
    """
    mu0 = float(params["mu0"])
    sigma0_sq = float(params["sigma0_sq"])
    x0_arr = np.asarray(x0, dtype=float)
    dt = delta_t(t)
    exp_t = np.exp(-t)
    exp_2t = np.exp(-2.0 * t)

    v_t_inv = 1.0 / sigma0_sq + exp_2t / dt
    v_t = 1.0 / v_t_inv
    m_t = v_t * (mu0 / sigma0_sq + exp_t * x0_arr / dt)
    v_x = exp_2t * sigma0_sq + dt
    m_prime = v_t * exp_t / dt
    return m_t, v_t, v_x, m_prime



def _gaussian_expectation_from_posterior(
    x0: Array,
    x1: Array,
    t: float,
    params: Dict[str, float],
    order: int,
) -> Tuple[Array, Array, Array]:
    """Compute I0, I1, I2 using Gauss-Hermite quadrature."""
    alpha = float(params["alpha"])
    b = float(params["b"])
    dt = delta_t(t)
    exp_t = np.exp(-t)
    beta_t = exp_t * b
    c_t = exp_t * alpha

    m_t, v_t, _, _ = laplace_k1_posterior_a0_given_x0(x0, t, params)
    nodes, weights = _hermgauss_cached(order)

    a0 = m_t[..., None] + np.sqrt(2.0 * v_t) * nodes
    y = x1[..., None] - c_t * a0

    h0 = normal_laplace_density(y, beta_t, dt)
    h1 = normal_laplace_density_prime(y, beta_t, dt)
    h2 = normal_laplace_density_double_prime(y, beta_t, dt)

    factor = weights / np.sqrt(np.pi)
    i0 = np.tensordot(h0, factor, axes=([-1], [0]))
    i1 = np.tensordot(h1, factor, axes=([-1], [0]))
    i2 = np.tensordot(h2, factor, axes=([-1], [0]))
    return i0, i1, i2



def laplace_k1_I012(
    x0: Array | float,
    x1: Array | float,
    t: float,
    params: Dict[str, float],
    order: int = 80,
) -> Tuple[Array, Array, Array]:
    """Return the exact Gaussian expectations I_0, I_1, I_2."""
    x0_arr, x1_arr = _broadcast_pair(x0, x1)
    return _gaussian_expectation_from_posterior(x0_arr, x1_arr, t, params, int(order))



def laplace_k1_density_exact(
    x0: Array | float,
    x1: Array | float,
    t: float,
    params: Dict[str, float],
    order: int = 80,
) -> Array:
    """Exact noisy density p_t(x_0, x_1) for the K=1 Laplace AR(1) model."""
    x0_arr, x1_arr = _broadcast_pair(x0, x1)
    mu0 = float(params["mu0"])
    _, _, v_x, _ = laplace_k1_posterior_a0_given_x0(x0_arr, t, params)
    i0, _, _ = _gaussian_expectation_from_posterior(x0_arr, x1_arr, t, params, int(order))
    return gaussian_pdf(x0_arr, np.exp(-t) * mu0, v_x) * i0



def laplace_k1_score_and_hessian(
    x0: Array | float,
    x1: Array | float,
    t: float,
    params: Dict[str, float],
    order: int = 80,
) -> Dict[str, Array]:
    """Exact score and negative log-Hessian for the K=1 Laplace benchmark.

    Returns a dictionary with keys
        density, score0, score1, H00, H01, H11, I0, I1, I2.
    """
    x0_arr, x1_arr = _broadcast_pair(x0, x1)
    mu0 = float(params["mu0"])
    alpha = float(params["alpha"])
    exp_t = np.exp(-t)
    c_t = exp_t * alpha

    m_t, _, v_x, m_prime = laplace_k1_posterior_a0_given_x0(x0_arr, t, params)
    i0, i1, i2 = _gaussian_expectation_from_posterior(x0_arr, x1_arr, t, params, int(order))
    density = gaussian_pdf(x0_arr, exp_t * mu0, v_x) * i0

    score1 = i1 / i0
    score0 = -(x0_arr - exp_t * mu0) / v_x - c_t * m_prime * (i1 / i0)

    h11 = (i1 ** 2 - i0 * i2) / (i0 ** 2)
    h01 = -c_t * m_prime * h11
    h00 = 1.0 / v_x + (c_t ** 2) * (m_prime ** 2) * h11

    return {
        "density": density,
        "score0": score0,
        "score1": score1,
        "H00": h00,
        "H01": h01,
        "H11": h11,
        "I0": i0,
        "I1": i1,
        "I2": i2,
        "m_t": m_t,
        "v_x": np.asarray(v_x),
        "m_prime": np.asarray(m_prime),
    }


# -----------------------------------------------------------------------------
# Independent numerical checks
# -----------------------------------------------------------------------------


def laplace_k1_density_direct_quad(
    x0: float,
    x1: float,
    t: float,
    params: Dict[str, float],
) -> float:
    """Independent double-integral evaluation of the K=1 noisy density.

    This does not use the closed-form normal-Laplace density and is used only for
    numerical verification.
    """
    mu0 = float(params["mu0"])
    sigma0_sq = float(params["sigma0_sq"])
    alpha = float(params["alpha"])
    b = float(params["b"])
    dt = delta_t(t)
    exp_t = np.exp(-t)

    def inner_integrand(eps: float, a0: float) -> float:
        prior_eps = np.exp(-abs(eps) / b) / (2.0 * b)
        like_x1 = gaussian_pdf(x1, exp_t * (alpha * a0 + eps), dt)
        return float(prior_eps * like_x1)

    def outer_integrand(a0: float) -> float:
        prior_a0 = gaussian_pdf(a0, mu0, sigma0_sq)
        like_x0 = gaussian_pdf(x0, exp_t * a0, dt)
        inner_val, _ = integrate.quad(inner_integrand, -np.inf, np.inf, args=(a0,), epsabs=1e-10, epsrel=1e-10, limit=200)
        return float(prior_a0 * like_x0 * inner_val)

    val, _ = integrate.quad(outer_integrand, -np.inf, np.inf, epsabs=1e-10, epsrel=1e-10, limit=200)
    return float(val)


__all__ = [
    "delta_t",
    "gaussian_pdf",
    "gaussian_chain_covariance",
    "gaussian_chain_precision",
    "gaussian_ou_covariance",
    "gaussian_ou_precision",
    "deterministic_control_precision",
    "gaussian_k1_mean_cov_precision",
    "gaussian_k1_score_and_hessian",
    "normal_laplace_density",
    "normal_laplace_density_prime",
    "normal_laplace_density_double_prime",
    "laplace_k1_posterior_a0_given_x0",
    "laplace_k1_I012",
    "laplace_k1_density_exact",
    "laplace_k1_score_and_hessian",
    "laplace_k1_density_direct_quad",
]
