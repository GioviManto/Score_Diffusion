"""
Laplace AR(1) under Ornstein--Uhlenbeck corruption: closed-form K=1 formulas.

All symbols match laplace_ar1_ou_note.pdf. The exported API lets us compute the
joint density, score, curvature field, and their residual-ray summaries exactly.

References (equation numbers from the note):
    beta = e^{-t},  Delta = 1 - beta^2
    v    = beta^2 * sigma0^2 + Delta                        (12)
    btilde = beta * b                                       (34)
    s^2  = sigma0^2 * Delta / v                             (19)
    tau^2 = Delta + alpha^2 * beta^2 * s^2                  (25)
    rho  = alpha * beta^2 * sigma0^2 / v                    (27)
    nu   = alpha * beta * Delta * mu0 / v                   (27)
    c    = alpha * Delta / v                                (91)
    a    = tau / btilde                                     (55)
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


@dataclass(frozen=True)
class LaplaceAR1Params:
    """All structural constants for the Laplace AR(1) / OU model at time t."""
    alpha: float
    b: float                  # clean Laplace scale  (Var(eps) = 2 b^2)
    sigma0: float             # prior std of A_0
    mu0: float                # prior mean of A_0
    t: float                  # diffusion time

    # Derived (filled in by build_params)
    beta: float = 0.0
    Delta: float = 0.0
    v: float = 0.0
    btilde: float = 0.0
    s2: float = 0.0
    tau2: float = 0.0
    tau: float = 0.0
    rho: float = 0.0
    nu: float = 0.0
    c: float = 0.0
    a: float = 0.0
    # Gaussian benchmark (matched-variance innovation)
    tau2G: float = 0.0


def build_params(alpha, b, sigma0, mu0, t):
    beta = np.exp(-t)
    Delta = 1.0 - beta**2
    v = beta**2 * sigma0**2 + Delta
    btilde = beta * b
    # Guard against Delta == 0 at t==0 (then s^2 = 0 and rho = alpha)
    if v > 0:
        s2 = sigma0**2 * Delta / v
    else:
        s2 = 0.0
    tau2 = Delta + alpha**2 * beta**2 * s2
    tau = np.sqrt(tau2)
    rho = alpha * beta**2 * sigma0**2 / v if v > 0 else alpha
    nu = alpha * beta * Delta * mu0 / v if v > 0 else 0.0
    c = alpha * Delta / v if v > 0 else 0.0
    # a = tau / btilde, handle btilde -> 0 limit (t -> infinity)
    a = tau / btilde if btilde > 0 else np.inf
    tau2G = tau2 + 2.0 * btilde**2
    return LaplaceAR1Params(
        alpha=alpha, b=b, sigma0=sigma0, mu0=mu0, t=t,
        beta=beta, Delta=Delta, v=v, btilde=btilde,
        s2=s2, tau2=tau2, tau=tau,
        rho=rho, nu=nu, c=c, a=a, tau2G=tau2G,
    )


# ---------------------------------------------------------------------------
# Residual density h_t(r), its derivatives, score psi_t, curvature kappa_t
# All formulas from the note; we implement the numerically stable forms.
# ---------------------------------------------------------------------------

def _Lplus_log(r, p):
    """log L_+(r) = -r/btilde + log Phi(r/tau - a).  Vectorised."""
    z = r / p.tau - p.a
    return -r / p.btilde + norm.logcdf(z)


def _Lminus_log(r, p):
    """log L_-(r) = +r/btilde + log Phi(-r/tau - a)."""
    z = -r / p.tau - p.a
    return r / p.btilde + norm.logcdf(z)


def log_h_t(r, p):
    """Log of the Normal-Laplace residual density h_t(r).

    h_t(r) = exp(a^2/2) / (2 btilde) * [L_+(r) + L_-(r)]     (55)

    Numerically stable via log-sum-exp of L_+ and L_-.
    """
    lp = _Lplus_log(r, p)
    lm = _Lminus_log(r, p)
    M = np.maximum(lp, lm)
    sum_log = M + np.log(np.exp(lp - M) + np.exp(lm - M))
    return 0.5 * p.a**2 - np.log(2.0 * p.btilde) + sum_log


def h_t(r, p):
    return np.exp(log_h_t(r, p))


def psi_t(r, p):
    """Residual score psi_t(r) = h'_t / h_t.

    Exact closed form (67):
        psi_t(r) = (1/btilde) * (L_- - L_+) / (L_+ + L_-)

    Numerically stable using the log-ratios.
    """
    lp = _Lplus_log(r, p)
    lm = _Lminus_log(r, p)
    # Using tanh-like stable form:
    #   (L- - L+)/(L- + L+) = tanh( (log L- - log L+) / 2 ) * <matching overall sign>
    # Actually  (e^{lm} - e^{lp})/(e^{lm}+e^{lp}) = tanh((lm - lp)/2).
    return np.tanh(0.5 * (lm - lp)) / p.btilde


def kappa_t(r, p):
    """Residual curvature kappa_t(r) = -d^2/dr^2 log h_t(r) = -psi_t'(r).

    Exact formula (80):
        kappa_t(r) = psi_t(r)^2 - 1/btilde^2
                   + (1 / btilde^2) * N(r; 0, tau^2) / h_t(r)

    where N(r; 0, tau^2) is the standard normal density with variance tau^2.
    """
    log_gauss = -0.5 * r**2 / p.tau2 - 0.5 * np.log(2.0 * np.pi * p.tau2)
    gauss_over_h = np.exp(log_gauss - log_h_t(r, p))
    return psi_t(r, p) ** 2 - 1.0 / p.btilde**2 + gauss_over_h / p.btilde**2


# ---------------------------------------------------------------------------
# Joint density, full score, full Hessian field
# ---------------------------------------------------------------------------

def residual(x0, x1, p):
    """r = x1 - nu - rho x0."""
    return x1 - p.nu - p.rho * x0


def log_p_t(x0, x1, p):
    """Log of the joint noisy density (56).

    log p_t(x0,x1) = -(x0 - beta mu0)^2 / (2v) - 0.5*log(2 pi v) + log h_t(r)
    """
    lg = -0.5 * (x0 - p.beta * p.mu0) ** 2 / p.v - 0.5 * np.log(2 * np.pi * p.v)
    return lg + log_h_t(residual(x0, x1, p), p)


def score_t(x0, x1, p):
    """Return (S0, S1) at (x0, x1).  Equation (60)."""
    r = residual(x0, x1, p)
    psi = psi_t(r, p)
    S0 = -(x0 - p.beta * p.mu0) / p.v - p.rho * psi
    S1 = psi
    return S0, S1


def hessian_field(x0, x1, p):
    """Return the 2x2 curvature matrix H_t(x) at (x0, x1).  Equation (75).

    H_t(x) = diag(1/v, 0) + kappa_t(r) * [[rho^2, -rho], [-rho, 1]]
    """
    r = residual(x0, x1, p)
    kappa = kappa_t(r, p)
    H = np.empty(np.broadcast_shapes(np.shape(x0), np.shape(x1)) + (2, 2))
    H[..., 0, 0] = 1.0 / p.v + p.rho**2 * kappa
    H[..., 0, 1] = -p.rho * kappa
    H[..., 1, 0] = -p.rho * kappa
    H[..., 1, 1] = kappa
    return H


# ---------------------------------------------------------------------------
# Gaussian benchmark (matched variance): eps ~ N(0, 2 b^2)
# ---------------------------------------------------------------------------

def gaussian_benchmark(x0, x1, p):
    """Return log-density, score, and constant Hessian of the matched Gaussian.

    Residual is N(0, tau2G) with tau2G = tau2 + 2 btilde^2.
    """
    r = residual(x0, x1, p)
    lg0 = -0.5 * (x0 - p.beta * p.mu0)**2 / p.v - 0.5 * np.log(2 * np.pi * p.v)
    lg1 = -0.5 * r**2 / p.tau2G - 0.5 * np.log(2 * np.pi * p.tau2G)
    logp = lg0 + lg1
    S0 = -(x0 - p.beta * p.mu0) / p.v + p.rho * r / p.tau2G
    S1 = -r / p.tau2G
    Qt = np.array([[1.0 / p.v + p.rho**2 / p.tau2G, -p.rho / p.tau2G],
                   [-p.rho / p.tau2G, 1.0 / p.tau2G]])
    return logp, (S0, S1), Qt
