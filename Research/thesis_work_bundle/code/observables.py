"""
Scalar observables on the residual ray for the Laplace K=1 benchmark.

Each observable is defined analytically, given a closed form, and has a
numerical reference implementation used for validation.

Observables implemented
-----------------------
  * kappa_center(t)       kappa_t(0)                         [center curvature]
  * kappa_tail(t)         lim_{|r|->inf} kappa_t(r) = 0      [tail curvature]
  * psi_saturation(t)     lim_{r->inf} psi_t(r) = -1/btilde
  * gap(t)                kappa_t(0) - kappa_tail(t)
  * band_slope(t)         rho (slope of the high-curvature residual band)
  * band_fwhm(t)          full width at half maximum of kappa_t(r)
  * kappa_G(t)            Gaussian-benchmark constant curvature 1/tau_G^2
  * gap_vs_G(t)           kappa_t(0) - kappa_G(t)   (center-benchmark gap)

Closed-form center curvature
----------------------------
Let a = tau / btilde and R(a) = Phi(-a)/phi(a) denote Mills' ratio for the
standard normal.  Then by symmetry of h_t,  psi_t(0) = 0, and from
equation (80) of the note:

    kappa_t(0) = -1/btilde^2 + (1/btilde^2) * N(0;0,tau^2) / h_t(0).

Using h_t(0) = (exp(a^2/2)/btilde) * Phi(-a) we obtain the clean identity

    kappa_t(0) * btilde^2 = 1/(a * R(a)) - 1.

This is a single dimensionless quantity controlling the non-Gaussianity.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad
from laplace_k1 import (
    build_params, kappa_t, psi_t, h_t, log_h_t,
)


def mills_ratio(a):
    """Mills' ratio R(a) = Phi(-a) / phi(a)."""
    return np.exp(-norm.logpdf(a) + norm.logcdf(-a))


def kappa_center_closed(p):
    """Closed-form center curvature (see module docstring).

    kappa_t(0) = (1/btilde^2) * (1/(a * R(a)) - 1).
    Handles the t -> infinity limit (a -> infinity) smoothly.
    """
    a = p.a
    if not np.isfinite(a):
        return 0.0
    R = mills_ratio(a)
    return (1.0 / (a * R) - 1.0) / p.btilde**2


def kappa_tail_closed(p):
    """Tail curvature.  Exactly zero for every finite t because
    psi_t saturates to +-1/btilde and the Gaussian overlap decays.
    """
    return 0.0


def psi_saturation(p):
    """Sign-signed saturation values.  psi_t -> -1/btilde as r -> +infty."""
    return (-1.0 / p.btilde, 1.0 / p.btilde)


def kappa_G(p):
    """Gaussian-benchmark residual curvature 1 / tau_G^2."""
    return 1.0 / p.tau2G


def gap_vs_G(p):
    """Center-vs-benchmark stiffness gap kappa_t(0) - kappa_G(t)."""
    return kappa_center_closed(p) - kappa_G(p)


def band_fwhm(p, r_max=None):
    """Full width at half maximum of kappa_t(r).

    Defined as 2 r* where kappa_t(r*) = (1/2) * kappa_t(0).
    Located by a bracketed root search on [0, r_max].
    """
    k0 = kappa_center_closed(p)
    # Robust fallback: if gap is tiny (near-Gaussian regime), FWHM is ill-defined
    if k0 <= 1e-14:
        return np.nan
    if r_max is None:
        r_max = max(6.0 * np.sqrt(p.tau2), 6.0 * p.btilde)
    target = lambda r: kappa_t(np.array([r]), p)[0] - 0.5 * k0
    # Make sure we have a sign change
    if target(r_max) * target(0.0) > 0:
        # Extend until a sign change appears or we give up
        for _ in range(8):
            r_max *= 2.0
            if target(r_max) * target(0.0) < 0:
                break
        else:
            return np.nan
    rstar = brentq(target, 1e-8, r_max, xtol=1e-10)
    return 2.0 * rstar


# ---------------------------------------------------------------------------
# Numerical reference implementations (for validation)
# ---------------------------------------------------------------------------

def kappa_center_numerical(p, h=1e-3):
    """Central finite-difference reference value for kappa_t(0)."""
    r = np.array([-2 * h, -h, 0.0, h, 2 * h])
    lh = log_h_t(r, p)
    # standard 5-point second derivative
    d2 = (-lh[0] + 16 * lh[1] - 30 * lh[2] + 16 * lh[3] - lh[4]) / (12 * h**2)
    return -d2


def fourth_cumulant_residual(p):
    """Exact fourth cumulant of R_t.

    R_t = beta*eps + G_t,  eps ~ Laplace(0, b), G_t ~ N(0, tau^2).
    Cumulants add:  kappa_4(R_t) = beta^4 * kappa_4(eps) = beta^4 * 12 b^4.
    """
    return 12.0 * p.beta**4 * p.b**4


def var_residual(p):
    """Exact variance Var(R_t) = 2 btilde^2 + tau^2."""
    return 2.0 * p.btilde**2 + p.tau2


def gaussianity_ratio(p):
    """Dimensionless non-Gaussianity score: excess-kurtosis of R_t.

    excess_kurt(R_t) = kappa_4 / Var^2.
    Equals 0 for a Gaussian; equals 3 for a pure Laplace (beta=1, tau=0).
    """
    return fourth_cumulant_residual(p) / var_residual(p) ** 2
