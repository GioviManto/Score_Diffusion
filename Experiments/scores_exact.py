"""
scores_exact.py — Comprehensive numerical verification of Score Diffusion hypotheses.

Hypotheses from CLAUDE.md (updated after Mézard meeting 2026-03-12):

  H1 [exact]:   Joint score is linear in x for Gaussian AR(1):
                S(x,t) = -Σ_t^{-1}(x - μ_t).
                Verify: Kalman-smoother formula == precision-matrix formula.

  H2 [exact]:   Σ_0^{-1} is tridiagonal for AR(1).
                Verify: off-tridiagonal elements are zero to machine precision.

  H3 [hypothesis]: -Σ_t^{-1} is approximately Toeplitz for stationary AR(1).
                Measure: ‖Q_t - T(Q_t)‖_F / ‖Q_t‖_F vs. N and t.

  H4 [exact]:   μ_t contraction: μ_t = e^{-t} μ_0, k-th component = α^k e^{-t} μ_{a,0}.
                Verify: chain formula agrees with direct computation.

  H5 [exact/hypothesis]: Kalman smoother provides the EXACT propagator for Gaussian AR(1).
                Verify: RTS-smoother score == precision-matrix score for all x.
                Also compute the row-to-row ratio of Σ_t^{-1} rows (Toeplitz propagator).

  K2: M_{12} = −α Δ for the K=2 bivariate characteristic function.
      Verify numerically. Non-zero ⇒ exact product-of-1D-bonds fails for K≥2.

Legacy checks (re-run from run_checks.py):
  C1  Gaussian chain covariance × precision = I
  C2  Small-time Neumann expansion of Σ_t^{-1}
  C3  Deterministic-control precision
  C4  Normal-Laplace convolution vs quadrature
  C5  Laplace K=1 density vs direct double integral
  C6  Laplace K=1 score vs finite differences
  C7  Laplace K=1 Hessian vs finite differences

Usage:
  conda run -n vlm-sam3-py312 python Experiments/scores_exact.py
from Score_Diffusion root, OR
  python scores_exact.py
from Experiments/ with PYTHONPATH set.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, linalg

# ---------------------------------------------------------------------------
# Path setup — allow running from Score_Diffusion root or from Experiments/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_AUDIT_CODE = _HERE.parent / "Research" / "laplace_ar1_audit" / "code"
sys.path.insert(0, str(_AUDIT_CODE))

from ar1_diffusion_utils import (               # noqa: E402
    delta_t,
    gaussian_pdf,
    gaussian_chain_covariance,
    gaussian_chain_precision,
    gaussian_ou_covariance,
    gaussian_ou_precision,
    deterministic_control_precision,
    normal_laplace_density,
    laplace_k1_density_direct_quad,
    laplace_k1_density_exact,
    laplace_k1_score_and_hessian,
)

# Tolerance thresholds
TOL_MACHINE = 1e-10    # exact identities (matrix inverses, exact algebra)
TOL_GH      = 1e-7     # Gauss-Hermite integrals (80-120 nodes)
TOL_FD      = 2e-5     # 5-point finite-difference references
TOL_TOEPLITZ_LARGE = 0.05   # H3 Toeplitz deviation: threshold at N=100

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _pf(err: float, tol: float) -> str:
    return PASS if err <= tol else FAIL


# ===========================================================================
# Kalman / RTS smoother for Gaussian AR(1) + OU
# ===========================================================================

def kalman_rts_smoother(
    x: np.ndarray,          # shape (K,)
    alpha: float,
    sigma0_sq: float,
    sigma_eta_sq: float,
    t: float,
) -> np.ndarray:
    """RTS smoother: returns smoothed means E[a_k | x], shape (K,).

    Model:
        a_{k+1} = alpha * a_k + eta_k,  eta_k ~ N(0, sigma_eta_sq)
        x_k = exp(-t) * a_k + sqrt(Delta_t) * z_k
    Prior: a_0 ~ N(0, sigma0_sq).  [mu0=0 here; extend if needed]
    """
    K = len(x)
    et = math.exp(-t)
    dt = delta_t(t)

    # ---- Forward filter ---------------------------------------------------
    m_filt  = np.zeros(K)
    P_filt  = np.zeros(K)
    m_pred  = np.zeros(K)
    P_pred  = np.zeros(K)

    m_pred[0] = 0.0          # mu_0 = 0
    P_pred[0] = sigma0_sq

    for k in range(K):
        # Update
        S_k      = et**2 * P_pred[k] + dt          # innovation variance
        K_k      = et * P_pred[k] / S_k            # Kalman gain
        m_filt[k] = m_pred[k] + K_k * (x[k] - et * m_pred[k])
        P_filt[k] = (1.0 - et * K_k) * P_pred[k]
        # Predict (if not last)
        if k < K - 1:
            m_pred[k+1] = alpha * m_filt[k]
            P_pred[k+1] = alpha**2 * P_filt[k] + sigma_eta_sq

    # ---- Backward smoother (RTS) -------------------------------------------
    m_smooth = m_filt.copy()
    P_smooth = P_filt.copy()

    for k in range(K-2, -1, -1):
        G_k        = P_filt[k] * alpha / P_pred[k+1]   # smoother gain
        m_smooth[k] = m_filt[k] + G_k * (m_smooth[k+1] - alpha * m_filt[k])
        P_smooth[k] = P_filt[k] + G_k**2 * (P_smooth[k+1] - P_pred[k+1])

    return m_smooth


def kalman_score(
    x: np.ndarray,
    alpha: float,
    sigma0_sq: float,
    sigma_eta_sq: float,
    t: float,
    mu0: float = 0.0,
) -> np.ndarray:
    """Joint score via Kalman smoother: S_k = (exp(-t) E[a_k|x] - x_k) / Delta_t."""
    # Shift x by mu0 effect: the smoother uses mu0=0, then shift means
    # E[a_k|x] = E[a_k - alpha^k mu0 | x - exp(-t)*alpha^k mu0*1] + alpha^k mu0
    # Simpler: subtract the deterministic mean from x, smooth, re-add.
    K = len(x)
    et = math.exp(-t)
    dt = delta_t(t)
    alpha_pow = np.array([alpha**k for k in range(K)])
    mu_a = mu0 * alpha_pow          # E[a_k] for prior
    mu_t = et * mu_a                # E[x_k] = exp(-t) * E[a_k]

    x_centred = x - mu_t           # remove deterministic mean
    m_s = kalman_rts_smoother(x_centred, alpha, sigma0_sq, sigma_eta_sq, t)
    m_s = m_s + mu_a               # add back the mean of a_k

    return (et * m_s - x) / dt


# ===========================================================================
# Toeplitz projection helper
# ===========================================================================

def toeplitz_project(Q: np.ndarray) -> np.ndarray:
    """Return the Toeplitz matrix whose (i,j) entry is the mean of diagonal |i-j|."""
    N = Q.shape[0]
    T = np.zeros_like(Q)
    for d in range(N):
        vals = [Q[i, i+d] for i in range(N-d)]
        avg = float(np.mean(vals))
        for i in range(N-d):
            T[i, i+d] = avg
            T[i+d, i] = avg
    return T


def toeplitz_relative_error(Q: np.ndarray) -> float:
    """‖Q - T(Q)‖_F / ‖Q‖_F."""
    T = toeplitz_project(Q)
    return float(np.linalg.norm(Q - T, 'fro') / np.linalg.norm(Q, 'fro'))


# ===========================================================================
# H1 — Linear score identity  (Kalman smoother == precision matrix)
# ===========================================================================

def check_H1_linear_score(K=12, alpha=0.8, sigma0_sq=1.0, sigma_eta_sq=0.36,
                           t=0.4, n_trials=50, rng_seed=0) -> float:
    """Max absolute error between Kalman-score and precision-matrix score over n_trials."""
    rng = np.random.default_rng(rng_seed)
    sigma0 = gaussian_chain_covariance(K, alpha, sigma0_sq, sigma_eta_sq)
    Q_t = gaussian_ou_precision(sigma0, t)
    mu_a = np.zeros(K)          # mu0 = 0
    mu_t = math.exp(-t) * mu_a

    dt = delta_t(t)
    max_err = 0.0
    for _ in range(n_trials):
        x = rng.standard_normal(K)
        # Precision-matrix score
        S_prec = -Q_t @ (x - mu_t)
        # Kalman score
        S_kalm = kalman_score(x, alpha, sigma0_sq, sigma_eta_sq, t, mu0=0.0)
        err = float(np.max(np.abs(S_prec - S_kalm)))
        max_err = max(max_err, err)
    return max_err


# ===========================================================================
# H2 — Tridiagonal precision at t=0
# ===========================================================================

def check_H2_tridiagonal(K=20, alpha=0.75, sigma_eta_sq=0.5) -> float:
    """Off-tri-diagonal elements of Σ_0^{-1} should be exactly zero."""
    sigma0_sq = sigma_eta_sq / (1.0 - alpha**2)    # stationary
    Q0 = gaussian_chain_precision(K, alpha, sigma0_sq, sigma_eta_sq)
    off_tri = Q0.copy()
    # Zero out main diagonal and first off-diagonals
    for i in range(K):
        off_tri[i, i] = 0.0
        if i+1 < K:
            off_tri[i, i+1] = 0.0
            off_tri[i+1, i] = 0.0
    return float(np.max(np.abs(off_tri)))


# ===========================================================================
# H3 — Toeplitz structure of -Σ_t^{-1} in stationary regime
# ===========================================================================

def check_H3_toeplitz(alpha=0.8, sigma_eta_sq=0.36,
                       N_values=(10, 20, 50, 100),
                       t_values=(0.1, 0.5, 1.0, 2.0)) -> dict:
    """Compute Toeplitz relative error of Q_t = Σ_t^{-1} for various N and t."""
    sigma0_sq = sigma_eta_sq / (1.0 - alpha**2)    # stationary prior variance
    results = {}
    for N in N_values:
        for t in t_values:
            sigma0 = gaussian_chain_covariance(N, alpha, sigma0_sq, sigma_eta_sq)
            Q_t = gaussian_ou_precision(sigma0, t)
            err = toeplitz_relative_error(Q_t)
            results[(N, t)] = err
    return results


# ===========================================================================
# H4 — Mean contraction
# ===========================================================================

def check_H4_mean_contraction(K=10, alpha=0.7, mu0=2.3, t=0.6) -> float:
    """Verify μ_t^k = e^{-t} * α^k * μ_0 from two independent methods."""
    et = math.exp(-t)
    # Method A: direct formula from CLAUDE.md
    mu_direct = et * mu0 * np.array([alpha**k for k in range(K)])
    # Method B: iterate the AR(1) prior mean, then apply OU shrinkage
    mu_chain = np.zeros(K)
    mu_chain[0] = mu0
    for k in range(1, K):
        mu_chain[k] = alpha * mu_chain[k-1]
    mu_via_chain = et * mu_chain
    return float(np.max(np.abs(mu_direct - mu_via_chain)))


# ===========================================================================
# H5 — Kalman as EXACT propagator for Gaussian AR(1)
# ===========================================================================

def check_H5_kalman_exact(K=15, alpha=0.8, sigma0_sq=1.0, sigma_eta_sq=0.36,
                           t=0.4, n_trials=50, rng_seed=1) -> dict:
    """
    Two checks:
      (a) Kalman smoother == precision matrix score  [same as H1, larger K]
      (b) Row-to-row ratio of Q_t: measure if interior rows are 'almost' shifts.
    """
    rng = np.random.default_rng(rng_seed)
    sigma0 = gaussian_chain_covariance(K, alpha, sigma0_sq, sigma_eta_sq)
    Q_t = gaussian_ou_precision(sigma0, t)
    mu_t = np.zeros(K)

    dt = delta_t(t)
    max_score_err = 0.0
    for _ in range(n_trials):
        x = rng.standard_normal(K)
        S_prec = -Q_t @ (x - mu_t)
        S_kalm = kalman_score(x, alpha, sigma0_sq, sigma_eta_sq, t, mu0=0.0)
        err = float(np.max(np.abs(S_prec - S_kalm)))
        max_score_err = max(max_score_err, err)

    # Row-shift analysis: compare interior rows of Q_t
    # If Q_t were Toeplitz, row k+1 would be a shift of row k.
    # Measure max relative deviation from a 1-step shift for interior rows.
    row_shift_errs = []
    for k in range(1, K-1):    # interior rows only (exclude boundaries)
        row_k    = Q_t[k, :]
        row_km1  = Q_t[k-1, :]
        # Shift row_{k-1} by 1 position and compare to row_k
        # (Toeplitz: Q[i,j] depends only on i-j, so row k = row k-1 shifted by 1)
        shifted = np.zeros(K)
        shifted[1:] = row_km1[:-1]      # shift right by 1
        # Compute relative norm of deviation from perfect shift
        norm_ref = np.linalg.norm(row_k)
        if norm_ref > 1e-15:
            row_shift_errs.append(np.linalg.norm(row_k - shifted) / norm_ref)

    return {
        "max_score_err": max_score_err,
        "mean_row_shift_err": float(np.mean(row_shift_errs)) if row_shift_errs else np.nan,
        "max_row_shift_err": float(np.max(row_shift_errs)) if row_shift_errs else np.nan,
    }


# ===========================================================================
# K2 — Bivariate characteristic function: M_{12} = -alpha * Delta
# ===========================================================================

def check_K2_cross_term(alpha=0.7, sigma0_sq=1.0, sigma_eta_sq=0.51,
                         t=0.5) -> dict:
    """
    Numerically compute the Gaussian-sector covariance M in the ell-coordinates
    (ell_0, ell_1 = k_1 + alpha*k_2, ell_2 = k_2) for K=2 Laplace AR(1).

    The joint noisy density for K=2 clean frames (x_0, x_1, x_2) is:
        p_t(k_0, k_1, k_2) = Gaussian_factor * Rational_factor
    The Gaussian factor's exponent is -1/2 * (k_0, k_1, k_2)^T * Sigma_t * (k_0, k_1, k_2).

    In ell-coordinates the quadratic form becomes -1/2 * ell^T * M * ell.
    Theory (k2_attempt.py symbolic result) predicts: M_{12} = -alpha * Delta_t.

    We verify numerically by computing the 3x3 noisy covariance of (X_0, X_1, X_2)
    (the x-space covariance Sigma_t), then rotating to ell-space.
    Note: the characteristic-function Gaussian factor involves Sigma_t in x-space,
    and the ell-coordinate rotation changes the covariance to M = R^{-T} Sigma_t R^{-1}
    where R is the rotation from k to ell.
    """
    dt = delta_t(t)
    et = math.exp(-t)

    # Clean chain (K=3 frames: a_0, a_1, a_2) covariance
    Sigma0 = gaussian_chain_covariance(3, alpha, sigma0_sq, sigma_eta_sq)
    # Noisy covariance Sigma_t = exp(-2t)*Sigma0 + Delta_t * I
    Sigma_t = math.exp(-2*t) * Sigma0 + dt * np.eye(3)

    # Rotation from (k_0,k_1,k_2) to (ell_0,ell_1,ell_2):
    #   ell_0 = k_0
    #   ell_1 = k_1 + alpha * k_2
    #   ell_2 = k_2
    # Inverse: k_0=ell_0, k_1=ell_1-alpha*ell_2, k_2=ell_2
    # So k = R_inv @ ell with R_inv = [[1,0,0],[0,1,-alpha],[0,0,1]]
    R_inv = np.array([[1, 0,      0],
                      [0, 1, -alpha],
                      [0, 0,      1]], dtype=float)
    # The quadratic form in k is k^T Sigma_t k = ell^T (R_inv^T Sigma_t R_inv) ell
    M = R_inv.T @ Sigma_t @ R_inv

    predicted_M12 = -alpha * dt
    actual_M12    = M[1, 2]
    err_M12 = abs(actual_M12 - predicted_M12)

    return {
        "M": M,
        "M_12_actual": actual_M12,
        "M_12_predicted": predicted_M12,
        "M_12_error": err_M12,
        "nonzero": abs(actual_M12) > 1e-10,
    }


# ===========================================================================
# Legacy checks re-run (C1–C7) from run_checks.py
# ===========================================================================

def _finite_difference(f, x, y, h=1e-4):
    l00 = f(x, y)
    fxp, fxm = f(x+h, y), f(x-h, y)
    fyp, fym = f(x, y+h), f(x, y-h)
    fpp, fpm = f(x+h, y+h), f(x+h, y-h)
    fmp, fmm = f(x-h, y+h), f(x-h, y-h)
    d0 = (fxp - fxm) / (2*h)
    d1 = (fyp - fym) / (2*h)
    h00 = (fxp - 2*l00 + fxm) / h**2
    h11 = (fyp - 2*l00 + fym) / h**2
    h01 = (fpp - fpm - fmp + fmm) / (4*h**2)
    return d0, d1, h00, h01, h11


def check_C1_gaussian_chain_precision():
    n, alpha = 9, 0.73
    se2 = 1.0 - alpha**2
    s0  = se2 / (1.0 - alpha**2)
    sigma = gaussian_chain_covariance(n, alpha, s0, se2)
    q     = gaussian_chain_precision(n, alpha, s0, se2)
    return float(np.max(np.abs(sigma @ q - np.eye(n))))


def check_C2_neumann_expansion():
    n, alpha, se2, s0 = 11, 0.65, 0.8, 1.3
    sigma0 = gaussian_chain_covariance(n, alpha, s0, se2)
    q0     = gaussian_chain_precision(n, alpha, s0, se2)
    t  = 0.01
    ct = math.exp(2*t) - 1.0
    exact = gaussian_ou_precision(sigma0, t)
    approx = np.zeros_like(q0)
    term   = q0.copy()
    for m in range(20):
        if m == 0:
            term = q0.copy()
        else:
            term = term @ q0
        approx += ((-ct)**m) * term
    approx *= math.exp(2*t)
    return float(np.max(np.abs(exact - approx)))


def check_C3_deterministic_control():
    n, alpha, s0, t = 12, 0.84, 1.7, 0.27
    sigma_t, q_t, _ = deterministic_control_precision(n, alpha, s0, t)
    return float(np.max(np.abs(sigma_t @ q_t - np.eye(n))))


def check_C4_normal_laplace():
    beta, sigma_sq = 0.37, 0.42
    ys = np.linspace(-2.5, 2.5, 9)
    errs = []
    for y in ys:
        val_formula = float(normal_laplace_density(y, beta, sigma_sq))
        def integrand(u, _y=y):
            return math.exp(-abs(u)/beta)/(2*beta) * gaussian_pdf(_y, u, sigma_sq)
        val_quad, _ = integrate.quad(integrand, -np.inf, np.inf,
                                     epsabs=1e-11, epsrel=1e-11, limit=300)
        errs.append(abs(val_formula - val_quad))
    return float(max(errs))


def check_C5_k1_reduction():
    params = {"mu0": 0.15, "alpha": 0.78, "b": 0.45,
              "sigma0_sq": 2*0.45**2/(1-0.78**2)}
    t, x0, x1 = 0.22, 0.31, -0.28
    val_exact = float(laplace_k1_density_exact(x0, x1, t, params, order=120))
    val_quad  = float(laplace_k1_density_direct_quad(x0, x1, t, params))
    return abs(val_exact - val_quad)


def check_C6_C7_k1_score_hessian():
    params = {"mu0": 0.10, "alpha": 0.81, "b": 0.50,
              "sigma0_sq": 2*0.50**2/(1-0.81**2)}
    t, x0, x1 = 0.18, 0.23, -0.17
    exact = laplace_k1_score_and_hessian(x0, x1, t, params, order=120)

    def logp(a, b):
        return math.log(float(laplace_k1_density_exact(a, b, t, params, order=120)))

    fd = _finite_difference(logp, x0, x1, h=2e-4)
    score_err = float(np.max(np.abs(
        np.array([float(exact["score0"]), float(exact["score1"])]) - np.array([fd[0], fd[1]]))))
    hess_err  = float(np.max(np.abs(
        np.array([[float(exact["H00"]), float(exact["H01"])],
                  [float(exact["H01"]), float(exact["H11"])]]) -
        np.array([[-fd[2], -fd[3]], [-fd[3], -fd[4]]]))))
    return score_err, hess_err


# ===========================================================================
# Main
# ===========================================================================

def main():
    lines   = []
    summary = []

    def header(s):
        lines.append("\n" + "=" * 65 + "\n")
        lines.append(f"  {s}\n")
        lines.append("=" * 65 + "\n")

    def row(tag, val, tol, note=""):
        tag_status = _pf(val, tol)
        line = f"  [{tag}] {note:<46s}  err={val:.3e}  (tol={tol:.0e})  {tag_status}\n"
        lines.append(line)
        summary.append((tag, note, val, tol, val <= tol))

    # -------------------------------------------------------------------
    # Legacy checks C1–C7
    # -------------------------------------------------------------------
    header("LEGACY CHECKS (C1–C7) from run_checks.py")

    row("C1", check_C1_gaussian_chain_precision(), TOL_MACHINE,
        "Gaussian chain: Σ · Q = I")
    row("C2", check_C2_neumann_expansion(), TOL_MACHINE,
        "Small-t Neumann expansion of Σ_t^{-1}")
    row("C3", check_C3_deterministic_control(), TOL_MACHINE,
        "Deterministic-control precision: Σ_t · Q_t = I")
    row("C4", check_C4_normal_laplace(), TOL_MACHINE,
        "Normal-Laplace density vs quadrature")
    row("C5", check_C5_k1_reduction(), TOL_GH,
        "Laplace K=1 density vs direct double-integral")

    se, he = check_C6_C7_k1_score_hessian()
    row("C6", se, TOL_FD, "Laplace K=1 score vs 5-point FD")
    row("C7", he, TOL_FD, "Laplace K=1 Hessian vs 5-point FD")

    # -------------------------------------------------------------------
    # H1 — Linear score identity
    # -------------------------------------------------------------------
    header("H1 — Linear joint score (Kalman smoother == precision matrix)")
    for K, alpha, t in [(8, 0.8, 0.3), (15, 0.6, 0.7), (20, 0.9, 1.5)]:
        se2 = 1 - alpha**2
        s0  = se2 / (1 - alpha**2)
        err = check_H1_linear_score(K=K, alpha=alpha,
                                     sigma0_sq=s0, sigma_eta_sq=se2, t=t)
        row("H1", err, TOL_MACHINE,
            f"K={K}, α={alpha}, t={t:.1f}: S_kalman vs S_precision")

    # -------------------------------------------------------------------
    # H2 — Tridiagonal precision
    # -------------------------------------------------------------------
    header("H2 — Σ_0^{-1} tridiagonal for AR(1)")
    for K, alpha in [(10, 0.7), (20, 0.85), (30, 0.95)]:
        se2 = 1 - alpha**2
        err = check_H2_tridiagonal(K=K, alpha=alpha, sigma_eta_sq=se2)
        row("H2", err, TOL_MACHINE,
            f"K={K}, α={alpha}: off-tridiag max |entry|")

    # -------------------------------------------------------------------
    # H3 — Toeplitz structure
    # -------------------------------------------------------------------
    header("H3 — Toeplitz structure of -Σ_t^{-1} [hypothesis]")
    h3_res = check_H3_toeplitz(
        alpha=0.8, sigma_eta_sq=0.36,
        N_values=(10, 20, 50, 100),
        t_values=(0.1, 0.5, 1.0, 2.0),
    )
    lines.append(f"  {'N':>5}  {'t':>5}  {'‖Q-T(Q)‖/‖Q‖':>14}\n")
    lines.append(f"  {'-'*5}  {'-'*5}  {'-'*14}\n")
    for (N, t), err in sorted(h3_res.items()):
        lines.append(f"  {N:5d}  {t:5.1f}  {err:14.6f}\n")
        summary.append(("H3", f"N={N}, t={t}", err, None, None))

    # Also report if the large-N limit is small
    err_100_05 = h3_res.get((100, 0.5), 1.0)
    row("H3", err_100_05, TOL_TOEPLITZ_LARGE,
        "N=100, t=0.5: Toeplitz dev (boundary effects)")

    # -------------------------------------------------------------------
    # H4 — Mean contraction
    # -------------------------------------------------------------------
    header("H4 — Mean contraction  μ_t^k = e^{-t} α^k μ_0  [exact]")
    for K, alpha, mu0, t in [(10, 0.7, 2.0, 0.6), (15, 0.85, -1.5, 1.2)]:
        err = check_H4_mean_contraction(K=K, alpha=alpha, mu0=mu0, t=t)
        row("H4", err, TOL_MACHINE,
            f"K={K}, α={alpha}, μ_0={mu0}, t={t}")

    # -------------------------------------------------------------------
    # H5 — Kalman as exact propagator
    # -------------------------------------------------------------------
    header("H5 — Kalman smoother as EXACT propagator [exact for Gaussian AR(1)]")
    for K, alpha, t in [(15, 0.8, 0.4), (25, 0.7, 1.0)]:
        se2 = 1 - alpha**2
        s0  = se2 / (1 - alpha**2)
        h5  = check_H5_kalman_exact(K=K, alpha=alpha,
                                     sigma0_sq=s0, sigma_eta_sq=se2, t=t)
        row("H5", h5["max_score_err"], TOL_MACHINE,
            f"K={K}, α={alpha}, t={t}: score identity err")
        lines.append(
            f"       → Interior row-shift deviation:"
            f" mean={h5['mean_row_shift_err']:.4f},"
            f" max={h5['max_row_shift_err']:.4f}"
            f"  (0=perfect Toeplitz shift)\n"
        )

    # -------------------------------------------------------------------
    # K2 — M_{12} = -α Δ  (K=2 factorisation failure)
    # -------------------------------------------------------------------
    header("K2 — K=2 bivariate CF: M_{12} = -α Δ_t (factorisation fails)")
    for alpha, t in [(0.7, 0.5), (0.5, 1.0), (0.9, 0.2)]:
        r = check_K2_cross_term(alpha=alpha, sigma0_sq=1.0,
                                 sigma_eta_sq=1-alpha**2, t=t)
        row("K2", r["M_12_error"], TOL_MACHINE,
            f"α={alpha}, t={t}: |M_12 - (-αΔ)|  [M_12={r['M_12_actual']:.4f}]")
        nz = "NON-ZERO (factorisation fails)" if r["nonzero"] else "ZERO (factorises!)"
        lines.append(f"       → M_12 = {r['M_12_actual']:.6f},  predicted = {r['M_12_predicted']:.6f}  → {nz}\n")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    header("SUMMARY")
    n_pass = sum(1 for _, _, val, tol, ok in summary if ok is True)
    n_fail = sum(1 for _, _, val, tol, ok in summary if ok is False)
    n_info = sum(1 for _, _, val, tol, ok in summary if ok is None)
    lines.append(f"  PASS: {n_pass}   FAIL: {n_fail}   INFO (no threshold): {n_info}\n\n")
    for tag, note, val, tol, ok in summary:
        if ok is None:
            continue
        status = PASS if ok else FAIL
        lines.append(f"  {status}  [{tag}] {note}  err={val:.3e}\n")
    lines.append("\n")

    output = "".join(lines)
    print(output)

    # Write plain-text report (strip ANSI colours)
    import re
    plain = re.sub(r'\033\[\d+m', '', output)
    report_path = _HERE / "scores_exact_report.txt"
    report_path.write_text(plain, encoding="utf-8")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
