"""
Validation of the closed-form Laplace-K1 formulas against independent numerics.

Three independent cross-checks, each stressing a different aspect:

  (Q) Quadrature of the convolution definition of h_t(r).
  (F) Fourier inversion of hhat_t(ell) = exp(-tau^2 ell^2 /2) / (1 + btilde^2 ell^2).
  (M) Monte Carlo sampling of (X0, X1) and kernel-density / gradient estimates.

For every experiment we print the maximum relative error.  The manuscript
only claims 'exact' when (Q) and (F) agree to ~1e-12 at all tested points.
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import laplace, norm

from laplace_k1 import (
    build_params, h_t, log_h_t, psi_t, kappa_t, residual,
    log_p_t, score_t, hessian_field,
)
from observables import (
    kappa_center_closed, kappa_center_numerical,
    fourth_cumulant_residual, var_residual,
)


# ---------- (Q) quadrature reference ----------------------------------------

def h_t_quadrature(r, p, tol=1e-12):
    """h_t(r) = int_R (1/(2 btilde)) exp(-|u|/btilde) * N(r-u;0,tau^2) du (33)."""
    def integrand(u, r_value):
        return (1.0 / (2 * p.btilde)) * np.exp(-abs(u) / p.btilde) \
               * norm.pdf(r_value - u, loc=0.0, scale=p.tau)
    out = np.zeros_like(np.atleast_1d(r), dtype=float)
    for i, rv in enumerate(np.atleast_1d(r)):
        val, _ = quad(integrand, -np.inf, np.inf, args=(rv,),
                      epsabs=tol, epsrel=tol, limit=200)
        out[i] = val
    return out if np.ndim(r) else out[0]


# ---------- (F) Fourier inversion reference ---------------------------------

def h_t_fourier(r, p, L=200.0, N=2**16):
    """Inverse Fourier transform of hhat_t(ell).

    hhat_t(ell) = exp(-tau^2 ell^2 /2) / (1 + btilde^2 ell^2).   (105)

    We evaluate h_t(r) = (1 / 2 pi) int cos(ell r) hhat_t(ell) dell.
    Uses a trapezoidal rule on [-L, L]; both L and N must be large enough
    that both the Gaussian decay and rational decay are captured.
    """
    ell = np.linspace(-L, L, N)
    dell = ell[1] - ell[0]
    hhat = np.exp(-0.5 * p.tau2 * ell**2) / (1.0 + p.btilde**2 * ell**2)
    # broadcast over r
    r_arr = np.atleast_1d(r)
    cos_mat = np.cos(np.outer(r_arr, ell))
    integral = (cos_mat * hhat).sum(axis=1) * dell / (2 * np.pi)
    return integral if np.ndim(r) else integral[0]


# ---------- (M) Monte Carlo reference ---------------------------------------

def sample_joint(p, N=100_000, rng=None):
    """Draw N samples from p_t(x0, x1) by simulating the generative model."""
    rng = np.random.default_rng(rng)
    A0 = rng.normal(p.mu0, p.sigma0, size=N)
    eps = laplace.rvs(loc=0.0, scale=p.b, size=N, random_state=rng)
    A1 = p.alpha * A0 + eps
    X0 = p.beta * A0 + np.sqrt(p.Delta) * rng.standard_normal(N)
    X1 = p.beta * A1 + np.sqrt(p.Delta) * rng.standard_normal(N)
    return X0, X1


# ---------------------------------------------------------------------------
# Top-level validation experiments
# ---------------------------------------------------------------------------

def run_density_validation(r_grid, p):
    closed = h_t(r_grid, p)
    quad_ref = h_t_quadrature(r_grid, p)
    fou_ref = h_t_fourier(r_grid, p)
    err_q = np.max(np.abs(closed - quad_ref) / np.abs(quad_ref))
    err_f = np.max(np.abs(closed - fou_ref) / np.abs(fou_ref))
    return {"max_rel_err_quad": err_q, "max_rel_err_fourier": err_f,
            "closed": closed, "quad": quad_ref, "fourier": fou_ref}


def run_score_validation(r_grid, p, h=1e-5):
    psi_closed = psi_t(r_grid, p)
    # Finite-difference reference via log h_t
    lh_plus = log_h_t(r_grid + h, p)
    lh_minus = log_h_t(r_grid - h, p)
    psi_fd = (lh_plus - lh_minus) / (2 * h)
    err = np.max(np.abs(psi_closed - psi_fd))  # absolute: small-score regions
    return {"max_abs_err": err, "closed": psi_closed, "fd": psi_fd}


def run_curvature_validation(r_grid, p, h=1e-3):
    k_closed = kappa_t(r_grid, p)
    # 5-point finite difference of -log h_t
    lh = np.array([log_h_t(r_grid + k * h, p) for k in (-2, -1, 0, 1, 2)])
    d2 = (-lh[0] + 16 * lh[1] - 30 * lh[2] + 16 * lh[3] - lh[4]) / (12 * h**2)
    k_fd = -d2
    err = np.max(np.abs(k_closed - k_fd))
    return {"max_abs_err": err, "closed": k_closed, "fd": k_fd}


def run_kappa_center_closed_vs_numeric(param_grid):
    """Sweep parameters and compare closed-form kappa_t(0) with 5-point FD."""
    rows = []
    for (alpha, b, sigma0, t) in param_grid:
        p = build_params(alpha, b, sigma0, 0.0, t)
        k_closed = kappa_center_closed(p)
        k_num = kappa_center_numerical(p)
        rel = abs(k_closed - k_num) / max(abs(k_num), 1e-16)
        rows.append((alpha, b, sigma0, t, k_closed, k_num, rel))
    return rows


def run_monte_carlo_score_check(p, grid_r, N=1_000_000, bw=None, rng=0):
    """Validate psi_t(r) via kernel-density estimation of h_t and log-diff."""
    rng = np.random.default_rng(rng)
    eps = laplace.rvs(loc=0.0, scale=p.b, size=N, random_state=rng)
    G = rng.normal(0.0, p.tau, size=N)
    R = p.beta * eps + G  # samples of R_t
    if bw is None:
        bw = 1.06 * np.std(R) * N ** (-1 / 5)
    # Gaussian kernel estimate of h_t(r) on grid; chunk over BOTH dims
    r_arr = np.atleast_1d(grid_r)
    h_mc = np.zeros_like(r_arr, dtype=float)
    r_chunk = 32
    N_chunk = 50_000
    for rs in range(0, r_arr.size, r_chunk):
        rc = r_arr[rs:rs + r_chunk]
        accum = np.zeros_like(rc, dtype=float)
        for ns in range(0, N, N_chunk):
            R_chunk = R[ns:ns + N_chunk]
            diffs = rc[:, None] - R_chunk[None, :]
            accum += norm.pdf(diffs / bw).sum(axis=1) / bw
        h_mc[rs:rs + r_chunk] = accum / N
    h_closed = h_t(r_arr, p)
    err = np.max(np.abs(h_mc - h_closed) / h_closed)
    return {"max_rel_err_kde": err, "bw": bw, "h_mc": h_mc, "h_closed": h_closed}


if __name__ == "__main__":
    # Exercise on a representative midrange setting
    p = build_params(alpha=0.8, b=1.0, sigma0=1.0, mu0=0.0, t=0.5)
    r = np.linspace(-5, 5, 121)

    print("=== Density validation ===")
    d = run_density_validation(r, p)
    print(f"  max rel err (quad) : {d['max_rel_err_quad']:.3e}")
    print(f"  max rel err (FFT ) : {d['max_rel_err_fourier']:.3e}")

    print("=== Score validation (vs 2nd-order FD) ===")
    s = run_score_validation(r, p)
    print(f"  max abs err        : {s['max_abs_err']:.3e}")

    print("=== Curvature validation (vs 5-point FD) ===")
    c = run_curvature_validation(r, p)
    print(f"  max abs err        : {c['max_abs_err']:.3e}")

    print("=== Center curvature closed-form vs numerics ===")
    grid = [(0.8, 1.0, 1.0, t) for t in [0.1, 0.5, 1.0, 2.0, 4.0]]
    rows = run_kappa_center_closed_vs_numeric(grid)
    print(f"  {'alpha':>6} {'b':>4} {'sigma':>5} {'t':>6} "
          f"{'closed':>12} {'5-pt FD':>12} {'rel err':>10}")
    for row in rows:
        a0, bb, s0, tt, kc, kn, rel = row
        print(f"  {a0:6.2f} {bb:4.2f} {s0:5.2f} {tt:6.2f} "
              f"{kc:12.6e} {kn:12.6e} {rel:10.2e}")
