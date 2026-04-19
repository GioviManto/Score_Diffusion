"""Numerical checks for the corrected Gaussian and Laplace AR(1) formulas."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
from scipy import integrate

from ar1_diffusion_utils import (
    delta_t,
    deterministic_control_precision,
    gaussian_chain_covariance,
    gaussian_chain_precision,
    gaussian_ou_precision,
    gaussian_pdf,
    laplace_k1_density_direct_quad,
    laplace_k1_density_exact,
    laplace_k1_score_and_hessian,
    normal_laplace_density,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "code" / "check_report.txt"


def finite_difference_scalar(f, x, y, h=1e-4):
    l00 = f(x, y)
    fxp = f(x + h, y)
    fxm = f(x - h, y)
    fyp = f(x, y + h)
    fym = f(x, y - h)
    fpp = f(x + h, y + h)
    fpm = f(x + h, y - h)
    fmp = f(x - h, y + h)
    fmm = f(x - h, y - h)

    d0 = (fxp - fxm) / (2.0 * h)
    d1 = (fyp - fym) / (2.0 * h)
    h00 = (fxp - 2.0 * l00 + fxm) / (h ** 2)
    h11 = (fyp - 2.0 * l00 + fym) / (h ** 2)
    h01 = (fpp - fpm - fmp + fmm) / (4.0 * h ** 2)
    return d0, d1, h00, h01, h11



def check_gaussian_chain_precision() -> float:
    n = 9
    alpha = 0.73
    sigma_eta_sq = 1.0 - alpha ** 2
    sigma0_sq = sigma_eta_sq / (1.0 - alpha ** 2)
    sigma = gaussian_chain_covariance(n, alpha, sigma0_sq, sigma_eta_sq)
    q = gaussian_chain_precision(n, alpha, sigma0_sq, sigma_eta_sq)
    return float(np.max(np.abs(sigma @ q - np.eye(n))))



def check_neumann_expansion() -> float:
    n = 11
    alpha = 0.65
    sigma_eta_sq = 0.8
    sigma0_sq = 1.3
    sigma0 = gaussian_chain_covariance(n, alpha, sigma0_sq, sigma_eta_sq)
    q0 = gaussian_chain_precision(n, alpha, sigma0_sq, sigma_eta_sq)
    t = 0.01
    ct = math.exp(2.0 * t) - 1.0
    exact = gaussian_ou_precision(sigma0, t)
    approx = np.zeros_like(q0)
    term = q0.copy()
    for m in range(20):
        if m == 0:
            term = q0.copy()
        else:
            term = term @ q0
        approx += ((-ct) ** m) * term
    approx *= math.exp(2.0 * t)
    return float(np.max(np.abs(exact - approx)))



def check_deterministic_control() -> float:
    n = 12
    alpha = 0.84
    sigma0_sq = 1.7
    t = 0.27
    sigma_t, q_t, _ = deterministic_control_precision(n, alpha, sigma0_sq, t)
    return float(np.max(np.abs(sigma_t @ q_t - np.eye(n))))



def check_normal_laplace_density() -> float:
    beta = 0.37
    sigma_sq = 0.42
    ys = np.linspace(-2.5, 2.5, 9)
    errs = []
    for y in ys:
        val_formula = float(normal_laplace_density(y, beta, sigma_sq))

        def integrand(u):
            return math.exp(-abs(u) / beta) / (2.0 * beta) * gaussian_pdf(y, u, sigma_sq)

        val_quad, _ = integrate.quad(integrand, -np.inf, np.inf, epsabs=1e-11, epsrel=1e-11, limit=300)
        errs.append(abs(val_formula - val_quad))
    return float(max(errs))



def check_k1_reduction() -> float:
    params = {
        "mu0": 0.15,
        "alpha": 0.78,
        "b": 0.45,
        "sigma0_sq": 2.0 * 0.45 ** 2 / (1.0 - 0.78 ** 2),
    }
    t = 0.22
    x0 = 0.31
    x1 = -0.28
    val_exact = float(laplace_k1_density_exact(x0, x1, t, params, order=120))
    val_quad = float(laplace_k1_density_direct_quad(x0, x1, t, params))
    return abs(val_exact - val_quad)



def check_k1_score_hessian() -> tuple[float, float]:
    params = {
        "mu0": 0.10,
        "alpha": 0.81,
        "b": 0.50,
        "sigma0_sq": 2.0 * 0.50 ** 2 / (1.0 - 0.81 ** 2),
    }
    t = 0.18
    x0 = 0.23
    x1 = -0.17

    exact = laplace_k1_score_and_hessian(x0, x1, t, params, order=120)

    def logp(a, b):
        return math.log(float(laplace_k1_density_exact(a, b, t, params, order=120)))

    fd = finite_difference_scalar(logp, x0, x1, h=2.0e-4)
    score_fd = np.array([fd[0], fd[1]], dtype=float)
    hess_fd = np.array([[-fd[2], -fd[3]], [-fd[3], -fd[4]]], dtype=float)
    score_exact = np.array([float(exact["score0"]), float(exact["score1"])], dtype=float)
    hess_exact = np.array(
        [
            [float(exact["H00"]), float(exact["H01"])],
            [float(exact["H01"]), float(exact["H11"])],
        ],
        dtype=float,
    )
    score_err = float(np.max(np.abs(score_exact - score_fd)))
    hess_err = float(np.max(np.abs(hess_exact - hess_fd)))
    return score_err, hess_err



def main() -> None:
    results = []
    results.append(("Gaussian chain inverse check", check_gaussian_chain_precision()))
    results.append(("Small-time Neumann expansion check", check_neumann_expansion()))
    results.append(("Deterministic control inverse check", check_deterministic_control()))
    results.append(("Normal-Laplace convolution check", check_normal_laplace_density()))
    results.append(("Laplace K=1 reduction vs direct double integral", check_k1_reduction()))
    score_err, hess_err = check_k1_score_hessian()
    results.append(("Laplace K=1 score vs finite differences", score_err))
    results.append(("Laplace K=1 Hessian vs finite differences", hess_err))

    lines = []
    lines.append("Corrected Gaussian and Laplace AR(1) checks\n")
    lines.append("=" * 54 + "\n")
    for name, value in results:
        lines.append(f"{name}: {value:.6e}\n")

    REPORT.write_text("".join(lines), encoding="utf-8")
    print("".join(lines))


if __name__ == "__main__":
    main()
