"""
Generate publication-quality figures for the manuscript.

Every figure answers a specific mathematical question; the caption in the
manuscript states that question.  Figures are saved as PDF under
../manuscript/figures/ with consistent naming.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from laplace_k1 import (
    build_params, h_t, psi_t, kappa_t, residual, score_t, hessian_field,
    gaussian_benchmark,
)
from observables import (
    kappa_center_closed, kappa_tail_closed, kappa_G, gap_vs_G,
    band_fwhm, var_residual, fourth_cumulant_residual, gaussianity_ratio,
)


OUT = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures")
os.makedirs(OUT, exist_ok=True)

# Matplotlib global style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ---------------------------------------------------------------------------
# Figure 1: residual density, score, curvature profile -- at several t
# ---------------------------------------------------------------------------

def fig_residual_profiles():
    alpha, b, sigma0 = 0.8, 1.0, 1.0
    ts = [0.05, 0.3, 1.0, 3.0]
    r = np.linspace(-6, 6, 801)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for t in ts:
        p = build_params(alpha, b, sigma0, 0.0, t)
        axes[0].plot(r, h_t(r, p), label=f"t={t}")
        axes[1].plot(r, psi_t(r, p), label=f"t={t}")
        axes[2].plot(r, kappa_t(r, p), label=f"t={t}")
    axes[0].set_title(r"$h_t(r)$ residual density")
    axes[0].set_xlabel("r"); axes[0].set_ylabel("density")
    axes[1].set_title(r"$\psi_t(r) = (\log h_t)'(r)$ residual score")
    axes[1].set_xlabel("r"); axes[1].set_ylabel(r"$\psi_t$")
    axes[2].set_title(r"$\kappa_t(r) = -(\log h_t)''(r)$ residual curvature")
    axes[2].set_xlabel("r"); axes[2].set_ylabel(r"$\kappa_t$")
    for ax in axes:
        ax.legend(loc="best")
    fig.savefig(os.path.join(OUT, "fig_residual_profiles.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: curvature heatmap in (x0, x1) space -- reproduces user's Fig. 4
# and explicitly overlays the residual band r = 0.
# ---------------------------------------------------------------------------

def fig_curvature_heatmaps():
    alpha, b, sigma0, t = 0.8, 1.0, 1.0, 0.5
    p = build_params(alpha, b, sigma0, 0.0, t)
    x = np.linspace(-3, 3, 201)
    X0, X1 = np.meshgrid(x, x)
    H = hessian_field(X0, X1, p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].pcolormesh(x, x, H[..., 1, 1], shading="auto",
                              cmap="viridis")
    axes[0].set_title(r"$H_{11}(x_0, x_1) = \kappa_t(r)$")
    im1 = axes[1].pcolormesh(x, x, H[..., 0, 1], shading="auto",
                              cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title(r"$H_{01}(x_0, x_1) = -\rho\,\kappa_t(r)$")
    # Overlay r = 0 line: x1 = nu + rho x0
    line_x0 = np.array([-3, 3])
    line_x1 = p.nu + p.rho * line_x0
    for ax in axes:
        ax.plot(line_x0, line_x1, 'w--', lw=1.5, alpha=0.8)
        ax.set_xlabel(r"$x_0$"); ax.set_ylabel(r"$x_1$")
        ax.set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])
    fig.colorbar(im1, ax=axes[1])
    fig.suptitle(r"Two-dimensional curvature field with residual ray $r=0$"
                 " overlaid", fontsize=11)
    fig.savefig(os.path.join(OUT, "fig_curvature_heatmaps.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: scalar observables vs diffusion time
# ---------------------------------------------------------------------------

def fig_scalar_observables_vs_t():
    alpha, b, sigma0 = 0.8, 1.0, 1.0
    ts = np.geomspace(1e-2, 8.0, 60)
    kc, kG, gap, varR, ex_kurt = [], [], [], [], []
    for t in ts:
        p = build_params(alpha, b, sigma0, 0.0, t)
        kc.append(kappa_center_closed(p))
        kG.append(kappa_G(p))
        gap.append(gap_vs_G(p))
        varR.append(var_residual(p))
        ex_kurt.append(gaussianity_ratio(p))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].loglog(ts, kc, label=r"$\kappa_t(0)$")
    axes[0].loglog(ts, kG, ls="--", label=r"$\kappa_t^{(G)} = 1/\tau_G^2$")
    axes[0].set_xlabel("t"); axes[0].set_title("Center vs Gaussian curvature")
    axes[0].legend()
    axes[1].loglog(ts, np.abs(gap),
                   label=r"$|\kappa_t(0) - \kappa_t^{(G)}|$")
    axes[1].set_xlabel("t"); axes[1].set_title("Center-benchmark gap")
    axes[1].legend()
    axes[2].loglog(ts, ex_kurt, label="excess kurtosis of $R_t$")
    axes[2].axhline(3.0, ls=":", color="gray", label="Laplace (t=0)")
    axes[2].axhline(0.0, ls=":", color="k")
    axes[2].set_xlabel("t"); axes[2].set_title("Non-Gaussianity persistence")
    axes[2].legend()
    fig.savefig(os.path.join(OUT, "fig_scalar_observables_vs_t.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: eigen-decomposition of H_t(x) along the residual ray
# Confirms the rank-1 structure of the 'bond matrix' M_rho
# ---------------------------------------------------------------------------

def fig_hessian_eigenvalues_along_ray():
    alpha, b, sigma0 = 0.8, 1.0, 1.0
    ts = [0.05, 0.3, 1.0, 3.0]
    fig, axes = plt.subplots(1, len(ts), figsize=(14, 3.2), sharey=True)
    # Sample along the residual ray: x0 varies, x1 = nu + rho x0 + r for r=0
    r_grid = np.linspace(-5, 5, 401)
    x0_fixed = 0.0
    for ax, t in zip(axes, ts):
        p = build_params(alpha, b, sigma0, 0.0, t)
        x1 = r_grid + p.nu + p.rho * x0_fixed
        H = hessian_field(np.full_like(r_grid, x0_fixed), x1, p)
        # Eigenvalues of 2x2 Hessian
        eigs = np.linalg.eigvalsh(H)
        ax.plot(r_grid, eigs[:, 1], label=r"$\lambda_+$")
        ax.plot(r_grid, eigs[:, 0], label=r"$\lambda_-$")
        # Gaussian benchmark (horizontal reference for comparison)
        _, _, Qt = gaussian_benchmark(np.array([x0_fixed]), np.array([p.nu]), p)
        eG = np.linalg.eigvalsh(Qt)
        ax.axhline(eG[0], ls=":", color="gray")
        ax.axhline(eG[1], ls=":", color="gray")
        ax.set_title(f"t={t}")
        ax.set_xlabel("r")
        ax.set_yscale("symlog", linthresh=1e-2)
    axes[0].set_ylabel(r"eigenvalues of $H_t$")
    axes[0].legend()
    fig.suptitle("Eigenvalues of the curvature field along the residual ray "
                 r"(dotted: Gaussian benchmark $Q_t$ eigenvalues)", fontsize=11)
    fig.savefig(os.path.join(OUT, "fig_hessian_eigenvalues_along_ray.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: FWHM band-width of kappa_t vs t  (new quantitative geometry)
# ---------------------------------------------------------------------------

def fig_bandwidth_vs_t():
    alpha, b, sigma0 = 0.8, 1.0, 1.0
    ts = np.geomspace(5e-3, 8.0, 60)
    fwhm = []; tau_vals = []; btilde_vals = []
    for t in ts:
        p = build_params(alpha, b, sigma0, 0.0, t)
        fwhm.append(band_fwhm(p))
        tau_vals.append(p.tau)
        btilde_vals.append(p.btilde)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.loglog(ts, fwhm, "o-", label="FWHM of $\\kappa_t(r)$")
    ax.loglog(ts, tau_vals, "--", label=r"$\tau(t)$")
    ax.loglog(ts, btilde_vals, "--", label=r"$\tilde b(t) = \beta b$")
    ax.set_xlabel("t")
    ax.set_title("Band width vs. diffusion time")
    ax.legend()
    fig.savefig(os.path.join(OUT, "fig_bandwidth_vs_t.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Phase portrait kappa_t(0) in (alpha, t) plane
# ---------------------------------------------------------------------------

def fig_phase_portrait():
    alphas = np.linspace(-0.95, 0.95, 60)
    ts = np.geomspace(5e-3, 8.0, 60)
    K = np.zeros((len(ts), len(alphas)))
    for i, t in enumerate(ts):
        for j, a in enumerate(alphas):
            p = build_params(a, 1.0, 1.0, 0.0, t)
            K[i, j] = kappa_center_closed(p)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    im = ax.pcolormesh(alphas, ts, np.log10(K + 1e-10),
                       shading="auto", cmap="viridis")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("t")
    ax.set_title(r"$\log_{10}\,\kappa_t(0)$: phase portrait in $(\alpha, t)$")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}\,\kappa_t(0)$")
    fig.savefig(os.path.join(OUT, "fig_phase_portrait.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: monte carlo validation of h_t(r)
# ---------------------------------------------------------------------------

def fig_validation_mc():
    from validation import run_monte_carlo_score_check
    alpha, b, sigma0, t = 0.8, 1.0, 1.0, 0.5
    p = build_params(alpha, b, sigma0, 0.0, t)
    r = np.linspace(-4, 4, 81)
    res = run_monte_carlo_score_check(p, r, N=500_000)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(r, res["h_closed"], "-", lw=2, label="closed-form h_t(r)")
    ax.plot(r, res["h_mc"], ".", ms=5, label=f"MC KDE (N=5e5)")
    ax.set_xlabel("r"); ax.set_ylabel(r"$h_t(r)$")
    ax.set_title(f"Monte Carlo validation: max rel err = {res['max_rel_err_kde']:.2e}")
    ax.legend()
    fig.savefig(os.path.join(OUT, "fig_validation_mc.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    print("Generating figures...")
    fig_residual_profiles();         print("  -> residual_profiles")
    fig_curvature_heatmaps();        print("  -> curvature_heatmaps")
    fig_scalar_observables_vs_t();   print("  -> scalar_observables_vs_t")
    fig_hessian_eigenvalues_along_ray(); print("  -> hessian_eigenvalues_along_ray")
    fig_bandwidth_vs_t();            print("  -> bandwidth_vs_t")
    fig_phase_portrait();            print("  -> phase_portrait")
    fig_validation_mc();             print("  -> validation_mc")
    print("done.")
