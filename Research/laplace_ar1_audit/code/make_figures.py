"""Generate the figures used in the Laplace-focused presentation note."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from ar1_diffusion_utils import (
    delta_t,
    gaussian_k1_score_and_hessian,
    gaussian_pdf,
    laplace_k1_score_and_hessian,
    normal_laplace_density,
)


ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
    }
)


def savefig(fig, name: str) -> None:
    fig.savefig(FIGDIR / name, bbox_inches="tight")
    plt.close(fig)


params = {
    "mu0": 0.0,
    "alpha": 0.80,
    "b": 0.50,
    "sigma0_sq": 2.0 * 0.50 ** 2 / (1.0 - 0.80 ** 2),
}


def figure_innovations_and_kernel() -> None:
    b = params["b"]
    t = 0.20
    dt = delta_t(t)
    u = np.linspace(-3.2, 3.2, 800)

    laplace_clean = np.exp(-np.abs(u) / b) / (2.0 * b)
    gaussian_clean = gaussian_pdf(u, 0.0, 2.0 * b ** 2)

    beta_t = np.exp(-t) * b
    laplace_noisy = normal_laplace_density(u, beta_t, dt)
    gaussian_noisy = gaussian_pdf(u, 0.0, np.exp(-2.0 * t) * 2.0 * b ** 2 + dt)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    axes[0].plot(u, laplace_clean, lw=2.2, label="Laplace innovation")
    axes[0].plot(u, gaussian_clean, lw=2.0, ls="--", label="matched Gaussian")
    axes[0].set_xlabel("innovation value")
    axes[0].set_ylabel("density")
    axes[0].set_title("Clean innovation benchmark")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(u, laplace_noisy, lw=2.2, label="exact noised Laplace kernel")
    axes[1].plot(u, gaussian_noisy, lw=2.0, ls="--", label="matched Gaussian after OU")
    axes[1].set_xlabel("innovation channel after OU")
    axes[1].set_ylabel("density")
    axes[1].set_title(r"Noised innovation at $t=0.20$")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("The cusp is smoothed immediately, but finite-time non-Gaussianity remains", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig01_innovations_and_kernel.pdf")



def figure_score_slices() -> None:
    t = 0.20
    s = np.linspace(-3.0, 3.0, 401)
    lap = laplace_k1_score_and_hessian(s, s, t, params, order=80)
    g0, g1, _ = gaussian_k1_score_and_hessian(s, s, t, params)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)

    axes[0].plot(s, lap["score1"], lw=2.2, label="Laplace exact")
    axes[0].plot(s, g1, lw=2.0, ls="--", label="matched Gaussian")
    axes[0].set_xlabel(r"diagonal slice $s$ with $(x_0,x_1)=(s,s)$")
    axes[0].set_ylabel(r"$\partial_{x_1}\log p_t(s,s)$")
    axes[0].set_title("Second-coordinate score")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(s, lap["score0"], lw=2.2, label="Laplace exact")
    axes[1].plot(s, g0, lw=2.0, ls="--", label="matched Gaussian")
    axes[1].set_xlabel(r"diagonal slice $s$ with $(x_0,x_1)=(s,s)$")
    axes[1].set_ylabel(r"$\partial_{x_0}\log p_t(s,s)$")
    axes[1].set_title("First-coordinate score")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("The exact Laplace score is nonlinear; the Gaussian score is affine", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig02_score_slices.pdf")



def figure_hessian_diagonal() -> None:
    t = 0.20
    s = np.linspace(-3.0, 3.0, 401)
    lap = laplace_k1_score_and_hessian(s, s, t, params, order=80)
    _, _, q = gaussian_k1_score_and_hessian(s, s, t, params)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)

    axes[0].plot(s, lap["H11"], lw=2.2, label="Laplace exact")
    axes[0].axhline(q[1, 1], lw=2.0, ls="--", label="matched Gaussian")
    axes[0].set_xlabel(r"diagonal slice $s$ with $(x_0,x_1)=(s,s)$")
    axes[0].set_ylabel(r"$\mathcal{H}_{11}(s,s)$")
    axes[0].set_title("Diagonal curvature entry")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(s, lap["H01"], lw=2.2, label="Laplace exact")
    axes[1].axhline(q[0, 1], lw=2.0, ls="--", label="matched Gaussian")
    axes[1].set_xlabel(r"diagonal slice $s$ with $(x_0,x_1)=(s,s)$")
    axes[1].set_ylabel(r"$\mathcal{H}_{01}(s,s)$")
    axes[1].set_title("Off-diagonal curvature entry")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("For Laplace innovations the effective precision depends on position", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig03_hessian_diagonal.pdf")



def figure_hessian_heatmaps() -> None:
    t = 0.20
    grid = np.linspace(-2.6, 2.6, 101)
    x0, x1 = np.meshgrid(grid, grid, indexing="xy")
    lap = laplace_k1_score_and_hessian(x0, x1, t, params, order=60)
    h11 = lap["H11"]
    h01 = lap["H01"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    im0 = axes[0].imshow(
        h11,
        extent=[grid[0], grid[-1], grid[0], grid[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set_xlabel(r"$x_0$")
    axes[0].set_ylabel(r"$x_1$")
    axes[0].set_title(r"$\mathcal{H}_{11}(x_0,x_1)$")
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.88)
    cb0.ax.set_ylabel("curvature")

    vmax = np.max(np.abs(h01))
    im1 = axes[1].imshow(
        h01,
        extent=[grid[0], grid[-1], grid[0], grid[-1]],
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
    )
    axes[1].set_xlabel(r"$x_0$")
    axes[1].set_ylabel(r"$x_1$")
    axes[1].set_title(r"$\mathcal{H}_{01}(x_0,x_1)$")
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.88)
    cb1.ax.set_ylabel("curvature")

    fig.suptitle("The Gaussian constant precision is replaced by a full x-dependent curvature field", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig04_hessian_heatmaps.pdf")



def main() -> None:
    figure_innovations_and_kernel()
    figure_score_slices()
    figure_hessian_diagonal()
    figure_hessian_heatmaps()
    print(f"Saved figures to {FIGDIR}")


if __name__ == "__main__":
    main()
