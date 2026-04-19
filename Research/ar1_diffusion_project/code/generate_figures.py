"""Generate publication-quality figures for the AR(1) diffusion manuscript."""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import eigh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

K = 59
N = K + 1
alpha = 0.9
sigma_0_sq = 1.0
sigma_eta_sq = 1.0 - alpha**2

def delta_t(t):
    return 1.0 - np.exp(-2.0 * np.asarray(t))

def build_sigma0(N, alpha, sigma_0_sq, sigma_eta_sq):
    sigma_sq = np.zeros(N)
    sigma_sq[0] = sigma_0_sq
    for k in range(1, N):
        sigma_sq[k] = alpha**2 * sigma_sq[k - 1] + sigma_eta_sq
    S = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            m = min(j, k)
            S[j, k] = alpha**abs(j - k) * sigma_sq[m]
    return S

def build_precision0(N, alpha, sigma_0_sq, sigma_eta_sq):
    P = np.zeros((N, N))
    P[0, 0] = 1.0 / sigma_0_sq + alpha**2 / sigma_eta_sq
    for k in range(1, N - 1):
        P[k, k] = (1.0 + alpha**2) / sigma_eta_sq
    P[N - 1, N - 1] = 1.0 / sigma_eta_sq
    for k in range(N - 1):
        P[k, k + 1] = -alpha / sigma_eta_sq
        P[k + 1, k] = -alpha / sigma_eta_sq
    return P

def build_sigma_t(S0, t):
    return np.exp(-2.0 * t) * S0 + float(delta_t(t)) * np.eye(S0.shape[0])

def set_clean(ax):
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

Sigma0 = build_sigma0(N, alpha, sigma_0_sq, sigma_eta_sq)
Prec0 = build_precision0(N, alpha, sigma_0_sq, sigma_eta_sq)
eigvals0, eigvecs0 = eigh(Sigma0)
order = eigvals0.argsort()[::-1]
eigvals0 = eigvals0[order]
U = eigvecs0[:, order]

# Figure 1
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
for ax in axes:
    set_clean(ax)
vmax0 = np.max(np.abs(Sigma0))
im0 = axes[0].imshow(Sigma0, cmap="RdBu_r", norm=TwoSlopeNorm(0, vmin=-vmax0, vmax=vmax0))
axes[0].set_title(r"Clean covariance $\Sigma_0$ (dense)", fontsize=12)
axes[0].set_xlabel(r"frame index $k$", fontsize=11)
axes[0].set_ylabel(r"frame index $j$", fontsize=11)
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
vmax1 = np.max(np.abs(Prec0))
im1 = axes[1].imshow(Prec0, cmap="RdBu_r", norm=TwoSlopeNorm(0, vmin=-vmax1, vmax=vmax1))
axes[1].set_title(r"Clean precision $\Sigma_0^{-1}$ (tridiagonal)", fontsize=12)
axes[1].set_xlabel(r"frame index $k$", fontsize=11)
axes[1].set_ylabel(r"frame index $j$", fontsize=11)
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
fig.savefig(os.path.join(FIG_DIR, "fig01_sigma0_vs_precision0.pdf"), bbox_inches="tight")
plt.close(fig)

# Figure 2
t_values = [0.01, 0.30, 0.70, 1.50]
fig, axes = plt.subplots(2, len(t_values), figsize=(13.6, 6.4), constrained_layout=True)
for row in axes:
    for ax in row:
        set_clean(ax)
for col, t in enumerate(t_values):
    St = build_sigma_t(Sigma0, t)
    St_inv = np.linalg.inv(St)
    vmax_cov = np.max(np.abs(St))
    vmax_prec = np.max(np.abs(St_inv))
    imt = axes[0, col].imshow(St, cmap="RdBu_r", norm=TwoSlopeNorm(0, vmin=-vmax_cov, vmax=vmax_cov))
    axes[0, col].set_title(rf"$t={t:.2f}$", fontsize=11)
    axes[0, col].set_xlabel(r"$k$", fontsize=10)
    if col == 0:
        axes[0, col].set_ylabel(r"$j$", fontsize=10)
    fig.colorbar(imt, ax=axes[0, col], fraction=0.046, pad=0.03)
    imb = axes[1, col].imshow(St_inv, cmap="RdBu_r", norm=TwoSlopeNorm(0, vmin=-vmax_prec, vmax=vmax_prec))
    axes[1, col].set_xlabel(r"$k$", fontsize=10)
    if col == 0:
        axes[1, col].set_ylabel(r"$j$", fontsize=10)
    fig.colorbar(imb, ax=axes[1, col], fraction=0.046, pad=0.03)
axes[0, 0].annotate(r"$\Sigma_t$", xy=(0.02, 1.18), xycoords="axes fraction", fontsize=12, fontweight="bold")
axes[1, 0].annotate(r"$\Sigma_t^{-1}$", xy=(0.02, 1.18), xycoords="axes fraction", fontsize=12, fontweight="bold")
fig.savefig(os.path.join(FIG_DIR, "fig02_sigma_t_panel.pdf"), bbox_inches="tight")
plt.close(fig)

# Figure 3
t_grid = np.linspace(0.001, 4.0, 300)
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)
for ax in axes:
    set_clean(ax)
for m in range(N):
    lam_t = np.exp(-2 * t_grid) * eigvals0[m] + delta_t(t_grid)
    axes[0].plot(t_grid, lam_t, color=plt.cm.viridis(m / N), lw=0.8)
axes[0].axhline(1.0, color="crimson", lw=1.2, ls="--", label=r"noise floor $=1$")
axes[0].set_xlabel(r"diffusion time $t$", fontsize=11)
axes[0].set_ylabel(r"$\lambda_i(t)$", fontsize=11)
axes[0].set_title(r"Eigenvalues of $\Sigma_t$", fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25)
for m in range(N):
    lam_inv = 1.0 / (np.exp(-2 * t_grid) * eigvals0[m] + delta_t(t_grid))
    axes[1].plot(t_grid, lam_inv, color=plt.cm.viridis(m / N), lw=0.8)
axes[1].axhline(1.0, color="crimson", lw=1.2, ls="--", label=r"noise precision $=1$")
axes[1].set_xlabel(r"diffusion time $t$", fontsize=11)
axes[1].set_ylabel(r"$\lambda_i(t)^{-1}$", fontsize=11)
axes[1].set_title(r"Eigenvalues of $\Sigma_t^{-1}$", fontsize=12)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.25)
fig.savefig(os.path.join(FIG_DIR, "fig03_eigenvalue_trajectories.pdf"), bbox_inches="tight")
plt.close(fig)

# Figure 4
t_range = np.linspace(0.001, 4.0, 250)
tridiag_frac = []
cond_numbers = []
for t in t_range:
    St = build_sigma_t(Sigma0, t)
    St_inv = np.linalg.inv(St)
    total = np.sum(np.abs(St_inv))
    tridiag = np.sum(np.abs(np.diag(St_inv)))
    tridiag += np.sum(np.abs(np.diag(St_inv, 1))) + np.sum(np.abs(np.diag(St_inv, -1)))
    tridiag_frac.append(tridiag / total)
    cond_numbers.append(np.linalg.cond(St))
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)
for ax in axes:
    set_clean(ax)
axes[0].plot(t_range, tridiag_frac, color="navy", lw=2.0)
axes[0].set_xlabel(r"diffusion time $t$", fontsize=11)
axes[0].set_ylabel(r"tridiagonal fraction of $\|\Sigma_t^{-1}\|_1$", fontsize=11)
axes[0].set_title(r"Loss of tridiagonal concentration", fontsize=12)
axes[0].grid(alpha=0.25)
axes[1].semilogy(t_range, cond_numbers, color="darkgreen", lw=2.0)
axes[1].set_xlabel(r"diffusion time $t$", fontsize=11)
axes[1].set_ylabel(r"$\kappa(\Sigma_t)$", fontsize=11)
axes[1].set_title(r"Condition number of $\Sigma_t$", fontsize=12)
axes[1].grid(alpha=0.25)
fig.savefig(os.path.join(FIG_DIR, "fig04_tridiag_fraction_condition.pdf"), bbox_inches="tight")
plt.close(fig)

# Figure 5
mid = N // 2
fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
set_clean(ax)
for t in [0.01, 0.20, 0.50, 1.00, 2.00, 3.00]:
    St_inv = np.linalg.inv(build_sigma_t(Sigma0, t))
    row = St_inv[mid, :]
    ax.plot(np.arange(N), row, marker="o", ms=2.8, lw=1.2, label=rf"$t={t:.2f}$")
ax.axvline(mid, color="gray", ls="--", lw=1.0)
ax.set_xlabel(r"frame index $k$", fontsize=11)
ax.set_ylabel(rf"$(\Sigma_t^{{-1}})_{{{mid},k}}$", fontsize=11)
ax.set_title(r"Coupling profile of a central row of $\Sigma_t^{-1}$", fontsize=12)
ax.grid(alpha=0.25)
ax.legend(fontsize=8, ncol=2)
fig.savefig(os.path.join(FIG_DIR, "fig05_precision_row_profile.pdf"), bbox_inches="tight")
plt.close(fig)

# Figure 6
fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.7), constrained_layout=True)
for ax in axes:
    set_clean(ax)
for col, t in enumerate([0.01, 0.50, 1.50, 3.00]):
    St_inv = np.linalg.inv(build_sigma_t(Sigma0, t))
    rotated = U.T @ St_inv @ U
    im = axes[col].imshow(np.abs(rotated), cmap="hot_r", vmin=0.0)
    axes[col].set_title(rf"$|U^\top\Sigma_t^{{-1}}U|$, $t={t:.2f}$", fontsize=10)
    axes[col].set_xlabel(r"mode index", fontsize=10)
    if col == 0:
        axes[col].set_ylabel(r"mode index", fontsize=10)
    fig.colorbar(im, ax=axes[col], fraction=0.046, pad=0.03)
fig.savefig(os.path.join(FIG_DIR, "fig06_precision_eigenbasis.pdf"), bbox_inches="tight")
plt.close(fig)

# Figure 7
fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
set_clean(ax)
for m in range(5):
    ax.plot(np.arange(N), U[:, m], lw=1.5, label=rf"$i={m+1}$, $\lambda_i={eigvals0[m]:.3f}$")
ax.set_xlabel(r"frame index $k$", fontsize=11)
ax.set_ylabel(r"$u_i(k)$", fontsize=11)
ax.set_title(r"Leading eigenvectors of $\Sigma_0$", fontsize=12)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
fig.savefig(os.path.join(FIG_DIR, "fig07_eigenvectors.pdf"), bbox_inches="tight")
plt.close(fig)
