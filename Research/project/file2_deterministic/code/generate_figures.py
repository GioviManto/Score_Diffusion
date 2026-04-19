import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import eigh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

ALPHA = 0.90
SIGMA0_SQ = 1.0
N = 60


def delta_t(t: float) -> float:
    return 1.0 - np.exp(-2.0 * t)


def build_sigma0_gaussian(n: int, alpha: float, sigma0_sq: float, sigma_eta_sq: float) -> np.ndarray:
    sigma_sq = np.zeros(n)
    sigma_sq[0] = sigma0_sq
    for k in range(1, n):
        sigma_sq[k] = alpha**2 * sigma_sq[k - 1] + sigma_eta_sq
    sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sigma[i, j] = alpha ** abs(i - j) * sigma_sq[min(i, j)]
    return sigma


def build_sigma0_det(n: int, alpha: float, sigma0_sq: float) -> np.ndarray:
    v = alpha ** np.arange(n)
    return sigma0_sq * np.outer(v, v)


def build_sigma_t(sigma0: np.ndarray, t: float) -> np.ndarray:
    return np.exp(-2.0 * t) * sigma0 + delta_t(t) * np.eye(sigma0.shape[0])


def build_qt_det(n: int, alpha: float, sigma0_sq: float, t: float) -> np.ndarray:
    beta = np.exp(-2.0 * t)
    dt = delta_t(t)
    v = alpha ** np.arange(n)
    norm2 = float(np.dot(v, v))
    coeff = beta * sigma0_sq / (dt * (dt + beta * sigma0_sq * norm2))
    return np.eye(n) / dt - coeff * np.outer(v, v)


def sym_heat(ax, mat, title: str, ticks: bool = False):
    vmax = np.max(np.abs(mat))
    vmax = max(vmax, 1e-12)
    im = ax.imshow(mat, cmap='RdBu_r', norm=TwoSlopeNorm(0.0, vmin=-vmax, vmax=vmax))
    ax.set_title(title, fontsize=11)
    if ticks:
        ax.set_xlabel('column index')
        ax.set_ylabel('row index')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    return im


sigma0_det = build_sigma0_det(N, ALPHA, SIGMA0_SQ)

# Figure 1: rank-one covariance and eigenvalues
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
im = sym_heat(axes[0], sigma0_det, r'deterministic clean covariance $\Sigma_0^{\mathrm{det}}$', ticks=True)
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
vals = eigh(sigma0_det, eigvals_only=True)
vals = vals[::-1]
axes[1].semilogy(np.arange(1, N + 1), np.maximum(vals, 1e-16), marker='o', markersize=2.3, linewidth=1.0)
axes[1].set_xlabel('eigenvalue index')
axes[1].set_ylabel('eigenvalue magnitude')
axes[1].set_title('One large eigenvalue, the rest collapse to zero')
axes[1].grid(alpha=0.25, which='both')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig01_rankone_covariance.pdf'))
plt.close(fig)

# Figure 2: deterministic Q_t heatmaps
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
for ax, t in zip(axes, [0.05, 0.30, 1.00]):
    qt = build_qt_det(N, ALPHA, SIGMA0_SQ, t)
    im = sym_heat(ax, qt, rf'$Q_t^{{\mathrm{{det}}}}$, $t={t:.2f}$')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig02_deterministic_qt.pdf'))
plt.close(fig)

# Figure 3: deterministic vs stochastic at the same t
sigma_eta_cmp = 0.15
sigma0_gauss_cmp = build_sigma0_gaussian(N, ALPHA, SIGMA0_SQ, sigma_eta_cmp)
qt_gauss_cmp = np.linalg.inv(build_sigma_t(sigma0_gauss_cmp, 0.30))
qt_det_cmp = build_qt_det(N, ALPHA, SIGMA0_SQ, 0.30)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
im0 = sym_heat(axes[0], qt_det_cmp, r'deterministic $Q_t$, $t=0.30$', ticks=True)
im1 = sym_heat(axes[1], qt_gauss_cmp, rf'Gaussian AR(1) $Q_t$, $t=0.30$, $\sigma_\eta^2={sigma_eta_cmp}$', ticks=True)
for ax, im in zip(axes, [im0, im1]):
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig03_compare_gaussian_vs_det.pdf'))
plt.close(fig)

# Figure 4: limit comparison and convergence
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
mid = N // 2
ks = np.arange(N)
qt_det = build_qt_det(N, ALPHA, SIGMA0_SQ, 0.40)
axes[0].semilogy(ks, np.abs(qt_det[mid, :]), marker='o', markersize=2.5, linewidth=1.2, label='deterministic')
for sigma_eta_sq in [0.20, 0.05, 0.01]:
    sigma0_g = build_sigma0_gaussian(N, ALPHA, SIGMA0_SQ, sigma_eta_sq)
    qt_g = np.linalg.inv(build_sigma_t(sigma0_g, 0.40))
    axes[0].semilogy(ks, np.abs(qt_g[mid, :]), marker='o', markersize=2.0, linewidth=1.0, label=rf'Gaussian $\sigma_\eta^2={sigma_eta_sq}$')
axes[0].set_xlabel('column index')
axes[0].set_ylabel(r'$|Q_{t,mk}|$ with $m=N/2$')
axes[0].set_title('Row profile: global latent coupling vs decaying Markov fill-in')
axes[0].grid(alpha=0.25, which='both')
axes[0].legend(frameon=False, fontsize=8)

sigma_vals = np.logspace(-3, -0.3, 18)
errs = []
for sigma_eta_sq in sigma_vals:
    sigma0_g = build_sigma0_gaussian(N, ALPHA, SIGMA0_SQ, sigma_eta_sq)
    qt_g = np.linalg.inv(build_sigma_t(sigma0_g, 0.40))
    rel = np.linalg.norm(qt_g - qt_det, ord='fro') / np.linalg.norm(qt_det, ord='fro')
    errs.append(rel)
axes[1].loglog(sigma_vals, errs, marker='o', markersize=3.0, linewidth=1.1)
axes[1].set_xlabel(r'innovation variance $\sigma_\eta^2$')
axes[1].set_ylabel(r'relative Frobenius error to $Q_t^{\mathrm{det}}$')
axes[1].set_title('Regularized Gaussian model converges to the deterministic benchmark')
axes[1].grid(alpha=0.25, which='both')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig04_limit_convergence.pdf'))
plt.close(fig)

with open(os.path.join(FIG, 'summary.txt'), 'w', encoding='utf-8') as fh:
    fh.write(f'alpha={ALPHA}\n')
    fh.write(f'sigma0_sq={SIGMA0_SQ}\n')
    fh.write('Generated figures for deterministic-limit study.\n')
