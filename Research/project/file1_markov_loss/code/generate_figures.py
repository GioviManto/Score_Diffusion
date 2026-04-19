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

ALPHA = 0.88
SIGMA_ETA_SQ = 1.0 - ALPHA**2
SIGMA0_SQ = SIGMA_ETA_SQ / (1.0 - ALPHA**2)  # stationary => 1


def delta_t(t: float) -> float:
    return 1.0 - np.exp(-2.0 * t)


def build_sigma0(n: int, alpha: float, sigma0_sq: float, sigma_eta_sq: float) -> np.ndarray:
    sigma_sq = np.zeros(n)
    sigma_sq[0] = sigma0_sq
    for k in range(1, n):
        sigma_sq[k] = alpha**2 * sigma_sq[k - 1] + sigma_eta_sq
    sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            m = min(i, j)
            sigma[i, j] = alpha ** abs(i - j) * sigma_sq[m]
    return sigma


def build_precision0(n: int, alpha: float, sigma0_sq: float, sigma_eta_sq: float) -> np.ndarray:
    q = np.zeros((n, n))
    q[0, 0] = 1.0 / sigma0_sq + alpha**2 / sigma_eta_sq
    for k in range(1, n - 1):
        q[k, k] = (1.0 + alpha**2) / sigma_eta_sq
    q[n - 1, n - 1] = 1.0 / sigma_eta_sq
    off = -alpha / sigma_eta_sq
    for k in range(n - 1):
        q[k, k + 1] = off
        q[k + 1, k] = off
    return q


def build_sigma_t(sigma0: np.ndarray, t: float) -> np.ndarray:
    beta = np.exp(-2.0 * t)
    return beta * sigma0 + delta_t(t) * np.eye(sigma0.shape[0])


def build_precision_t(sigma0: np.ndarray, t: float) -> np.ndarray:
    return np.linalg.inv(build_sigma_t(sigma0, t))


def distance_profile(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    out = []
    for d in range(1, n):
        vals = np.abs(np.diag(mat, d))
        out.append(vals.mean() if len(vals) else 0.0)
    return np.array(out)


def offdiag_band_share(mat: np.ndarray, r: int) -> float:
    n = mat.shape[0]
    total = 0.0
    band = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = abs(mat[i, j])
            total += val
            if abs(i - j) <= r:
                band += val
    return band / total if total > 0 else 1.0


def total_offdiag_l1(mat: np.ndarray) -> float:
    return np.sum(np.abs(mat)) - np.sum(np.abs(np.diag(mat)))


def sym_heat(ax, mat, title: str, show_ticks: bool = True):
    vmax = np.max(np.abs(mat))
    vmax = max(vmax, 1e-12)
    im = ax.imshow(mat, cmap='RdBu_r', norm=TwoSlopeNorm(0.0, vmin=-vmax, vmax=vmax))
    ax.set_title(title, fontsize=11)
    if show_ticks:
        ax.set_xlabel('column index')
        ax.set_ylabel('row index')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    return im


# Figure 1: covariance vs precision at t=0
n_large = 60
sigma0_large = build_sigma0(n_large, ALPHA, SIGMA0_SQ, SIGMA_ETA_SQ)
q0_large = build_precision0(n_large, ALPHA, SIGMA0_SQ, SIGMA_ETA_SQ)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
im0 = sym_heat(axes[0], sigma0_large, r'clean covariance $\Sigma_0$')
im1 = sym_heat(axes[1], q0_large, r'clean precision $Q_0=\Sigma_0^{-1}$')
for ax, im in zip(axes, [im0, im1]):
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig01_cov_vs_prec.pdf'))
plt.close(fig)

# Figure 2: small-N fill-in heatmaps
n_small = 5
sigma0_small = build_sigma0(n_small, ALPHA, SIGMA0_SQ, SIGMA_ETA_SQ)
small_times = [0.0, 0.08, 0.30, 1.00]
fig, axes = plt.subplots(1, len(small_times), figsize=(3.3 * len(small_times), 3.1))
for ax, t in zip(axes, small_times):
    qt = build_precision_t(sigma0_small, t) if t > 0 else build_precision0(n_small, ALPHA, SIGMA0_SQ, SIGMA_ETA_SQ)
    im = sym_heat(ax, qt, rf'$Q_t$, $t={t:.2f}$', show_ticks=False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig02_small_fillin.pdf'))
plt.close(fig)

# Figure 3: large-N fill-in heatmaps
large_times = [0.0, 0.08, 0.30, 1.00]
fig, axes = plt.subplots(1, len(large_times), figsize=(3.3 * len(large_times), 3.1))
for ax, t in zip(axes, large_times):
    qt = build_precision_t(sigma0_large, t) if t > 0 else q0_large
    im = sym_heat(ax, qt, rf'$Q_t$, $t={t:.2f}$', show_ticks=False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig03_large_fillin.pdf'))
plt.close(fig)

# Figure 4: band shares and distance profiles
trange = np.linspace(0.001, 2.8, 160)
share1 = []
share2 = []
share4 = []
off_l1 = []
for t in trange:
    qt = build_precision_t(sigma0_large, t)
    share1.append(offdiag_band_share(qt, 1))
    share2.append(offdiag_band_share(qt, 2))
    share4.append(offdiag_band_share(qt, 4))
    off_l1.append(total_offdiag_l1(qt))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
axes[0].plot(trange, share1, label=r'$R_1(t)$')
axes[0].plot(trange, share2, label=r'$R_2(t)$')
axes[0].plot(trange, share4, label=r'$R_4(t)$')
axes[0].set_xlabel(r'diffusion time $t$')
axes[0].set_ylabel('fraction of off-diagonal L1 mass')
axes[0].set_title('How much coupling stays near the diagonal?')
axes[0].set_ylim(0.0, 1.02)
axes[0].grid(alpha=0.25)
axes[0].legend(frameon=False)

sel_times = [0.05, 0.20, 0.60, 1.50]
d = np.arange(1, n_large)
for t in sel_times:
    prof = distance_profile(build_precision_t(sigma0_large, t))
    axes[1].semilogy(d, prof, marker='o', markersize=2.8, linewidth=1.2, label=rf'$t={t:.2f}$')
axes[1].set_xlabel(r'distance $|i-j|$')
axes[1].set_ylabel(r'average $|Q_{t,ij}|$')
axes[1].set_title('Off-diagonal decay of the noisy precision')
axes[1].grid(alpha=0.25, which='both')
axes[1].legend(frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig04_metrics.pdf'))
plt.close(fig)

# Figure 5: spectral flattening
lam, _ = eigh(sigma0_large)
lam = lam[::-1]
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
axes[0].plot(np.arange(1, n_large + 1), lam, marker='o', markersize=2.5, linewidth=1.2)
axes[0].set_xlabel('mode index (sorted)')
axes[0].set_ylabel(r'eigenvalue of $\Sigma_0$')
axes[0].set_title('Clean covariance spectrum')
axes[0].grid(alpha=0.25)

for t in [0.0, 0.08, 0.30, 1.00, 2.00]:
    beta = np.exp(-2.0 * t)
    wt = 1.0 / (beta * lam + delta_t(t)) if t > 0 else 1.0 / lam
    axes[1].plot(np.arange(1, n_large + 1), wt, marker='o', markersize=2.0, linewidth=1.2, label=rf'$t={t:.2f}$')
axes[1].set_xlabel('mode index (sorted)')
axes[1].set_ylabel(r'mode weight in $Q_t$')
axes[1].set_title(r'Weights $1/(e^{-2t}\lambda_m + \Delta_t)$')
axes[1].grid(alpha=0.25)
axes[1].legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig05_spectrum.pdf'))
plt.close(fig)

# Save a small text summary used by the note if desired.
with open(os.path.join(FIG, 'summary.txt'), 'w', encoding='utf-8') as fh:
    fh.write(f'alpha={ALPHA}\n')
    fh.write(f'sigma_eta_sq={SIGMA_ETA_SQ}\n')
    fh.write('Generated figures for Gaussian AR(1) Markov-loss study.\n')
