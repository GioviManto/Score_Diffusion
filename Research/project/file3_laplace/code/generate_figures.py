import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.special import logsumexp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

ALPHA = 0.80
SIGMA0_SQ = 1.0
B = 0.45
N = 4               # K=3, so four frames
T = 0.45
SHRINK = np.exp(-T)
BETA = np.exp(-2.0 * T)
DELTA = 1.0 - BETA
NSAMPLES = 35000
RNG = np.random.default_rng(1234)


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


def sample_clean(n_samples: int) -> np.ndarray:
    a0 = RNG.normal(0.0, np.sqrt(SIGMA0_SQ), size=n_samples)
    eta = RNG.laplace(0.0, B, size=(n_samples, N - 1))
    a = np.zeros((n_samples, N))
    a[:, 0] = a0
    for k in range(N - 1):
        a[:, k + 1] = ALPHA * a[:, k] + eta[:, k]
    return a


def posterior_mean_batch(xs: np.ndarray, samples: np.ndarray, batch: int = 32) -> np.ndarray:
    out = np.zeros_like(xs)
    for start in range(0, xs.shape[0], batch):
        stop = min(xs.shape[0], start + batch)
        xb = xs[start:stop]                                    # (B, N)
        diff = xb[:, None, :] - SHRINK * samples[None, :, :]   # (B, M, N)
        logw = -0.5 / DELTA * np.sum(diff * diff, axis=2)      # (B, M)
        norm = logsumexp(logw, axis=1, keepdims=True)
        w = np.exp(logw - norm)
        out[start:stop] = w @ samples
    return out


def posterior_cov(x: np.ndarray, samples: np.ndarray) -> np.ndarray:
    diff = x[None, :] - SHRINK * samples
    logw = -0.5 / DELTA * np.sum(diff * diff, axis=1)
    logw = logw - logsumexp(logw)
    w = np.exp(logw)
    mean = np.sum(w[:, None] * samples, axis=0)
    centered = samples - mean[None, :]
    cov = centered.T @ (w[:, None] * centered)
    return mean, cov


def score_from_mean(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return -(x - SHRINK * mean) / DELTA


def hessian_from_cov(cov: np.ndarray) -> np.ndarray:
    return -np.eye(N) / DELTA + BETA * cov / (DELTA ** 2)


def offdiag_band_share(mat: np.ndarray, r: int = 1) -> float:
    num = 0.0
    den = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i == j:
                continue
            val = abs(mat[i, j])
            den += val
            if abs(i - j) <= r:
                num += val
    return num / den if den > 0 else 1.0


def sym_heat(ax, mat, title: str):
    vmax = max(np.max(np.abs(mat)), 1e-12)
    im = ax.imshow(mat, cmap='RdBu_r', norm=TwoSlopeNorm(0.0, vmin=-vmax, vmax=vmax))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('column index')
    ax.set_ylabel('row index')
    return im


samples = sample_clean(NSAMPLES)

# Gaussian comparator with matched second moments.
sigma_eta_sq_match = 2.0 * B**2
sigma0_gauss = build_sigma0_gaussian(N, ALPHA, SIGMA0_SQ, sigma_eta_sq_match)
qt_gauss = np.linalg.inv(BETA * sigma0_gauss + DELTA * np.eye(N))

# Figure 1: innovation densities (matched variance)
xs = np.linspace(-3.0, 3.0, 500)
lap = 0.5 / B * np.exp(-np.abs(xs) / B)
gsig = np.sqrt(sigma_eta_sq_match)
gauss = np.exp(-xs**2 / (2.0 * gsig**2)) / np.sqrt(2.0 * np.pi * gsig**2)
fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(xs, lap, linewidth=2.0, label=rf'Laplace$(0,{B:.2f})$')
ax.plot(xs, gauss, linewidth=2.0, linestyle='--', label=rf'Gaussian var $={sigma_eta_sq_match:.3f}$')
ax.set_xlabel('innovation value')
ax.set_ylabel('density')
ax.set_title('Matched-variance innovations: Laplace is sharper and heavier-tailed')
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig01_innovation_compare.pdf'))
plt.close(fig)

# Figure 2: score nonlinearity along a one-dimensional slice
mid = 1
u = np.linspace(-3.0, 3.0, 161)
Xslice = np.zeros((len(u), N))
Xslice[:, mid] = u
means = posterior_mean_batch(Xslice, samples, batch=24)
scores = -(Xslice - SHRINK * means) / DELTA
s_lap = scores[:, mid]
s_gauss = -(Xslice @ qt_gauss.T)[:, mid]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(u, s_lap, linewidth=2.0, label='Laplace AR(1)')
ax.plot(u, s_gauss, linewidth=2.0, linestyle='--', label='Gaussian AR(1), same covariance')
ax.set_xlabel(r'slice parameter $u$ in $x=u e_2$')
ax.set_ylabel(r'central score component $S_2(x,t)$')
ax.set_title(r'Non-Gaussian score is nonlinear; Gaussian score is affine')
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig02_score_slice.pdf'))
plt.close(fig)

# Figure 3: Hessian heatmaps at two points, compared with Gaussian constant precision
v = ALPHA ** np.arange(N)
direction = v / np.linalg.norm(v)
state0 = np.zeros(N)
state1 = 2.0 * direction
_, cov0 = posterior_cov(state0, samples)
_, cov1 = posterior_cov(state1, samples)
H0 = hessian_from_cov(cov0)
H1 = hessian_from_cov(cov1)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
im0 = sym_heat(axes[0], qt_gauss, r'Gaussian benchmark $H_t(x)=Q_t$')
im1 = sym_heat(axes[1], -H0, r'Laplace: $-H_t(x)$ at $x=0$')
im2 = sym_heat(axes[2], -H1, r'Laplace: $-H_t(x)$ at $x=2v/\|v\|$')
for ax, im in zip(axes, [im0, im1, im2]):
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig03_hessian_heatmaps.pdf'))
plt.close(fig)

# Figure 4: nearest-neighbour share of the Hessian off-diagonal mass along a state-space line
u2 = np.linspace(-2.8, 2.8, 81)
shares = []
for amp in u2:
    _, cov = posterior_cov(amp * direction, samples)
    H = hessian_from_cov(cov)
    shares.append(offdiag_band_share(H, r=1))
share_gauss = offdiag_band_share(qt_gauss, r=1)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(u2, shares, linewidth=2.0, label='Laplace Hessian band share')
ax.axhline(share_gauss, linestyle='--', linewidth=2.0, label='Gaussian constant band share')
ax.set_xlabel(r'amplitude $u$ in $x=u v/\|v\|$')
ax.set_ylabel(r'$R_1(x)=\frac{\sum_{|i-j|=1}|H_{ij}(x)|}{\sum_{i\ne j}|H_{ij}(x)|}$')
ax.set_title('A locality proxy becomes state-dependent in the Laplace model')
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig04_hessian_bandshare.pdf'))
plt.close(fig)

with open(os.path.join(FIG, 'summary.txt'), 'w', encoding='utf-8') as fh:
    fh.write(f'alpha={ALPHA}\n')
    fh.write(f'b={B}\n')
    fh.write(f't={T}\n')
    fh.write('Generated figures for Laplace AR(1) study.\n')
