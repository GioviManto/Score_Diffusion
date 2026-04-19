import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Core linear-algebra setup
# ------------------------
def delta_t(t):
    return 1.0 - np.exp(-2.0 * t)


def build_sigma0(N, alpha, sigma0_sq, sigma_eta_sq):
    sigma_sq = np.zeros(N)
    sigma_sq[0] = sigma0_sq
    for k in range(1, N):
        sigma_sq[k] = alpha**2 * sigma_sq[k - 1] + sigma_eta_sq
    S = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            m = min(j, k)
            S[j, k] = alpha**abs(j - k) * sigma_sq[m]
    return S


def build_precision0(N, alpha, sigma0_sq, sigma_eta_sq):
    Q = np.zeros((N, N))
    Q[0, 0] = 1.0 / sigma0_sq + alpha**2 / sigma_eta_sq
    for k in range(1, N - 1):
        Q[k, k] = (1.0 + alpha**2) / sigma_eta_sq
    Q[N - 1, N - 1] = 1.0 / sigma_eta_sq
    for k in range(N - 1):
        Q[k, k + 1] = -alpha / sigma_eta_sq
        Q[k + 1, k] = -alpha / sigma_eta_sq
    return Q


def build_sigma_t(Sigma0, t):
    return np.exp(-2.0 * t) * Sigma0 + delta_t(t) * np.eye(Sigma0.shape[0])


def build_precision_t(Sigma0, t):
    return np.linalg.inv(build_sigma_t(Sigma0, t))


def deterministic_sigma0(N, alpha, sigma0_sq):
    v = alpha ** np.arange(N)
    return sigma0_sq * np.outer(v, v), v


def deterministic_sigma_t(N, alpha, sigma0_sq, t):
    S0, v = deterministic_sigma0(N, alpha, sigma0_sq)
    St = np.exp(-2.0 * t) * S0 + delta_t(t) * np.eye(N)
    c = np.exp(-2.0 * t) * sigma0_sq
    denom = 1.0 + (c / delta_t(t)) * np.dot(v, v)
    Qt = (1.0 / delta_t(t)) * np.eye(N) - (c / (delta_t(t) ** 2 * denom)) * np.outer(v, v)
    return St, Qt, v


def band_mask(N, r):
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return np.abs(i - j) <= r


def band_mass_fraction(Q, r):
    mask = band_mask(Q.shape[0], r)
    num = np.sum(np.abs(Q[mask]))
    den = np.sum(np.abs(Q))
    return num / den


def effective_bandwidth(Q, rho=0.95):
    N = Q.shape[0]
    for r in range(N):
        if band_mass_fraction(Q, r) >= rho:
            return r
    return N - 1


def mean_interaction_range(Q):
    N = Q.shape[0]
    idx = np.arange(N)
    vals = []
    for k in range(N):
        w = np.abs(Q[k, :])
        vals.append(np.sum(np.abs(idx - k) * w) / np.sum(w))
    return float(np.mean(vals))


def posterior_gain_matrix(Sigma0, t):
    Q = build_precision_t(Sigma0, t)
    return np.exp(-t) * Sigma0 @ Q


def avg_abs_by_distance(M):
    N = M.shape[0]
    out = np.zeros(N)
    for d in range(N):
        vals = []
        for i in range(N - d):
            vals.append(abs(M[i, i + d]))
            if d > 0:
                vals.append(abs(M[i + d, i]))
        out[d] = np.mean(vals)
    return out


def score_correlation(Q, i, j):
    return Q[i, j] / np.sqrt(Q[i, i] * Q[j, j])


# Parameters
N = 60
alpha = 0.90
sigma0_sq = 1.0
sigma_eta_sq = 1.0 - alpha**2
Sigma0 = build_sigma0(N, alpha, sigma0_sq, sigma_eta_sq)
Q0 = build_precision0(N, alpha, sigma0_sq, sigma_eta_sq)
lam0, U = eigh(Sigma0)
lam0 = lam0[::-1]
U = U[:, ::-1]

# ------------------------
# Figure 1: covariance vs precision at t=0
# ------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
im0 = axes[0].imshow(Sigma0, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0.0))
axes[0].set_title(r'$\Sigma_0$ (dense covariance)')
axes[0].set_xlabel('frame index $k$')
axes[0].set_ylabel('frame index $j$')
plt.colorbar(im0, ax=axes[0], shrink=0.82)

im1 = axes[1].imshow(Q0, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0.0))
axes[1].set_title(r'$\Sigma_0^{-1}$ (tridiagonal precision)')
axes[1].set_xlabel('frame index $k$')
axes[1].set_ylabel('frame index $j$')
plt.colorbar(im1, ax=axes[1], shrink=0.82)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig01_sigma0_vs_precision0.pdf', bbox_inches='tight')
plt.close(fig)

# ------------------------
# Figure 2: time evolution in linear color scale
# ------------------------
t_values = [0.01, 0.15, 0.4, 1.0, 2.5]
fig, axes = plt.subplots(2, len(t_values), figsize=(3.4 * len(t_values), 6.7))
for c, t in enumerate(t_values):
    St = build_sigma_t(Sigma0, t)
    Qt = np.linalg.inv(St)
    vmax_s = np.max(np.abs(St))
    vmax_q = np.max(np.abs(Qt))
    ims = axes[0, c].imshow(St, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax_s, vmax=vmax_s))
    imq = axes[1, c].imshow(Qt, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax_q, vmax=vmax_q))
    axes[0, c].set_title(rf'$\Sigma_t$, $t={t}$')
    axes[1, c].set_title(rf'$\Sigma_t^{{-1}}$, $t={t}$')
    axes[1, c].set_xlabel('frame index')
    if c == 0:
        axes[0, c].set_ylabel('frame index')
        axes[1, c].set_ylabel('frame index')
    plt.colorbar(ims, ax=axes[0, c], shrink=0.72)
    plt.colorbar(imq, ax=axes[1, c], shrink=0.72)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig02_sigma_t_panel.pdf', bbox_inches='tight')
plt.close(fig)

# ------------------------
# Figure 3: eigenvalue trajectories
# ------------------------
t_grid = np.linspace(0.0, 4.0, 300)
fig, axes = plt.subplots(1, 2, figsize=(13.8, 5))
for m in range(N):
    lam_t = np.exp(-2.0 * t_grid) * lam0[m] + (1.0 - np.exp(-2.0 * t_grid))
    axes[0].plot(t_grid, lam_t, lw=0.85)
    axes[1].plot(t_grid, 1.0 / lam_t, lw=0.85)
axes[0].axhline(1.0, ls='--', lw=1.2)
axes[1].axhline(1.0, ls='--', lw=1.2)
axes[0].set_title(r'eigenvalues of $\Sigma_t$')
axes[1].set_title(r'eigenvalues of $\Sigma_t^{-1}$')
for ax in axes:
    ax.set_xlabel(r'diffusion time $t$')
    ax.grid(alpha=0.3)
axes[0].set_ylabel(r'$\lambda_m(t)$')
axes[1].set_ylabel(r'$\lambda_m(t)^{-1}$')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig03_eigenvalue_trajectories.pdf', bbox_inches='tight')
plt.close(fig)

# ------------------------
# Figure 4: tridiagonal fraction and conditioning
# ------------------------
t_grid = np.linspace(1e-3, 4.0, 240)
tri = []
cond = []
fullness = []
for t in t_grid:
    Qt = build_precision_t(Sigma0, t)
    tri.append(band_mass_fraction(Qt, 1))
    cond.append(np.linalg.cond(build_sigma_t(Sigma0, t)))
    fullness.append(1.0 - band_mass_fraction(Qt, 1))
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
axes[0].plot(t_grid, tri, lw=2, label='tridiagonal mass fraction')
axes[0].plot(t_grid, fullness, lw=1.8, label='off-band mass')
axes[0].set_xlabel(r'diffusion time $t$')
axes[0].set_ylabel('fraction of entrywise $\ell_1$ mass')
axes[0].set_ylim(0, 1.02)
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=9)
axes[1].semilogy(t_grid, cond, lw=2)
axes[1].set_xlabel(r'diffusion time $t$')
axes[1].set_ylabel(r'$\kappa(\Sigma_t)$')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig04_tridiag_fraction_condition.pdf', bbox_inches='tight')
plt.close(fig)

# ------------------------
# Figure 5: row profile of the precision
# ------------------------
mid = N // 2
fig, ax = plt.subplots(figsize=(9.2, 5.1))
for t in [0.01, 0.10, 0.30, 0.70, 1.50, 3.00]:
    Qt = build_precision_t(Sigma0, t)
    ax.plot(np.arange(N), Qt[mid, :], marker='o', ms=2.8, lw=1.2, label=rf'$t={t}$')
ax.axvline(mid, ls='--', alpha=0.5)
ax.axvline(mid - 1, ls=':', alpha=0.3)
ax.axvline(mid + 1, ls=':', alpha=0.3)
ax.set_xlabel('frame index $j$')
ax.set_ylabel(rf'$(\Sigma_t^{{-1}})_{{{mid},j}}$')
ax.set_title(r'row profile of the noisy precision')
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, ncol=2)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig05_precision_row_profile.pdf', bbox_inches='tight')
plt.close(fig)

# ------------------------
# Figure 6: precision in eigenbasis
# ------------------------
fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
for c, t in enumerate([0.01, 0.30, 1.00, 2.50]):
    Qt = build_precision_t(Sigma0, t)
    R = U.T @ Qt @ U
    im = axes[c].imshow(np.abs(R), cmap='hot_r')
    axes[c].set_title(rf'$|U^\top \Sigma_t^{{-1}} U|$, $t={t}$')
    axes[c].set_xlabel('mode')
    if c == 0:
        axes[c].set_ylabel('mode')
    plt.colorbar(im, ax=axes[c], shrink=0.78)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig06_precision_eigenbasis.pdf', bbox_inches='tight')
plt.close(fig)

# ------------------------
# Figure 7: a few eigenvectors
# ------------------------
fig, ax = plt.subplots(figsize=(9.2, 5.0))
for m in range(6):
    ax.plot(np.arange(N), U[:, m], lw=1.4, label=rf'$m={m}$')
ax.set_xlabel('frame index')
ax.set_ylabel(r'$u_m(k)$')
ax.set_title(r'leading eigenvectors of $\Sigma_0$')
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, ncol=3)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig07_eigenvectors.pdf', bbox_inches='tight')
plt.close(fig)

# ------------------------
# Figure 8: symlog heatmaps to expose weak fill-in
# ------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
for c, t in enumerate([0.01, 0.15, 0.40]):
    Qt = build_precision_t(Sigma0, t)
    vmax = np.max(np.abs(Qt))
    im = axes[c].imshow(Qt, cmap='RdBu_r', norm=SymLogNorm(linthresh=1e-2, linscale=1.0, vmin=-vmax, vmax=vmax, base=10))
    axes[c].set_title(rf'$\Sigma_t^{{-1}}$, symlog scale, $t={t}$')
    axes[c].set_xlabel('frame index')
    if c == 0:
        axes[c].set_ylabel('frame index')
    plt.colorbar(im, ax=axes[c], shrink=0.78)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig08_precision_symlog_heatmaps.pdf', bbox_inches='tight')
plt.close(fig)

# ------------------------
# Figure 9: deterministic vs stochastic benchmark
# ------------------------
fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.0))
for c, t in enumerate([0.05, 0.30, 1.00]):
    Qt_stoch = build_precision_t(Sigma0, t)
    _, Qt_det, _ = deterministic_sigma_t(N, alpha, sigma0_sq, t)
    vmax_s = np.max(np.abs(Qt_stoch))
    vmax_d = np.max(np.abs(Qt_det))
    ims = axes[0, c].imshow(Qt_stoch, cmap='RdBu_r', norm=SymLogNorm(linthresh=1e-3, linscale=1.0, vmin=-vmax_s, vmax=vmax_s, base=10))
    imd = axes[1, c].imshow(Qt_det, cmap='RdBu_r', norm=SymLogNorm(linthresh=1e-3, linscale=1.0, vmin=-vmax_d, vmax=vmax_d, base=10))
    axes[0, c].set_title(rf'stochastic AR(1), $t={t}$')
    axes[1, c].set_title(rf'deterministic propagation, $t={t}$')
    axes[1, c].set_xlabel('frame index')
    if c == 0:
        axes[0, c].set_ylabel('frame index')
        axes[1, c].set_ylabel('frame index')
    plt.colorbar(ims, ax=axes[0, c], shrink=0.72)
    plt.colorbar(imd, ax=axes[1, c], shrink=0.72)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig09_deterministic_vs_stochastic_precision.pdf', bbox_inches='tight')
plt.close(fig)

# ------------------------
# Figure 10: locality observables and posterior mean gain
# ------------------------
t_grid = np.linspace(1e-3, 4.0, 260)
tri = []
bw95 = []
meanrng = []
scorecorr = []
postrng = []
for t in t_grid:
    Qt = build_precision_t(Sigma0, t)
    tri.append(band_mass_fraction(Qt, 1))
    bw95.append(effective_bandwidth(Qt, rho=0.95))
    meanrng.append(mean_interaction_range(Qt))
    scorecorr.append(score_correlation(Qt, mid, mid + 1))
    Gt = posterior_gain_matrix(Sigma0, t)
    postrng.append(mean_interaction_range(Gt))

fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))
axes[0, 0].plot(t_grid, tri, lw=2)
axes[0, 0].set_title('tridiagonal mass fraction')
axes[0, 0].set_xlabel(r'$t$')
axes[0, 0].set_ylabel('mass fraction')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(t_grid, bw95, lw=2)
axes[0, 1].set_title(r'effective bandwidth $b_{0.95}(t)$')
axes[0, 1].set_xlabel(r'$t$')
axes[0, 1].set_ylabel('bandwidth')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(t_grid, meanrng, lw=2, label='precision range')
axes[1, 0].plot(t_grid, postrng, lw=1.8, label='posterior-mean range')
axes[1, 0].set_title('average interaction range')
axes[1, 0].set_xlabel(r'$t$')
axes[1, 0].set_ylabel('mean distance')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].legend(fontsize=9)

axes[1, 1].plot(t_grid, scorecorr, lw=2)
axes[1, 1].set_title(r'neighbor score correlation $\mathrm{Corr}(S_k,S_{k+1})$')
axes[1, 1].set_xlabel(r'$t$')
axes[1, 1].set_ylabel('correlation')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGDIR / 'fig10_locality_observables.pdf', bbox_inches='tight')
plt.close(fig)

# ------------------------
# Figure 11: posterior mean gain row profile
# ------------------------
fig, ax = plt.subplots(figsize=(9.2, 5.2))
for t in [0.01, 0.15, 0.40, 1.0, 2.5]:
    Gt = posterior_gain_matrix(Sigma0, t)
    ax.plot(np.arange(N), Gt[mid, :], marker='o', ms=2.6, lw=1.2, label=rf'$t={t}$')
ax.axvline(mid, ls='--', alpha=0.5)
ax.axvline(mid - 1, ls=':', alpha=0.3)
ax.axvline(mid + 1, ls=':', alpha=0.3)
ax.set_xlabel('frame index $j$')
ax.set_ylabel(rf'$(e^{{-t}}\Sigma_0\Sigma_t^{{-1}})_{{{mid},j}}$')
ax.set_title('posterior-mean sensitivity row profile')
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, ncol=2)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig11_posterior_mean_profile.pdf', bbox_inches='tight')
plt.close(fig)

# simple reproducibility report
report_path = ROOT / 'code' / 'figure_summary.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('Figure generation completed.\n')
    f.write(f'N={N}, alpha={alpha}, sigma0_sq={sigma0_sq}, sigma_eta_sq={sigma_eta_sq}\n')
    idx_max = int(np.argmax(meanrng))
    f.write(f'Max average precision interaction range at t={t_grid[idx_max]:.4f}\n')
    idx_min_tri = int(np.argmin(tri))
    f.write(f'Min tridiagonal mass fraction at t={t_grid[idx_min_tri]:.4f}\n')
    idx_peak_corr = int(np.argmax(np.abs(scorecorr)))
    f.write(f'Max absolute neighbor score correlation at t={t_grid[idx_peak_corr]:.4f}\n')

print('All figures written to', FIGDIR)
