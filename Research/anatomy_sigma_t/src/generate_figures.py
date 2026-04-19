"""
Figures for: The Anatomy of Σ_t and Σ_t⁻¹
AR(1) model, K=59 (60 frames), stationary regime.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
from scipy.linalg import eigh
import os

os.makedirs('/home/claude/refined/figures', exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ── Parameters ───────────────────────────────────────────────────────
K = 59
N = K + 1   # 60 frames
alpha = 0.9
sigma_eta_sq = 1.0 - alpha**2   # stationary: sigma_inf^2 = 1
sigma_inf_sq = 1.0

def delta_t(t):
    return 1.0 - np.exp(-2*t)

def gamma_t(t):
    e2t = np.exp(-2*t)
    return e2t / (1.0 - e2t) if t > 0 else np.inf

# ── Build matrices ───────────────────────────────────────────────────
def build_sigma0(N, alpha):
    """Stationary covariance: (Σ₀)_{jk} = α^|j-k|"""
    idx = np.arange(N)
    return alpha ** np.abs(idx[:, None] - idx[None, :])

def build_sigma_t(Sigma0, t):
    return np.exp(-2*t) * Sigma0 + delta_t(t) * np.eye(len(Sigma0))

Sigma0 = build_sigma0(N, alpha)
eigvals0, eigvecs0 = eigh(Sigma0)
# Sort descending
eigvals0 = eigvals0[::-1]
eigvecs0 = eigvecs0[:, ::-1]
U = eigvecs0

# Verify
print(f"N = {N}, alpha = {alpha}")
print(f"Eigenvalue range: [{eigvals0[-1]:.4f}, {eigvals0[0]:.4f}]")
print(f"Condition number: {eigvals0[0]/eigvals0[-1]:.1f}")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: The clean matrices Σ₀ and Σ₀⁻¹ side by side
# ══════════════════════════════════════════════════════════════════════
Prec0 = np.linalg.inv(Sigma0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(Sigma0, cmap='RdBu_r', norm=TwoSlopeNorm(0))
axes[0].set_title(r'$\Sigma_0$ — Covariance (dense)')
axes[0].set_xlabel('Frame $k$'); axes[0].set_ylabel('Frame $j$')
plt.colorbar(im0, ax=axes[0], shrink=0.78)

vmax = np.max(np.abs(Prec0))
im1 = axes[1].imshow(Prec0, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[1].set_title(r'$\Sigma_0^{-1}$ — Precision (tridiagonal)')
axes[1].set_xlabel('Frame $k$'); axes[1].set_ylabel('Frame $j$')
plt.colorbar(im1, ax=axes[1], shrink=0.78)

plt.tight_layout()
plt.savefig('/home/claude/refined/figures/fig1_clean.pdf', bbox_inches='tight')
plt.close()
print("Fig 1 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Σ_t⁻¹ at four key diffusion times (2x2 grid)
# ══════════════════════════════════════════════════════════════════════
t_vals = [0.05, 0.35, 1.0, 3.0]
labels = [
    r'$t = 0.05$ — near-clean',
    r'$t = 0.35$ — crossover ($\gamma_t \approx 1$)',
    r'$t = 1.0$ — noise-dominated',
    r'$t = 3.0$ — pure noise',
]

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for idx, (t, lab) in enumerate(zip(t_vals, labels)):
    ax = axes[idx // 2, idx % 2]
    St_inv = np.linalg.inv(build_sigma_t(Sigma0, t))
    vmax = np.max(np.abs(St_inv))
    im = ax.imshow(St_inv, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    dt = delta_t(t)
    ax.set_title(f'{lab}\n$\\Delta_t = {dt:.3f}$,  $\\gamma_t = {gamma_t(t):.2f}$', fontsize=10)
    ax.set_xlabel('Frame $k$'); ax.set_ylabel('Frame $j$')
    plt.colorbar(im, ax=ax, shrink=0.78)

plt.suptitle(r'$\Sigma_t^{-1}$ at four diffusion times ($K=59$, $\alpha=0.9$)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('/home/claude/refined/figures/fig2_prec_t.pdf', bbox_inches='tight')
plt.close()
print("Fig 2 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Row 30 of Σ_t⁻¹ — coupling profile
# ══════════════════════════════════════════════════════════════════════
mid = N // 2  # row 30
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# (a) Linear scale
for t in [0.05, 0.2, 0.5, 1.0, 2.0]:
    St_inv = np.linalg.inv(build_sigma_t(Sigma0, t))
    axes[0].plot(range(N), St_inv[mid, :], '-', lw=1.2, label=f'$t={t}$')
axes[0].axvline(mid, color='gray', ls='--', alpha=0.4, lw=0.8)
axes[0].set_xlabel('Frame $k$')
axes[0].set_ylabel(rf'$(\Sigma_t^{{-1}})_{{{mid},k}}$')
axes[0].set_title(f'Row {mid}: coupling to all frames')
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(True, alpha=0.2)

# (b) Log-scale decay with distance
for t in [0.1, 0.3, 0.5, 1.0, 2.0]:
    St_inv = np.linalg.inv(build_sigma_t(Sigma0, t))
    avg_by_dist = []
    for d in range(N):
        vals = [abs(St_inv[i, i+d]) for i in range(N-d)]
        avg_by_dist.append(np.mean(vals))
    axes[1].semilogy(range(N), avg_by_dist, 'o-', markersize=2, lw=1, label=f'$t={t}$')

axes[1].set_xlabel(r'Distance $|j-k|$')
axes[1].set_ylabel(r'Mean $|(\Sigma_t^{-1})_{j,k}|$')
axes[1].set_title('Off-diagonal decay (log scale)')
axes[1].legend(fontsize=8, ncol=2)
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/home/claude/refined/figures/fig3_coupling.pdf', bbox_inches='tight')
plt.close()
print("Fig 3 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Tridiagonal weight fraction & condition number
# ══════════════════════════════════════════════════════════════════════
t_range = np.linspace(0.005, 4.0, 300)

tridiag_frac = []
cond_nums = []
for t in t_range:
    St = build_sigma_t(Sigma0, t)
    St_inv = np.linalg.inv(St)
    # Tridiagonal weight
    d_sum = np.sum(np.abs(np.diag(St_inv)))
    od1 = np.sum(np.abs(np.diag(St_inv, 1))) + np.sum(np.abs(np.diag(St_inv, -1)))
    tot = np.sum(np.abs(St_inv))
    tridiag_frac.append((d_sum + od1) / tot)
    # Condition number from eigenvalues (fast)
    lam = np.exp(-2*t)*eigvals0 + delta_t(t)
    cond_nums.append(lam[0] / lam[-1])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(t_range, tridiag_frac, 'b-', lw=2)
axes[0].axhline(1.0, color='gray', ls=':', alpha=0.4)
axes[0].axvline(0.5*np.log(2), color='red', ls='--', alpha=0.5, label=r'$t = \frac{1}{2}\ln 2$ ($\gamma_t = 1$)')
axes[0].set_xlabel(r'Diffusion time $t$')
axes[0].set_ylabel(r'$\rho_{\mathrm{tri}}(t)$')
axes[0].set_title('Tridiagonal weight fraction')
axes[0].legend()
axes[0].set_xlim([0, 4]); axes[0].set_ylim([0, 1.05])
axes[0].grid(True, alpha=0.2)

axes[1].semilogy(t_range, cond_nums, 'b-', lw=2)
axes[1].axhline(1.0, color='gray', ls=':', alpha=0.4)
axes[1].axvline(0.5*np.log(2), color='red', ls='--', alpha=0.5, label=r'$\gamma_t = 1$')
axes[1].set_xlabel(r'Diffusion time $t$')
axes[1].set_ylabel(r'$\kappa(\Sigma_t)$')
axes[1].set_title('Condition number')
axes[1].legend()
axes[1].set_xlim([0, 4])
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/home/claude/refined/figures/fig4_diagnostics.pdf', bbox_inches='tight')
plt.close()
print("Fig 4 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Eigenvalue spectrum & eigenvectors
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) Eigenvalues of Σ₀
axes[0].bar(range(N), eigvals0, color='steelblue', alpha=0.8, width=1.0)
axes[0].set_xlabel('Mode index $m$')
axes[0].set_ylabel(r'$d_m$')
axes[0].set_title(r'Eigenvalues of $\Sigma_0$')
axes[0].grid(True, alpha=0.2)

# (b) Eigenvalue trajectories of Σ_t⁻¹
t_fine = np.linspace(0.01, 3, 200)
# Plot a selection of modes
modes_to_plot = [0, 1, 2, 5, 10, 20, 40, 59]
colors_mode = plt.cm.viridis(np.linspace(0, 1, len(modes_to_plot)))
for i, m in enumerate(modes_to_plot):
    lam_inv = 1.0 / (np.exp(-2*t_fine)*eigvals0[m] + (1 - np.exp(-2*t_fine)))
    axes[1].plot(t_fine, lam_inv, '-', lw=1.3, color=colors_mode[i], label=f'$m={m}$')
axes[1].axhline(1.0, color='red', ls='--', lw=0.8, alpha=0.6)
axes[1].set_xlabel(r'Diffusion time $t$')
axes[1].set_ylabel(r'$1/(e^{-2t}d_m + \Delta_t)$')
axes[1].set_title(r'Eigenvalues of $\Sigma_t^{-1}$ vs $t$')
axes[1].legend(fontsize=7, ncol=2)
axes[1].grid(True, alpha=0.2)

# (c) First few eigenvectors
for m in range(5):
    axes[2].plot(range(N), eigvecs0[:, m], '-', lw=1.0,
                 label=f'$m={m}$, $d_{m}={eigvals0[m]:.2f}$')
axes[2].set_xlabel('Frame index $k$')
axes[2].set_ylabel(r'$u_m(k)$')
axes[2].set_title('Eigenvectors of $\\Sigma_0$ (first 5)')
axes[2].legend(fontsize=7)
axes[2].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/home/claude/refined/figures/fig5_spectral.pdf', bbox_inches='tight')
plt.close()
print("Fig 5 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Σ_t⁻¹ in the eigenbasis (proves diagonality)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for col, t in enumerate([0.1, 1.0, 3.0]):
    St_inv = np.linalg.inv(build_sigma_t(Sigma0, t))
    rotated = U.T @ St_inv @ U
    im = axes[col].imshow(np.abs(rotated), cmap='hot_r', vmin=0)
    axes[col].set_title(f'$|U^\\top \\Sigma_t^{{-1}} U|$, $t={t}$', fontsize=10)
    axes[col].set_xlabel('Mode $m$')
    if col == 0:
        axes[col].set_ylabel('Mode $n$')
    plt.colorbar(im, ax=axes[col], shrink=0.78)

plt.suptitle(r'$\Sigma_t^{-1}$ is diagonal in the eigenbasis of $\Sigma_0$ for all $t$', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('/home/claude/refined/figures/fig6_eigenbasis.pdf', bbox_inches='tight')
plt.close()
print("Fig 6 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 7: Neumann series band-by-band construction
# ══════════════════════════════════════════════════════════════════════
t_neu = 0.5
gt = gamma_t(t_neu)
dt = delta_t(t_neu)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
exact_inv = np.linalg.inv(build_sigma_t(Sigma0, t_neu))

partial_sum = np.zeros((N, N))
titles = ['$n=0$: diagonal', '$n \\leq 1$: tridiagonal', '$n \\leq 3$: 7-diagonal', '$n \\leq 10$']
n_maxes = [0, 1, 3, 10]
Sigma0_power = np.eye(N)

all_terms = [np.eye(N)]  # n=0 term
for n in range(1, 11):
    Sigma0_power = Sigma0_power @ Sigma0
    all_terms.append((-gt)**n * Sigma0_power)

for col, n_max in enumerate(n_maxes):
    partial_sum = sum(all_terms[:n_max+1]) / dt
    err = np.max(np.abs(partial_sum - exact_inv))
    im = axes[col].imshow(partial_sum, cmap='RdBu_r', norm=TwoSlopeNorm(0))
    axes[col].set_title(f'{titles[col]}\n$\\|\\mathrm{{error}}\\|_\\infty = {err:.3f}$', fontsize=9)
    axes[col].set_xlabel('$k$')
    if col == 0:
        axes[col].set_ylabel('$j$')
    plt.colorbar(im, ax=axes[col], shrink=0.7)

plt.suptitle(f'Neumann series: building $\\Sigma_t^{{-1}}$ band by band ($t={t_neu}$, $\\gamma_t = {gt:.2f}$)', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/refined/figures/fig7_neumann.pdf', bbox_inches='tight')
plt.close()
print("Fig 7 done.")

print("\nAll figures generated.")
