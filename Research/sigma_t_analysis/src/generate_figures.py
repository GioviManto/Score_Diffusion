"""
Generate all figures for the Sigma_t analysis document.
AR(1) model: a_{k+1} = alpha * a_k + eta_k
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import eigh
import os

os.makedirs('/home/claude/figures', exist_ok=True)

# ─── Parameters ───────────────────────────────────────────────────────
K = 19           # 20 frames total (indices 0..19)
N = K + 1        # dimension
alpha = 0.9      # AR(1) coefficient
sigma_0_sq = 1.0 # initial variance
sigma_eta_sq = 1.0 - alpha**2  # stationary: sigma_inf^2 = sigma_eta^2/(1-alpha^2) = 1
mu_0 = 0.0       # zero mean for simplicity

# ─── Build Sigma_0 ───────────────────────────────────────────────────
def build_sigma0(N, alpha, sigma_0_sq, sigma_eta_sq):
    """Build the (N x N) clean covariance matrix."""
    sigma_sq = np.zeros(N)
    sigma_sq[0] = sigma_0_sq
    for k in range(1, N):
        sigma_sq[k] = alpha**2 * sigma_sq[k-1] + sigma_eta_sq
    Sigma0 = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            m = min(j, k)
            Sigma0[j, k] = alpha**abs(j-k) * sigma_sq[m]
    return Sigma0

def build_precision0(N, alpha, sigma_0_sq, sigma_eta_sq):
    """Build the tridiagonal precision matrix directly."""
    P = np.zeros((N, N))
    P[0, 0] = 1.0/sigma_0_sq + alpha**2/sigma_eta_sq
    for k in range(1, N-1):
        P[k, k] = (1.0 + alpha**2)/sigma_eta_sq
    P[N-1, N-1] = 1.0/sigma_eta_sq
    for k in range(N-1):
        P[k, k+1] = -alpha/sigma_eta_sq
        P[k+1, k] = -alpha/sigma_eta_sq
    return P

def delta_t(t):
    return 1.0 - np.exp(-2*t)

def build_sigma_t(Sigma0, t):
    return np.exp(-2*t) * Sigma0 + delta_t(t) * np.eye(len(Sigma0))

Sigma0 = build_sigma0(N, alpha, sigma_0_sq, sigma_eta_sq)
Prec0 = build_precision0(N, alpha, sigma_0_sq, sigma_eta_sq)

# Verify: Sigma0 @ Prec0 ≈ I
err = np.max(np.abs(Sigma0 @ Prec0 - np.eye(N)))
print(f"Verification: ||Sigma0 * Prec0 - I||_inf = {err:.2e}")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Sigma_0 and Sigma_0^{-1} side by side
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Sigma_0
im0 = axes[0].imshow(Sigma0, cmap='RdBu_r', norm=TwoSlopeNorm(0))
axes[0].set_title(r'$\Sigma_0$ (Covariance — Dense)', fontsize=13)
axes[0].set_xlabel('Frame index $k$')
axes[0].set_ylabel('Frame index $j$')
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# Sigma_0^{-1}
im1 = axes[1].imshow(Prec0, cmap='RdBu_r', norm=TwoSlopeNorm(0))
axes[1].set_title(r'$\Sigma_0^{-1}$ (Precision — Tridiagonal)', fontsize=13)
axes[1].set_xlabel('Frame index $k$')
axes[1].set_ylabel('Frame index $j$')
plt.colorbar(im1, ax=axes[1], shrink=0.8)

plt.tight_layout()
plt.savefig('/home/claude/figures/fig1_sigma0_vs_prec0.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Fig 1 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Sigma_t and Sigma_t^{-1} at multiple t values
# ══════════════════════════════════════════════════════════════════════
t_values = [0.01, 0.3, 0.7, 1.5, 3.0]
fig, axes = plt.subplots(2, len(t_values), figsize=(3.4*len(t_values), 7))

for col, t in enumerate(t_values):
    St = build_sigma_t(Sigma0, t)
    St_inv = np.linalg.inv(St)
    
    vmax_cov = np.max(np.abs(St))
    im_top = axes[0, col].imshow(St, cmap='RdBu_r', norm=TwoSlopeNorm(0, vmin=-vmax_cov, vmax=vmax_cov))
    axes[0, col].set_title(f'$\\Sigma_t$,  $t={t}$\n$\\Delta_t={delta_t(t):.3f}$', fontsize=10)
    if col == 0:
        axes[0, col].set_ylabel('Frame $j$')
    plt.colorbar(im_top, ax=axes[0, col], shrink=0.7)
    
    vmax_prec = np.max(np.abs(St_inv))
    im_bot = axes[1, col].imshow(St_inv, cmap='RdBu_r', norm=TwoSlopeNorm(0, vmin=-vmax_prec, vmax=vmax_prec))
    axes[1, col].set_title(f'$\\Sigma_t^{{-1}}$,  $t={t}$', fontsize=10)
    if col == 0:
        axes[1, col].set_ylabel('Frame $j$')
    axes[1, col].set_xlabel('Frame $k$')
    plt.colorbar(im_bot, ax=axes[1, col], shrink=0.7)

plt.suptitle(f'$K={K}$ frames, $\\alpha={alpha}$, $\\sigma_0^2={sigma_0_sq}$, $\\sigma_\\eta^2={sigma_eta_sq:.2f}$ (stationary)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/figures/fig2_sigma_t_panel.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Fig 2 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Tridiagonal weight fraction vs t
# ══════════════════════════════════════════════════════════════════════
t_range = np.linspace(0.001, 4.0, 200)

diag_weight = []      # sum of |diagonal entries|
offdiag1_weight = []  # sum of |first off-diagonal entries|
rest_weight = []      # sum of |all other entries|
total_weight = []

for t in t_range:
    St = build_sigma_t(Sigma0, t)
    St_inv = np.linalg.inv(St)
    
    d = np.sum(np.abs(np.diag(St_inv)))
    od1 = np.sum(np.abs(np.diag(St_inv, 1))) + np.sum(np.abs(np.diag(St_inv, -1)))
    tot = np.sum(np.abs(St_inv))
    rest = tot - d - od1
    
    diag_weight.append(d)
    offdiag1_weight.append(od1)
    rest_weight.append(rest)
    total_weight.append(tot)

diag_frac = np.array(diag_weight) / np.array(total_weight)
od1_frac = np.array(offdiag1_weight) / np.array(total_weight)
tridiag_frac = diag_frac + od1_frac
rest_frac = np.array(rest_weight) / np.array(total_weight)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(t_range, tridiag_frac, 'b-', lw=2, label=r'Tridiagonal band (diag + 1st off-diag)')
ax.plot(t_range, diag_frac, 'g--', lw=1.5, label=r'Diagonal only')
ax.plot(t_range, rest_frac, 'r:', lw=1.5, label=r'Beyond tridiagonal')
ax.set_xlabel(r'Diffusion time $t$', fontsize=12)
ax.set_ylabel(r'Fraction of $\|\Sigma_t^{-1}\|_1$ (entry-wise)', fontsize=12)
ax.set_title(r'Tridiagonal weight fraction of $\Sigma_t^{-1}$ vs $t$' + f'\n$K={K}$, $\\alpha={alpha}$', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim([0, 4])
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/figures/fig3_tridiag_fraction.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Fig 3 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Row profile of Sigma_t^{-1} (middle row) at different t
# ══════════════════════════════════════════════════════════════════════
mid = N // 2  # row index 10
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

t_vals_profile = [0.01, 0.2, 0.5, 1.0, 2.0, 3.0]
for t in t_vals_profile:
    St = build_sigma_t(Sigma0, t)
    St_inv = np.linalg.inv(St)
    row = St_inv[mid, :]
    ax.plot(range(N), row, 'o-', markersize=3, label=f'$t = {t}$')

ax.axvline(mid, color='gray', ls='--', alpha=0.5, label=f'$k = {mid}$ (self)')
ax.axvline(mid-1, color='gray', ls=':', alpha=0.3)
ax.axvline(mid+1, color='gray', ls=':', alpha=0.3)
ax.set_xlabel('Frame index $k$', fontsize=12)
ax.set_ylabel(rf'$(\Sigma_t^{{-1}})_{{{mid},k}}$', fontsize=12)
ax.set_title(rf'Row ${mid}$ of $\Sigma_t^{{-1}}$: coupling to all frames' + f'\n$\\alpha={alpha}$', fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/figures/fig4_row_profile.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Fig 4 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Eigenvalue spectrum of Sigma_0, Sigma_t, Sigma_t^{-1}
# ══════════════════════════════════════════════════════════════════════
eigvals_0, eigvecs_0 = eigh(Sigma0)  # ascending order
eigvals_0 = eigvals_0[::-1]
eigvecs_0 = eigvecs_0[:, ::-1]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel (a): eigenvalues of Sigma_0
axes[0].bar(range(N), eigvals_0, color='steelblue', alpha=0.8)
axes[0].set_xlabel('Eigenvalue index $m$', fontsize=11)
axes[0].set_ylabel(r'$d_m$ (eigenvalue of $\Sigma_0$)', fontsize=11)
axes[0].set_title(r'Eigenvalues of $\Sigma_0$', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Panel (b): eigenvalues of Sigma_t^{-1} for several t
for t in [0.01, 0.3, 1.0, 2.0]:
    lam_t_inv = 1.0 / (np.exp(-2*t) * eigvals_0 + delta_t(t))
    axes[1].plot(range(N), lam_t_inv, 'o-', markersize=3, label=f'$t = {t}$')
axes[1].set_xlabel('Eigenvalue index $m$', fontsize=11)
axes[1].set_ylabel(r'$\lambda_m^{-1}(t) = (e^{-2t}d_m + \Delta_t)^{-1}$', fontsize=11)
axes[1].set_title(r'Eigenvalues of $\Sigma_t^{-1}$', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Panel (c): eigenvectors (first 5)
for m in range(5):
    axes[2].plot(range(N), eigvecs_0[:, m], '-', lw=1.2, label=f'$m = {m}$ ($d_{m} = {eigvals_0[m]:.3f}$)')
axes[2].set_xlabel('Frame index $k$', fontsize=11)
axes[2].set_ylabel(r'$u_m(k)$', fontsize=11)
axes[2].set_title(r'First 5 eigenvectors of $\Sigma_0$' + '\n(shared by all $\\Sigma_t$)', fontsize=12)
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/figures/fig5_eigenvalues.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Fig 5 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Sigma_t^{-1} in the eigenbasis (should be diagonal)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
U = eigvecs_0  # columns are eigenvectors

t_vals_eig = [0.01, 0.5, 1.5, 3.0]
for col, t in enumerate(t_vals_eig):
    St_inv = np.linalg.inv(build_sigma_t(Sigma0, t))
    # Rotate: U^T Sigma_t^{-1} U  should be diagonal
    St_inv_rotated = U.T @ St_inv @ U
    
    im = axes[col].imshow(np.abs(St_inv_rotated), cmap='hot_r', vmin=0)
    axes[col].set_title(f'$|U^\\top \\Sigma_t^{{-1}} U|$, $t={t}$', fontsize=10)
    axes[col].set_xlabel('Eigenvector index')
    if col == 0:
        axes[col].set_ylabel('Eigenvector index')
    plt.colorbar(im, ax=axes[col], shrink=0.8)

plt.suptitle(r'$\Sigma_t^{-1}$ in the eigenbasis of $\Sigma_0$: diagonal for all $t$', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/figures/fig6_eigenbasis.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Fig 6 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 7: Eigenvalue trajectories as function of t
# ══════════════════════════════════════════════════════════════════════
t_fine = np.linspace(0.001, 4.0, 300)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Eigenvalues of Sigma_t
for m in range(N):
    lam_t = np.exp(-2*t_fine) * eigvals_0[m] + (1 - np.exp(-2*t_fine))
    axes[0].plot(t_fine, lam_t, lw=0.8, color=plt.cm.viridis(m/N))
axes[0].axhline(1.0, color='red', ls='--', lw=1, label=r'$\lambda = 1$ (pure noise)')
axes[0].set_xlabel(r'Diffusion time $t$', fontsize=12)
axes[0].set_ylabel(r'$\lambda_m(t) = e^{-2t} d_m + \Delta_t$', fontsize=12)
axes[0].set_title(r'Eigenvalues of $\Sigma_t$ vs $t$', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# (b) Eigenvalues of Sigma_t^{-1}
for m in range(N):
    lam_t_inv = 1.0 / (np.exp(-2*t_fine) * eigvals_0[m] + (1 - np.exp(-2*t_fine)))
    axes[1].plot(t_fine, lam_t_inv, lw=0.8, color=plt.cm.viridis(m/N))
axes[1].axhline(1.0, color='red', ls='--', lw=1, label=r'$\lambda^{-1} = 1$ (pure noise)')
axes[1].set_xlabel(r'Diffusion time $t$', fontsize=12)
axes[1].set_ylabel(r'$\lambda_m^{-1}(t)$', fontsize=12)
axes[1].set_title(r'Eigenvalues of $\Sigma_t^{-1}$ vs $t$', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/figures/fig7_eigenvalue_trajectories.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Fig 7 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 8: Off-diagonal decay of Sigma_t^{-1} (log scale)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

for t in [0.1, 0.3, 0.5, 1.0, 2.0]:
    St_inv = np.linalg.inv(build_sigma_t(Sigma0, t))
    # Average absolute value at each distance from diagonal
    avg_by_dist = []
    for d in range(N):
        vals = []
        for i in range(N):
            j = i + d
            if j < N:
                vals.append(abs(St_inv[i, j]))
        avg_by_dist.append(np.mean(vals) if vals else 0)
    ax.semilogy(range(N), avg_by_dist, 'o-', markersize=3, label=f'$t = {t}$')

ax.set_xlabel(r'Distance from diagonal $|j - k|$', fontsize=12)
ax.set_ylabel(r'Mean $|(\Sigma_t^{-1})_{j,k}|$ at distance $|j-k|$', fontsize=12)
ax.set_title(r'Off-diagonal decay of $\Sigma_t^{-1}$' + f' ($\\alpha={alpha}$, $K={K}$)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/figures/fig8_offdiag_decay.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Fig 8 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 9: Condition number of Sigma_t vs t
# ══════════════════════════════════════════════════════════════════════
cond_numbers = []
for t in t_range:
    St = build_sigma_t(Sigma0, t)
    cond_numbers.append(np.linalg.cond(St))

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.semilogy(t_range, cond_numbers, 'b-', lw=2)
ax.set_xlabel(r'Diffusion time $t$', fontsize=12)
ax.set_ylabel(r'$\kappa(\Sigma_t) = \lambda_{\max}/\lambda_{\min}$', fontsize=12)
ax.set_title(r'Condition number of $\Sigma_t$ vs $t$' + f'\n$\\alpha={alpha}$, $K={K}$', fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/figures/fig9_condition_number.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Fig 9 done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 10: Explicit 5x5 numerical matrices at a few t values
# ══════════════════════════════════════════════════════════════════════
K_small = 4
N_small = 5
Sigma0_small = build_sigma0(N_small, alpha, sigma_0_sq, sigma_eta_sq)
Prec0_small = build_precision0(N_small, alpha, sigma_0_sq, sigma_eta_sq)

for t in [0.0, 0.3, 1.0, 3.0]:
    if t == 0.0:
        St_small = Sigma0_small
        St_inv_small = Prec0_small
    else:
        St_small = build_sigma_t(Sigma0_small, t)
        St_inv_small = np.linalg.inv(St_small)
    
    print(f"\n--- t = {t} (Delta_t = {delta_t(t) if t > 0 else 0:.4f}) ---")
    print(f"Sigma_t (5x5):")
    for row in St_small:
        print("  " + "  ".join(f"{v:8.4f}" for v in row))
    print(f"Sigma_t^{{-1}} (5x5):")
    for row in St_inv_small:
        print("  " + "  ".join(f"{v:8.4f}" for v in row))

# ══════════════════════════════════════════════════════════════════════
# FIGURE 11: alpha sensitivity — vary alpha
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
alphas = [0.2, 0.5, 0.8, 0.95]
t_fixed = 0.5

for col, a in enumerate(alphas):
    se = 1.0 - a**2  # stationary noise
    S0 = build_sigma0(N, a, 1.0, se)
    St = build_sigma_t(S0, t_fixed)
    St_inv = np.linalg.inv(St)
    
    im0 = axes[0, col].imshow(St, cmap='RdBu_r', norm=TwoSlopeNorm(0))
    axes[0, col].set_title(f'$\\Sigma_t$, $\\alpha={a}$', fontsize=10)
    plt.colorbar(im0, ax=axes[0, col], shrink=0.7)
    
    im1 = axes[1, col].imshow(St_inv, cmap='RdBu_r', norm=TwoSlopeNorm(0))
    axes[1, col].set_title(f'$\\Sigma_t^{{-1}}$, $\\alpha={a}$', fontsize=10)
    plt.colorbar(im1, ax=axes[1, col], shrink=0.7)

plt.suptitle(f'Effect of $\\alpha$ at fixed $t = {t_fixed}$, $K={K}$', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/figures/fig11_alpha_sensitivity.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Fig 11 done.")

print("\nAll figures generated successfully!")
