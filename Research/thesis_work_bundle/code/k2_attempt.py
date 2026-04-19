"""
K=2 Laplace AR(1) under OU: exact attempt in Fourier space.

Purpose
-------
Test whether the K=1 'Gaussian site + 1D non-Gaussian bond' closure generalises
to two steps.  We work out the characteristic function, rotate to the
innovation-frequency coordinates (ell_0, ell_1, ell_2) = (k_0, k_1 + alpha k_2,
k_2), and compute the Gaussian-sector covariance M.

Central question: does the Gaussian cross-covariance between ell_1 and ell_2
(the two Laplace-innovation frequencies) vanish?  If yes, the density
factorises as a product of two independent 1D bonds.  If no, the factorisation
fails and we need bivariate nonlinearity.

Result
------
M_{12} = -alpha * Delta  (non-zero whenever alpha != 0 and t != 0).
So exact factorisation fails for K = 2 except in trivial limits.
"""

import numpy as np
import sympy as sp


def characteristic_function_K2_symbolic():
    """Work out hat p_t(k_0, k_1, k_2) symbolically.

    Returns the matrix M (Gaussian-sector covariance in ell-coordinates)
    symbolically, and its Schur complement after marginalising ell_0.
    """
    k0, k1, k2 = sp.symbols("k_0 k_1 k_2", real=True)
    alpha, beta, Delta, sigma0, mu0, b = sp.symbols(
        "alpha beta Delta sigma_0 mu_0 b", positive=True, real=True)
    btilde = beta * b

    # Clean CF:  p_0(k) = exp(i mu_0 q - sigma0^2 q^2 / 2) / (1 + b^2 (k1 + alpha k2)^2) / (1 + b^2 k2^2)
    q = k0 + alpha * k1 + alpha**2 * k2
    clean = sp.exp(sp.I * mu0 * q - sigma0**2 * q**2 / 2) \
            / (1 + b**2 * (k1 + alpha * k2)**2) \
            / (1 + b**2 * k2**2)

    # OU corruption:  p_t(k) = exp(-Delta ||k||^2 / 2) * p_0(beta k)
    q_beta = beta * q
    gauss_factor = sp.exp(-Delta * (k0**2 + k1**2 + k2**2) / 2
                          + sp.I * beta * mu0 * q
                          - beta**2 * sigma0**2 * q**2 / 2)
    rational_factor = 1 / (1 + btilde**2 * (k1 + alpha * k2)**2) \
                      / (1 + btilde**2 * k2**2)
    phat = gauss_factor * rational_factor

    # Rotate to innovation-frequency coordinates
    #   ell_0 = k_0,   ell_1 = k_1 + alpha * k_2,   ell_2 = k_2
    # Inverse:  k_0 = ell_0,  k_1 = ell_1 - alpha ell_2,  k_2 = ell_2
    ell0, ell1, ell2 = sp.symbols("ell_0 ell_1 ell_2", real=True)
    subs = {k0: ell0, k1: ell1 - alpha * ell2, k2: ell2}
    gauss_ell = sp.simplify(gauss_factor.subs(subs))

    # Extract the quadratic form -ell^T M ell / 2 from log of gauss_ell
    log_gauss = sp.logcombine(sp.log(gauss_ell), force=True)
    # We want just the quadratic part.  It's cleaner to recompute directly:
    q_sub = ell0 + alpha * (ell1 - alpha * ell2) + alpha**2 * ell2
    q_sub = sp.simplify(q_sub)     # = ell0 + alpha*ell1  (beautiful!)
    ksq = ell0**2 + (ell1 - alpha * ell2)**2 + ell2**2
    # Quadratic form
    quad_form = sp.expand(Delta * ksq + beta**2 * sigma0**2 * q_sub**2)

    # Collect coefficients into symmetric matrix
    ells = [ell0, ell1, ell2]
    M = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            if i == j:
                M[i, j] = sp.Rational(1, 1) * sp.diff(quad_form, ells[i], 2) / 2
            else:
                M[i, j] = sp.Rational(1, 2) * sp.diff(quad_form, ells[i], ells[j])
    M = sp.simplify(M)

    # Schur complement after marginalising ell_0
    Mtilde = sp.zeros(2, 2)
    for i in (1, 2):
        for j in (1, 2):
            Mtilde[i - 1, j - 1] = M[i, j] - M[i, 0] * M[0, j] / M[0, 0]
    Mtilde = sp.simplify(Mtilde)

    return {
        "q_sub": q_sub,         # linear term in (ell_0, ell_1, ell_2): only ell_0 + alpha ell_1
        "M": M,
        "Mtilde": Mtilde,
        "rational_factor_ell": 1 / (1 + btilde**2 * ell1**2) / (1 + btilde**2 * ell2**2),
    }


def pretty_print():
    res = characteristic_function_K2_symbolic()
    print("Quadratic form in ell_0, ell_1, ell_2 (K=2)")
    print("  linear phase term involves only q = ell_0 + alpha ell_1")
    print(f"  q_sub = {res['q_sub']}")
    print()
    print("M (3x3 covariance of the Gaussian sector in ell-coords):")
    sp.pprint(res["M"])
    print()
    print("M_tilde (Schur complement; effective covariance of (ell_1, ell_2)):")
    sp.pprint(res["Mtilde"])
    print()
    print("Rational non-Gaussian factor in ell-coords:")
    sp.pprint(res["rational_factor_ell"])
    print()
    print("Conclusion:")
    print("  M[1,2] = M[2,1] = -alpha * Delta")
    print("  This cross-term vanishes iff alpha=0 (trivial) or Delta=0 (t=0).")
    print("  Hence exact 'product of two 1D bonds' factorisation fails for K=2.")


# --------------------------------------------------------------------------
# Numerical K=2 density via 3D Fourier inversion (the correct tool)
# --------------------------------------------------------------------------

def p_K2_fourier(x0, x1, x2, alpha, b, sigma0, mu0, t,
                 L=20.0, N=64):
    """Evaluate p_t(x_0, x_1, x_2) by 3D inverse Fourier transform.

    Uses the exact closed-form characteristic function (no truncation of the
    rational factor needed).  The grid is (N)^3 over [-L, L]^3 in each
    frequency axis; chosen modest to avoid OOM.  For verification, pair with
    Monte Carlo sampling below.
    """
    beta = np.exp(-t)
    Delta = 1 - beta**2
    btilde = beta * b
    q_axis = np.linspace(-L, L, N)
    dq = q_axis[1] - q_axis[0]
    K0, K1, K2 = np.meshgrid(q_axis, q_axis, q_axis, indexing="ij")
    q = K0 + alpha * K1 + alpha**2 * K2
    # Complex characteristic function
    gauss = np.exp(-Delta * (K0**2 + K1**2 + K2**2) / 2
                   + 1j * beta * mu0 * q
                   - beta**2 * sigma0**2 * q**2 / 2)
    rat = 1.0 / (1 + btilde**2 * (K1 + alpha * K2)**2) / (1 + btilde**2 * K2**2)
    phat = gauss * rat
    # phase for target point
    phase = np.exp(-1j * (K0 * x0 + K1 * x1 + K2 * x2))
    integrand = phat * phase
    val = np.real(integrand.sum()) * dq**3 / (2 * np.pi)**3
    return val


def monte_carlo_K2(alpha, b, sigma0, mu0, t, N=100_000, rng=0):
    """Draw N samples from the K=2 joint law p_t(x_0, x_1, x_2)."""
    from scipy.stats import laplace
    rng = np.random.default_rng(rng)
    A0 = rng.normal(mu0, sigma0, size=N)
    eps1 = laplace.rvs(loc=0.0, scale=b, size=N, random_state=rng)
    eps2 = laplace.rvs(loc=0.0, scale=b, size=N, random_state=rng)
    A1 = alpha * A0 + eps1
    A2 = alpha * A1 + eps2
    beta = np.exp(-t)
    Delta = 1 - beta**2
    Z = rng.standard_normal(size=(3, N))
    X0 = beta * A0 + np.sqrt(Delta) * Z[0]
    X1 = beta * A1 + np.sqrt(Delta) * Z[1]
    X2 = beta * A2 + np.sqrt(Delta) * Z[2]
    return X0, X1, X2


if __name__ == "__main__":
    pretty_print()

    # Numerical K=2 density at a point, cross-checked by 3D KDE
    alpha, b, sigma0, mu0, t = 0.6, 1.0, 1.0, 0.0, 0.5
    print("\n--- Numerical K=2 check ---")
    pt_point = p_K2_fourier(0.2, 0.3, 0.1, alpha, b, sigma0, mu0, t,
                             L=15.0, N=56)
    print(f"  p_t(0.2, 0.3, 0.1) via 3D Fourier inversion = {pt_point:.6f}")
    X0, X1, X2 = monte_carlo_K2(alpha, b, sigma0, mu0, t, N=2_000_000)
    bw = 0.08  # fixed bandwidth for this 3D density estimate
    from scipy.stats import norm
    d0 = (0.2 - X0) / bw
    d1 = (0.3 - X1) / bw
    d2 = (0.1 - X2) / bw
    # 3D Gaussian kernel product
    kern = norm.pdf(d0) * norm.pdf(d1) * norm.pdf(d2) / bw**3
    pt_mc = float(kern.mean())
    print(f"  p_t(0.2, 0.3, 0.1) via 3D MC+KDE          = {pt_mc:.6f}")
    print(f"  rel. discrepancy                            = {abs(pt_point - pt_mc) / pt_mc:.2%}")
