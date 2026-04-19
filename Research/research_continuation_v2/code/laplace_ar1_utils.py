"""
Laplace AR(1) diffusion model utilities.

Core mathematical functions for computing exact noisy densities and scores
under Laplace innovation shocks, combined with OU noising.
"""

import numpy as np
from scipy import special
from scipy import integrate
from scipy.ndimage import gaussian_filter1d


def delta_t(t):
    """Time-dependent variance contribution from OU noise."""
    return 1.0 - np.exp(-2.0 * t)


def laplace_gaussian_convolution(y, b_eff, sigma_sq):
    """
    Closed-form density of X = L + Z, where:
      L ~ Laplace(0, b_eff)
      Z ~ N(0, sigma_sq)

    Evaluated at y.

    Parameters
    ----------
    y : float or array
        Evaluation point(s)
    b_eff : float
        Laplace scale parameter
    sigma_sq : float
        Gaussian variance

    Returns
    -------
    float or array
        Probability density value(s)
    """
    if sigma_sq <= 0 or b_eff <= 0:
        return np.zeros_like(y, dtype=float)

    y = np.asarray(y)
    sigma = np.sqrt(sigma_sq)
    r = sigma_sq / (b_eff ** 2)  # ratio

    # Compute using the closed form:
    # density = (1/(4*b_eff)) * exp(σ²/(2b²)) * [exp(y/b)*erfc(a) + exp(-y/b)*erfc(b)]
    # where a = y/b + σ²/(√2 b²), b = -y/b + σ²/(√2 b²)

    arg1 = y / b_eff + sigma_sq / (np.sqrt(2.0) * b_eff ** 2)
    arg2 = -y / b_eff + sigma_sq / (np.sqrt(2.0) * b_eff ** 2)

    # Compute erfc values safely
    erfc1_val = special.erfc(arg1)
    erfc2_val = special.erfc(arg2)

    # Compute terms with exponentials
    exp_factor1 = y / b_eff
    exp_factor2 = -y / b_eff

    # Use safe multiplication to avoid overflow
    with np.errstate(over='ignore', under='ignore'):
        # Try to compute directly first
        term1 = np.exp(exp_factor1) * erfc1_val
        term2 = np.exp(exp_factor2) * erfc2_val
        total = term1 + term2

        # For cases where overflow occurred, use log-sum-exp
        mask_bad = ~np.isfinite(total)
        if np.any(mask_bad):
            # Log-sum-exp: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(min(a,b) - max(a,b)))
            with np.errstate(divide='ignore'):
                log1 = exp_factor1 + np.log(erfc1_val)
                log2 = exp_factor2 + np.log(erfc2_val)

            max_log = np.maximum(log1, log2)
            logtotal = max_log + np.log(np.exp(log1 - max_log) + np.exp(log2 - max_log))
            # Recompute using logs for bad entries
            result = np.where(mask_bad, np.exp(logtotal), total)
            total = result

    # Compute density
    log_const = -np.log(4.0 * b_eff) + r / 2.0
    log_density = log_const + np.log(np.maximum(total, 1e-300))

    return np.exp(log_density)


def laplace_k1_density(x0, x1, t, params):
    """
    Exact noisy density p_t(x_0, x_1) for K=1 (two frames).

    Clean chain: a_0 ~ N(μ_0, σ_0²), a_1 = α a_0 + ε (ε ~ Laplace(0, b))
    After OU noising: x_k = e^{-t} a_k + √Δ_t z_k

    Parameters
    ----------
    x0, x1 : float or array
        Observations
    t : float
        Diffusion time
    params : dict
        Must contain: 'mu0', 'sigma0_sq', 'alpha', 'b' (Laplace scale)

    Returns
    -------
    float or array
        Joint density p_t(x_0, x_1)
    """
    mu0 = params['mu0']
    sigma0_sq = params['sigma0_sq']
    alpha = params['alpha']
    b = params['b']

    x0 = np.asarray(x0)
    x1 = np.asarray(x1)

    dt = delta_t(t)
    exp_t = np.exp(-t)
    exp_2t = np.exp(-2.0 * t)

    # Posterior of a_0 given x_0
    # p(a_0) * p(x_0|a_0) ~ N(a_0; m_post, v_post)
    v_post_inv = 1.0 / sigma0_sq + exp_2t / dt
    v_post = 1.0 / v_post_inv
    m_post_unnorm = mu0 / sigma0_sq + exp_t * x0 / dt
    m_post = v_post * m_post_unnorm

    # Normalization constant for p(a_0) * p(x_0|a_0)
    # This is sqrt(2π * v_post) times the constant from Gaussian product
    log_norm_post = 0.5 * (
        -np.log(2.0 * np.pi * sigma0_sq)
        - mu0 ** 2 / sigma0_sq
        - np.log(2.0 * np.pi * dt)
        - x0 ** 2 / dt
    ) - 0.5 * np.log(v_post_inv / (2.0 * np.pi))

    # Use Gauss-Hermite quadrature to integrate over a_0
    # ∫ N(a_0; m_post, v_post) * h(x_1 - exp_t*α*a_0) da_0

    # Gauss-Hermite nodes and weights (n=20 is usually enough)
    from numpy.polynomial.hermite import hermgauss
    nodes, weights = hermgauss(20)

    # Transform to a_0 space: a_0 = m_post + √(2*v_post) * node
    a0_vals = m_post + np.sqrt(2.0 * v_post) * nodes

    # Compute Laplace-Gaussian convolution at each node
    # h(y) where y = x_1 - exp_t*α*a_0
    y_vals = x1 - exp_t * alpha * a0_vals
    b_eff = exp_t * b
    h_vals = laplace_gaussian_convolution(y_vals, b_eff, dt)

    # Quadrature
    integral = np.sum(weights * h_vals) / np.sqrt(np.pi)

    # Final density: integral * normalization
    density = np.exp(log_norm_post) * integral

    return density


def laplace_k1_score(x0, x1, t, params, eps=1e-5):
    """
    Score (gradient of log density) for K=1 model via numerical differentiation.

    Parameters
    ----------
    x0, x1 : float or array
        Observations
    t : float
        Diffusion time
    params : dict
        Model parameters
    eps : float
        Finite difference step

    Returns
    -------
    tuple of (float or array, float or array)
        (∂/∂x_0 log p_t, ∂/∂x_1 log p_t)
    """
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)

    # Gradient w.r.t. x0
    p_center = laplace_k1_density(x0, x1, t, params)
    p_plus_x0 = laplace_k1_density(x0 + eps, x1, t, params)
    p_minus_x0 = laplace_k1_density(x0 - eps, x1, t, params)

    score_x0 = (np.log(p_plus_x0 + 1e-300) - np.log(p_minus_x0 + 1e-300)) / (2.0 * eps)

    # Gradient w.r.t. x1
    p_plus_x1 = laplace_k1_density(x0, x1 + eps, t, params)
    p_minus_x1 = laplace_k1_density(x0, x1 - eps, t, params)

    score_x1 = (np.log(p_plus_x1 + 1e-300) - np.log(p_minus_x1 + 1e-300)) / (2.0 * eps)

    return score_x0, score_x1


def laplace_k2_density(x, t, params):
    """
    Exact noisy density p_t(x_0, x_1, x_2) for K=2 (three frames).

    Uses sequential 1D integration over a_0 and a_1.

    Parameters
    ----------
    x : array-like, shape (3,) or (n, 3)
        Observations [x_0, x_1, x_2]
    t : float
        Diffusion time
    params : dict
        Must contain: 'mu0', 'sigma0_sq', 'alpha', 'b'

    Returns
    -------
    float or array
        Joint density p_t(x)
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]

    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]

    mu0 = params['mu0']
    sigma0_sq = params['sigma0_sq']
    alpha = params['alpha']
    b = params['b']

    dt = delta_t(t)
    exp_t = np.exp(-t)
    exp_2t = np.exp(-2.0 * t)

    # We integrate: ∫∫ p(a_0) g(x_0|a_0) h(x_1 - exp_t*α*a_0)
    #                     × ∫ h(x_2 - exp_t*α*a_1) da_1 da_0
    #
    # The inner integral over a_1 is: ∫ h(x_2 - exp_t*α*a_1) da_1
    # But we also have M(a_1|a_0) = Laplace(a_1; α*a_0, b)
    # So: ∫ M(a_1|a_0) g(x_1|a_1) h(x_2 - exp_t*α*a_1) da_1

    def integrand_a1(a1, a0):
        """Integrand over a_1 for fixed a_0."""
        # M(a_1|a_0) = Laplace density
        laplace_term = np.exp(-np.abs(a1 - alpha * a0) / b) / (2.0 * b)

        # g(x_1|a_1) = Normal density
        gaussian_term = np.exp(-(x1 - exp_t * a1) ** 2 / (2.0 * dt)) / np.sqrt(2.0 * np.pi * dt)

        # h(x_2 - exp_t*α*a_1)
        y2 = x2 - exp_t * alpha * a1
        b_eff = exp_t * b
        h_term = laplace_gaussian_convolution(y2, b_eff, dt)

        return laplace_term * gaussian_term * h_term

    def integrand_a0(a0):
        """Integrand over a_0 (after integrating a_1)."""
        # Prior * Likelihood for x_0
        prior_term = np.exp(-(a0 - mu0) ** 2 / (2.0 * sigma0_sq)) / np.sqrt(2.0 * np.pi * sigma0_sq)
        gaussian_x0_term = np.exp(-(x0 - exp_t * a0) ** 2 / (2.0 * dt)) / np.sqrt(2.0 * np.pi * dt)

        # Integrate over a_1 with tighter tolerance for speed
        integral_a1, _ = integrate.quad(integrand_a1, -5, 5, args=(a0,), limit=30, epsabs=1e-5, epsrel=1e-4)

        return prior_term * gaussian_x0_term * integral_a1

    # Integrate over a_0 with tighter tolerance for speed
    density, _ = integrate.quad(integrand_a0, -5, 5, limit=30, epsabs=1e-5, epsrel=1e-4)

    return density


def laplace_k2_score(x, t, params, eps=1e-5):
    """
    Score for K=2 model via numerical differentiation.

    Parameters
    ----------
    x : array-like, shape (3,)
        Observations [x_0, x_1, x_2]
    t : float
        Diffusion time
    params : dict
        Model parameters
    eps : float
        Finite difference step

    Returns
    -------
    array, shape (3,)
        Score vector (∂/∂x_0, ∂/∂x_1, ∂/∂x_2) log p_t
    """
    x = np.asarray(x)

    p_center = laplace_k2_density(x, t, params)

    score = np.zeros(3)
    for i in range(3):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps

        p_plus = laplace_k2_density(x_plus, t, params)
        p_minus = laplace_k2_density(x_minus, t, params)

        score[i] = (np.log(p_plus + 1e-300) - np.log(p_minus + 1e-300)) / (2.0 * eps)

    return score


def gaussian_k1_density(x0, x1, t, params):
    """
    Gaussian benchmark: p_t(x_0, x_1) for AR(1) with Gaussian innovations.

    Clean: a_0 ~ N(μ_0, σ_0²), a_1 = α a_0 + ε (ε ~ N(0, σ_η²))
    After OU: x_k = e^{-t} a_k + √Δ_t z_k
    """
    mu0 = params['mu0']
    sigma0_sq = params['sigma0_sq']
    alpha = params['alpha']
    sigma_eta_sq = params['sigma_eta_sq']

    x0 = np.asarray(x0)
    x1 = np.asarray(x1)

    dt = delta_t(t)
    exp_2t = np.exp(-2.0 * t)

    # Joint distribution of (a_0, a_1):
    # Var(a_0) = sigma0_sq
    # Var(a_1) = α² sigma0_sq + sigma_eta_sq
    # Cov(a_0, a_1) = α sigma0_sq

    var_a0 = sigma0_sq
    var_a1 = alpha ** 2 * sigma0_sq + sigma_eta_sq
    cov_a01 = alpha * sigma0_sq

    # After OU noising:
    # x_k ~ N(exp(-t) a_k, Δ_t)
    # So (x_0, x_1) is jointly Gaussian

    var_x0 = exp_2t * var_a0 + dt
    var_x1 = exp_2t * var_a1 + dt
    cov_x01 = exp_2t * cov_a01

    mean_x0 = 0.0  # assuming zero mean (μ_0 = 0 for simplicity, or set it)
    mean_x1 = 0.0

    # 2D Gaussian density
    sigma_matrix = np.array([
        [var_x0, cov_x01],
        [cov_x01, var_x1]
    ])

    det_sigma = var_x0 * var_x1 - cov_x01 ** 2
    if det_sigma <= 0:
        return np.zeros_like(x0, dtype=float)

    inv_sigma = np.linalg.inv(sigma_matrix)

    dx0 = x0 - mean_x0
    dx1 = x1 - mean_x1

    # Quadratic form
    quad = (
        inv_sigma[0, 0] * dx0 ** 2
        + 2.0 * inv_sigma[0, 1] * dx0 * dx1
        + inv_sigma[1, 1] * dx1 ** 2
    )

    density = np.exp(-0.5 * quad) / np.sqrt((2.0 * np.pi) ** 2 * det_sigma)

    return density


def gaussian_k1_score(x0, x1, t, params):
    """Exact score for Gaussian K=1 (closed form)."""
    mu0 = params['mu0']
    sigma0_sq = params['sigma0_sq']
    alpha = params['alpha']
    sigma_eta_sq = params['sigma_eta_sq']

    x0 = np.asarray(x0)
    x1 = np.asarray(x1)

    dt = delta_t(t)
    exp_2t = np.exp(-2.0 * t)

    var_a0 = sigma0_sq
    var_a1 = alpha ** 2 * sigma0_sq + sigma_eta_sq
    cov_a01 = alpha * sigma0_sq

    var_x0 = exp_2t * var_a0 + dt
    var_x1 = exp_2t * var_a1 + dt
    cov_x01 = exp_2t * cov_a01

    det_sigma = var_x0 * var_x1 - cov_x01 ** 2
    inv_sigma = np.linalg.inv(np.array([[var_x0, cov_x01], [cov_x01, var_x1]]))

    score_x0 = -(inv_sigma[0, 0] * x0 + inv_sigma[0, 1] * x1)
    score_x1 = -(inv_sigma[1, 0] * x0 + inv_sigma[1, 1] * x1)

    return score_x0, score_x1


def gaussian_k2_density(x, t, params):
    """Gaussian K=2: exact density as multivariate Gaussian."""
    mu0 = params['mu0']
    sigma0_sq = params['sigma0_sq']
    alpha = params['alpha']
    sigma_eta_sq = params['sigma_eta_sq']

    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]

    dt = delta_t(t)
    exp_2t = np.exp(-2.0 * t)

    # Build covariance of (a_0, a_1, a_2)
    var_a0 = sigma0_sq
    var_a1 = alpha ** 2 * sigma0_sq + sigma_eta_sq
    var_a2 = alpha ** 2 * var_a1 + sigma_eta_sq

    cov_a01 = alpha * sigma0_sq
    cov_a02 = alpha ** 2 * sigma0_sq
    cov_a12 = alpha * var_a1

    cov_a = np.array([
        [var_a0, cov_a01, cov_a02],
        [cov_a01, var_a1, cov_a12],
        [cov_a02, cov_a12, var_a2]
    ])

    # Covariance of (x_0, x_1, x_2)
    cov_x = exp_2t * cov_a + dt * np.eye(3)

    # Gaussian density
    from scipy.stats import multivariate_normal
    dist = multivariate_normal(mean=np.zeros(3), cov=cov_x)

    if x.shape[0] == 1:
        return dist.pdf(x)[0]
    else:
        return dist.pdf(x)


def gaussian_k2_score(x, t, params):
    """Exact score for Gaussian K=2."""
    mu0 = params['mu0']
    sigma0_sq = params['sigma0_sq']
    alpha = params['alpha']
    sigma_eta_sq = params['sigma_eta_sq']

    x = np.asarray(x)

    dt = delta_t(t)
    exp_2t = np.exp(-2.0 * t)

    var_a0 = sigma0_sq
    var_a1 = alpha ** 2 * sigma0_sq + sigma_eta_sq
    var_a2 = alpha ** 2 * var_a1 + sigma_eta_sq

    cov_a01 = alpha * sigma0_sq
    cov_a02 = alpha ** 2 * sigma0_sq
    cov_a12 = alpha * var_a1

    cov_a = np.array([
        [var_a0, cov_a01, cov_a02],
        [cov_a01, var_a1, cov_a12],
        [cov_a02, cov_a12, var_a2]
    ])

    cov_x = exp_2t * cov_a + dt * np.eye(3)
    inv_cov_x = np.linalg.inv(cov_x)

    score = -inv_cov_x @ x

    return score


def monte_carlo_score_estimate(N_samples, K, t, params, kernel_bw=0.1):
    """
    Estimate score via Monte Carlo sampling + kernel gradient estimation.

    Parameters
    ----------
    N_samples : int
        Number of samples
    K : int
        Number of AR steps (K=1 → N=2 frames, K=2 → N=3 frames)
    t : float
        Diffusion time
    params : dict
        Model parameters including 'distribution' = 'laplace' or 'gaussian'
    kernel_bw : float
        Kernel bandwidth for density/score estimation

    Returns
    -------
    dict
        Keys: 'x' (sampled points), 'score_estimate' (kernel gradient)
    """
    N = K + 1
    mu0 = params['mu0']
    sigma0_sq = params['sigma0_sq']
    alpha = params['alpha']

    # Sample from the clean process
    a_samples = np.zeros((N_samples, N))
    a_samples[:, 0] = np.random.normal(mu0, np.sqrt(sigma0_sq), N_samples)

    dist_type = params.get('distribution', 'gaussian')

    for k in range(1, N):
        if dist_type == 'laplace':
            b = params['b']
            eps = np.random.laplace(0, b, N_samples)
        else:  # gaussian
            sigma_eta_sq = params['sigma_eta_sq']
            eps = np.random.normal(0, np.sqrt(sigma_eta_sq), N_samples)

        a_samples[:, k] = alpha * a_samples[:, k - 1] + eps

    # Add OU noise
    dt = delta_t(t)
    exp_t = np.exp(-t)
    x_samples = exp_t * a_samples + np.sqrt(dt) * np.random.normal(0, 1, (N_samples, N))

    # Estimate score via kernel gradient
    from scipy.spatial.distance import cdist

    # Use a grid of test points
    x_grid = np.linspace(np.percentile(x_samples, 5), np.percentile(x_samples, 95), 20)

    scores_grid = np.zeros((20, N))

    for i, x_test in enumerate(x_grid):
        # Gaussian kernel: exp(-||x - x_test||^2 / (2*bw^2))
        if N == 2:
            # For K=1, use 1D slices
            distances = np.abs(x_samples[:, 0] - x_test)
            kernel_weights = np.exp(-distances ** 2 / (2.0 * kernel_bw ** 2))

            # Kernel density estimate
            density_est = np.mean(kernel_weights) / (np.sqrt(2.0 * np.pi) * kernel_bw)

            # Score = ∇ log p ≈ (1/p) ∇p
            # ∇p via finite differences of kernel average
            eps_kern = 0.01
            distances_plus = np.abs(x_samples[:, 0] - (x_test + eps_kern))
            kernel_weights_plus = np.exp(-distances_plus ** 2 / (2.0 * kernel_bw ** 2))
            density_plus = np.mean(kernel_weights_plus) / (np.sqrt(2.0 * np.pi) * kernel_bw)

            distances_minus = np.abs(x_samples[:, 0] - (x_test - eps_kern))
            kernel_weights_minus = np.exp(-distances_minus ** 2 / (2.0 * kernel_bw ** 2))
            density_minus = np.mean(kernel_weights_minus) / (np.sqrt(2.0 * np.pi) * kernel_bw)

            grad_density = (density_plus - density_minus) / (2.0 * eps_kern)
            score_est = grad_density / (density_est + 1e-10)
            scores_grid[i, 0] = score_est
        # For higher dimensions, similar approach but more complex

    return {
        'x': x_grid,
        'x_samples': x_samples,
        'score_estimate': scores_grid
    }
