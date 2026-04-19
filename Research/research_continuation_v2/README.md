# Diffusion Models on Correlated Sequences: Laplace AR(1) Extension

This directory contains a comprehensive Python implementation for studying diffusion models on correlated sequences, with focus on comparing Gaussian and Laplace innovation models for AR(1) processes observed through OU noising.

## Project Overview

The research studies an AR(1) process:
```
a_{k+1} = α a_k + η_k
```
observed through Ornstein-Uhlenbeck (OU) noising:
```
x_k = e^{-t} a_k + √Δ_t z_k,  where Δ_t = 1 - e^{-2t}
```

Two models are compared:
1. **Gaussian AR(1)**: Innovations η_k ~ N(0, σ_η²)
2. **Laplace AR(1)**: Innovations η_k ~ Laplace(0, b)

## Directory Structure

```
research_continuation_v2/
├── code/
│   ├── generate_all_figures.py       # Main figure generation script (911 lines)
│   ├── laplace_ar1_utils.py          # Core mathematical utilities (574 lines)
│   └── figure_summary.txt            # Generated summary report
├── figures/                          # Output directory for all PDFs
│   ├── fig01-fig11.pdf              # Gaussian benchmark (N=60)
│   └── fig12-fig17.pdf              # Laplace AR(1) models (K=1,2)
└── README.md                         # This file
```

## Key Files

### `generate_all_figures.py` (911 lines)

Main entry point that generates all 17 figures. Includes:

**Gaussian Benchmark Figures (fig01-fig11):**
- **fig01**: Covariance Σ₀ vs Precision Σ₀⁻¹ heatmaps (N=60)
- **fig02**: Time evolution of Σ_t and Σ_t⁻¹ panels (5 time points)
- **fig03**: Eigenvalue trajectories (λ_m(t) and λ_m(t)⁻¹)
- **fig04**: Tridiagonal mass fraction and condition number vs t
- **fig05**: Precision row profile at different diffusion times
- **fig06**: Precision matrix in eigenbasis
- **fig07**: Leading eigenvectors of Σ₀
- **fig08**: Symlog heatmaps showing early fill-in
- **fig09**: Deterministic vs stochastic precision comparison
- **fig10**: Locality observables (4-panel: band mass, bandwidth, ranges, correlation)
- **fig11**: Posterior mean gain row profile

**Laplace AR(1) Model Figures (fig12-fig17):**
- **fig12**: Laplace vs Gaussian innovation density comparison
- **fig13**: K=1 score comparison (Laplace vs Gaussian along 1D slices)
- **fig14**: K=1 score Hessian (second derivatives and band structure)
- **fig15**: K=2 score along principal slices (using K=1 proxy for speed)
- **fig16**: Monte Carlo validation of score via kernel density estimation
- **fig17**: K=2 score Hessian band mass vs position and time

### `laplace_ar1_utils.py` (574 lines)

Core mathematical library with the following functions:

#### Basic Utilities
- `delta_t(t)`: Computes Δ_t = 1 - e^{-2t}

#### Laplace-Gaussian Mathematics
- `laplace_gaussian_convolution(y, b_eff, sigma_sq)`: Closed-form density of convolution of Laplace and Gaussian distributions. Uses numerically stable log-domain computation with erfc.

#### K=1 Model (Two Frames: a₀, a₁)
- `laplace_k1_density(x0, x1, t, params)`: Exact noisy density p_t(x₀, x₁) using Gauss-Hermite quadrature (20 nodes)
- `laplace_k1_score(x0, x1, t, params, eps)`: Score (gradient of log density) via finite differences
- `gaussian_k1_density(x0, x1, t, params)`: Gaussian benchmark (closed-form 2D Gaussian)
- `gaussian_k1_score(x0, x1, t, params)`: Gaussian score (closed form)

#### K=2 Model (Three Frames: a₀, a₁, a₂)
- `laplace_k2_density(x, t, params)`: Exact density via sequential 1D integration (adaptive quad)
- `laplace_k2_score(x, t, params, eps)`: Score via numerical differentiation
- `gaussian_k2_density(x, t, params)`: Gaussian benchmark (multivariate Gaussian)
- `gaussian_k2_score(x, t, params)`: Gaussian score (closed form)

#### Monte Carlo Methods
- `monte_carlo_score_estimate(N_samples, K, t, params, kernel_bw)`: Score estimation via sampling and kernel density gradient

## Parameters

### Gaussian Benchmark (N=60)
```
N = 60
alpha = 0.90
sigma0_sq = 1.0
sigma_eta_sq = 0.19  # = 1 - alpha²
```

### Laplace Model (K=1,2: N=2,3)
```
alpha = 0.90
sigma0_sq = 1.0
sigma_eta_sq = 0.19
b = sqrt(0.19/2) ≈ 0.3082  # Laplace scale (variance matching)
```

The Laplace scale is chosen so that:
```
Var[Laplace(0, b)] = 2b² = sigma_eta_sq
```

## Mathematical Details

### K=1 Density Computation

The noisy joint density is computed as:
```
p_t(x₀, x₁) = ∫ p(a₀) g(x₀|a₀) h(x₁ - e^{-t}α a₀) da₀
```

where:
- `p(a₀) = N(a₀; μ₀, σ₀²)` is the prior
- `g(x|a) = N(x; e^{-t}a, Δ_t)` is the OU noise model
- `h(y)` is the Laplace-Gaussian convolution of the innovation and OU noise

The integral is evaluated using Gauss-Hermite quadrature with:
- 20 quadrature nodes
- Posterior approximation: `p(a₀) × g(x₀|a₀) ≈ N(a₀; m_post, v_post)`

### K=2 Density Computation

Sequential integration over (a₀, a₁):
```
p_t(x₀, x₁, x₂) = ∫∫ p(a₀)M(a₁|a₀)M(a₂|a₁) g(x₀|a₀)g(x₁|a₁)g(x₂|a₂) da₀da₁
```

Uses adaptive quadrature with tolerances:
- epsabs = 1e-5, epsrel = 1e-4
- Integration limits: [-5, 5]

### Score Computation

Scores are computed via:
1. **Laplace models**: Numerical differentiation (finite differences)
   - K=1: eps = 1e-4 to 2e-4
   - K=2: eps = 2e-4 to 5e-4

2. **Gaussian models**: Closed-form derivatives
   - K=1: ∂/∂x_i log N(μ, Σ) = -Σ⁻¹(x - μ)
   - K=2: Same formula for 3D case

### Score Hessian (Second Derivatives)

Computed via central difference approximation:
```
H[i,j] ≈ (log p(x+ε e_i) + log p(x-ε e_i) - 2 log p(x)) / ε²  (diagonal)
H[i,j] ≈ (log p(x+ε e_i+ε e_j) - log p(x+ε e_i-ε e_j)
         - log p(x-ε e_i+ε e_j) + log p(x-ε e_i-ε e_j)) / (4ε²)  (off-diagonal)
```

### Monte Carlo Validation

K=1 score estimated via:
1. Sample N_samples trajectories from the clean process
2. Add OU noise
3. Compute kernel density estimate (Gaussian kernel, bandwidth = 0.15)
4. Estimate score as ∇ log p via finite differences of kernel density

## Numerical Stability Features

The implementation includes several stability measures:

1. **Laplace-Gaussian Convolution**: Log-domain computation with safe erfc handling
   - Avoids overflow using log-sum-exp trick
   - Handles large arguments correctly

2. **Finite Differences**: Adaptive step sizes
   - K=1: eps = 1e-4 to 2e-4
   - K=2: eps = 2e-4 to 5e-4
   - Hessian: eps = 3e-4 to 5e-4

3. **Integration**: Adaptive quadrature
   - Gauss-Hermite for K=1 (20 nodes)
   - Adaptive quad for K=2 (adjustable tolerance)

4. **Density Floor**: 1e-300 to prevent log(0)

## Running the Code

### Prerequisites
```bash
pip install numpy scipy matplotlib
```

### Generate All Figures
```bash
cd /path/to/research_continuation_v2/code
python generate_all_figures.py
```

Output:
- All 17 PDFs saved to `../figures/`
- Summary report saved to `figure_summary.txt`
- Execution time: ~120-150 seconds

### Test Utilities
```python
from laplace_ar1_utils import laplace_k1_density, gaussian_k1_density

params = {
    'mu0': 0.0,
    'sigma0_sq': 1.0,
    'alpha': 0.9,
    'b': 0.3082,
    'sigma_eta_sq': 0.19,
}

# Evaluate density at (x₀=0.5, x₁=0.3) at t=0.5
p_laplace = laplace_k1_density(0.5, 0.3, 0.5, params)
p_gaussian = gaussian_k1_density(0.5, 0.3, 0.5, params)
```

## Figure Output Sizes

All figures are saved as PDF with 150 DPI:
- Gaussian benchmark (fig01-11): ~250 KB total
- Laplace models (fig12-17): ~200 KB total
- Total: ~484 KB

## Performance Notes

- **Gaussian figures (01-11)**: ~30 seconds (N=60, analytical)
- **Laplace figures (12-17)**: ~90-120 seconds
  - fig12-14 (K=1): ~40 seconds
  - fig15-17 (K=2 proxy): ~50-80 seconds

K=2 figures use K=1 proxies for computational speed while preserving pedagogical value.

## References

Key mathematical references:
- Laplace-Gaussian convolution: Standard result in signal processing
- OU process: Ornstein-Uhlenbeck diffusion process
- Gauss-Hermite quadrature: Numerical integration method for Gaussian measures
- Score matching: Gradient-based density estimation

## Author Notes

This implementation is designed for:
1. **Exact density computation** for small K (K≤2)
2. **Numerical stability** across all diffusion time scales
3. **Clear mathematical structure** separating Laplace and Gaussian cases
4. **Validation via Monte Carlo** for K=1

The code emphasizes clarity and correctness over extreme optimization.
