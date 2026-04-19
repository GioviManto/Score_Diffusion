# The Anatomy of Σ_t and Σ_t⁻¹

> How Diffusion Noise Reshapes the Score of an AR(1) Sequence

## What this is

A focused 12-page analysis of the matrices Σ_t and Σ_t⁻¹ that control the joint score S(x,t) = −Σ_t⁻¹(x − μ_t) in a diffusion model applied to an AR(1) Gaussian sequence.

## Contents

```
anatomy_sigma_t/
├── README.md
├── anatomy_sigma_t.tex          # LaTeX source
├── anatomy_sigma_t.pdf          # Compiled document
├── figures/                     # All figures (PDF)
│   ├── fig1_clean.pdf           # Σ₀ vs Σ₀⁻¹
│   ├── fig2_prec_t.pdf          # Σ_t⁻¹ at four t values
│   ├── fig3_coupling.pdf        # Coupling profiles and decay
│   ├── fig4_diagnostics.pdf     # Tridiag fraction + condition number
│   ├── fig5_spectral.pdf        # Eigenvalues and eigenvectors
│   ├── fig6_eigenbasis.pdf      # Diagonality in eigenbasis
│   └── fig7_neumann.pdf         # Neumann series band-by-band
└── src/
    └── generate_figures.py      # Reproduces all figures
```

## Reproduce

```bash
pip install numpy matplotlib scipy
cd src && python generate_figures.py
cd .. && pdflatex anatomy_sigma_t.tex && pdflatex anatomy_sigma_t.tex
```

## Parameters

K = 59 (60 frames), α = 0.9, σ²_η = 0.19 (stationary with σ²_∞ = 1)
