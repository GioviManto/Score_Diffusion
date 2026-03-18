# Structure of Σ_t and Σ_t⁻¹ in the AR(1) Diffusion Model

> Part of the MSc Thesis on Score-Based Diffusion Models for Sequences  
> Giovanni Mantovani & Marco Lomelé — Bocconi University, 2025–2026  
> Supervisors: Prof. Marc Mézard, Tutor: Jérôme Garnier-Brun

## Overview

This folder contains the analysis of the noisy covariance matrix Σ_t = e^{-2t} Σ₀ + Δ_t I and its inverse Σ_t⁻¹, which encodes the **entire structure of the joint score** S(x, t) = −Σ_t⁻¹(x − μ_t) for a diffusion model applied to an AR(1) Gaussian sequence.

The central research question (Mézard, March 18 2026):
> *"What is the impact of the Markovian process underlying the sequence of frames on the structure of the score?"*

### Key results

1. **Four closed-form representations of Σ_t⁻¹** — factored, Mézard's form, Woodbury, and spectral.
2. **Numerical study (K=19 frames)** — heatmaps, tridiagonal weight fraction, off-diagonal decay, row profiles, condition number.
3. **Spectral decomposition** — Σ_t⁻¹ shares eigenvectors with Σ₀ for all t. In the eigenbasis the problem fully decouples into K+1 independent scalar denoisers. This is a consequence of **Gaussianity**, not Markovianity.

## Repository contents

```
sigma_t_analysis/
├── README.md                    # This file
├── structure_sigma_t.tex        # Full LaTeX source (18 pages)
├── structure_sigma_t.pdf        # Compiled document
├── figures/                     # All generated figures (PDF)
│   ├── fig1_sigma0_vs_prec0.pdf
│   ├── fig2_sigma_t_panel.pdf
│   ├── fig3_tridiag_fraction.pdf
│   ├── fig4_row_profile.pdf
│   ├── fig5_eigenvalues.pdf
│   ├── fig6_eigenbasis.pdf
│   ├── fig7_eigenvalue_trajectories.pdf
│   ├── fig8_offdiag_decay.pdf
│   ├── fig9_condition_number.pdf
│   └── fig11_alpha_sensitivity.pdf
└── src/
    └── generate_figures.py      # Python script to reproduce all figures
```

## How to reproduce

### Requirements

```bash
pip install numpy matplotlib scipy
```

LaTeX compilation requires: `texlive-latex-extra`, `texlive-science` (for `physics.sty`), `texlive-latex-recommended`.

### Generate figures

```bash
cd sigma_t_analysis/src
python generate_figures.py
```

This creates all figures in `../figures/`. The script also prints the explicit 5×5 numerical matrices at t = 0, 0.3, 1.0, 3.0 to stdout.

### Compile the document

```bash
cd sigma_t_analysis
pdflatex structure_sigma_t.tex
pdflatex structure_sigma_t.tex   # second pass for TOC and cross-refs
```

## Parameters used

| Parameter | Value | Meaning |
|-----------|-------|---------|
| K | 19 | 20 frames total |
| α | 0.9 | AR(1) contraction coefficient |
| σ₀² | 1.0 | Initial variance |
| σ_η² | 1 − α² = 0.19 | Innovation variance (stationary regime) |
| σ_∞² | 1.0 | Stationary variance |

## Modifying the analysis

To change parameters, edit the top of `src/generate_figures.py`:

```python
K = 19           # number of AR(1) steps
alpha = 0.9      # AR(1) coefficient
sigma_0_sq = 1.0 # initial variance
sigma_eta_sq = 1.0 - alpha**2  # stationary noise
```

Then re-run the script and recompile the LaTeX.

## Connection to previous work

This document builds on the derivation in `Draft-AR1.pdf` (Joint Score of a Diffusion Model Applied to an AR(1) Sequence), which established:
- The joint score S(x, t) = −Σ_t⁻¹(x − μ_t)
- The equivalence with Mézard's conditional expectation formula
- The Kalman smoother as the exact propagator

The present analysis answers the follow-up question: *what does Σ_t⁻¹ actually look like?*
