# CLAUDE.md — Score-Based Diffusion for Dynamic Objects

## Project Aim

We study how the **score field** `s(x, u, t) = ∇_x log p_t(x | u)` of a generative
diffusion model evolves across **internal time** `u` (frame index of a dynamic object
such as a video or trajectory), for a fixed **diffusion time** `t`.

The central hypothesis is the existence of a **propagator** `P` such that:

```
S_{:,u}(x) ≈ P( S_{:,u−1}(x) )
```

where `P` may be linear, stationary (independent of `u`), and related to the generator
of the underlying SDE. We pursue this with **fully solvable Gaussian toy models** —
no neural networks, closed-form scores throughout.

**Post-Achilli-call direction (March 2026):** write a closed SDE/ODE for the score
coefficients evolving in `u` — treating (mean, covariance) as order parameters.

---

## Repo Map

```
Score_Diffusion/
├── Toy-models/
│   ├── Toy-model_1/ou_diffusion_explainer.html     # 2D Gaussian mixture + OU animation
│   ├── toy-model-2/toy-model-2.html                 # 1D trimodal joint p(x0,x1), score field
│   ├── Toy-model_3/rotation_ou_score_lab.html       # 2D rotating AR(1), Jacobian eigenvalues
│   ├── toy-model-4/bouncing_cube_score_lab.html     # 3D billiard, marginal vs conditional score
│   ├── Toy-model_5/score_two_clocks_lab.html        # MAIN: (u,t) plane, 4 labs, full analytics
  └── Toy-model_6/circle_score_lab.html           # NEW: circle model, order param SDE, prop. error
├── reports/
│   └── mezard_update.html                           # Email report for Prof. Mézard
├── README.md
└── CLAUDE.md                                        # (this file)
```

**Parent directory** (`Desktop/Diffusion/`):
```
Notes/Problem_formulation.pdf    # thesis direction notes (circle model, propagator def.)
Notes/2nd-meeting-notes.pdf      # board exercises: circle, 1D kernel, fixed-u slice
Papers/dynamical-regimes_diff.pdf # Biroli-Bonnaire-de Bortoli-Mézard regimes paper
Papers/Speciation-Transition.pdf  # speciation transition
thesis_achilli_final.pdf          # Achilli thesis (relevant background)
Generative_diffusion_updated_notes_MM.pdf  # Mézard lecture notes
iap-diffusion-labs/labs/          # MIT IAP 2026 Jupyter notebooks (OU, Langevin, diffusion)
```

---

## How to Run

**No build step.** All toy models are self-contained HTML files. Open directly:

```bash
# Open any toy model in browser
open Score_Diffusion/Toy-models/Toy-model_5/score_two_clocks_lab.html

# Or serve locally (avoids CDN dependency)
cd Score_Diffusion
python3 -m http.server 8080
# then visit http://localhost:8080/Toy-models/Toy-model_5/score_two_clocks_lab.html
```

**GitHub Pages** (live, no install):
```
https://giovimanto.github.io/Score_Diffusion/Toy-models/Toy-model_5/score_two_clocks_lab.html
https://giovimanto.github.io/Score_Diffusion/Toy-models/Toy-model_3/rotation_ou_score_lab.html
```

**Python environment** (for notebooks in `iap-diffusion-labs/`):
```bash
cd Desktop/Diffusion
source venv/bin/activate        # virtualenv present
pip install numpy scipy matplotlib jupyter celluloid
jupyter notebook iap-diffusion-labs/labs/lab_one.ipynb
```

---

## Key Theory Pointers

| Document | What it contributes |
|----------|---------------------|
| `Notes/Problem_formulation.pdf` | Core thesis framing: circle toy model, propagator hypothesis `S_{:,u} ≈ P(S_{:,u−1})`, research questions Q1–Q7 |
| `Notes/2nd-meeting-notes.pdf` | Board exercises: 2D rotation score (Exercise 1), 1D one-step kernel joint score (Exercise 2), radial/elliptic structure (Exercise 3) |
| `Papers/dynamical-regimes_diff.pdf` | Biroli–Bonnaire–de Bortoli–Mézard: speciation transition `t_S`, collapse `t_C`, four regimes of reverse diffusion |
| `Generative_diffusion_updated_notes_MM.pdf` | Mézard lecture notes: OU process, backward SDE, score formulas, speciation, collapse |
| `thesis_achilli_final.pdf` | Achilli: background on score learning, intermediate-t optimality, phase transitions |

---

## Research Questions & Metrics

### Propagator hypotheses

**H1 (Linear propagator):** For Gaussian dynamics, `s(x,u,t)` is linear in `x`.
The propagator `P_t` maps linear score fields to linear score fields.
Exact for all Gaussian models. ✓

**H2 (Rotation propagator):** For rotating AR(1) with isotropic covariance,
`[P · s_u](x) = R_{Δθ} · s(R_{−Δθ} x, u, t)`. Exact at stationarity; approximate for anisotropic Σ. ✓

**H3 (Mean contraction):** The mean component of the score propagates as
`μ_{u+1}(t) = α · μ_u(t)`, i.e. a pure α-contraction independent of `t`. Exact. ✓

**H4 (Score ODE):** The score coefficients (A(u,t), b(u,t)) satisfy closed ODEs in `u`:
```
dA/du = −A² · ∂σ²_u(t)/∂u           (Riccati-type)
db/du = log(α) · b(u,t) + (noise correction)
```
*Direction being developed (post-Achilli call).*

### Evaluation metrics (to implement)

- **Relative L2 error:** `‖s(·,u,t) − P·s(·,u−1,t)‖ / ‖s(·,u,t)‖`  vs. `t`
- **Correlation:** Pearson correlation between consecutive score fields across ensemble
- **Spectral gap:** `λ_1(J_s) − λ_2(J_s)` as function of `(u,t)` — measures anisotropy
- **Regime boundaries:** values of `(u,t)` where spectral gap drops to 1/e of its max

### Spectral / Jacobian diagnostics

- `J_s(u,t) = −Σ_u(t)^{−1}` (exact for Gaussians)
- Triad identity: `I(u,t) = −J_s = 1/σ²_u(t)` (Fisher = −Jacobian = inverse variance)
- Dimensional collapse: all `λ_k(J_s) → −1` as `t → ∞`

### Four regimes on (u,t) plane

| Phase | Region | Description |
|-------|--------|-------------|
| I — Sharp | small u, small t | High anisotropy, data-specific score |
| II — Best window | any u, intermediate t | Near-stationary covariance; propagator clearest |
| III — Stationary | large u, small t | σ²_u → σ²_∞; variance-part of P ≈ 1 |
| IV — Universal noise | large t | s ≈ −x; all dynamical info erased |

---

## Design System (all toy-model HTML files)

```
Fonts:  EB Garamond (body),  JetBrains Mono (code/labels)
Colors: --bg:#f7f4ef  --ink:#1a1410  --accent:#b5341a  --accent2:#1a3a5c
Libs:   MathJax 3.2.2,  Plotly 2.27.0  (both CDN)
Width:  max ~900–1050px, academic paper style
```

---

## Open TODOs / Next Milestones

- [ ] **Derive closed ODE for score coefficients** `(A(u,t), b(u,t))` in `u` for Gaussian AR(1)+OU
- [ ] **Circle toy model (TM6):** stochastic circle dynamics `dr=(1−r)du+√(2T_r)dB_r`, `dθ=ωdu+√(2T_θ)dB_θ`; exact analytic score via radial×angular factorisation
- [ ] **Quantitative propagator metrics:** Python script for L2 error ‖P·s_{u−1} − s_u‖ vs `t`
- [ ] **Non-Gaussian extension:** 2-component rotating mixture; measure deviation from linear-P hypothesis
- [ ] **Connect to speciation/collapse:** identify Phase II boundary sharpness in large-d limit
- [ ] **Report update:** add quantitative propagator error plots to Mézard report

---

## Notation (use consistently)

| Symbol | Meaning |
|--------|---------|
| `u` | Internal time (frame index, trajectory parameter) |
| `t` | Diffusion time (OU noise level) |
| `s(x,u,t)` | Per-frame score = ∇_x log p_t(x\|u) |
| `P` or `P_t` | Propagator mapping s_{u−1} → s_u |
| `α` | AR(1) contraction coefficient |
| `σ²_∞` | Stationary variance = σ²_η / (1−α²) |
| `R_{Δθ}` | 2D rotation matrix, angle Δθ |
| `J_s` | Jacobian ∂s/∂x = −Σ(u,t)^{−1} |
| `I(u,t)` | Fisher information = 1/σ²_u(t) |
