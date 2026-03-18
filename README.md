# Score-Based Generative Diffusion Models for Dynamic Objects

**Studying the joint score of a diffusion model over entire sequences**

*Giovanni Mantovani & Marco Lomele · MSc Thesis · Data Science, Bocconi University · 2025–2026*
*Supervisors: Prof. Marc Mézard · Tutor: Jérôme Garnier*

---

## What This Repo Is About

Standard diffusion models generate **static** objects (images). We study the **dynamic** setting:
a *dynamic object* is a sequence of frames indexed by *internal time* `u = 0, 1, ..., K−1`
(a video, a rotating digit, a stochastic trajectory). The naive approach embeds the full
trajectory in ℝ^{dK} and learns a single score — wasteful, because consecutive frames are
highly correlated.

**Our goal:** understand and exploit the temporal structure of the **joint score field**

```
S(x, t) = ∇_{(x_0,...,x_{K-1})} log P_t(x_0, x_1, ..., x_{K-1})
```

where `t` is the diffusion (OU noise) time applied independently per frame.
We hypothesize a **propagator** `P` such that consecutive block components satisfy:

```
S_{:,u}(x)  ≈  P_t( S_{:,u−1}(x) )
```

We attack this with **fully solvable Gaussian toy models** — exact closed-form scores,
interactive browser-based laboratories, no neural networks.

---

## Key Conceptual Point (Mézard meeting, 2026-03-12)

> **The score must be computed from the joint distribution `P_t(x_0,...,x_K)`, not from
> the product of marginals `∏_u p_t(x_u | u)`.**

For a Markov sequence with transition kernel `M(a_{u+1} | a_u)`:

```
P_0(a_0,...,a_K) = p_0(a_0) · M(a_1|a_0) · ... · M(a_K|a_{K-1})
```

After independent OU diffusion of each frame at time `t`:

```
P_t(x_0,...,x_K) = Σ_{a} P_0(a) × ∏_k N(x_k ; a_k e^{−t}, Δ_t)
```

The k-th component of the joint score is (Mézard, board, 2026-03-12):

```
(∇ log P_t)_k = ⟨(a_k e^{−t} − x_k) / Δ_t⟩_{a | x}
```

a conditional expectation over the **full posterior** of clean frames.
For Gaussian AR(1) this is **exactly linear in x** and solvable by Kalman smoothing.

---

## Quickstart

All toy models are self-contained HTML files. No installation required.

```bash
git clone https://github.com/giovimanto/Score_Diffusion.git
cd Score_Diffusion

# Open any toy model directly
open Toy-models/Toy-model_5/score_two_clocks_lab.html

# Or serve locally (if CDN is unavailable)
python3 -m http.server 8080
# then visit http://localhost:8080/...
```

**Live demos** (GitHub Pages):

| Model | Link | Status |
|-------|------|--------|
| TM3 — Rotation + OU (per-frame) | [rotation\_ou\_score\_lab.html](https://giovimanto.github.io/Score_Diffusion/Toy-models/Toy-model_3/rotation_ou_score_lab.html) | old formulation |
| TM5 — Two Clocks (per-frame) | [score\_two\_clocks\_lab.html](https://giovimanto.github.io/Score_Diffusion/Toy-models/Toy-model_5/score_two_clocks_lab.html) | old formulation |
| TM6 — Circle & Propagator | [circle\_score\_lab.html](https://giovimanto.github.io/Score_Diffusion/Toy-models/Toy-model_6/circle_score_lab.html) | old formulation |

---

## Toy Models

All models use exact analytical scores, MathJax-rendered equations, and Plotly
interactive visualisations. No build step — open HTML in browser.

| # | Model | Dimension | Internal dynamics | Score type | Key observable |
|---|-------|-----------|-------------------|------------|----------------|
| 1 | [OU Diffusion Explainer](Toy-models/Toy-model_1/ou_diffusion_explainer.html) | 2 | Static (no sequence) | Marginal | 2D Gaussian mixture forward/reverse animation |
| 2 | [Trimodal Joint Distribution](Toy-models/toy-model-2/toy-model-2.html) | 1 | `x_{u+1} = x_u + c + η` | **Joint (2-frame)** | Joint score field on (x_0, x_1) plane |
| 3 | [Rotation + OU](Toy-models/Toy-model_3/rotation_ou_score_lab.html) | 2 | `a(u+1) = R_{Δθ} a(u) + ξ_u` | Marginal [old] | Score quiver + Jacobian eigenvalues |
| 4 | [Bouncing Particle](Toy-models/toy-model-4/bouncing_cube_score_lab.html) | 3 | Billiard in [0,L]³ | Marginal [old] | Marginal vs conditional score |
| 5 | [Score in Two Clocks](Toy-models/Toy-model_5/score_two_clocks_lab.html) ⭐ | 1–2 | AR(1): `x_{u+1} = αx_u + η_u` | Marginal [old] | (u,t) phase diagram; 4 interactive labs |
| 6 | [Score on the Circle](Toy-models/Toy-model_6/circle_score_lab.html) | 2 | Circle SDE | Marginal [old] | Separable score; order parameter ODE |

> **Note:** TM3, TM4, TM5, TM6 computed the *per-frame marginal* score `s(x_u,u,t) = ∇ log p_t(x_u|u)`.
> The correct object for the propagator is the **joint** score over the full sequence.
> TM2 already uses a 2-frame joint distribution — it is closest to the correct formulation.

### What TM5 (Two Clocks) showed — per-frame perspective

- AR(1) dynamics with exact Gaussian marginals at each `u`
- Four regimes on the `(u, t)` plane: sharp (I), best window (II), stationary (III), universal noise (IV)
- Jacobian `J_s = −1/σ²_u(t)`, Fisher information triad `I = −J = 1/σ²`
- This per-frame picture remains useful as a *single-frame baseline* but misses inter-frame coupling in the score

---

## Theory Notes

Key documents (in parent `Diffusion/` directory):

| File | Content |
|------|---------|
| `Notes/Problem_formulation.pdf` | Thesis framing: circle model, propagator hypothesis, Q1–Q7 |
| `Notes/2nd-meeting-notes.pdf` | Board exercises: 2D rotation (Ex.1), 1D 2-frame joint (Ex.2), fixed-u slice (Ex.3) |
| `Papers/dynamical-regimes_diff.pdf` | Biroli–Bonnaire–de Bortoli–Mézard: speciation `t_S`, collapse `t_C` |
| `Generative_diffusion_updated_notes_MM.pdf` | Mézard lecture notes: OU, score, speciation, collapse |
| `thesis_achilli_final.pdf` | Achilli: score learning background, intermediate-t optimality |

---

## Reports

- [`reports/mezard_update.html`](reports/mezard_update.html) — Progress report for Prof. Marc Mézard (March 2026)

---

## Research Questions (updated after 2026-03-12 meeting)

**Core question:**
> Given a Markov sequence of length K with known dynamics, does the joint score field
> `S(x,t) = −Σ_t^{-1}(x − μ_t)` admit a sparse/propagator structure that can be
> exploited for efficient generation?

Sub-questions:

- **Q1.** What is the exact joint score for K-frame AR(1)? Compute Σ_t and its inverse analytically.
- **Q2.** Is Σ_t^{-1} approximately tridiagonal for intermediate t? (It is exactly so at t→0.)
- **Q3.** Does the k-th block of the joint score depend primarily on its neighbors x_{k-1}, x_{k+1}?
- **Q4.** Can the Kalman smoother provide a closed-form propagator P that maps S_{k-1} → S_k?
- **Q5.** How does the inter-frame coupling in the score decay with t? At what t does it reach 1/e?
- **Q6.** For non-Gaussian (mixture) priors, does the joint score remain approximately linear?
- **Q7.** Does knowing P enable better/faster reverse sampling for the full trajectory?

---

## Contributing / Development

- Follow the design system in `CLAUDE.md` (fonts, colors, layout, notation)
- New toy models go in `Toy-models/Toy-model_N/`
- Keep scores closed-form; label `[exact]` or `[hypothesis]` explicitly
- **Always state whether the score computed is joint or marginal**
- See `CLAUDE.md` for full notation, hypotheses, and milestones
