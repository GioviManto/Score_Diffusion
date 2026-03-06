# Score-Based Generative Diffusion Models for Dynamic Objects

**Exploring the stochastic internal dynamics of the score function**

*Giovanni Manto & Marco · MSc Thesis Direction · 2025–2026*

---

## What This Repo Is About

Standard score-based diffusion models generate static objects (images). We study the
**dynamic** setting: a *dynamic object* is a sequence of frames
`a_u ∈ ℝ^d` indexed by *internal time* `u = 0, 1, …, T−1` (a video, a trajectory, a
rotating digit). Naively, one embeds the full trajectory in ℝ^{dT} and learns a single
score — wasteful, because consecutive frames are highly correlated.

Our goal is to understand and exploit the **temporal structure of the score field**
`s(x, u, t) = ∇_x log p_t(x | u)`,
where `t` is the diffusion (OU noise) time applied independently per frame.
Specifically, we hypothesize a **propagator** `P` such that

```
S_{:,u}(x)  ≈  P( S_{:,u−1}(x) )
```

and ask: Is `P` linear? Stationary (independent of `u`)? Related to the generator of
the underlying SDE? Can it be read off analytically?

We attack these questions with **fully solvable Gaussian toy models** — no neural
networks, exact closed-form scores — and interactive browser-based laboratories.

---

## Quickstart

All toy models are self-contained HTML files. No installation required.

```bash
git clone https://github.com/giovimanto/Score_Diffusion.git
cd Score_Diffusion

# Open the main deliverable directly
open Toy-models/Toy-model_5/score_two_clocks_lab.html

# Or serve locally (if CDN is unavailable)
python3 -m http.server 8080
# then visit http://localhost:8080/Toy-models/Toy-model_5/score_two_clocks_lab.html
```

**Live demos** (GitHub Pages, no install):

| Model | Link |
|-------|------|
| TM3 — Rotation + OU | [rotation\_ou\_score\_lab.html](https://giovimanto.github.io/Score_Diffusion/Toy-models/Toy-model_3/rotation_ou_score_lab.html) |
| TM5 — Two Clocks (main) | [score\_two\_clocks\_lab.html](https://giovimanto.github.io/Score_Diffusion/Toy-models/Toy-model_5/score_two_clocks_lab.html) |
| TM6 — Circle & Propagator | [circle\_score\_lab.html](https://giovimanto.github.io/Score_Diffusion/Toy-models/Toy-model_6/circle_score_lab.html) |

---

## Toy Models

All models use exact analytical scores, MathJax-rendered equations, and Plotly
interactive visualisations. No build step — open HTML in browser.

| # | Model | Dimension | Internal dynamics | Key observable |
|---|-------|-----------|-------------------|----------------|
| 1 | [OU Diffusion Explainer](Toy-models/Toy-model_1/ou_diffusion_explainer.html) | 2 | Static (no u) | 2D Gaussian mixture forward/reverse animation |
| 2 | [Trimodal Joint Distribution](Toy-models/toy-model-2/toy-model-2.html) | 1 | `x_{u+1} = x_u + c + η` | Joint score field (x_0, x_1) plane |
| 3 | [Rotation + OU Score Lab](Toy-models/Toy-model_3/rotation_ou_score_lab.html) | 2 | `a(u+1) = R_{Δθ} a(u) + ξ_u` | Score quiver field + Jacobian eigenvalues |
| 4 | [Bouncing Particle](Toy-models/toy-model-4/bouncing_cube_score_lab.html) | 3 | Billiard in [0,L]³ | Marginal (KDE) vs conditional (folded Gaussian) score |
| 5 | [Score in Two Clocks](Toy-models/Toy-model_5/score_two_clocks_lab.html) ⭐ | 1–2 | AR(1): `x_{u+1} = αx_u + η_u` | Full (u,t) phase diagram; 4 interactive labs |
| 6 | [Score on the Circle](Toy-models/Toy-model_6/circle_score_lab.html) 🆕 | 2 | Circle SDE: `dr=(1−r)du+√(2T_r)dB`, `dθ=ωdu+√(2T_θ)dB` | Separable exact score; 4-ODE order parameter; analytical propagator error ε(u,t)=2T_θ/(2T_θu+1−e^{−2t}) |

### Toy Model 5 — The Two-Clock Lab (main deliverable)

The central laboratory. Two independent time parameters:
- `u` — internal time (semantic evolution of the dynamic object)
- `t` — diffusion time (OU noise clock, independent per frame)

**Four interactive labs:**
1. **Lab 1:** 1D distribution p_u(x) + score field + score norm vs t (sliders: u, t, α, μ₀, σ₀, σ_η)
2. **Lab 2:** (u,t) plane heatmap of score norm / entropy / Fisher / variance — the four-regime phase diagram
3. **Lab 3:** 2D rotating AR(1) score vector field + live Jacobian eigenvalues in status bar
4. **Lab 4:** Forward noising paths (black) + reverse generative paths (pink) at fixed u

---

## Theory Notes

Key documents (in parent `Diffusion/` directory, not tracked by this repo):

| File | Content |
|------|---------|
| `Notes/Problem_formulation.pdf` | Core thesis framing: circle model, propagator hypothesis, research questions Q1–Q7 |
| `Notes/2nd-meeting-notes.pdf` | Board exercises: 2D rotation score, 1D one-step kernel, fixed-u slice |
| `Papers/dynamical-regimes_diff.pdf` | Biroli–Bonnaire–de Bortoli–Mézard: speciation `t_S`, collapse `t_C` |
| `Generative_diffusion_updated_notes_MM.pdf` | Mézard lecture notes on score, OU, speciation, collapse |

---

## Reports

- [`reports/mezard_update.html`](reports/mezard_update.html) — Email-ready progress report for Prof. Marc Mézard (March 2026)

---

## How to Reproduce Results

```bash
# TM3: open and adjust sliders for Δθ, α, σ_η; watch eigenvalues in status bar
open Toy-models/Toy-model_3/rotation_ou_score_lab.html

# TM5 Lab 2: adjust u_max, t_max sliders; observe four-regime phase diagram
open Toy-models/Toy-model_5/score_two_clocks_lab.html

# TM5 Lab 3: adjust Δθ, α sliders; watch eigenvalue collapse as t increases
# (Lab 3 is on the same page, scroll down)
```

All computations are in-browser JavaScript. No Python, no GPU, no training.

---

## Contributing / Development

- Follow the design system in `CLAUDE.md` (fonts, colors, layout, notation)
- New toy models go in `Toy-models/Toy-model_N/`
- Keep scores closed-form; label hypotheses explicitly with `[hypothesis]` if unverified
- See `CLAUDE.md` for full research questions, metrics, and open TODOs
