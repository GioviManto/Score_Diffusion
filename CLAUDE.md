# CLAUDE.md — Score-Based Diffusion for Dynamic Objects

## Project Aim

We study the **joint score field** of a diffusion model applied to an entire sequence
(video, trajectory). The sequence is a dynamic object: frames `a(0), a(1), ..., a(K)`
evolve under a known Markov process with internal time `u` and are **jointly** noised
by independent Ornstein–Uhlenbeck (OU) diffusion at diffusion time `t`.

**Central hypothesis:** a propagator `P` exists such that consecutive *blocks* of the
joint score are approximately related:

```
S_{:,u}(x)  ≈  P_t( S_{:,u−1}(x) )
```

We attack this with **fully solvable Gaussian toy models** — no neural networks,
exact closed-form everywhere.

---

## Critical Correction (Mézard meeting, 2026-03-12)

### What was wrong before

Previous toy models (TM1–TM5) computed the **per-frame marginal score**:

```
s(x_u, u, t) = ∇_{x_u} log p_t(x_u | u)        ← WRONG
```

This marginalises over all other frames, discarding all inter-frame correlations from the
score. It implicitly treats frames as independently generated — i.e. assumes
`P_0(a_0,...,a_K) = ∏_u p_0(a_u | u)`, which is false for any correlated process.

### Correct formulation

The **joint score** lives in `ℝ^{dK}` (all frames simultaneously):

```
S(x_0,...,x_K, t) = ∇_{(x_0,...,x_K)} log P_t(x_0,...,x_K)
```

The k-th block component is (exact formula, board 2026-03-12):

```
S_k(x, t) = ⟨(a_k e^{−t} − x_k) / Δ_t⟩_{a | x}
```

where the expectation is over the full posterior:

```
P(a_0,...,a_K | x_0,...,x_K)  ∝  P_0(a_0,...,a_K) × ∏_k N(x_k ; a_k e^{−t}, Δ_t)
```

and the Markov prior factorises as:

```
P_0(a_0,...,a_K) = p_0(a_0) · M(a_1|a_0) · M(a_2|a_1) · ... · M(a_K|a_{K-1})
```

### Why Gaussian AR(1) is now the perfect testbed

For AR(1): `a_{u+1} = α a_u + η_u`, `η_u ~ N(0, σ_η²)`:

- The joint prior `P_0(a_0,...,a_K)` is a `K`-dim Gaussian with covariance
  `Σ_0^{ij} = α^{|i−j|} σ_∞²` (stationary) or computed from initial conditions.
- The posterior `P(a|x)` is also Gaussian → **Kalman smoother** gives exact
  conditional mean `⟨a_k⟩_{a|x}` as a **linear function of the full x**.
- The joint score is **linear** in `x ∈ ℝ^K`:
  `S(x, t) = −Σ_t^{−1}(x − μ_t)` where `Σ_t = e^{−2t} Σ_0 + Δ_t I`.
- The precision `Σ_0^{−1}` is **tridiagonal** for AR(1) → sparse structure in the score.

---

## Repo Map

```
Score_Diffusion/
├── Toy-models/
│   ├── Toy-model_1/ou_diffusion_explainer.html     # 2D Gaussian mixture + OU (static)
│   ├── toy-model-2/toy-model-2.html                # 1D 2-frame joint p(x0,x1) — uses correct 2-frame joint!
│   ├── Toy-model_3/rotation_ou_score_lab.html      # 2D rotating AR(1), per-frame score [OLD]
│   ├── toy-model-4/bouncing_cube_score_lab.html    # 3D billiard, marginal score [OLD]
│   ├── Toy-model_5/score_two_clocks_lab.html       # (u,t) plane, 4 labs — per-frame score [OLD]
│   └── Toy-model_6/circle_score_lab.html           # circle model — per-frame score [OLD]
├── reports/
│   └── mezard_update.html                          # progress report for Prof. Mézard (March 2026)
├── README.md
└── CLAUDE.md                                       # (this file)
```

**Parent directory** (`Desktop/Diffusion/`):
```
Notes/Problem_formulation.pdf    # thesis framing, propagator hypothesis, Q1–Q7
Notes/2nd-meeting-notes.pdf      # board exercises: circle, 1D joint kernel, fixed-u slice
Papers/dynamical-regimes_diff.pdf # Biroli–Bonnaire–de Bortoli–Mézard regimes
Generative_diffusion_updated_notes_MM.pdf  # Mézard lecture notes
thesis_achilli_final.pdf          # Achilli thesis background
```

---

## How to Run

No build step — all toy models are self-contained HTML files:

```bash
open Score_Diffusion/Toy-models/Toy-model_5/score_two_clocks_lab.html
# or serve locally:
cd Score_Diffusion && python3 -m http.server 8080
```

GitHub Pages: `https://giovimanto.github.io/Score_Diffusion/`

---

## Core Mathematics (Correct Formulation)

### Joint distribution

For a K-frame sequence with AR(1) dynamics (`a_{u+1} = α a_u + η_u`):

```
P_0(a_0,...,a_{K-1}) = N(a_0; μ_0, σ_0²) × ∏_{u=0}^{K-2} N(a_{u+1}; α a_u, σ_η²)
```

After independent OU diffusion of each frame at time `t` (with `Δ_t = 1 − e^{−2t}`):

```
P_t(x_0,...,x_{K-1}) = ∫ da P_0(a) × ∏_k N(x_k ; a_k e^{−t}, Δ_t)
```

This is a K-dim Gaussian with:
```
μ_t = e^{−t} μ_a
Σ_t = e^{−2t} Σ_0 + Δ_t I_K
```

### Joint score (exact)

```
S(x, t) = −Σ_t^{−1}(x − μ_t)      ∈ ℝ^K
```

The k-th component couples x_k to ALL other frames through Σ_t^{−1}.

### Sparsity / Markov structure

- Σ_0^{−1} (precision of the clean AR(1) chain) is **tridiagonal**:
  ```
  (Σ_0^{−1})_{kk}   = (1 + α²)/σ_η²   for interior k
  (Σ_0^{−1})_{k,k±1} = −α/σ_η²
  ```
- Σ_t^{−1} is **not** sparse in general, but at `t → 0` it approaches the
  tridiagonal Σ_0^{−1}.
- At `t → ∞`: `Σ_t → Δ_t I`, score → `−x/Δ_t` (all correlations erased).

### Kalman smoother identity

For the k-th score component:
```
S_k(x, t) = (e^{−t} ⟨a_k⟩_{a|x} − x_k) / Δ_t
```
where `⟨a_k⟩_{a|x}` is the Kalman-smoothed estimate of the k-th clean frame
given the full noisy trajectory. This is linear: `⟨a_k⟩_{a|x} = C_k · x + d_k`
for computable matrices `C_k` (functions of α, σ_η, t, K).

---

## Propagator — Correct Reformulation

### What the propagator means now

The propagator `P_t` maps the k-th block of the joint score at position `u` to
the (k+1)-th block. More precisely, ask: how does adding frame `u+1` to the
conditioned trajectory change the score at frame `u`?

This is now **a question about the conditional precision structure** of Σ_t^{−1}.

### Hypotheses (updated)

**H1 (Linear joint score):** `S(x,t) = A(t) x + b(t)` for `A = −Σ_t^{-1}`,
`b = Σ_t^{-1} μ_t`. Exact for all Gaussian models. ✓

**H2 (Tridiagonal precision at t=0):** For AR(1), `Σ_0^{-1}` is tridiagonal,
so each score component depends only on its two neighbours. Exact. ✓

**H3 (Score propagation as row shift):** In the stationary regime,
`−Σ_t^{−1}` is Toeplitz → the k-th row of the propagator is a shift of the
(k−1)-th row. To be verified.

**H4 (Mean contraction):** `μ_t^{(u+1)} = α μ_t^{(u)}` is t-independent. Exact. ✓

**H5 (Kalman propagator):** The Kalman filter/smoother provides the exact
propagator: adding one new observed frame updates all smoothed estimates
via the Kalman gain. This is the exact `P`.

---

## Next Milestones

### Immediate (before meeting Wed 2026-03-18)

- [ ] **TM7 — 1D 2-frame joint score (K=2, AR(1)):**
  - Compute `P_t(x_0, x_1)` exactly as 2D Gaussian
  - Plot the 2D joint score field `(S_0, S_1)` as a function of `(x_0, x_1)` at fixed `t`
  - Show how `S_0` couples to `x_1` (new effect absent in old formulation)
  - Slider for `t`: watch coupling appear at small `t`, vanish at large `t`
  - Interactive lab showing Σ_t^{−1} entries vs `t`

- [ ] **K-frame extension (K=3, 4, 5):**
  - Implement exact Kalman smoother for AR(1) + OU
  - Plot each score component `S_k(x, t)` vs all `x_j`
  - Visualize the precision matrix Σ_t^{−1} as a heatmap (show tridiagonality at small t)

- [ ] **Propagator metric:**
  `ε(t) = ‖S_k(x,t) − P̂_t · S_{k−1}(x,t)‖ / ‖S_k(x,t)‖`
  where `P̂_t` is learned/approximate vs exact Kalman

### Medium term

- [ ] **Non-Gaussian extension:** mixture prior → nonlinear score, test linear-P hypothesis
- [ ] **Connect to speciation:** at what `t` does the coupling (off-diagonal Σ_t^{-1}) vanish?
- [ ] **Circle model (TM6):** redo with correct joint score over sequence of K circle frames

---

## Design System (all HTML toy models)

```
Fonts:  EB Garamond (body),  JetBrains Mono (code/labels)
Colors: --bg:#f7f4ef  --ink:#1a1410  --accent:#b5341a  --accent2:#1a3a5c
Libs:   MathJax 3.2.2,  Plotly 2.27.0  (both CDN)
Width:  max ~900–1050px, academic paper style
```

---

## Notation (use consistently)

| Symbol | Meaning |
|--------|---------|
| `u` or `k` | Internal time index (frame number) |
| `t` | Diffusion time (OU noise level) |
| `a(u)` or `a_u` | Clean frame at internal time u |
| `x(u)` or `x_u` | Noisy observation of frame u at diffusion time t |
| `K` | Sequence length (number of frames) |
| `S(x, t)` | **Joint** score = ∇_x log P_t(x_0,...,x_{K-1}) ∈ ℝ^{dK} |
| `S_k(x, t)` | k-th block component of joint score |
| `s(x_u, u, t)` | **Marginal** per-frame score [OLD, do not use for propagator] |
| `P` or `P_t` | Propagator mapping S_{k−1} → S_k |
| `Σ_0` | Joint covariance of clean chain (K×K) |
| `Σ_t` | Joint covariance at diffusion time t: `e^{-2t}Σ_0 + Δ_t I` |
| `Δ_t` | OU noise variance = `1 − e^{-2t}` |
| `α` | AR(1) contraction coefficient |
| `σ_η²` | AR(1) noise variance |
| `σ_∞²` | Stationary variance = `σ_η²/(1−α²)` |
| `M(a'|a)` | Markov transition kernel |

---

## Workflow Rules

- Always prefer **closed-form** over numerical approximation.
- New toy models go in `Toy-models/Toy-model_N/`.
- Label hypotheses explicitly with `[hypothesis]` or `[exact]` when unverified vs proven.
- No neural networks. No black boxes. Analytical everything.
- When computing the score, **always specify whether it is the joint or marginal score**.
- Self-contained HTML with MathJax + Plotly, no build step.
