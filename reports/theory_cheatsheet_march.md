# Pre-Call Theory Cheatsheet
## Score Fields for Dynamic Objects — March 2026

---

## 0. The Central Problem (30-second version)

A **dynamic object** (video, trajectory) is a sequence of frames $a^\mu = \{a_u\}_{u=0}^{T-1}$. You want a generative model that can denoise any frame. Score-based diffusion does this by learning $s(x, u, t) = \nabla_x \log p_t(x \mid u)$. The key question: does the score field *propagate* predictably from one frame to the next, and if so, what is the operator $P$ that does it?

---

## 1. Score-Based Diffusion (Static Baseline)

**Definition.** For a distribution $p_0$, the forward (noising) OU process produces $p_t$ at time $t$. The score is:
$$s(x, t) = \nabla_x \log p_t(x)$$
Denoising = running the reverse SDE driven by $s$.

**Why it matters.** The score tells you: *"which direction makes this sample more probable?"* Training a score network is equivalent to training a denoiser (Tweedie identity). This is the foundation everything else builds on.

**Intuition.** Think of the score as a vector field on $\mathbb{R}^d$ pointing toward high-density regions. At large $t$, all information is erased and $s \approx -x$. At small $t$, the field has sharp structure reflecting the data geometry.

---

## 2. The OU Forward Process

**Definition.** Each frame $a_u \in \mathbb{R}^d$ is independently corrupted:
$$X_t(u) = a_u \, e^{-t} + \sqrt{1 - e^{-2t}} \, Z(u), \quad Z(u) \sim \mathcal{N}(0, I_d)$$

**Implication.** The marginal at time $t$ is:
$$p_t(x \mid u) = \mathcal{N}(x;\; e^{-t} a_u,\; (1 - e^{-2t}) I_d)$$
For Gaussian priors $p_0(a_u) = \mathcal{N}(\mu_u, \Sigma_u)$, the noised marginal stays Gaussian — this is what makes all scores closed-form.

**Key limits.** $t=0$: $X = a_u$ (no noise). $t \to \infty$: $X \sim \mathcal{N}(0, I_d)$ (pure noise, all data erased).

**Connection.** The OU process is the simplest continuous analog of the VP-SDE used in DDPM. All toy models use this same forward process, so $t$ means exactly the same thing everywhere.

---

## 3. The Score Function $s(x, u, t)$

**Definition.** The *per-frame score* at frame $u$, diffusion time $t$, evaluated at point $x$:
$$s(x, u, t) := \nabla_x \log p_t(x \mid u)$$

**For a Gaussian prior** $p_0(\cdot \mid u) = \mathcal{N}(\mu_u, \Sigma_u)$, the noised density is $\mathcal{N}(\mu_u(t), \Sigma_u(t))$ where:
$$\mu_u(t) = e^{-t} \mu_u, \qquad \Sigma_u(t) = e^{-2t} \Sigma_u + (1 - e^{-2t}) I$$
So the score is linear in $x$:
$$s(x, u, t) = -\Sigma_u(t)^{-1}(x - \mu_u(t))$$

**Implication for the project.** Because the score is linear in $x$ for all Gaussian models (Hypothesis H1 ✓), the propagator $P$ only needs to map *linear functions of $x$* to linear functions of $x$ — a massive structural simplification.

**Intuition.** The score is a *shifted linear map*: it points from $x$ toward the conditional mean $\mu_u(t)$, scaled by the precision $\Sigma_u(t)^{-1}$. The mean carries the "signal" (where is frame $u$ centered?), the covariance carries the "uncertainty" (how spread out are frames at step $u$?).

---

## 4. Two-Clock Architecture

**Definition.** Two independent indices govern the problem:

| Clock | Symbol | Role |
|-------|--------|------|
| Internal time | $u \in \{0, \ldots, T-1\}$ | Semantic: frame index, trajectory position |
| Diffusion time | $t \in [0, \infty)$ | Noise level: how corrupted is the frame |

**Implication.** The score $s(x, u, t)$ lives on a *two-dimensional* time plane. Its behavior in $u$ and $t$ is structurally different: $t$ controls the noise level of any single frame; $u$ controls the correlations *between* frames through the internal dynamics.

**Key insight.** These clocks commute: the OU noising is applied *after* the internal dynamics, and independently per frame. This means $p_t(x \mid u)$ is just the noised version of $p_0(a_u)$ — the internal dynamics enter only through the marginal $p_0(\cdot \mid u)$.

---

## 5. Propagator Hypothesis

**Definition.** We hypothesize an operator $P$ (possibly linear, possibly $t$-dependent) such that:
$$s(\cdot, u, t) \approx P\bigl[s(\cdot, u-1, t)\bigr]$$

**Why this matters.** If exact, the score at all $T$ frames is determined by $s(\cdot, 0, t)$ plus $P$ — reducing the learning problem from $\mathcal{O}(dT)$ parameters to $\mathcal{O}(d)$. This is the central hypothesis of the project.

**What has been confirmed so far:**
- **H1 (Linear P):** For Gaussian dynamics, the propagator maps linear fields to linear fields. Exact. ✓
- **H2 (Rotation P):** For rotating AR(1) at isotropic covariance, $[P \cdot s_u](x) = R_{\Delta\theta} \cdot s(R_{-\Delta\theta} x, u, t)$. Exact at stationarity. ✓
- **H3 (Mean contraction):** $\mu_{u+1}(t) = \alpha \cdot \mu_u(t)$, independent of $t$. Exact. ✓

**Intuition.** $P$ is the operator that "moves the score field" the same way the underlying dynamics move the data. For rotation: rotate the input point backward, apply the score, rotate the output forward.

---

## 6. AR(1) Chain (Internal Dynamics)

**Definition.** The simplest Markovian sequential model:
$$a_{u+1} = \alpha \, a_u + \eta_u, \qquad \eta_u \sim \mathcal{N}(0, \sigma_\eta^2)$$
with $|\alpha| < 1$ for stability.

**Closed-form marginals.**
- Mean: $\mu_u = \alpha^u \mu_0$
- Variance: $\sigma_u^2 = \alpha^{2u} \sigma_0^2 + \sigma_\eta^2 \frac{1 - \alpha^{2u}}{1 - \alpha^2}$
- Stationary variance: $\sigma_\infty^2 = \sigma_\eta^2 / (1 - \alpha^2)$

**Implication.** Everything about the score is determined by $(\mu_u, \sigma_u^2)$. These satisfy simple recursions in $u$, and after composing with OU: $\sigma_u^2(t) = 1 + e^{-2t}(\sigma_u^2 - 1)$.

**Connection.** This is TM5 — the "two clocks" lab. The AR(1) is the canonical model for studying the $(u,t)$ phase diagram because all quantities are 1D and explicit.

---

## 7. Rotating AR(1) — TM3

**Definition.** 2D generalization: replace scalar contraction by rotation plus noise:
$$a(u+1) = R_{\Delta\theta} \, a(u) + \xi_u, \qquad \xi_u \sim \mathcal{N}(0, \sigma_\eta^2 I_2)$$

**Closed-form score.** Starting from $a(0) \sim \mathcal{N}(\mu_0, \Sigma_0)$, the marginal at frame $u$ is Gaussian with:
- $\mu_u = R_{\Delta\theta}^u \mu_0$
- $\Sigma_u \to \sigma_\infty^2 I$ as $u \to \infty$ (isotropic stationary distribution, because the noise $\sigma_\eta^2 I$ is rotationally symmetric)

The score after OU diffusion:
$$s(x, u, t) = -\Sigma_u(t)^{-1}(x - \mu_u(t))$$

**Exact propagator result (H2).** In the isotropic limit $\Sigma_u(t) = \sigma^2(t) I$:
$$s(x, u+1, t) = R_{\Delta\theta} \cdot s(R_{-\Delta\theta} x, u, t)$$
The propagator acts as: *pull back the evaluation point, apply, push forward the output.*

**Why the covariance must be isotropic.** When $\Sigma_u$ is anisotropic, the precision $\Sigma_u^{-1}$ does not commute with $R_{\Delta\theta}$, and the propagator picks up a correction. The error is controlled by the *spectral gap* of $\Sigma_u(t)$ — how far it is from a scalar multiple of identity.

---

## 8. Jacobian of the Score $J_s$

**Definition.** The matrix of partial derivatives of the score with respect to $x$:
$$J_s(x, u, t) := \frac{\partial s}{\partial x}(x, u, t) = -\Sigma_u(t)^{-1}$$
(exact and *position-independent* for all Gaussian models).

**Implication.** For Gaussians, $J_s$ is a negative definite matrix with eigenvalues $\{-1/\sigma_k^2(t)\}$ where $\sigma_k^2$ are the eigenvalues of $\Sigma_u(t)$.

**Why it matters.** The Jacobian is a spectral diagnostic:
- Its eigenvalues measure *how informative* the score is along each principal direction
- Anisotropy of $J_s$ = propagator error for $P = R_{\Delta\theta}$
- All eigenvalues $\to -1$ as $t \to \infty$ (universal noise limit, Phase IV)

**Triad identity.** For Gaussians: Fisher information = $-$Jacobian eigenvalue = inverse variance:
$$\mathcal{I}(u,t) = -\lambda_k(J_s) = 1/\sigma_u^2(t)$$
Three equivalent ways to measure "how much information about $u$ is left in the score at diffusion time $t$."

---

## 9. Fisher Information $\mathcal{I}(u, t)$

**Definition.** Amount of information about the frame parameter $u$ in the noised distribution $p_t(x \mid u)$. For the Gaussian per-frame score: $\mathcal{I}(u,t) = 1/\sigma_u^2(t)$.

**Implication for the project.** Fisher information quantifies *when the propagator is most useful*:
- High $\mathcal{I}$: small $t$, score is sharp and data-specific
- Low $\mathcal{I}$: large $t$, score washed out
- **Phase II** (intermediate $t$) is where $\mathcal{I}$ is in the "sweet spot" — signal-to-noise balanced, propagator structure is most legible

**Intuition.** Think of Fisher information as the "sharpness" of the score. At $t=0$, it equals $1/\sigma_u^2$ (the raw data variance). As $t$ increases, noise dilutes the signal and $\mathcal{I} \to 1$ (the noise floor). The optimal window for fitting $P$ is where $\mathcal{I}$ transitions between these regimes.

---

## 10. Four-Regime Phase Diagram on $(u, t)$ — TM5

The 1D AR(1) score $s(x, u, t) = -(x - \mu_u(t))/\sigma_u^2(t)$ organizes into four qualitative regimes:

| Phase | Region | Score behavior | Propagator status |
|-------|--------|---------------|-------------------|
| **I — Sharp** | small $u$, small $t$ | Tracks initial condition; strong anisotropy; high Fisher info | $P$ exists but covariance anisotropy may break exact formula |
| **II — Best window** | intermediate $t$, any $u$ | Variance balances signal and noise; near-isotropic | **Optimal regime for $P$**: most legible, least error |
| **III — Stationary** | large $u$, small $t$ | $\sigma_u^2 \to \sigma_\infty^2$; variance part of $P \approx 1$; only mean evolves | Variance propagator trivial; mean propagator $= \alpha$ |
| **IV — Universal noise** | large $t$ | $\sigma_u^2(t) \to 1$ for all $u$; $s \approx -x$ | All dynamical information erased; $P$ becomes identity |

**Propagator decomposition (exact for AR(1)).** The mean component propagates as $\mu_{u+1}(t) = \alpha \cdot \mu_u(t)$ (pure $\alpha$-contraction, *independent of $t$*), and the variance component $\sigma_u^2(t)/\sigma_{u+1}^2(t) = 1$ exactly at stationarity.

**Connection to Biroli et al.** Their "speciation" and "collapse" transitions in static score models correspond to the $(u,t)$ regime boundaries here — the question is whether these sharpen into genuine phase transitions as $d \to \infty$.

---

## 11. Circle Model (TM6) — First Non-Gaussian

**Definition.** 2D stochastic dynamics in polar coordinates:
$$\mathrm{d}r = (1 - r)\,\mathrm{d}u + \sqrt{2T_r}\,\mathrm{d}B_r, \qquad \mathrm{d}\theta = \omega\,\mathrm{d}u + \sqrt{2T_\theta}\,\mathrm{d}B_\theta$$
Radial and angular processes are *independent*, so the joint density factorizes: $p_0(r, \theta \mid u) = p_r(r \mid u) \cdot p_\theta(\theta \mid u)$.

**Why non-Gaussian.** The radial SDE has a mean-reverting drift toward $r=1$ (a circle), not a Gaussian attractor. The marginal $p_r$ is not Gaussian.

**Score in Cartesian coordinates.** Using the Jacobian of the polar change of variables:
$$s(x, u) = \left(s_r(r, u) - \frac{1}{r}\right)\hat{e}_r + \frac{s_\theta(\theta, u)}{r}\,\hat{e}_\theta$$
where $s_r = -(r - m_r)/\sigma_r^2$, $s_\theta = -(\theta - \theta_\text{mean})/\sigma_\theta^2$, and the $-1/r$ correction is from the Jacobian (the coordinate change introduces curvature into the score).

**Implication.** Even though the *model* is non-Gaussian, the score is still determined by a finite-dimensional **order parameter** $(m_r, \sigma_r^2, \theta_\text{mean}, \sigma_\theta^2)$ — four real numbers that evolve in $u$. This is the "order parameter hypothesis": the complexity of the score field is hidden in a low-dimensional dynamical system.

---

## 12. Order Parameter SDE

**Definition.** The score at any $(u, t)$ is *fully determined* by the tuple $(m_r, \sigma_r^2, \theta_\text{mean}, \sigma_\theta^2)$. Each component satisfies a closed ODE in $u$ (derivable from Fokker-Planck or moment equations).

**Why this is powerful.** Instead of tracking an infinite-dimensional function $s(\cdot, u, t)$ as $u$ evolves, you track 4 real numbers. The propagator $P$ reduces to a map $\mathbb{R}^4 \to \mathbb{R}^4$.

**Connection to H4 (Score ODE).** The post-Achilli call direction is to make this explicit for the Gaussian AR(1)+OU case: write closed ODEs for $(A(u,t), b(u,t))$ where $s(x,u,t) = A(u,t) x + b(u,t)$. This is the "Riccati" direction.

---

## 13. Riccati Equation for Precision $\kappa$

**Definition.** Define precision $\kappa = 1/\sigma^2 = -\lambda(J_s)$. For the AR(1) internal dynamics with contraction rate $\gamma = -\log \alpha$, the precision satisfies:
$$\frac{\mathrm{d}\kappa}{\mathrm{d}u} = 2\gamma \kappa - \sigma_\eta^2 \kappa^2$$

**This is a Riccati equation** (first-order ODE, quadratic in $\kappa$). It is exactly solvable.

**Fixed point.** Setting $\mathrm{d}\kappa/\mathrm{d}u = 0$: $\kappa_* = 2\gamma/\sigma_\eta^2 = 1/\sigma_\infty^2$ (inverse of the stationary variance). The precision converges to the stationary value.

**Special cases:**
- Circle model ($\gamma = 0$, no mean reversion for angle): $\mathrm{d}\kappa_\theta/\mathrm{d}u = -2T_\theta \kappa_\theta^2$, giving $\kappa_\theta(u) = 1/(2T_\theta u)$ — algebraic decay (precision falls off like $1/u$)
- AR(1) with $\gamma > 0$: exponential approach to $\kappa_*$

**Unification.** The same equation governs precision for *all* Gaussian internal dynamics. For the rotating AR(1), it extends to the matrix Riccati: $\mathrm{d}K/\mathrm{d}u = 2\gamma K - \sigma_\eta^2 K^2$ for the precision tensor $K = \Sigma^{-1}$.

**Why it matters.** The Riccati equation is the ODE that the "variance part of the propagator" satisfies. At stationarity ($K = K_\infty$), the variance component of $P$ is trivial (identity), and only the mean part matters. This directly connects to Phase III.

---

## 14. Propagator Relative Error $\varepsilon(u, t)$

**Definition.** For the circle model, the relative $L^2$ error of the rotation propagator $P = R_\omega$ on the tangential score component is:
$$\varepsilon(u, t) = \frac{2T_\theta}{2T_\theta \, u + (1 - e^{-2t})}$$

**Behavior:**
- At $t = 0$: $\varepsilon = 1/u$ — algebraic decay in internal time (the longer you run, the better $P$ works)
- At intermediate $t$ (Phase II): the denominator grows further, reducing error
- As $u \to \infty$: $\varepsilon \to 0$ — propagator becomes exact at stationarity

**Intuition.** Error vanishes when either (a) the chain has run long enough that the covariance is isotropic (large $u$), or (b) noise has homogenized the covariance anisotropy (intermediate $t$). Phase II works because OU diffusion acts as a "covariance isotropizer."

---

## 15. Speciation and Collapse Transitions (Biroli et al.)

**Context.** For *static* score-based models, Biroli–Bonnaire–de Bortoli–Mézard identify two critical diffusion times in the reverse process:
- **Collapse transition $t_C$:** above this, the reverse SDE converges to pure noise; below, it starts to localize
- **Speciation transition $t_S < t_C$:** below this, individual modes of $p_0$ separate out; the score starts "choosing" between clusters

**Connection to this project.** The four-regime diagram in TM5 is a *dynamic analog*:
- Phase IV (universal noise) corresponds to $t > t_C$: no structural information
- Phase II (best window) is analogous to the intermediate regime between $t_S$ and $t_C$
- The open question: do the regime boundaries in the $(u,t)$ plane sharpen into genuine phase transitions as $d \to \infty$? In static models they do (mean-field/large-$d$ limit).

**Why it matters for direction.** If the Phase II boundary sharpens as $d \to \infty$, then there is a *phase transition in the propagator accuracy* — a clean signal for the theory. This is the large-$d$ extension (Next Steps item 3).

---

## 16. Key Connections Map

```
OU forward process
    ↓
Noised Gaussian marginal p_t(x|u)
    ↓
Score s(x,u,t) = -Σ_u(t)⁻¹(x - μ_u(t))
    |                    |
    ↓                    ↓
Jacobian J_s        Order parameter
= -Σ_u(t)⁻¹         (μ_u, σ²_u) or (m_r, σ²_r, θ_mean, σ²_θ)
    |                    |
    ↓                    ↓
Fisher info         Riccati equation
I = 1/σ²_u(t)       dκ/du = 2γκ - σ²_η κ²
    |                    |
    └─────────┬──────────┘
              ↓
        (u,t) phase diagram
        4 regimes (Phases I–IV)
              |
    ┌─────────┴──────────┐
    ↓                    ↓
Propagator P         Biroli et al.
(exact in Ph. II)    (speciation/collapse)
    |
    ↓
Error ε(u,t) = 2Tθ / (2Tθu + (1-e⁻²ᵗ))
```

---

## 17. Quick Reference: What Each Toy Model Proves

| Model | Key proof | Regime |
|-------|-----------|--------|
| **TM1** Gaussian mixture | Baseline: score softens from anisotropic to isotropic as $t \uparrow$ | Static (no $u$) |
| **TM2** Drift+noise | Joint score = prior gradient + transition correction; exact decomposition | 1D, 2 frames |
| **TM3** Rotating AR(1) | $P = R_{\Delta\theta}$ exact at isotropic covariance; Jacobian eigenvalue = Fisher info | Gaussian, 2D |
| **TM4** Billiard | Analytic folded-Gaussian conditional score matches KDE marginal | Non-Markovian, 3D |
| **TM5** Two clocks | Four-regime $(u,t)$ phase diagram; mean/variance propagator decomposition | Gaussian, 1D |
| **TM6** Circle | Non-Gaussian; polar decomposition; order parameter SDE; Riccati; $\varepsilon(u,t)$ formula | Non-Gaussian, 2D |

---

## 18. Key Phrases for the Call

- **"Two-clock architecture"** — $u$ is semantic time, $t$ is noise time; they're independent
- **"Propagator hypothesis"** — $s(\cdot, u, t) \approx P[s(\cdot, u-1, t)]$; reduces $\mathcal{O}(dT)$ to $\mathcal{O}(d)$
- **"Order parameter"** — the score is determined by a finite-dimensional dynamical system (mean, covariance); the Riccati equation is the ODE for the variance/precision
- **"Phase II is the sweet spot"** — intermediate $t$, near-isotropic covariance, propagator error is smallest
- **"Riccati equation unifies everything"** — same equation for all Gaussian internal dynamics, special cases are AR(1) and circle model
- **"Next direction: score coefficient ODE"** — write $s(x,u,t) = A(u,t)x + b(u,t)$, derive the closed ODE for $(A, b)$ in $u$, this is the H4 direction from the Achilli call
