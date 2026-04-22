"""
K=2 Monte Carlo validation — HPC batch experiment.

For the K=2 Laplace AR(1) model, validates the numerical 3D Fourier inversion
(p_K2_fourier from k2_attempt.py) against large-N Monte Carlo + 3D KDE at
a grid of (x0, x1, x2) evaluation points.

Also computes:
  - The M_12 = -alpha * Delta cross-term for a parameter sweep
  - The factorisation failure magnitude: |M_12| / M_11

Output:
  Experiments/results/k2_monte_carlo.txt
  Experiments/results/k2_fourier_vs_mc.csv
"""

from __future__ import annotations
import sys, time, csv
from pathlib import Path
import numpy as np
from scipy.stats import norm as spnorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Research" / "thesis_work_bundle" / "code"))
sys.path.insert(0, str(ROOT / "Research" / "laplace_ar1_audit" / "code"))
sys.path.insert(0, str(ROOT / "Experiments"))

RESULTS = ROOT / "Experiments" / "results"
RESULTS.mkdir(exist_ok=True)

from ar1_diffusion_utils import gaussian_chain_covariance, gaussian_ou_covariance
from scores_exact import check_K2_cross_term

try:
    from k2_attempt import p_K2_fourier, monte_carlo_K2
    HAS_K2 = True
except ImportError:
    HAS_K2 = False
    print("WARNING: k2_attempt not available. Running reduced checks.")


def kde_3d_chunked(eval_pts, samples, bw, chunk_n=20_000):
    """Gaussian KDE in 3D, chunked to avoid OOM for large N."""
    N = samples.shape[1]
    n_eval = eval_pts.shape[0]
    dens = np.zeros(n_eval)
    for ns in range(0, N, chunk_n):
        chunk = samples[:, ns:ns + chunk_n]  # (3, chunk)
        for i, pt in enumerate(eval_pts):
            diff = (pt[:, None] - chunk) / bw
            kern = np.prod(spnorm.pdf(diff), axis=0)
            dens[i] += kern.sum() / bw**3
    return dens / N


def run_k2_mc(N_mc=2_000_000, bw=0.12):
    """Validate 3D Fourier inversion vs Monte Carlo at a set of test points."""
    alpha, b, sigma0, mu0, t = 0.6, 1.0, 1.0, 0.0, 0.5

    print(f"K=2 Monte Carlo validation: N={N_mc:,}, bw={bw}")
    print(f"  Parameters: α={alpha}, b={b}, σ₀={sigma0}, t={t}")

    # Evaluation points
    eval_pts = np.array([
        [0.0,  0.0,  0.0],
        [0.5,  0.3,  0.1],
        [-0.5, 0.2, -0.1],
        [1.0,  0.5,  0.3],
        [-0.3,-0.5,  0.4],
    ])

    if not HAS_K2:
        print("k2_attempt not available — skipping MC vs Fourier comparison.")
        return

    # Monte Carlo samples
    t0 = time.time()
    rng_seed = 42
    X0, X1, X2 = monte_carlo_K2(alpha, b, sigma0, mu0, t, N=N_mc, rng=rng_seed)
    samples = np.stack([X0, X1, X2], axis=0)  # (3, N)
    print(f"  MC sampling done in {time.time()-t0:.1f}s")

    # KDE + Fourier at each eval point
    rows = []
    print(f"\n  {'Point':25s}  {'Fourier':>12s}  {'MC+KDE':>12s}  {'rel_err':>10s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*12}  {'-'*10}")
    for pt in eval_pts:
        x0, x1, x2 = pt

        t_fou = time.time()
        p_fou = p_K2_fourier(x0, x1, x2, alpha, b, sigma0, mu0, t, L=15.0, N=56)
        time_fou = time.time() - t_fou

        t_kde = time.time()
        diff = (pt[:, None] - samples) / bw
        kern = np.prod(spnorm.pdf(diff), axis=0)
        p_mc = float(kern.mean()) / bw**3
        time_kde = time.time() - t_kde

        rel_err = abs(p_fou - p_mc) / max(abs(p_mc), 1e-12)
        label = f"({x0:.1f},{x1:.1f},{x2:.1f})"
        print(f"  {label:25s}  {p_fou:12.6f}  {p_mc:12.6f}  {rel_err:10.3%}")
        rows.append({
            "x0": x0, "x1": x1, "x2": x2,
            "p_fourier": p_fou, "p_mc_kde": p_mc, "rel_err": rel_err,
            "bw": bw, "N_mc": N_mc,
        })

    # Write CSV
    csv_path = RESULTS / "k2_fourier_vs_mc.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV written: {csv_path}")
    return rows


def run_m12_sweep():
    """Sweep (alpha, t) and compute the factorisation failure metric |M_12|/M_11."""
    alphas = np.linspace(0.1, 0.95, 18)
    t_vals = [0.1, 0.5, 1.0, 2.0, 3.0]

    print("\nM_12 = -α·Δ_t parameter sweep:")
    rows = []
    for alpha in alphas:
        for t in t_vals:
            r = check_K2_cross_term(
                alpha=alpha, sigma0_sq=1.0,
                sigma_eta_sq=max(1-alpha**2, 1e-4), t=t
            )
            M = r["M"]
            rel_coup = abs(r["M_12_actual"]) / (M[1,1] + 1e-15)
            rows.append({
                "alpha": float(alpha), "t": float(t),
                "M12": r["M_12_actual"],
                "M12_pred": r["M_12_predicted"],
                "M12_err": r["M_12_error"],
                "M11": float(M[1,1]),
                "rel_coupling": rel_coup,
            })

    csv_path = RESULTS / "k2_m12_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # Print summary
    max_err = max(r["M12_err"] for r in rows)
    max_coup = max(r["rel_coupling"] for r in rows)
    print(f"  Max |M_12 - (-αΔ)|:          {max_err:.2e}  (all should be ~0)")
    print(f"  Max |M_12|/M_11 (coupling %): {max_coup*100:.1f}%")
    print(f"  CSV written: {csv_path}")
    return rows


def main():
    t_start = time.time()

    print("=" * 60)
    print("K=2 Monte Carlo & Factorisation Failure Experiment")
    print("=" * 60)

    # Part 1: Fourier vs MC
    mc_rows = run_k2_mc(N_mc=2_000_000, bw=0.12)

    # Part 2: M_12 sweep
    m12_rows = run_m12_sweep()

    # Summary report
    elapsed = time.time() - t_start
    lines = [
        "K=2 Monte Carlo Experiment — Summary\n",
        "=" * 50 + "\n",
        f"Runtime: {elapsed:.1f}s\n\n",
    ]
    if mc_rows:
        rel_errs = [r["rel_err"] for r in mc_rows]
        lines += [
            "Fourier vs MC validation:\n",
            f"  N_mc = {mc_rows[0]['N_mc']:,}, bw = {mc_rows[0]['bw']}\n",
            f"  Mean rel. error: {np.mean(rel_errs)*100:.2f}%\n",
            f"  Max  rel. error: {np.max(rel_errs)*100:.2f}%\n",
            f"  (KDE bandwidth bias is O(bw^2) ≈ {mc_rows[0]['bw']**2:.4f})\n\n",
        ]
    if m12_rows:
        max_err = max(r["M12_err"] for r in m12_rows)
        lines += [
            "M_12 = -α·Δ_t sweep:\n",
            f"  Max verification error: {max_err:.2e}  (should be ~1e-16)\n",
            "  Conclusion: M_12 ≠ 0 for all α>0, t>0 → K=2 factorisation FAILS.\n",
        ]

    report_path = RESULTS / "k2_monte_carlo.txt"
    report_path.write_text("".join(lines))
    print("\n" + "".join(lines))
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
