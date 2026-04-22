"""
Laplace K=1 parameter sweep — HPC batch experiment.

Sweeps a grid of (alpha, b, sigma0, t) and, for each setting:
  - Computes kappa_t(0) via the closed-form Mills-ratio identity
  - Compares with 5-point finite-difference reference
  - Computes kappa_G(t) (matched-Gaussian benchmark)
  - Saves full results to CSV + summary TXT

Output files (written to Score_Diffusion/Experiments/results/):
  sweep_laplace_k1.csv       — full row-level results
  sweep_laplace_k1_summary.txt — pass/fail and statistics
"""

from __future__ import annotations
import sys, math, time, csv
from pathlib import Path
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Research" / "laplace_ar1_audit" / "code"))
sys.path.insert(0, str(ROOT / "Research" / "thesis_work_bundle" / "code"))

RESULTS = ROOT / "Experiments" / "results"
RESULTS.mkdir(exist_ok=True)

from ar1_diffusion_utils import delta_t

try:
    from laplace_k1 import build_params, kappa_t, log_h_t
    from observables import kappa_center_closed, kappa_center_numerical, kappa_G

    def _kappa_t_at_zero(p):
        return kappa_center_closed(p)

    def _kappa_t_at_zero_num(p):
        return kappa_center_numerical(p)

    def _kappa_G(p):
        return kappa_G(p)

    HAS_LAPLACE_K1 = True
except ImportError:
    HAS_LAPLACE_K1 = False
    print("WARNING: laplace_k1 / observables not available. Running reduced checks.")


def run_sweep():
    alphas  = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    bs      = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    sigma0s = [0.5, 1.0, 1.5, 2.0]
    t_vals  = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
    mu0     = 0.0

    total = len(alphas) * len(bs) * len(sigma0s) * len(t_vals)
    print(f"Sweep: {total} parameter combinations")
    print(f"Output: {RESULTS}")

    fieldnames = [
        "alpha", "b", "sigma0", "t",
        "kappa_center_closed", "kappa_center_num", "rel_err_kappa_center",
        "kappa_G", "gap_vs_G", "a", "btilde", "tau",
        "pass_kappa",
    ]

    rows = []
    t0 = time.time()

    for alpha in alphas:
        for b in bs:
            for sigma0 in sigma0s:
                for t in t_vals:
                    if not HAS_LAPLACE_K1:
                        continue
                    try:
                        p = build_params(alpha, b, sigma0, mu0, t)
                        k_closed = _kappa_t_at_zero(p)
                        k_num    = _kappa_t_at_zero_num(p)
                        k_g      = _kappa_G(p)
                        rel_err  = abs(k_closed - k_num) / max(abs(k_num), 1e-15)
                        gap      = k_closed - k_g
                        passed   = rel_err < 1e-5
                        rows.append({
                            "alpha": alpha, "b": b, "sigma0": sigma0, "t": t,
                            "kappa_center_closed": k_closed,
                            "kappa_center_num":    k_num,
                            "rel_err_kappa_center": rel_err,
                            "kappa_G":   k_g,
                            "gap_vs_G":  gap,
                            "a":         p.a,
                            "btilde":    p.btilde,
                            "tau":       p.tau,
                            "pass_kappa": passed,
                        })
                    except Exception as e:
                        print(f"  ERROR at α={alpha} b={b} σ0={sigma0} t={t}: {e}")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(rows)} rows in {elapsed:.1f}s")

    if not rows:
        print("No rows produced (laplace_k1 not available).")
        return

    # Write CSV
    csv_path = RESULTS / "sweep_laplace_k1.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV written: {csv_path}")

    # Summary
    n_pass  = sum(1 for r in rows if r["pass_kappa"])
    n_fail  = sum(1 for r in rows if not r["pass_kappa"])
    rel_errs = [r["rel_err_kappa_center"] for r in rows]
    gaps     = [r["gap_vs_G"] for r in rows]

    lines = [
        "Laplace K=1 parameter sweep — results summary\n",
        "=" * 55 + "\n",
        f"Total combinations: {len(rows)}\n",
        f"PASS (rel_err < 1e-5): {n_pass}\n",
        f"FAIL:                  {n_fail}\n",
        f"\nkappa_center rel error:\n",
        f"  mean: {np.mean(rel_errs):.3e}\n",
        f"  max:  {np.max(rel_errs):.3e}\n",
        f"  min:  {np.min(rel_errs):.3e}\n",
        f"\nGap kappa_center - kappa_G:\n",
        f"  mean: {np.mean(gaps):.6f}\n",
        f"  max:  {np.max(gaps):.6f} (most non-Gaussian)\n",
        f"  min:  {np.min(gaps):.6f}\n",
        f"\nRuntime: {elapsed:.1f}s\n",
    ]

    # Top 5 most non-Gaussian
    rows_sorted = sorted(rows, key=lambda r: -r["gap_vs_G"])
    lines.append("\nTop-5 most non-Gaussian settings (highest kappa gap):\n")
    lines.append(f"  {'alpha':>6} {'b':>5} {'sigma0':>6} {'t':>5} {'gap':>10} {'a':>8}\n")
    for r in rows_sorted[:5]:
        lines.append(f"  {r['alpha']:6.2f} {r['b']:5.2f} {r['sigma0']:6.2f} "
                     f"{r['t']:5.2f} {r['gap_vs_G']:10.6f} {r['a']:8.4f}\n")

    txt_path = RESULTS / "sweep_laplace_k1_summary.txt"
    txt_path.write_text("".join(lines))
    print("".join(lines))
    print(f"Summary written: {txt_path}")


if __name__ == "__main__":
    run_sweep()
