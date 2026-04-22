#!/bin/bash
#SBATCH --job-name=score_diff_k2_mc
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/3164542/Score_Diffusion/Experiments/jobs/logs/k2_mc_%j.out
#SBATCH --error=/home/3164542/Score_Diffusion/Experiments/jobs/logs/k2_mc_%j.err

# --------------------------------------------------------------------------
# K=2 Monte Carlo validation experiment.
# Tasks:
#  1. Draw N=2M samples from the K=2 Laplace AR(1) model
#  2. Estimate density at 5 evaluation points via 3D Gaussian KDE
#  3. Cross-validate against 3D Fourier inversion (p_K2_fourier)
#  4. Sweep (alpha, t) grid and confirm M_12 = -alpha * Delta_t
#
# Output: Experiments/results/k2_{fourier_vs_mc.csv, m12_sweep.csv, monte_carlo.txt}
#
# Memory: N=2M samples × 3 arrays × float64 ≈ 48MB → 32G is very conservative
# Partition: defq (CPU-only — 3D Fourier uses pure NumPy, no GPU)
# Expected runtime: 30-60 minutes (dominated by KDE evaluation)
# --------------------------------------------------------------------------

set -euo pipefail
set +u; source /home/3164542/.bashrc; set -u

mkdir -p /home/3164542/Score_Diffusion/Experiments/jobs/logs
mkdir -p /home/3164542/Score_Diffusion/Experiments/results

cd /home/3164542/Score_Diffusion

echo "========================================================"
echo "Score Diffusion: K=2 Monte Carlo experiment"
echo "Host:    $(hostname)"
echo "Started: $(date)"
echo "CPUs available: $(nproc)"
echo "========================================================"

conda activate vlm-sam3-py312

python Experiments/jobs/k2_monte_carlo_script.py

echo ""
echo "Finished: $(date)"
