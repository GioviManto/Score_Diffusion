#!/bin/bash
#SBATCH --job-name=score_diff_sweep_k1
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3164542/Score_Diffusion/Experiments/jobs/logs/sweep_k1_%j.out
#SBATCH --error=/home/3164542/Score_Diffusion/Experiments/jobs/logs/sweep_k1_%j.err

# --------------------------------------------------------------------------
# Laplace K=1 parameter sweep.
# Grid: 6 × 7 × 4 × 6 = 1008 parameter combinations.
# Each: closed-form kappa_center vs 5-point FD + Gaussian benchmark gap.
# Output: Experiments/results/sweep_laplace_k1.{csv,txt}
#
# Partition: defq (CPU only — pure NumPy/SciPy, no GPU needed)
# Expected runtime: < 10 minutes
# --------------------------------------------------------------------------

set -euo pipefail
set +u; source /home/3164542/.bashrc; set -u

mkdir -p /home/3164542/Score_Diffusion/Experiments/jobs/logs
mkdir -p /home/3164542/Score_Diffusion/Experiments/results

cd /home/3164542/Score_Diffusion

echo "========================================================"
echo "Score Diffusion: Laplace K=1 parameter sweep"
echo "Host:    $(hostname)"
echo "Started: $(date)"
echo "========================================================"

conda activate vlm-sam3-py312

python Experiments/jobs/sweep_laplace_k1_script.py

echo ""
echo "Finished: $(date)"
