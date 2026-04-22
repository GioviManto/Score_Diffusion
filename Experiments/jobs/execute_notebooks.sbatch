#!/bin/bash
#SBATCH --job-name=score_diff_notebooks
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3164542/Score_Diffusion/Experiments/jobs/logs/notebooks_%j.out
#SBATCH --error=/home/3164542/Score_Diffusion/Experiments/jobs/logs/notebooks_%j.err

# --------------------------------------------------------------------------
# Execute all four Score Diffusion experiment notebooks via nbconvert.
# Each notebook runs end-to-end, saves figures to Experiments/figures/,
# and produces an executed .ipynb with all outputs embedded.
#
# Partition: defq (3-day limit, CPU-only cnode — no GPU needed)
# --------------------------------------------------------------------------

set -euo pipefail
set +u; source /home/3164542/.bashrc; set -u

mkdir -p /home/3164542/Score_Diffusion/Experiments/jobs/logs
mkdir -p /home/3164542/Score_Diffusion/Experiments/figures

cd /home/3164542/Score_Diffusion

echo "========================================================"
echo "Score Diffusion: executing notebooks"
echo "Host:    $(hostname)"
echo "Started: $(date)"
echo "========================================================"

conda activate vlm-sam3-py312

NBCONVERT="jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 \
    --inplace"

NB_DIR="/home/3164542/Score_Diffusion/Experiments"

for nb in nb1_gaussian_foundations nb2_score_field_k2 nb3_toeplitz_propagator nb4_laplace_k1_k2; do
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "Executing: ${nb}.ipynb"
    echo "──────────────────────────────────────────────────────"
    $NBCONVERT "${NB_DIR}/${nb}.ipynb" && echo "DONE: ${nb}" || echo "ERROR in ${nb}"
done

echo ""
echo "========================================================"
echo "All notebooks executed. Finished: $(date)"
echo "Figures saved to: Experiments/figures/"
ls Experiments/figures/*.png 2>/dev/null | wc -l | xargs -I{} echo "  {} PNG files produced"
echo "========================================================"
