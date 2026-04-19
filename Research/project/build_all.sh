#!/usr/bin/env bash
set -eu
for d in file1_markov_loss file2_deterministic file3_laplace; do
  (cd "$d" && python3 code/generate_figures.py && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
done
