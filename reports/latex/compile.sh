#!/bin/bash
set -e

echo "=================================================="
echo "Compiling IMRaD LaTeX Research Report"
echo "=================================================="

if ! command -v pdflatex &> /dev/null; then
    echo "[!] pdflatex not found on this machine."
    echo "[i] You can upload the generated 'reports/latex/' directory or 'recsys_rl_imrad_report.zip' directly to Overleaf (https://www.overleaf.com) to compile into PDF."
    exit 0
fi

pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "[+] Compilation successful: main.pdf generated!"
