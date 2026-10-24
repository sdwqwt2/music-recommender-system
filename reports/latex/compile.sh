#!/bin/bash
set -e

echo "=================================================="
echo "Compiling IMRaD LaTeX Research Report (English)"
echo "=================================================="

if ! command -v pdflatex &> /dev/null; then
    echo "[!] pdflatex not found on this machine."
    echo "[i] You can upload 'recsys_rl_report_en.zip' directly to Overleaf (https://www.overleaf.com) to compile into PDF."
    exit 0
fi

pdflatex -interaction=nonstopmode main_en.tex
bibtex main_en
pdflatex -interaction=nonstopmode main_en.tex
pdflatex -interaction=nonstopmode main_en.tex

echo "[+] Compilation successful: main_en.pdf generated!"
