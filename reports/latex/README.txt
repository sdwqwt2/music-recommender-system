# Instructions for Compiling the English LaTeX Preprint

## Option A: Overleaf (Fastest, 1-Click)
1. Go to https://www.overleaf.com and log in / create a free account.
2. Click **New Project -> Upload Project**.
3. Select `recsys_rl_report_en.zip`.
4. In Overleaf Settings, ensure the compiler is set to **pdfLaTeX** and main document is **main_en.tex**.
5. Click **Recompile**.

## Option B: Local Command Line
If you have MacTeX / TeXLive installed:
```bash
./compile.sh
```
or
```bash
make
```
This produces `main_en.pdf`.
