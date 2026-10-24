TARGET_RU = main_ru
TARGET_EN = main_en

all: $(TARGET_RU).pdf $(TARGET_EN).pdf main.pdf

$(TARGET_RU).pdf: $(TARGET_RU).tex references.bib sections_ru/*.tex
	pdflatex -interaction=nonstopmode $(TARGET_RU) || true
	bibtex $(TARGET_RU) || true
	pdflatex -interaction=nonstopmode $(TARGET_RU) || true
	pdflatex -interaction=nonstopmode $(TARGET_RU) || true

$(TARGET_EN).pdf: $(TARGET_EN).tex references.bib sections_en/*.tex
	pdflatex -interaction=nonstopmode $(TARGET_EN) || true
	bibtex $(TARGET_EN) || true
	pdflatex -interaction=nonstopmode $(TARGET_EN) || true
	pdflatex -interaction=nonstopmode $(TARGET_EN) || true

main.pdf: main.tex references.bib sections_ru/*.tex
	pdflatex -interaction=nonstopmode main || true
	bibtex main || true
	pdflatex -interaction=nonstopmode main || true
	pdflatex -interaction=nonstopmode main || true

ru: $(TARGET_RU).pdf

en: $(TARGET_EN).pdf

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz *.fdb_latexmk *.fls sections_ru/*.aux sections_en/*.aux sections/*.aux

.PHONY: all ru en clean
