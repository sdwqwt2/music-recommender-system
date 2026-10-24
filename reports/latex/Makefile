TARGET = main
PDF = $(TARGET).pdf

all: $(PDF)

$(PDF): $(TARGET).tex references.bib sections/*.tex
	pdflatex -interaction=nonstopmode $(TARGET) || true
	bibtex $(TARGET) || true
	pdflatex -interaction=nonstopmode $(TARGET) || true
	pdflatex -interaction=nonstopmode $(TARGET) || true

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz *.fdb_latexmk *.fls sections/*.aux

.PHONY: all clean
