TARGET = main_en

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex references.bib sections_en/*.tex
	pdflatex -interaction=nonstopmode $(TARGET) || true
	bibtex $(TARGET) || true
	pdflatex -interaction=nonstopmode $(TARGET) || true
	pdflatex -interaction=nonstopmode $(TARGET) || true

zip:
	zip -r recsys_rl_report_en.zip main_en.tex references.bib sections_en/ figures/ compile.sh Makefile README.txt

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz *.fdb_latexmk *.fls sections_en/*.aux

.PHONY: all zip clean
