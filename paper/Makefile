LATEXMK=latexmk -pdf -interaction=nonstopmode -halt-on-error

all: main.pdf slides.pdf

main.pdf: main.tex
	$(LATEXMK) main.tex

slides.pdf: slides.tex
	$(LATEXMK) slides.tex

clean:
	latexmk -C main.tex
	latexmk -C slides.tex
