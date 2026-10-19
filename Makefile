# Build and check the released chapters. Requires: TeX Live (latexmk, biber), Python venv in .venv, R, GNU Octave.
PY  ?= .venv/bin/python
CH  ?= 01
LTX  = latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build

.PHONY: all code check book clean
all: check

code:
	$(PY) code/ch$(CH)/run_all.py
	$(PY) scripts/cover.py
	Rscript code/ch$(CH)/crosscheck.R
	if [ -f code/ch$(CH)/octave/ch$(CH)_check.m ]; then octave-cli --no-gui code/ch$(CH)/octave/ch$(CH)_check.m; fi

check: code
	$(PY) -m pytest -q code/ch$(CH)
	$(MAKE) book

book:
	$(LTX) -jobname=quant-finance-book main.tex

clean:
	latexmk -C -outdir=build main.tex; rm -rf build
