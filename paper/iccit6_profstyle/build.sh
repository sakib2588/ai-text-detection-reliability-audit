#!/usr/bin/env bash
# Deterministic ICCIT build: pdflatex -> bibtex -> pdflatex x2.
# Reports page count and any undefined ref/citation. Latexmk mis-orders bibtex
# from a clean aux, so the sequence is spelled out.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex >/tmp/iccit_p1.log 2>&1 || true
bibtex main >/tmp/iccit_bib.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/tmp/iccit_p2.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/tmp/iccit_p3.log 2>&1 || true
echo "pages: $(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}')"
u=$(grep -icE "undefined (reference|citation)|there were undefined" main.log || true)
echo "undefined refs/cites: $u"
