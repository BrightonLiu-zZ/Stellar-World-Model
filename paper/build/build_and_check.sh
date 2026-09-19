#!/usr/bin/env bash
# Rebuild main.pdf and print the three format checks: overfull boxes, page count, page-4/5 boundary.
# Run from anywhere: bash paper/build/build_and_check.sh
cd "$(dirname "$0")/.." || exit 1
pdflatex -interaction=nonstopmode main.tex | grep -E "^!|Warning"
bibtex main | grep -iE "warning|error"
pdflatex -interaction=nonstopmode main.tex | grep -E "^!"
pdflatex -interaction=nonstopmode main.tex | grep -E "^!|undefined"
echo "overfull boxes: $(grep -c Overfull main.log)"
pdfinfo main.pdf | grep Pages
echo "--- last lines of page 4"
pdftotext -layout -f 4 -l 4 main.pdf - | grep -v '^\s*$' | tail -n 3
echo "--- first lines of page 5 (first must be References)"
pdftotext -layout -f 5 -l 5 main.pdf - | grep -v '^\s*$' | head -n 2
