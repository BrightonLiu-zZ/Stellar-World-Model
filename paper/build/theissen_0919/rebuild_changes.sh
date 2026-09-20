#!/usr/bin/env bash
# Rebuild changes_0919.pdf: cover (comment-by-comment list + Figure 1 before/after) + one latexdiff page
# showing the title, abstract and first Introduction paragraph. Run from anywhere.
cd "$(dirname "$0")/../.." || exit 1          # paper/
W=build/theissen_0919
cp main.tex $W/main_after.tex
cp figures/fig1_deltas_xgb.pdf $W/fig1_after.pdf
latexdiff-so $W/main_before.tex main.tex > $W/main_diff.tex
python - "$W" <<'EOF'
import sys
from pathlib import Path
w = Path(sys.argv[1])
src = (w / "main_diff.tex").read_text(encoding="utf-8")
cut = src.index("In this domain, the baseline that a learned encoder")
bs = chr(92)
tail = f"\n{bs}clearpage\n{bs}bibliographystyle{{plainnat}}\n{bs}bibliography{{refs}}\n{bs}end{{document}}\n"
Path("excerpt_diff.tex").write_text(src[:cut] + tail, encoding="utf-8")
EOF
pdflatex -interaction=batchmode excerpt_diff.tex | grep -aE "^!"
bibtex excerpt_diff | grep -aiE "error"
pdflatex -interaction=batchmode excerpt_diff.tex | grep -aE "^!"
pdflatex -interaction=batchmode excerpt_diff.tex | grep -aE "^!"
(cd $W && pdflatex -interaction=batchmode cover.tex | grep -aE "^!")
pdfseparate -f 1 -l 1 excerpt_diff.pdf $W/ex1.pdf 2>&1 | grep -av "recursive"
pdfunite $W/cover.pdf $W/ex1.pdf $W/changes_0919.pdf
mv excerpt_diff.tex $W/
rm -f excerpt_diff.* $W/ex1.pdf $W/cover.pdf $W/cover.aux $W/cover.log
pdfinfo $W/changes_0919.pdf | grep Pages
