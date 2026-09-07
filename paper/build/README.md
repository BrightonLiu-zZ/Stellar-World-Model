# ML4PS 2026 submission — build notes

`paper/main.tex` is the single source of truth.

## Layout, and the rule that produced it

`paper/` holds **only what Overleaf compiles**, so the whole directory can be uploaded without
picking through it. Everything that is a tool, a provenance record or a screen copy lives here in
`paper/build/`, which is never uploaded.

| path | what it is |
|---|---|
| `paper/main.tex` | the paper. NeurIPS 2026 template, unmodified `.sty`; the workshop footer is set by a `\renewcommand` in the preamble, not by editing the style file |
| `paper/neurips_2026.sty` | official template, downloaded 2026-09-04 from `media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`. **Do not edit** — a modified template is a desk reject |
| `paper/refs.bib` | 20 entries, every one resolved against the arXiv API by id |
| `paper/tables/table1_scorecard.tex` | generated. Do not hand-edit |
| `paper/figures/fig1_deltas.pdf` | generated. The PDF is what LaTeX embeds |
| `paper/main.pdf` | the compiled draft, kept for sending to collaborators. Overleaf builds its own |
| `build/check_refs.py` | regenerates `refs_verified.txt`. Re-run after any citation change |
| `build/refs_verified.txt` | what the arXiv API returned for each entry (title, authors, date). Diff `refs.bib` against this, do not trust it. All 20 rows read `[OK]`; the trailing block is provenance for the two ids originally found by title search, not an outstanding queue |
| `build/table1_data.csv` | every number printed in Table 1 and Figure 1, with provenance |
| `build/dynamics_ablation.csv` | the per-task dynamics-off contrast behind the Results ablation paragraph, paired by seed |
| `build/fig1_deltas.png` | screen copy of Figure 1 for slides and chat. LaTeX never reads it |

The Word mirror (`ML4PS2026_draft1.docx` and its `main_docx.tex`) was **deleted 2026-09-07** once
review moved to Overleaf. To regenerate it, see the pandoc recipe below.

## Rebuilding

Table and figure (uses the `swm` env — `astro`'s matplotlib fails to load its DLLs in this shell):

```
python experiments/plot_ml4ps_scorecard.py
```

PDF, from inside `paper/` (MiKTeX; `latexmk` is unavailable because MiKTeX cannot find a perl engine):

```
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Word mirror, if it is ever wanted again (portable pandoc at `C:/tmp/pandoc-3.6.3/`; the figure has to
be swapped to the PNG in `build/`, because Word cannot embed a PDF):

```
sed 's#figures/fig1_deltas.pdf#build/fig1_deltas.png#' main.tex > build/main_docx.tex
pandoc build/main_docx.tex --citeproc --bibliography=refs.bib --resource-path=.:build -o build/draft.docx
```

## Venue constraints this build satisfies

- 4 pages excluding references: body ends on page 4, references run pages 5-6.
- Footer reads exactly "Submitted to the 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026). Do not distribute."
- Fully anonymized; no code link (optional at this venue for the Research track, verified against the 2026 guidelines page). Replication is served by a text recipe in §4 plus the sizes given there, not by a repository.
- Generative-AI use disclosed, as the guidelines require.

## Page budget

The body fills page 4 to its last line. Any addition needs a matching cut, and
`pdftoppm -png -f 4 -l 5 main.pdf pg` is the check — the page count alone does not tell you whether
the body spilled, because the references occupy two pages either way. This has already bitten once:
hand edits on 2026-09-07 pushed two lines onto page 5 while the page count stayed at 6.

## Overleaf

Upload the whole of `paper/` except `build/`. Nothing in `build/` is needed to compile.
