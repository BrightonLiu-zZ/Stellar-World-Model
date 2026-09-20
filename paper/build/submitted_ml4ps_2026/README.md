# The version submitted to ML4PS 2026

This directory is a frozen copy of the paper as it was uploaded to OpenReview for the 9th Workshop
on Machine Learning and the Physical Sciences, deadline Sat 2026-09-19 23:59 AoE (= 2026-09-20
11:59 UTC). It exists so that every later change — the arXiv version above all — can be diffed
against a fixed baseline. Nothing here is ever edited; the live source stays `paper/main.tex`.

| file | what it is |
|---|---|
| `main.tex` | source, written 2026-09-20 00:18 local (07:18 UTC), i.e. inside the submission window |
| `main.pdf` | the compiled PDF, built 2026-09-20 00:19 local |
| `figures/fig1_deltas_xgb.pdf` | Figure 1 as submitted; the other figures and all tables are unchanged since `2d32dbb` and are not duplicated here |

**How this snapshot relates to the git history.** `2d32dbb` ("rebuild the compiled draft") is the
state *before* the 2026-09-19 review wave with Prof. Theissen. The submitted version is that commit
plus:

- the review-wave edits archived in `../theissen_0919/` (abstract and introduction wording, the TESS
  cadence sentence, the ten tasks spelled out, alternate-row shading in Figure 1);
- two late corrections made at 00:18: "one percent of the labels" → "a few percent" with the count
  "6 of 10" → "7 of 10", and the Figure 4 caption's "Spearman's" → "Spearman's rank correlation".

**What is deliberately *not* here.** No de-anonymised author block, no `preprint` option, no
acknowledgements: the submission was double-blind and carried the workshop's "Do not distribute"
footer. Those changes belong to the arXiv version, tracked in
`docs/roadmap/2026-09-19-arxiv-version-roadmap.md`, Phase 2.

The float-spacing `\setlength` overrides in the preamble *are* here, because they were in the
submitted file. They existed only to hold the body to four pages and are removed for arXiv
(roadmap 2.4).
