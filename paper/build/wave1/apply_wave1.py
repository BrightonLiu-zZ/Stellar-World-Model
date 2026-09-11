"""Apply Yue Ma's wave-1 PairTeX feedback to main.tex. Every replacement must match exactly once."""
from pathlib import Path
import re
import sys

src = Path("main.tex")
text = src.read_text(encoding="utf-8")

edits = [
    # ---- #1 A: abstract long sentence -> three sentences; "star embedding" -> learned features
    ("""We train a
Conv1D variational encoder with a recurrent latent-dynamics model on 195{,}883 unlabelled TESS
two-minute light curves, freeze it, and evaluate the resulting star embedding on ten labelled tasks
spanning variability classification, asteroseismology and rotation, against three baselines: the
engineered features alone, an architecture-matched supervised Conv1D trained end-to-end, and an MLP
on raw flux. Concatenating the learned embedding with the engineered features improves a frozen
linear readout beyond $2\\,\\mathrm{SE}$ on 7 of 10 tasks, while an untrained encoder does not.""",
     """We train a
Conv1D variational encoder with a recurrent latent-dynamics model on 195{,}883 unlabelled TESS
two-minute light curves and freeze it. We then evaluate the features it has learned for each star on
ten labelled tasks spanning variability classification, asteroseismology and rotation. Each task is
scored against three baselines: the engineered features alone, an architecture-matched supervised
Conv1D trained end-to-end, and an MLP on raw flux. Concatenating the learned and engineered features
improves a frozen linear readout beyond $2\\,\\mathrm{SE}$ on 7 of 10 tasks, while an untrained encoder
does not."""),
    # ---- #2 A: bold conclusion; unify on "engineered"
    ("""\\textbf{A learned representation and hand-built features are
complementary rather than competing, and once combined, plain logistic regression on frozen
features matches what end-to-end deep networks reach on the labels that exist.}""",
     """\\textbf{Self-supervision adds to engineered features rather than replacing them. Combined under a
plain logistic regression, the two match end-to-end deep networks trained on the same labelled stars,
669 to 16{,}002 per task.}"""),
    # ---- #1 A consistency: the remaining "embedding" occurrences that are ours
    ("freeze it, and ask what its embedding adds to a 25-feature engineered stack",
     "freeze it, and ask what the features it learns add to a 25-feature engineered stack"),
    ("\\item \\textbf{The learned embedding is complementary to engineered features, not a replacement for\nthem.}",
     "\\item \\textbf{The learned features are complementary to the engineered ones, not a replacement for\nthem.}"),
    ("\\caption{The two claims. \\textbf{(a)} What the learned embedding adds to the engineered features under",
     "\\caption{The two claims. \\textbf{(a)} What the learned features add to the engineered features under"),
    ("\\paragraph{The learned embedding adds to the engineered features.}",
     "\\paragraph{The learned features add to the engineered features.}"),
    # name mu as "the learned features" where it is defined
    ("\\paragraph{Representations and readout.} A star is represented by $\\mu$, the mean over its first\nsegment (16--20 windows) of the per-window posterior means.",
     "\\paragraph{Representations and readout.} A star's learned features are $\\mu$, the mean over its first\nsegment (16--20 windows) of the per-window posterior means."),
    # ---- arms (1): define the word at first use, and count all six
    ("""We report four
arms -- \\texttt{features}, $\\mu$, \\texttt{features}\\,$\\oplus\\,\\mu$, and the same fusion built on an
\\emph{untrained} encoder as a control.""",
     """We compare four
arms, each the same readout on a different feature set: \\texttt{features}, $\\mu$,
\\texttt{features}\\,$\\oplus\\,\\mu$, and, as a control, the same fusion built on an \\emph{untrained}
encoder. The two supervised baselines of the next paragraph are the fifth and sixth arms."""),
    # ---- #3 A: heading + diary-tone opener
    ("""\\paragraph{Selecting an encoder is harder than training one.} Our largest practical difficulty was
not optimisation but model selection.""",
     """\\paragraph{Model selection must be gated on a held-out probe.} Model selection, not optimisation, is
where this pipeline is fragile."""),
]

for old, new in edits:
    n = text.count(old)
    if n != 1:
        sys.exit(f"expected exactly 1 match, found {n}:\n{old[:120]}")
    text = text.replace(old, new)

# ---- #4 + #5: replace the whole Limitations paragraph (arms (2) and (3) live inside it)
start = text.index("\\paragraph{\\textbf{Limitations.}}")
end = text.index("\\paragraph{Reproducibility and disclosure.}")
new_lim = r"""\paragraph{\textbf{Limitations and next steps.}} The four linear-readout arms (\texttt{features},
$\mu$, \texttt{features}\,$\oplus\,\mu$, and the untrained control) share one linear model. It has few
parameters, so it cannot rescue weak features, and it is identical across the four, so the comparison
is between feature sets. Our claim is made for a linear readout, and we make no claim about more
powerful readouts. Our corpus is the bright ($T < 10$) two-minute sample, and every arm, the two
supervised baselines included, sees one 5.7-day segment per star. The fainter and far more numerous
stars, and longer light curves spanning several TESS sectors, are the obvious next experiments; the
latter would let rotation periods longer than 5.7 days be measured, which one segment cannot do. Our
negatives are stars absent from the label catalogues, not stars verified to lack the signal, so a
correct detection the catalogue missed counts against us; every detection score is therefore a lower
bound. The flare label marks stars that have flared somewhere in their TESS data, not a flare inside
the 5.7-day segment we encode, so the task is recognising a flaring star, not finding a flare. When
the readout is trained on one percent of the labels, the advantage narrows on 6 of 10 tasks, so how
much pre-training helps when labels are scarce remains an open question.

"""
text = text[:start] + new_lim + text[end:]
src.write_text(text, encoding="utf-8")
left = [m.start() for m in re.finditer(r"embedding", text)]
print("applied; remaining 'embedding' occurrences (should be the 2 cited ones):", len(left))
