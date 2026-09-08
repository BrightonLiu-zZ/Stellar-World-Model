# S1 — label-efficiency curves on cached µ (2026-09-01)

`experiments/analyze_s1_label_efficiency.py` → `experiments/s1_label_efficiency/`
(`s1_probe.csv` 11,265 rows · `s1_summary.csv` 340 · `s1_growth.csv` · `s1_footing.csv` ·
`s1_below_floor.csv` · `control_cvC/` 1,272 rows · three PNGs from
`experiments/plot_s1_label_efficiency.py`).
Roadmap stretch row **S1** (`docs/roadmap/2026-08-25-ml4ps-pivot-roadmap.md`), provenance Yue Ma
**W6**. Zero GPU, ~2.5 min CPU for the headline grid + ~9 min for the readout control.

> **Paper status: measurement complete, figure NOT committed.** The user's call 2026-09-01 is that
> which S1 panel (if any) enters the 4-page draft is **R5's decision**, taken with C1/C2/C3 in hand
> against the figure budget. This file records the result and lays out the candidate claims; it does
> not pick one, and `docs/ml4ps-paper-framing.md` is deliberately NOT edited.

*(`experiments/*.csv` is gitignored in this repo, so this file is the committed record of the verdict.)*

## The question, and the answer in one line

W6: a *semi-supervised* claim has to be paid for in label reduction. F1 measured the fusion advantage
(`features ⊕ µ` over `features`) at one point on the label axis — 100 % of the train split. S1 asks
whether it **grows as labels shrink**.

> **It does not.** Of the 10 printable tasks at the frozen readout, **6 narrow, 3 are flat, 1 widens**;
> under the readout control, **5 narrow, 3 flat, 0 widen**. **S1-E2 fired; S1-E1 did not, under either
> readout.** The fusion advantage is a *large-label* phenomenon.

The pre-registered expectation was the opposite, and it was written into the script's docstring before
anything was scored.

**The other half of the answer, which needed a control to find.** The *size* of the low-label deficit
is largely an artifact of the frozen readout's fixed penalty, not a property of µ — see
§"The readout control" below. So the sentence this experiment supports is the narrow one:

> the fusion advantage does not grow as labels shrink

and **not** the wider one it looked like at first:

> ~~fusion is worse than the engineered features when labels are scarce~~ — does not survive the control.

## Design, frozen before scoring (user decisions 2026-09-01)

| axis | choice | why not the alternative |
|---|---|---|
| x-axis | **union ladder**: fractions {1, 3, 10, 30, 100} % ∪ absolute {50, 100, 300, 1000, 3000} train stars, capped at each task's n_train | a fraction means 20× different label counts across tasks (1 % = 160 stars for `osc_giant`, 8 for `rgb_vs_heb`), so a pure fraction axis puts non-comparable budgets on the same tick. The roadmap's `{1, 10, 100} %` points are contained exactly and ride in the CSV as `frac` |
| floor | **n_train ≥ 50 AND (regression, or n_train_pos ≥ 10)** | the fusion arm has 25 + 128 = 153 columns; below ~50 rows the fit is degenerate whatever the prevalence. Rejected budgets are recorded with their reason in `s1_below_floor.csv`, not silently dropped |
| arms (5) | `features_only` · `µ` · `features ⊕ µ` (headline) · `untrained µ` · `features ⊕ untrained µ` | the dyn-off arm would double runtime to tell a second story in a one-panel figure |
| readout | `mean` pooling, `linear` family — the F1 headline cell | |
| draws | 10 stratified train resamples per budget; 1 at the full budget (nothing to resample, deterministic readout) | |
| spreads | **seed spread is the error bar** (average over draws, then sd over the 6 encoder seeds); **draw spread reported beside it, never merged** | one is representation noise, the other is which stars got labelled |

**Subsample the TRAIN split only, at star level, stratified** (class label; target quartile for the two
regressions). The test set is byte-identical at every budget, and the draw index is shared across all
five arms within a cell so a delta is differenced on identical rows.

**Population.** Inherited from F1 unchanged: the v1 packed subset (9,428 / 2,021 stars) and the
new-task pool (16,002 / 3,429). Both **case-control** — every prevalence here is inflated relative to a
survey (R8 measured absolutes falling 60–71 % under `survey_matched`, deltas keeping sign but not
magnitude). Carried as `population_note` on every row. No absolute score in this table transfers to a
survey population.

**`flare`** is scored with F1's label set (`flare_ever`) and stays **unprintable** until L1's visual
gate lands (STATUS 2026-08-26d, reporting rule B). It carries `printable=False`, is drawn grey in the
figure, and is excluded from the S1-E1 count.

## Footing, both gates before any curve was read

| gate | what it tests | result |
|---|---|---|
| **FOOTING-2** | the re-derived per-task keep masks reproduce F1's published `n_test` / `n_test_pos` | **exact on all 11** (eb 2021/196 · transit 2021/122 · ijspeert 2021/93 · rgb_vs_heb 161/113 · flare 3429/304 · numax_hon 1313 · rotation_period 150) |
| **FOOTING-1a** | the full-budget `features_only` row reproduces `f1_absolute.csv` | **exact, 0.0 on all 11** |
| **FOOTING-1b** | the full-budget µ-bearing rows reproduce `f1_absolute.csv` | 29 of 55 rows exact, 45 under 1e-4, **max 1.0e-3** |

The full-budget row is not a new measurement — same stars, same columns, same readout as F1 — so it
*must* reproduce. It does, up to one term that was diagnosed rather than waved at:

**The 1e-3 residual is BLAS thread count, demonstrated not assumed.** `lbfgs` on the 153-column fusion
design matrix is ill-conditioned enough that the reduction order inside the BLAS dot products changes
the fitted coefficients. Measured on `flare` / `untrained` / `features ⊕ µ`:

```
  main process, multithreaded BLAS  0.4798055587265005   <-- F1's regime; matches f1_absolute to 16 digits
  main process, OMP_NUM_THREADS=1   0.4808140093533278   <-- matches a joblib worker to 16 digits
```

joblib pins its workers to one thread, so the whole S1 grid ran single-threaded. The residual ordering
follows conditioning exactly — `untrained` fusion (1.0e-3) > trained fusion (3.2e-4) > µ-only (1.6e-4)
> features (0.0) — which is what the mechanism predicts and drift would not.

Two consequences, both acted on:

1. **The script now pins BLAS to one thread at import.** Left unpinned it returned different numbers at
   `--jobs 1` than at `--jobs 12`; that was a real reproducibility bug, not a cosmetic one.
2. **The term cancels in every S1-internal comparison.** Each cell on each curve is fitted in the same
   single-threaded regime, so it drops out of every fusion delta. It survives only in this
   cross-artifact check against a differently-threaded run. Same class of thing F1 already recorded for
   its two µ caches (3.5e-4 from cuDNN nondeterminism, moving 6-seed PR-AUC by ≤1e-4).

## The result — `S1-E1` does not fire, and it fails in the same direction on almost everything

`features ⊕ µ` − `features`, readout `mean`, linear, `hann0p3_fbwd` 6 seeds. `growth` = delta at the
smallest admissible budget minus delta at the full budget, **paired per encoder seed** (the same six
seeds appear at both budgets, so its 2·SE is the sd of the six differences — not the two levels' bars
added, which would double-count the shared seed effect).

| task | n_low | Δ at n_low | Δ at full n | growth | 2·SE | call |
|---|---|---|---|---|---|---|
| rotation | 283 | −0.0635 | +0.0157 | **−0.0793** | 0.0191 | narrows |
| solar_like_osc | 100 | −0.0322 | +0.0707 | **−0.1030** | 0.0058 | narrows |
| pulsating | 94 | +0.0010 | +0.0571 | **−0.0561** | 0.0146 | narrows |
| transit | 283 | −0.0237 | +0.0274 | **−0.0511** | 0.0114 | narrows |
| eb | 283 | −0.0124 | +0.0331 | **−0.0455** | 0.0047 | narrows |
| osc_giant | 50 | −0.0358 | −0.0028 | **−0.0330** | 0.0070 | narrows |
| ijspeert | 283 | −0.0669 | −0.0546 | −0.0123 | 0.0315 | flat |
| numax_hon | 50 | +0.0310 | +0.0358 | −0.0048 | 0.0067 | flat |
| rgb_vs_heb | 50 | −0.0540 | −0.0505 | −0.0035 | 0.0146 | flat |
| rotation_period | 67 | +0.0581 | +0.0132 | **+0.0449** | 0.0187 | widens |
| *flare (unprintable)* | *160* | *−0.0351* | *+0.0506* | *−0.0857* | — | *narrows* |

**Not one task narrows in the other direction.** Every `growth` is negative except `rotation_period`,
and every "narrows" call clears 2·SE comfortably. This is a strong negative, not a noisy one.

### The untrained control says the low-budget loss is dilution, not µ

At the smallest budget, `features ⊕ untrained µ` − `features` is **more negative** than the trained
arm's delta on 8 of 11 tasks (numax_hon −0.147 vs +0.031; rotation_period −0.125 vs +0.058;
rgb_vs_heb −0.071 vs −0.054; solar_like_osc −0.064 vs −0.032; transit −0.045 vs −0.024). So adding 128
columns to a linear readout with few rows costs score **generically**, and trained µ costs *less* than
random µ. What collapses at low n is the fusion arm's advantage over the engineered features — not the
representation.

### What did NOT hold, stated because it is the tempting over-read

The trained-vs-untrained µ contrast is positive at the smallest budget on **9 of 11** tasks — but it is
**larger** at low n on only **3 of 11** (`numax_hon` 0.566 vs 0.372, `rotation_period` 0.454 vs 0.300,
`osc_giant` 0.148 vs 0.138). On the other eight it shrinks with the labels too. "The representation
matters more when labels are scarce" is therefore **not** a general result here; it is a property of
three tasks, two of which are the regressions.

## The readout control — added after the curves were read, and it changed the claim

**Why it exists.** The fixedC result splits **9–2 along the readout, not along the task**. All nine
classification probes use `LogisticRegression(C=1.0)`, whose penalty does *not* depend on n, and all
nine lose their advantage at small budgets. Both regression probes use `RidgeCV`, whose alpha *is*
chosen by leave-one-out CV at every budget, and both keep a positive delta down to 50 and 67 training
stars. That is a perfect alignment with the readout and none with the task type, so "the fusion
advantage is a large-label phenomenon" had an untested rival reading: *a readout with n-independent
regularisation overfits 153 columns on few rows*.

**What was run.** `--readout cvC`: one knob changed (C selected by cross-validation over sklearn's own
default `Cs=10` grid — a hand-picked range would make this tuning rather than a check), everything else
held identical, on the nine classification probes at the ladder's fraction points. Fold count capped by
the rarer class, because a CV that cannot estimate its own objective would be a *worse* readout than the
one it audits, which would bias the control toward flattering the pre-registration. Reported **beside**
the frozen headline, never replacing it (the C3b precedent); the linear-probe lock governs the headline
probe, which this does not touch. Artifacts in `control_cvC/`; it inherits FOOTING-1/2 from the fixedC
run, since F1 published no cvC cell to reproduce.

**Result — the control confirms the slope and refutes the level.**

| task | budget | Δ fixed C=1.0 | Δ C by CV | shift |
|---|---|---|---|---|
| eb | 283 (3 %) | −0.0124 | **+0.0340** | +0.046 |
| eb | 943 (10 %) | −0.0342 | **+0.0433** | +0.078 |
| rotation | 283 | −0.0635 | **+0.0146** | +0.078 |
| rotation | 943 | −0.0754 | −0.0102 | +0.065 |
| pulsating | 94 (1 %) | +0.0010 | **+0.0365** | +0.035 |
| ijspeert | 943 | −0.1317 | −0.0126 | +0.119 |
| transit | 283 | −0.0237 | −0.0341 | −0.010 |
| solar_like_osc | 160 | −0.0259 | −0.0318 | −0.006 |
| osc_giant | 160 | −0.0534 | −0.0287 | +0.025 |
| **every task** | **full budget** | — | — | **\|shift\| ≤ 0.008 on 8 of 9** |

Three things follow, and they should be quoted together:

1. **The control does not disturb F1.** At the full budget the two readouts agree to ≤0.008 on 8 of 9
   tasks (`eb` +0.0331 → +0.0368, `solar_like_osc` +0.0707 → +0.0682, `osc_giant` −0.0028 → −0.0028).
   Regularisation stops mattering once there are rows, which is both expected and a sanity check on the
   control itself.
2. **The low-budget deficit is mostly the fixed penalty.** At each task's lowest matched budget the
   delta is positive on **3 of 9** under cvC against **1 of 9** under fixedC, and the shift is positive
   on 8 of 9. On the v1 block `eb`, `rotation` and `pulsating` are *above* the engineered baseline at
   94–283 labelled stars once the penalty can adapt.
3. **The slope survives intact.** At the highest matched budget the delta is positive on **6 of 9 under
   both**. Zero tasks widen under either readout. S1-E1 fails robustly.

`transit` and `solar_like_osc` are the two tasks the control does **not** rescue — their deficits are
slightly *worse* under cvC. Recorded rather than smoothed: the readout explanation is partial, not
total.

## Candidate claims, none selected (R5' decides)

| # | claim | figure | risk |
|---|---|---|---|
| A | *(null)* "the fusion advantage grows with label count; across 1–100 % of training labels there is no small-label regime where it is larger" | none — one Limitations sentence | weakens the W6 semi-supervised angle; costs no figure slot |
| B | "with a readout whose regularisation adapts to n, µ's contribution is already present at a few hundred labelled stars on the v1 detection block (`eb` +0.034 at 283 labels vs +0.037 at 9,428) while the asteroseismic pool probes need thousands" | `s1_delta_curves.png` + control | rests on a readout that is **not** the frozen protocol; needs both curves shown or it reads as cherry-picked |
| C | "the apparent low-label collapse of a frozen linear probe is a fixed-regularisation artifact" | `s1_readout_control.png` | a finding about our probe, not about SSL; spends a slot arguing with our own protocol |

ML4PS tells reviewers they need not read appendices (standing rule 6, amended), so a figure kept only
as backup buys little — in-or-out is a real decision, and it is R5's.


## Reading the two figures

`s1_delta_curves.png` — the paper candidate. Fusion delta vs labelled training stars, log x, one line
per task, seed-paired 2·SE bars, zero line. Almost every line rises to the **right**.

`s1_arm_curves.png` — 11 small multiples, all five arms in absolute score. This exists so the delta
panel cannot be misread: a widening delta can come from fusion improving *or* from the baseline
collapsing, and only the absolute panel distinguishes them. It shows the engineered baseline (black)
above the fusion arm (red) at low n on most classification probes, with the two crossing as n grows.

`s1_readout_control.png` — 9 small multiples, the fusion delta under both penalties on matched budgets
only. Drawn on matched budgets because the control ran the ladder's fraction points and not the
absolute infill; plotting each curve over its own budget set would put the two lines at different x and
invite reading a budget difference as a readout difference.

## Scope calls recorded rather than buried

- **The floor is applied to the realised draw size, not the target.** `rotation_period`'s n=50 level
  draws 49 stars (rounding across four target quartiles, each guaranteed one row) and is therefore
  rejected; its ladder starts at 67. Recorded because a 49-vs-50 rejection looks arbitrary otherwise.
- **`ijspeert_excl_villanova` is not scored** (ADR-0010: one probe per physical quantity).
- **The full-budget level runs one draw**, and its draw spread is 0 by construction, not by
  measurement. The CSV's `draw_kind` column says so on every row.
- **Regression strata are target quartiles**, the analogue of class strata. Without them a 67-star draw
  from `rotation_period` can miss the long-period tail entirely and the spread would be the draw's, not
  the budget's.
