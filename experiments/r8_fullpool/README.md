# R8 - survey-prevalence rescore of the v1 tasks (2026-08-12)

Roadmap task R8 (`docs/plans/2026-08-11-pre-freeze-roadmap.md`). Converts the claims-matrix
disclosure "prevalence inflated ~16x; gaps valid, absolutes not; stretch = full-pool rescore" into a
measured row. Script: `experiments/analyze_r8_fullpool.py` (stages `pool check pilot extract score`,
resumable per shard and per arm). Artifacts: `fullpool_scores.csv` (1,080 rows = 18 arms x 3 poolings
x 4 tasks x 5 populations), `fullpool_summary.csv`, `sizing_note.md`, `shard_manifest.csv`,
`replay_vs_packed_check.csv`.

## What was run

18 arms x 6 seeds' worth of encoders over **106,284 added stars** (every corpus star with windows
minus the 9,428 subset-train and 2,021 subset-val stars): `exp07_hann0p3_fbwd` x 6 seeds,
`exp07_hann0p3_off` x 6 seeds, and `untrained_w256` at **6 distinct inits** (the single-fixed-init
caveat of R7/C4 does not apply to this table). 59 min wall-clock on the RTX 4060.

**The probe is frozen.** Train split, estimator, standardization, and the amplitude residualization map
are all fitted on train only, so expanding the test population cannot change the fitted model. Only the
test-side negative population moves.

### Two protocol gates, both exact

1. **Replay vs packed.** Added negatives are replayed from `processed/sequences/*.npz` while existing
   test stars come from the packed memmap. On 200 stars present in both: max flux diff **0.0**, max mu
   diff **0.0**, max amplitude-feature diff **0.0**. The two readers are the same reader.
2. **Published-table reproduction.** The `subset_test` population reproduces
   `exp07_aux_gap_6seed.csv` on all **144** joined rows to **0.0**, all three poolings.

### Corrections to the handoff's premise

- Quiet pool is **102,008**, not 166,610; corpus stars with windows are **119,754**, not 195,883.
- Prevalence inflation on eb is **8.5x**, not 16x.
- **"Full pool as negatives" is not survey prevalence** - it overshoots. Only 15% of positives sit in
  test while 100% of the added negatives would, so `survey_full` lands at 0.18% against a 1.07% corpus
  rate. The reportable row is `survey_matched`, which subsamples negatives to the corpus base rate over
  20 draws and therefore carries a draw spread as well as a seed spread.

## Populations

| name | negatives | why |
|---|---|---|
| `subset_test` | existing 2,021-star test split | the published row; the reproduction gate |
| `quiet_matched` | catalogue-clean stars, thinned to corpus rate | prevalence corrected, pool composition still idealized |
| `survey_matched` | **every non-positive corpus star**, thinned to corpus rate | what "survey prevalence" actually claims |
| `quiet_full` / `survey_full` | all available negatives | deterministic maximum-dilution; **below** survey rate, do not quote as survey |

## Headline: `hann0p3_fbwd`, `mean_resid`, 6 seeds

| task | subset_test | quiet_matched | **survey_matched** | prevalence |
|---|---|---|---|---|
| eb | 0.590 | 0.242 | **0.172** +/- 0.008 seed, 0.010 draw | 9.7% -> 1.07% |
| pulsating | 0.801 | 0.322 | **0.266** +/- 0.015 seed, 0.010 draw | 10.7% -> 1.20% |
| transit | 0.166 | 0.069 | **0.066** +/- 0.014 seed, 0.006 draw | 6.0% -> 0.68% |
| rotation | 0.431 | 0.495 | **0.495** +/- 0.010 seed, 0.010 draw | 8.9% -> 12.46% |

Under `mean` pooling the same ordering holds at higher absolutes (eb 0.771 -> 0.344, pulsating
0.805 -> 0.275, rotation 0.559 -> 0.631, transit 0.144 -> 0.032).

**The pre-registered expectation held: absolutes drop hard, and that is the point.** eb loses 71% of
its PR-AUC, pulsating 67%, transit 60%.

## Three findings the rescore produced

**F1 - paired deltas are NOT prevalence-invariant, only their sign is.** The disclosure sentence
"gaps valid, absolutes not" is too strong as written. In absolute PR-AUC the gaps compress about 4x:

| contrast (mean_resid) | subset_test | survey_matched |
|---|---|---|
| eb, fbwd - off | +0.080 +/- 0.007 | +0.012 +/- 0.005 |
| eb, fbwd - untrained | +0.134 +/- 0.011 | +0.035 +/- 0.010 |
| rotation, fbwd - off | +0.046 +/- 0.013 | +0.043 +/- 0.012 |
| pulsating, fbwd - off | +0.001 +/- 0.010 (ns) | +0.020 +/- 0.011 (1.9 SE) |
| transit, fbwd - off | +0.020 +/- 0.011 | +0.013 +/- 0.008 (ns) |

Sign and (mostly) significance survive; magnitude does not. Any sentence quoting a gap in PR-AUC units
must name the prevalence it was measured at. At the over-diluted `survey_full` the eb arm contrast
disappears entirely (-0.001 +/- 0.002), which bounds how far the dilution argument can be pushed.

**F2 - the confusable-negatives cost is real and separable from prevalence.** `quiet_matched` and
`survey_matched` sit at the same base rate and differ only in pool composition, so their difference is
the price of negatives that contain rotators and other-class variables: eb **-0.070**, pulsating
**-0.056**, transit -0.003. Roughly a quarter of eb's remaining score at survey prevalence is an
artifact of a negative pool that excludes everything confusable. This was invisible in every prior
table.

**F3 - rotation moves the other way, which quantifies Q9 for the first time.** Quiet is defined as
"matched in no catalogue", so it excludes rotators; rotation positives reach the subset only through
the transit/eb/pulsating strata. The test split therefore sits at 8.9% rotation against a 12.46%
corpus rate - **below** survey prevalence - and matching requires thinning negatives, not adding them.
Rotation rises 0.431 -> 0.495 and becomes the best-scoring v1 task at survey prevalence, ahead of
pulsating. This is a property of the negative pool, not of the model. The rotation row must never be
quoted without it. (`quiet_matched` and `survey_matched` are identical for rotation by construction:
its survey negatives are the non-rotators, which is the quiet pool plus other-class positives, and the
thinning draws from the same fixed set.)

## Consequence for the paper

- Report `survey_matched` as the survey-prevalence absolute row, with `n_pos`/`n_neg`/prevalence
  columns printed so it can never be confused with the case-control row.
- Keep the case-control absolutes as the primary table (standard design), with this row as the
  prevalence disclosure made concrete rather than promised.
- Amend the disclosure sentence: gaps keep their **sign**, not their **magnitude**.
- The probe is fit `class_weight="balanced"` and is therefore badly calibrated at survey prevalence.
  Irrelevant to PR-AUC, which is rank-based, but state it once so no reviewer trips on it.

## Retention

`mu_cache/` holds 18 arms x 22 shards of per-star pooled mu (~2 GB). Keep until the paper's tables are
frozen; it is the only artifact that makes a re-cut of the negative population free. `features/`
(engineered amplitude features for 106,284 stars) is arm-independent and worth keeping regardless -
R16 and any future pool-scale probe can reuse it.
