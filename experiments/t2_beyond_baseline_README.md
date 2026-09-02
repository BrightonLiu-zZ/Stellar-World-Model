# T2 - beyond-baseline rotation probe (Q17 limb A), 2026-08-27

Lane R, roadmap task **T2**. Script: `experiments/analyze_t2_beyond_baseline_rotation.py`
(stages `features score gate`, resumable, CPU-only, no GPU). Artifacts under
`experiments/t2_beyond_baseline/`: `t2_bucket.csv` (1,184 rows) · `t2_summary.csv` (224) ·
`t2_gate.csv` · `t2_saturation.csv` · `t2_population.csv` · `t2_degradation.png`. Feature side-artifact:
`experiments/r8_fullpool/features25/shard_*.parquet` (25 features x 106,284 stars, reusable).

**The question.** The encoder's input baseline is `seq_len * window_size * cadence = 16 * 256 * 2 min
= 5.69 d`. TARS labels our corpus from 0.10 to 23.59 d, so across roughly 5.7-23.6 d we hold labels for
a quantity no tensor the encoder sees contains one full cycle of. Can a linear readout on frozen mu
recover them anyway, and is any recovery period inference rather than activity amplitude?

---

## 1. Verdict

**G-rot, evaluated exactly as pre-registered: PASS in both beyond-baseline buckets.**

**That PASS does not support the words "period inference", and this write-up does not use them.** The
gate names R2; the question the gate was written to answer ("period inference, or activity amplitude?")
is answered by rank correlation, and the two disagree in every beyond-baseline cell. The pre-registered
verdict is reported as the verdict. The disagreement is reported beside it, at equal prominence, and
neither is quietly swapped for the other.

The substantive reading is G-rot's own pre-registered fallback sentence, reached by a route the gate did
not anticipate:

> **the beyond-baseline signal is amplitude/activity, not period inference**

with one addition the gate could not have predicted: past the baseline, the amplitude-only control is
**better** at ordering periods than the trained latent is.

**The positive finding is a different one, and it is sharp:** the readout's predicted period **saturates
at the model's own input baseline**. Averaged over 6 seeds it returns 5.567 d (seed sd 0.122) for
5.7-13 d stars whose true mean is 8.60 d, and 5.623 d (seed sd 0.086) for >=13 d stars whose true mean
is 17.34 d, against an input baseline of **5.689 d**. Two buckets whose truths differ by 2x receive the
same answer, and that answer sits within 1-2% of the baseline. This is
`docs/architecture.md`'s pre-registered reading made quantitative:

> Monotonic degradation across buckets is a paper finding in its own right - it demonstrates that
> encoder behavior matches the physical limit of the input rather than overfitting to label artifacts.

---

## 2. What was run

| axis | values |
|---|---|
| population | **`r8_added`** = the 106,284 R8 `added` stars --> **13,789 TARS rotators**, 0.10-23.59 d |
| buckets | `<=2 d / 2-5.7 d / 5.7-13 d / >=13 d`, edges `(0, 2, 5.69, 13, inf]`, verbatim from T1 |
| split | 70/30 stratified by bucket, drawn ONCE at seed 0, reused across every arm/seed/control |
| fit | one global RidgeCV over all rotators, R2 sliced per bucket; a beyond-baseline-only refit is the labelled secondary |
| target | `log10(P_rot)` headline and gate; linear days reported beside it |
| arms | `exp07_hann0p3_fbwd` x 6, `exp07_hann0p3_off` x 6, `untrained_i0..i5`, at `mean` and `mean_std` |
| controls | untrained-mu floor · amplitude-only (4 cols) · A1 25-feature ceiling · mu(+)feats · untrained-mu(+)feats |
| estimator | `new_task_scorecard.score_regression` verbatim (RidgeCV, StandardScaler, CV alpha) |

Bucket sizes and the denominator that governs every R2 in this file:

| bucket | n | test n | log10 target SD |
|---|---|---|---|
| `<=2 d` | 5,681 | 1,704 | 0.365 |
| `2-5.7 d` | 3,587 | 1,076 | 0.131 |
| **`5.7-13 d`** | 3,337 | 1,001 | **0.093** |
| **`>=13 d`** | 1,184 | 355 | **0.067** |
| all | 13,789 | 4,136 | 0.594 |

### Both footing gates passed before any new number was read

1. **Estimator.** This script's ridge path, run in v1-compat mode (frozen v1 subset, ADR-0004's P <= 5 d
   cap, linear target, `mean`, the subset's own split), reproduces F1's published `rotation_period`
   R2 = 0.680302 to **2.4e-08**.
2. **Features.** The re-extracted amplitude columns equal R8's cached `features/*.parquet` to **0.0**
   across all **106,284** stars. `analyze_r8_fullpool.py` had computed all 25 features per star and
   persisted only 4, so this stage recovers 21 discarded columns rather than measuring anything new,
   which is why the tolerance is exact rather than approximate.

---

## 3. The two metrics disagree, and the disagreement is the result

`exp07_hann0p3_fbwd`, `mu`, `mean` readout, log10 target, global fit. Positive delta = trained mu better.

| bucket | baseline | metric | trained | baseline | delta | 2*SE | > 2*SE |
|---|---|---|---|---|---|---|---|
| 5.7-13 d | untrained | **R2 (gate)** | -5.630 | -18.044 | **+12.414** | 1.303 | **yes** |
| 5.7-13 d | amplitude-only | **R2 (gate)** | -5.630 | -29.487 | **+23.857** | 0.255 | **yes** |
| >=13 d | untrained | **R2 (gate)** | -58.635 | -86.754 | **+28.118** | 3.351 | **yes** |
| >=13 d | amplitude-only | **R2 (gate)** | -58.635 | -92.511 | **+33.876** | 1.010 | **yes** |
| 5.7-13 d | untrained | rho (reported) | 0.174 | 0.154 | +0.019 | 0.038 | no |
| 5.7-13 d | amplitude-only | rho (reported) | 0.174 | **0.270** | **-0.096** | 0.024 | no |
| >=13 d | untrained | rho (reported) | -0.080 | 0.101 | **-0.181** | 0.033 | no |
| >=13 d | amplitude-only | rho (reported) | -0.080 | **0.335** | **-0.415** | 0.026 | no |

Read the two halves together. On R2 the trained latent beats both baselines by enormous margins. On
rank correlation it beats neither, is **outranked by the 4-column amplitude control in both buckets**,
and is slightly *anti*-correlated with period past 13 d.

### Why: R2 in these buckets is a bias statistic, not a discrimination statistic

Decomposing the held-out predictions (log10 space, global fit, means over 6 seeds) separates level from
scatter:

| bucket | arm | true mean | predicted mean | bias | rho |
|---|---|---|---|---|---|
| 5.7-13 d | fbwd mu | 0.935 | 0.746 | **-0.189** | 0.174 |
| 5.7-13 d | off mu | 0.935 | 0.666 | -0.269 | 0.114 |
| 5.7-13 d | untrained mu | 0.935 | 0.573 | -0.362 | 0.154 |
| 5.7-13 d | amplitude-only | 0.935 | 0.460 | -0.475 | **0.270** |
| >=13 d | fbwd mu | 1.239 | 0.750 | **-0.489** | -0.080 |
| >=13 d | off mu | 1.239 | 0.689 | -0.550 | 0.077 |
| >=13 d | untrained mu | 1.239 | 0.653 | -0.586 | 0.101 |
| >=13 d | amplitude-only | 1.239 | 0.614 | -0.625 | **0.335** |

The bias column is monotone in exactly the order the R2 column is, and the rho column is not. That is
the whole disagreement in one table.

Every arm under-predicts these stars badly. The trained latent under-predicts *least*, so it wins R2;
the amplitude control orders them *best*, so it wins rho. Because each bucket's target SD is 0.067-0.093
dex while these biases are 0.19-0.63 dex, the squared-error term is dominated by the level offset and
the gate's R2 margin is almost entirely a bias margin. This is a property of the regime, not a defect in
anyone's arithmetic: G-rot was written before the band's variance structure was measured, and R2 turns
out not to discriminate the two hypotheses the gate names.

`t2_bucket.csv` carries `bias`, `pred_mean`, `truth_mean` and `pred_sd` on every row so this
decomposition is reproducible without re-fitting.

---

## 4. The saturation result

Predicted period in days (geometric mean over the bucket), 6 arms per family, `mean` readout, global fit.
`t2_saturation.csv`.

| bucket | true | **fbwd mu** | untrained mu | 25 features | amplitude-only |
|---|---|---|---|---|---|
| `<=2 d` | 0.61 | 0.85 | 1.36 | 0.85 | 1.93 |
| `2-5.7 d` | 3.39 | 4.35 | 3.26 | 3.66 | 2.39 |
| **`5.7-13 d`** | 8.60 | **5.57** (sd 0.12) | 3.74 | 6.74 | 2.88 |
| **`>=13 d`** | 17.34 | **5.62** (sd 0.09) | 4.50 | 6.11 | 4.11 |

The trained readout tracks the target across the two within-baseline buckets (0.85 --> 4.35 d against a
true 0.61 --> 3.39 d), then stops. Both beyond-baseline buckets receive the same answer, 5.57 and 5.62 d,
differing by less than half a seed SD while their truths differ by a factor of 2. The ceiling sits
**2.1% and 1.2% below the 5.689 d input baseline** respectively.

**The control that makes this non-trivial.** A shrinkage estimator with no usable signal reverts to the
unconditional mean, and the training set's geometric-mean period is **2.40 d**. The observed ceiling is
**2.3x above** that, so the readout is not merely reverting: it correctly identifies these stars as slow
rotators and then cannot say how slow. The untrained encoder does not show the same behaviour (3.74 and
4.50 d, not a common value), so the ceiling is a property of the trained representation.

**What this does not establish.** That the ceiling is *caused* by the 5.69 d input baseline is strongly
suggested by the coincidence but not proven here; the falsifiable version is that changing `seq_len` or
`window_size` should move the ceiling with it. That is a GPU-scale test, out of T2's scope, and is
recorded as a follow-up rather than claimed.

---

## 5. The secondary fit: is there signal in the band at all?

Refitting on the beyond-baseline rotators alone removes the level confound entirely (every arm is now
centred on the band). `mean` readout, log10 target, `fit_scope = beyond`.

| arm | rho 5.7-13 d | rho >=13 d |
|---|---|---|
| fbwd mu | 0.291 | 0.372 |
| off mu | 0.283 | 0.364 |
| untrained mu | 0.273 | 0.364 |
| amplitude-only | 0.287 | 0.350 |
| **25 features** | **0.398** | **0.393** |
| fbwd mu (+) feats | 0.397 | 0.399 |

Three readings, all of which matter more than the gate:

1. **There is real signal in the band.** Every arm reaches rho 0.27-0.40, so beyond-baseline periods are
   partially recoverable from a 5.69 d window. The monotonic collapse in section 3 is a property of the
   *global map*, not of the band: locally refitted, the `>=13 d` bucket is no harder than `5.7-13 d`.
2. **Almost none of that signal is SSL-specific.** Paired against the untrained floor, fbwd's rho
   increment is **+0.018 +/- 0.005** at 5.7-13 d and **+0.008 +/- 0.012 (ns)** at >=13 d. Against the
   amplitude control, +0.004 +/- 0.004 and +0.022 +/- 0.008. Real but negligible.
3. **The engineered features beat every mu arm in the band**, and fusion does not exceed features alone
   (0.397/0.399 vs 0.398/0.393). Whatever recovers beyond-baseline period here, the 25-feature basis
   already has it and mu adds nothing on top. This is consistent with C3's pre-registered risk firing
   and with D19's scoping rule: the honest claim is scoped, not across-the-board.

---

## 6. Disclosures

- **Population substitution.** G-rot pre-registers the `survey_matched` pool. That population cannot
  carry a regression: its positives are the 2,021-star subset-test split, i.e. 25 beyond-baseline
  rotators (T1). The gate was run on **`r8_added`** instead. Stated here rather than silently
  re-derived; the old population remains recoverable through the cited F1 row in section 7.
- **`r8_added`, not T1's `r8_scoreable`.** T1 named `added + existing_test` = 108,305. `existing_test`'s
  mu lives in a different cache layout and contributes **25 of the 4,546** beyond-baseline rotators
  (0.55%). One reader, one population; the 25 stars are omitted and named here.
- **A new split, not the frozen v1 probe.** R8's frozen-probe property does not transfer, because
  `added` stars were never in any split and a regression needs its own train set.
- **The cap is T2's own.** `new_task_scorecard` carries two different caps in one file (5.0 d for
  `score_rotation_period_from_mu`, 5.7 d for `prot_kounkel`). T2 inherits **neither**; its population is
  every added star with a TARS period, uncapped, bounded above by the data at 23.59 d. **T3** reconciles
  this with ADR-0004; T2 does not.
- **Catalogue purity, not a control.** `added` contains **zero** transit/eb/pulsating positives (the v1
  subset absorbed all of them), so these are catalogue-pure rotators and no cross-class contamination
  control was needed. This was measured, not assumed.
- **TARS circularity still applies** (T0). The target is TARS's period; T2 measures agreement with TARS
  inside a band TARS itself resolves. Nothing here speaks to limb B.
- **`mean_std` is reported, not gated.** G-rot names `mean`. At `mean_std` the same disagreement holds
  (fbwd mu rho 0.232 / -0.170 in the two buckets against amplitude-only's 0.270 / 0.335).

---

## 7. The v1 contrast row, cited not recomputed

From `experiments/f1_fusion_scorecard/f1_probe.csv` (frozen v1 subset, ADR-0004's P <= 5 d cap, linear
target, n_test = 150). Reported so the pre-substitution population stays visible:

| arm | R2 @ mean | R2 @ mean_std |
|---|---|---|
| hann0p3_fbwd mu | 0.677 +/- 0.008 | 0.756 +/- 0.009 |
| hann0p3_off mu | 0.537 +/- 0.017 | 0.685 +/- 0.015 |
| untrained mu | 0.377 (single init in that cache) | 0.707 |
| features only | 0.703 | 0.703 |
| fbwd mu (+) feats | 0.717 +/- 0.010 | 0.773 +/- 0.005 |

The contrast is the point: **inside** the baseline the trained latent beats the untrained floor by
+0.30 R2 and is competitive with the engineered basis. **Past** the baseline (sections 3-5) it beats the
untrained floor only on a bias statistic and is outranked by 4 amplitude columns. T1 established that
the v1 subset's own beyond-baseline row (25 test rotators) is not computable, and it was not attempted.

---

## 8. What T2 hands forward

- **T3** gets its per-bucket table. ADR-0004's 5.0 d cap is now measurable against the 5.69 d physical
  edge, and section 4 gives T3 a physical reason to prefer one: the readout's own ceiling is at 5.6 d.
- **R6' / the Theissen meeting** gets a real answer to limb A: past its input baseline the model knows
  a star is a slow rotator and cannot say how slow, with the ceiling at the baseline. It does not get a
  "we beat the baseline" claim, because there isn't one.
- **The paper** gets a limitation, not a result. Under D19's scoping rule this is a scoped negative on a
  supplementary task; it does not touch the D16 fusion spine.
- **A new open question** (proposed, not registered here): does the saturation ceiling move with
  `seq_len`/`window_size`? That is the falsifiable form of section 4's causal claim and it needs GPU.
