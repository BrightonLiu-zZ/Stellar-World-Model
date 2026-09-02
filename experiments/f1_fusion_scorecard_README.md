# F1 — the fusion readout on all 11 tasks, at the N1 readout ladder (2026-08-25)

`experiments/analyze_f1_fusion_scorecard.py` → `experiments/f1_fusion_scorecard/f1_{probe,summary,absolute}.csv`
(1,395 / 210 / 330 rows). Roadmap row **F1** (`docs/roadmap/2026-08-25-ml4ps-pivot-roadmap.md`),
provenance Yue Ma **W11** plus the carried **N1** readout switch. This is the ML4PS paper's headline
table under spine **D16**.

*(`docs/` and `*.csv` are gitignored in this repo, so this file is the committed record of the verdict.)*

## What did not exist before

Both halves of the measurement existed, in different shapes, and neither could be printed beside the
other:

| | fusion readouts | pooling | code path |
|---|---|---|---|
| v1, 4 tasks | `exp07_channel_probe.csv` | `mean` only | `encoder_mu_table`'s own encode pass |
| menu, 7 probes | `exp08_menu_channel/` (R1) | `mean` only | `new_task_scorecard` scorers |

So no task had `mean_std`, the two blocks ran different code, and **no single artifact carried all
11**. F1 produces that artifact **from the caches alone — zero GPU**, because both R1 µ caches store
per-window blocks, which makes the `std` half of `mean_std` derivable on CPU.

## Controls, all exact

| control | result |
|---|---|
| `eb` @ `mean`, `hann0p3_fbwd`, 6 seeds vs published `exp07_aux_gap_6seed.csv` **0.7710** | **0.7711**, \|diff\| **1.3e-4** |
| all 7 menu fusion deltas vs R1's published table | reproduce to **≤5e-4** (e.g. solar_like_osc +0.0707 vs +0.071; ijspeert −0.0543 vs −0.054) |

The second control matters more than it looks: F1 reaches the menu numbers through a **different
pooling and residualisation path** than R1 did, so agreeing to 5e-4 is an independent reimplementation
check, not a tautology.

A third, non-exact observation worth recording: the two v1 µ caches (`exp07_forensics/mu_cache`,
mean/std-pooled, and `exp08_menu_channel/subset_mu_cache`, per-window blocks) agree on population and
star order exactly but their mean-pooled µ differs by up to **3.5e-4**, i.e. 0.2–1.0 % of a µ sd. That
is batching/cuDNN nondeterminism between two encode passes of the same checkpoint, and it moves the
6-seed PR-AUC by ≤1e-4. Recorded so nobody later reads it as cache drift.

## Readouts (N1 / D2), and one decision about what `mean ⊥ amp` can mean

`mean` is the headline, `mean_std` the second, `mean ⊥ amp` an appendix robustness row that is never
a headline score. It is **not a pooling** — it is `mean` with the periodicity-free amplitude basis
projected out — and it is named accordingly.

**`mean ⊥ amp` is emitted for µ-only arms and never for fusion arms.** Projecting the 4-scalar
amplitude basis out of µ and then concatenating all 25 engineered features — which *contain* that
basis — back in re-introduces exactly what was removed, so the quantity would not mean anything. The
fusion arm's confound control is `µ ⊥ all-25` instead, which is what R1 used. Because that readout has
no `features_only` arm by construction it yields no *delta* rows at all, which is why `f1_absolute.csv`
exists: absolute score is the selection quantity anyway (F4), and a summary carrying only deltas would
have silently dropped the appendix readout after computing it.

## Verdict — fusion wins on 7 of 11, and the four v1 tasks are the strongest block

`features ⊕ µ` − `features`, `hann0p3_fbwd`, 6 seeds, readout `mean`. Every delta names its prevalence
(R8-F1). `features_only` is seedless, so the spread is the µ side's alone — stated, not dressed up as
a two-sided SE.

| block | task | n_test (pos) | prevalence | features alone | fusion Δ | 2·SE | >2·SE |
|---|---|---|---|---|---|---|---|
| menu | solar_like_osc | 3429 (417) | 0.122 | 0.320 | **+0.0707** | 0.0058 | yes |
| v1 | pulsating | 2021 (216) | 0.107 | 0.789 | **+0.0572** | 0.0105 | yes |
| menu | flare | 3429 (304) | 0.089 | 0.474 | **+0.0506** | 0.0078 | yes |
| menu | numax_hon (R²) | 1313 | — | 0.831 | **+0.0358** | 0.0022 | yes |
| v1 | eb | 2021 (196) | 0.097 | 0.742 | **+0.0330** | 0.0030 | yes |
| v1 | transit | 2021 (122) | 0.060 | 0.190 | **+0.0274** | 0.0125 | yes |
| menu | rotation_period (R²) | 150 | — | 0.703 | **+0.0132** | 0.0100 | yes |
| v1 | rotation | 2021 (180) | 0.089 | 0.540 | +0.0155 | 0.0163 | no (0.95×) |
| menu | osc_giant | 3429 (1313) | 0.383 | 0.920 | −0.0028 | 0.0019 | no |
| menu | rgb_vs_heb (ROC) | 161 (113) | 0.702 | 0.758 | −0.0505 | 0.0186 | no |
| menu | ijspeert | 2021 (93) | 0.046 | 0.508 | −0.0543 | 0.0289 | no |

**All four v1 tasks are positive**, three of them beyond 2·SE — a stronger block than the menu, and a
picture R1 could not see because it never ran v1. `rotation` misses at 0.95× the bar.

Where fusion costs score the pattern is R1's and is unchanged: the feature baseline is already strong
(`osc_giant` 0.920) or the probe is small (`rgb_vs_heb` 161 stars, `ijspeert` 93 positives) — and the
**untrained arm loses there by the same or a larger margin** (ijspeert −0.096, rgb_vs_heb −0.038),
which is the signature of extra collinear columns diluting a linear readout rather than of µ carrying
misleading content.

### The readout changes two rows, which is why every table must name it

At `mean_std`, `rotation` flips from +0.0155 to **−0.0099** and `rotation_period` jumps from +0.0132
to **+0.0695**. Seven of eleven still clear 2·SE, but not the same seven. Quoting a fusion number
without its readout is not a reportable statement.

## The dynamics-specificity contrast is the robust half, and it generalises further than R1 showed

`Δ(fbwd) − Δ(off)`, paired per seed. `features_only` cancels exactly, so this is the one statistic here
carrying **both** arms' seed spreads (F17).

| task | Δ of Δ | 2·SE | >2·SE | | task | Δ of Δ | 2·SE | >2·SE |
|---|---|---|---|---|---|---|---|---|
| solar_like_osc | +0.0472 | 0.0123 | yes | | pulsating | +0.0065 | 0.0193 | no |
| ijspeert | +0.0275 | 0.0260 | yes | | osc_giant | +0.0031 | 0.0035 | no |
| rotation | +0.0262 | 0.0163 | yes | | rgb_vs_heb | +0.0015 | 0.0298 | no |
| numax_hon | +0.0254 | 0.0054 | yes | | transit | +0.0092 | 0.0207 | no |
| eb | +0.0206 | 0.0117 | yes | | | | | |
| rotation_period | +0.0186 | 0.0178 | yes | | | | | |
| flare | +0.0176 | 0.0133 | yes | | | | | |

**Every one of the eleven is positive**; 7 clear 2·SE. The menu half reproduces R1's 5 of 7 exactly,
and v1 adds `eb` and `rotation`. The dynamics term buys complementarity to the engineered basis on
every task measured, even the three where fusion's *level* is negative.

## µ alone vs features alone — a new asymmetry between the two blocks

| | µ alone beats features |
|---|---|
| v1 (4 tasks) | **3 of 4** — pulsating +0.017, eb +0.030, rotation +0.020; transit −0.046 |
| menu (7 probes) | **1 of 7** — solar_like_osc +0.016 only |

The menu row reproduces R1's standing result ("features alone still beat µ alone on 6 of 7 probes, the
exception being solar_like_osc"). The v1 row is new and points the other way: on the variability tasks
µ is competitive **on its own**, while on the asteroseismic menu the engineered features dominate and
µ can only *add*. Both halves of "engineered features beat SSL alone" and "SSL adds to engineered
features" remain simultaneously true, but which one leads now depends on the block.

## Scope calls made in building this, recorded rather than buried

- **`hann0p3_off` is an arm.** The session handoff listed four arms and omitted it; that list is a
  minimum, and without the off arm the contrast above — R1's most robust result — cannot be computed.
- **Encoder-agnostic by construction.** Cell name and both cache dirs are parameters, so a D17 switch
  re-runs this with zero code change. The exp09 µ-cache trap applies: caches key on `{arm}.npz` with
  no checkpoint in the key and short-circuit on `exists()`, so any new extraction needs its **own**
  `--cache-dir`. The resolved cache paths are written into `f1_probe.csv` so a mislabelled table is
  detectable after the fact.
- **Prevalence is NaN, not zero, for the two regression probes.** R8-F1 requires every PR-AUC delta to
  name its prevalence; it says nothing about R², and inventing a number there would put a meaningless
  column beside a meaningful one.

## C3b — the same-readout control, and what it does to the claim

`--families gbm mlp`, 6 `hann0p3_fbwd` seeds + untrained, readout `mean` →
`experiments/f1_nonlinear_control/`. Approved 2026-08-25 as a **control**, reported beside the linear
headline and never replacing it; it does not touch the v1 headline probe on µ that CLAUDE.md's
linear-probe lock governs.

**Why it had to exist.** C3 showed GBM on the 25 features alone beating the linear fusion arm on 10 of
11 tasks. But that comparison changes readout **and** input at once, so it cannot separate "µ carries
nothing extra" from "µ was compensating for a linear readout". Only the same-readout cell can.

**Estimator note — a flaw found by a control and fixed before the numbers were read.** F1 scores the
engineered arm once, which is right under `linear` (logistic/ridge are deterministic) and **wrong**
under GBM/MLP, which carry their own seeds. The first pass differenced 6 fusion seeds against a single
features fit; a cross-script check against C3 read **0.0062** instead of 0.0, the same size as two of
the four surviving margins. The delta is now **paired per seed** and the cross-script check reads
**exactly 0.0** on both families. Verdicts were unchanged, but they were not safe until this was fixed.

**Result — 4 of 11 under GBM, 0 of 11 under MLP:**

| task | linear Δ | **GBM Δ** | 2·SE | >2·SE | untrained Δ (GBM) |
|---|---|---|---|---|---|
| **transit** | +0.027 | **+0.041** | 0.014 | yes | +0.014 |
| eb | +0.033 | **+0.016** | 0.004 | yes | −0.004 |
| rgb_vs_heb | −0.051 | **+0.014** | 0.009 | yes | −0.012 |
| rotation_period | +0.013 | **+0.013** | 0.010 | yes | −0.032 |
| numax_hon | +0.036 | +0.000 | 0.001 | no | −0.005 |
| flare | +0.051 | −0.001 | 0.006 | no | −0.038 |
| pulsating | +0.057 | −0.005 | 0.011 | no | +0.011 |
| osc_giant | −0.003 | −0.005 | 0.004 | no | +0.002 |
| ijspeert | −0.054 | −0.009 | 0.009 | no | +0.004 |
| solar_like_osc | +0.071 | −0.019 | 0.008 | no | −0.031 |
| rotation | +0.016 | −0.029 | 0.010 | no | −0.046 |

Under **MLP**, 0 of 11 — and the untrained arm loses by more on almost every probe, which is the
signature of 128 extra collinear columns diluting the fit rather than of µ misleading it.

**All four survivors beat their untrained control**, so what survives is not generic column addition.

### The claim this licenses

Most of what µ contributed to the **linear** fusion readout was compensating for that readout, exactly
as the 08-15 pre-registration warned. The exception is the interesting part: **`transit` is the only
task where µ's contribution GROWS under the stronger readout** (+0.027 → +0.041, the largest GBM delta
in the table). That is mechanistically coherent — transit is the localized task, where global
engineered features cannot represent the signal no matter how much capacity the readout is given, and
it is the same asymmetry the MIL work kept finding (pooling gains are transit-only, ADR-0008-lite).

Consequently the D16 spine sentence — *"SSL µ adds information that engineered features and a
supervised CNN do not already carry"* — **is not supported as written** once the baseline is allowed a
nonlinear readout. The defensible, W8-scoped version is:

> µ adds to a linear readout on engineered features across most of the 11 tasks; under a nonlinear
> readout on the same features that contribution largely disappears, except on the localized `transit`
> task, where it survives and grows. The dynamics term's complementarity (fbwd vs off) is positive on
> all 11 tasks under the linear readout.

This lands **before** C1/C2 (the supervised arms), which can only narrow it further. It follows from
Yue Ma's own W12/W13 baseline discipline, so it is her advice working rather than a setback — but it
is a smaller claim than the roadmap assumed, and it is the user's call whether to carry it to her now
or at draft review. **`docs/ml4ps-paper-framing.md` has deliberately NOT been edited.**

## C3b over four encoder cells (2026-08-29) — `experiments/f1_exp09_cells/`

The published C3b table carries **one** encoder. This extension runs the identical measurement over
four, to answer whether its verdict is a property of the readout or of the incumbent.

**Cell set — the union of two clauses, fixed before the table was read.** Top 3 exp09 cells on `mean4`
(the unweighted mean of the four v1 PR-AUCs, at `mean` pooling — see the 08-29 readout decision below),
plus the three named by the user:

| cell | mean4 | rank | in via |
|---|---|---|---|
| `exp09_dpss_impulse_w0p03_fb0p01` | 0.5783 | 1 | both clauses |
| `exp09_hann_w0p10` | 0.5768 | 2 | top-3 only |
| `exp09_dpss_impulse_w0p025_fb0p01` | 0.5746 | 3 | both clauses |
| `exp07_hann0p3_fbwd` (incumbent) | 0.5712 | 5 | named only |

**Readout decision, 2026-08-29 (user):** the reported probe score is **`mean`**, never `mean_std` and
never `max(mean, mean_std)` per task — the last selects the readout on the test score, which inflates
every cell and inflates the noisiest cells most. Under `max`, the mean4 ranking is a *different* one
(w0p025_fb0.01 first, the incumbent third), which is exactly why it is not used.

**Cost.** 18 arms (3 cells x 6 seeds) x two star populations = 36 mu caches, ~35 min GPU; then F1 over
25 arms x readout `mean` x {linear, gbm, mlp} = 75 jobs, 1 h 41 min CPU. ~5 GB of cache, written into
the existing `exp08_menu_channel` dirs (safe: the mu-cache trap is same-arm-name/different-checkpoint,
and every new arm name is unique to its cell).

**Checkpoint — `best_recon_aux` for all four, and the confound is measured, not assumed.**
`exp07_hann0p3_fbwd` seeds 0-5 **predate** `best_recon_only.pt` and have none, so `best_recon_aux` is
the only checkpoint the four cells share and the only one that keeps this table on a single selection
rule. That runs against A3 (an aux-swept cell should be selected by an aux-free metric), so the size of
the disagreement is reported rather than argued: against `exp09_menu_mu/f1_style`, which holds the two
`fb0.01` cells at `best_recon_only`, mean |delta| **0.0007 / 0.0008** and max **0.0030** over the 40
linear rows the two artifacts share.

**Controls, both exact.**

| control | result |
|---|---|
| `eb` @ `mean`, `hann0p3_fbwd`, 6 seeds vs published **0.7710** | **0.7711**, \|diff\| 1.3e-4 |
| all 90 incumbent + features rows vs `f1_fusion_scorecard` / `f1_nonlinear_control` | max \|diff\| **8.3e-17** (one row; float round-off) |
| `features` column, this artifact vs `exp09_menu_mu` (independent run) | **0.0** on all 10 |

### Result 1 — the readout decides which arm wins, in every cell

Argmax arm over the 10 reportable tasks (`flare` dropped):

| cell | linear (fus/feat/mu) | gbm | mlp |
|---|---|---|---|
| `hann w=0.30` | **6** / 3 / 1 | 5 / 5 / 0 | 1 / **9** / 0 |
| `hann w=0.10` | **6** / 3 / 1 | **6** / 4 / 0 | 0 / **9** / 1 |
| `dpss w=0.03 fb0.01` | **6** / 2 / 2 | **6** / 4 / 0 | 1 / **7** / 2 |
| `dpss w=0.025 fb0.01` | **6** / 2 / 2 | **6** / 4 / 0 | 0 / **8** / 2 |

`features (+) mu` tops **6 of 10 under `linear` in all four cells**; `features` alone tops 7-9 of 10
under `mlp` in all four. **Swapping the encoder does not move the C3b verdict** — it is a readout
effect, which the single-cell table could assert but not show.

### Result 2 — "which cell is best" is not well posed without naming the readout

Cell topping each arm, of 10 tasks:

| arm | linear | gbm | mlp |
|---|---|---|---|
| `mu` | w0p025_fb **6** | hann w=0.10 **5** | w0p03_fb **7** |
| `features (+) mu` | incumbent **5**, w0p025_fb **5** | 3/3/2/2, no leader | w0p025_fb **6** |

Three readouts, three different orderings of the same four encoders — and none of them is the `mean4`
ranking that selected the cells. These are **argmax counts, not tests**: no error bar is attached, and
the margins behind them are frequently smaller than the seed spread.

### Result 3 — mu alone still loses to the engineered features, in every cell

`mu - features` is positive on 3-4 of 10 tasks under `linear` and on 0-3 under the nonlinear readouts,
for all four cells. The two floored cells recover a little under `mlp` (2 of 10 each, against 0 for the
incumbent) — the only place on this table where the cell axis changes the qualitative picture.

**Scope.** The fusion **deltas** (with 1*SE / 2*SE) remain incumbent-only: widening them would add 30
rows no pre-registration asks about, and the cross-cell question they invite is a different statistic.
The rows exist in `f1_exp09_cells/f1_summary.csv` if it is ever asked. Rendered in section 3 of
`src/notebooks/exp09_probe_score_study.ipynb`.
