# exp10 pre-design forensics — six zero-GPU tests that were meant to pick the training lever (2026-08-29/30)

Run against the handoff `tmp/handoff/2026-08-29-exp10-forensics-handoff.md`. Everything here is scored
on **cached µ only, zero GPU**, on the shipping encoder `hann0p3_fbwd` (6 seeds) plus its untrained twin.
Scripts: `experiments/exp10_build_derived_mu.py`, `experiments/exp10_footing_checks.py`,
`experiments/analyze_exp10_{fb_mu_predictability,fc_mil,fe_subsample,ff_gbm_fairness}.py`,
`experiments/exp10_forensics_verdict.py`. Outputs: `experiments/exp10_forensics/` (1.4 GB, one subdir
per forensic). Wall time ≈ **4 h 45 min** CPU, 2–5 concurrent jobs on 16 cores.

*(`docs/` and `*.csv` are gitignored in this repo, so this file is the committed record of the verdict.)*

**Headline, before any detail: no lever tested here moves the C3b survival count above 4 of 11.** The
count under GBM is 4 for the incumbent, 5 for the best PCA truncation (and that +1 is two
negligible-magnitude rows in exchange for two real ones), 4 for the best window statistic, and 4 after
fair tuning of both arms. exp10's provisional success condition (a) — *survive a nonlinear readout on
more than 4/11* — is not reachable by any of the readout-side moves these forensics could test.

---

## 0. Footing gates — all pass, and by a wider margin than the handoff asked for

`experiments/exp10_forensics/footing/footing_gates.csv`. The derived-cache path (write a `{arm}.npz`
holding one star-level row per star, then run `analyze_f1_fusion_scorecard.py` against it unchanged) is
not merely consistent with the published artifacts, it is **bit-identical** to them.

| gate | rows | max abs diff | tol | passed |
|---|---|---|---|---|
| named target: `eb` @ `mean`, µ arm, 6 seeds vs published 0.7710 | 6 | **1.3e-4** | 5e-4 | yes |
| identity cache vs `f1_fusion_scorecard/f1_probe.csv` (linear) | 330 | **0.0** | 5e-4 | yes |
| identity cache vs `f1_nonlinear_control/f1_probe.csv` (gbm) | 300 | **0.0** | 5e-4 | yes |
| `features_only` under GBM vs `c3_feature_controls` | 90 | **0.0** | 0.0 | yes |
| resolved cache paths recorded in every output CSV | 630 | — | — | yes |

**One wording mismatch in the handoff, recorded rather than snapped.** §1 footing gate 1 calls 0.7711
"F1's published `eb` @ `mean` linear **fusion**". It is the **µ-alone** arm; the fusion `eb` number is
0.775. The gate was run against the µ arm, which is what `REPRO_TARGET` in the F1 script selects and
what `exp07_aux_gap_6seed.csv` holds.

**Two further controls that were free and are worth keeping.**

- The derived `mean_std` arm scored at readout `mean` reproduces F1's own `mean_std` **readout** on the
  raw cache exactly (`rotation` −0.0099, `rotation_period` +0.0695 — the two rows the published README
  singles out as readout-sensitive). That is an independent check of the derived-cache path at 256
  columns, not just at the 128-column passthrough.
- F-F's `default` column, computed by a separately written script, reproduces **all eleven** published
  C3b GBM deltas (eb +0.0164, transit +0.0413, rgb_vs_heb +0.0142, rotation_period +0.0125, …).

---

## 1. The fourteen pre-registered rules, scored verbatim

`experiments/exp10_forensics/verdict/decision_rules.csv`.

| rule | fired | one-line reason |
|---|---|---|
| R-A1 | **no** | best transform `pca16` survives 5/11, the bar is 6 |
| R-A2 | **no** | best `ijspeert` −0.0003 and best `rgb_vs_heb` −0.0126 at k ≤ 32; neither reaches ≥ 0 |
| R-A3 | **no** | `pca16` *does* improve GBM (4 → 5), so "no k improves either family" is false |
| R-B1 | **no** | gbm − ridge = **0.0299**, the bar is 0.15 |
| R-B2 | **yes** | unpredictable variance 0.063 < 0.20 and probe-weighted 0.6491 > unweighted 0.6366 |
| R-B3 | **no** | the unpredictable fraction is small, not large |
| R-C1 | **yes** | `mean+std+max` gains **+0.0392 ± 0.0111** over `features ⊕ mean` on transit, vs its untrained twin's +0.0126 |
| R-D1 | **no** | the 6-seed concat beats the best single seed on 2/11 |
| R-D2 | **no** | it sits below the single-seed mean on 5/11; "most" needs 6 |
| R-E1 | **no** | at the rule's own level n ≈ 755 only `eb` dies (1 of 3) |
| R-E2 | **yes** | fires as R-E1's complement, but with an exception — see §5 |
| R-F1 | **no** | 4/11 → 4/11, the count moves by 0 |
| R-F2 | **yes** | count stays in 3–5 and does not move |
| R-F3 | **yes** | and its answer is *uses-µ-without-gain*, not *ignores µ* |

**Three of the six forensics land between their own branches** (F-A: all three rules miss; F-D: both
rules miss; F-E: fires only at the level the rule does not name). Per P8 / P11-F / P12 precedent these
are recorded as mismatches below and not assigned to the nearest branch.

---

## 2. F-A — dilution is a **linear**-readout problem, and truncation is not the lever

`experiments/exp10_forensics/fa_pca/`, `fa_pca_untrained/`, `verdict/fa_*.csv`.
Arms: `pca{4,8,16,32,64}` and `whiten128` on the star-level mean-pooled µ, PCA fit **on train stars
only, per seed, per population**; the same six transforms on the untrained twin.

### GBM survival count, 11 reportable tasks

| transform | n survive | survivors |
|---|---|---|
| **pca16** | **5** | eb, flare, ijspeert, numax_hon, transit |
| `mean` (incumbent) | 4 | eb, rgb_vs_heb, rotation_period, transit |
| pca4 | 4 | eb, numax_hon, pulsating, transit |
| pca32 | 3 | eb, ijspeert, transit |
| pca8 | 3 | eb, numax_hon, transit |
| whiten128 | 3 | eb, ijspeert, transit |
| pca64 | 1 | eb |

R-A1 needs ≥ 6 with transit retained and ≥ 1 recovery among {osc_giant, ijspeert, rgb_vs_heb}. `pca16`
satisfies **two of the three clauses** (transit retained; ijspeert recovers, −0.0085 → **+0.0298**) and
misses the count by one. **R-A1 does not fire.**

**The +1 is soft, and this matters more than the count.** `pca16`'s two new survivors are
`flare` **+0.0037 ± 0.0024** and `numax_hon` **+0.0011 ± 0.0006** (on an R² of 0.92) — statistically
significant, practically nothing — while it *loses* `rgb_vs_heb` (+0.0142 → −0.0091) and
`rotation_period` (+0.0125 → −0.0022), both of which were real. The single substantive move in the
whole table is ijspeert.

### The untrained twin says what PCA is actually doing

| transform | untrained n survive | untrained mean delta over 11 tasks |
|---|---|---|
| pca4 | 0 | **+0.0066** |
| pca8 | 0 | +0.0036 |
| pca16 | 0 | +0.0003 |
| pca32 | 0 | −0.0053 |
| pca64 | 0 | −0.0075 |
| `mean` | 0 | **−0.0125** |
| whiten128 | 0 | −0.0144 |

Truncation lifts the **untrained** arm monotonically as k falls, by about the same amount it lifts the
trained one, and never produces a single survivor. What PCA removes is therefore generic
extra-column harm, not a µ-specific pathology — which is the dilution hypothesis confirmed as a
*mechanism* and refuted as a *lever*.

### R-A2: the linear losses shrink but never cross zero

| transform | ijspeert (linear Δ) | rgb_vs_heb (linear Δ) |
|---|---|---|
| `mean` | −0.0543 | −0.0505 |
| pca32 | −0.0352 | −0.0193 |
| pca16 | −0.0432 | −0.0173 |
| pca8 | −0.0296 | −0.0186 |
| **pca4** | **−0.0003** | −0.0126 |
| pca64 | −0.0582 | −0.0405 |
| whiten128 | −0.0617 | −0.0814 |

Monotone improvement down to k = 4, a ~4× reduction on both probes, and **neither reaches ≥ 0**, so
R-A2 does not fire on the letter. It is confirmed in direction and refuted in magnitude. The linear
survival count is 7/11 at every k that helps and never rises above the incumbent's 7 (which also
reproduces F1's published "7 of 11 clear 2·SE").

> **Mismatch recorded.** R-A1, R-A2 and R-A3 all fail to fire. The domain the rules were written for
> has a gap: "an improvement occurred, but below R-A1's bar and consisting of negligible-magnitude
> rows". The reading the evidence supports is R-A3's *intent* (dilution is not the lever) reached
> through a route R-A3's *condition* excludes. Note also that R-A1 carried the prediction "trained
> bottleneck ≥ PCA"; nothing here tests that prediction, so this table does not license promoting
> z32/z16 and does not license predicting that a trained bottleneck would beat PCA either.

---

## 3. F-B — µ is an almost **linear** recoding of the features, and one dim carries almost all of it

`experiments/exp10_forensics/fb_predictability/`. Per-dim regressions µ_d ~ 25 features on train stars,
scored on test; ridge with a per-target CV alpha, GBM under 6 random_states, over 6 encoder seeds and
both populations. **Population control exact on all 11 tasks** (`fb_population_control.csv`): every
rebuilt keep-mask reproduces the n_test F1 recorded.

| aggregate | ridge | gbm | gbm − ridge |
|---|---|---|---|
| µ-variance-weighted, pool | 0.9104 ± 0.0058 | 0.9384 ± 0.0025 | +0.0280 |
| µ-variance-weighted, subset | 0.9036 ± 0.0045 | 0.9354 ± 0.0021 | +0.0318 |
| unweighted per-dim mean | 0.5110 | 0.6366 | +0.126 |
| probe-coefficient-weighted (mean over 11 tasks) | 0.5290 | 0.6491 | +0.120 |

**R-B1 does not fire** (0.030 ≪ 0.15). The map from features to µ is *not* substantially nonlinear, so
the handoff's pre-declaration that a linear decorrelation penalty would be insufficient is **not**
triggered — on this evidence a linear penalty is the appropriate instrument, not a discredited one.

**R-B2 fires on the letter** (unpredictable variance 1 − 0.937 = **0.063** < 0.20; probe-weighted 0.6491
> unweighted 0.6366).

> **Mismatch recorded — the number that fires the rule is one latent dimension.** On seed 0, **dim 51
> alone holds 84 % (subset) / 86 % (pool) of µ's total test variance and is 99 % predictable** (ridge R²
> 0.989, GBM 0.998). The top five dims hold 89–90 %. The µ-variance-weighted aggregate is therefore
> close to a measurement of that single amplitude-like dim, not of µ. The distribution behind it:
> median per-dim GBM R² **0.676**, 5th percentile 0.24–0.29, and the top-variance quartile (93 % of all
> variance) averages only 0.69. R-B2's trigger is satisfied; its stated conclusion — "µ's used content
> is mostly redundant" — is only about **65 %** true, because the probe-coefficient-weighted aggregate
> (0.649) sits next to the *unweighted* mean, not the variance-weighted one. The probes spread their
> weight onto dims that are not the easily-predicted high-variance ones, leaving ~35 % of the content
> the probes actually use unrecoverable by a GBM on the 25 features.

R-B3 does not fire (the unpredictable fraction is small, not large), though the quantity it points at is
real: `mu_perp_full` over its untrained control averages **+0.112** across the 11 tasks.

---

## 4. F-C — the localized channel is real on `transit`, and the MIL operator is not the way to reach it

`experiments/exp10_forensics/fc_windowstats/`, `fc_windowstats_untrained/`, `fc_mil/`,
`verdict/fc_*.csv`. Gains are differenced **within an encoder seed** against the `features ⊕ mean`
control, so the error bar is a paired one.

### GBM, the two localized tasks

| task | variant | Δ vs features | gain over `mean` | paired 2·SE | untrained twin's gain |
|---|---|---|---|---|---|
| transit | mean+std | +0.0620 | **+0.0207** | 0.0043 | +0.0053 |
| transit | mean+std+q10+q90 | +0.0739 | **+0.0326** | 0.0151 | +0.0090 |
| **transit** | **mean+std+max** | **+0.0805** | **+0.0392** | 0.0111 | +0.0126 |
| eb | mean+std | +0.0181 | +0.0017 | 0.0054 | −0.0119 |
| eb | mean+std+max | +0.0178 | +0.0014 | 0.0016 | −0.0076 |
| transit | `window_score` (MIL) | +0.0308 | **−0.0105** | 0.0169 | +0.0152 |
| eb | `window_score` (MIL) | +0.0086 | **−0.0078** | 0.0046 | +0.0107 |

**R-C1 fires**, on transit, for all three window statistics: each clears its own paired 2·SE and each
exceeds its untrained twin's gain by roughly 3×. `eb` gains nothing. The incumbent transit delta of
+0.041 becomes **+0.081** from a pooling change alone, with no training — the largest single number this
forensics wave produced.

**The MIL half fails and reproduces the ADR-0008-lite warning exactly.** `features ⊕ window_score` on
transit is +0.0308 under GBM while its **untrained twin is +0.0290**: trained minus untrained is
+0.0018. The one-column MIL score is also below the 128-column mean pooling (−0.0105 against it). The
operator, not the representation, is doing the work — which is why the untrained twin was mandatory.

*Deviation, stated not buried:* the MIL score needed a value for train stars too, and the published
operator's train scores are in-sample. The train column is built **out-of-fold** (5 folds over stars, so
a star's windows never straddle a fold); the test column is the published operator verbatim, and the
`ws_only` arm reported beside it is the published MIL number. `random_state=0` stays frozen throughout.

### What the richer pooling does NOT do

| pooling | GBM survivors (of 11) | linear survivors (of 11) |
|---|---|---|
| `mean` | 4 | 7 |
| mean+std | 4 | 7 |
| mean+std+q10+q90 | 4 | 6 |
| mean+std+max | 3 | 7 |

It deepens transit and buys `rgb_vs_heb` +0.012 to +0.016, but **the survival count does not move**. The
gain is depth, not breadth. And under ADR-0012 decision 3 a pooling change is a **readout** change: an
external baseline that can sit beside the linear headline and can never enter the spine sentence.

---

## 5. F-D — column count hurts the linear readout hard and the GBM barely at all

`experiments/exp10_forensics/fd_ensemble/` (gbm, 24 arms) and `fd_ensemble_linear/` (linear, 4 arms).
The four variants have **no encoder-seed spread by construction**: the 6 seed-named arms of each variant
are hardlinks of one file, present only so the GBM picks up random_state 0–5 paired against the
engineered arm. Under `linear` they are one deterministic fit and carry no error bar at all.

### GBM

| task | ens_concat (768) | ens_pca128 | ens_pca64 | ens_seedmean | incumbent (single seed) |
|---|---|---|---|---|---|
| transit | **+0.0561** | +0.0435 | +0.0277 | +0.0509 | +0.0413 |
| eb | +0.0233 | +0.0275 | **+0.0324** | +0.0224 | +0.0164 |
| rgb_vs_heb | **+0.0374** | +0.0232 | −0.0017 | +0.0105 | +0.0142 |
| rotation_period | +0.0312 | −0.0081 | +0.0113 | **+0.0396** | +0.0125 |
| ijspeert | −0.0226 | +0.0227 | **+0.0590** | −0.0079 | −0.0085 |
| rotation | −0.0384 | −0.0345 | −0.0365 | −0.0210 | −0.0294 |
| solar_like_osc | −0.0189 | −0.0273 | −0.0243 | −0.0160 | −0.0190 |

### Linear — the same concat collapses

| task | ens_concat (768) | ens_pca64 | ens_seedmean | incumbent (128) |
|---|---|---|---|---|
| ijspeert | **−0.1123** | −0.0609 | −0.0552 | −0.0543 |
| rgb_vs_heb | **−0.1001** | −0.0376 | −0.0570 | −0.0505 |
| rotation | **−0.0788** | +0.0405 | +0.0132 | +0.0155 |
| eb | −0.0078 | +0.0422 | +0.0431 | +0.0330 |
| transit | +0.0006 | +0.0309 | +0.0226 | +0.0274 |

> **Mismatch recorded.** Neither R-D1 nor R-D2 fires: concat beats the best single seed on **2/11** and
> falls below the single-seed **mean** on **5/11**, so neither "most" threshold (6) is met. The two
> rules also do not exhaust the space — the honest reading is *task-dependent*: the seeds carry
> complementary content on transit, eb and rgb_vs_heb and carry noise on rotation and ijspeert. The
> linear table is the unambiguous half: 768 columns roughly **double** the two small-probe losses and
> flip `rotation` from +0.016 to −0.079, and PCA-64/128 of the same 768 columns undoes it and lands
> *above* the incumbent on eb (+0.042 vs +0.033) and rotation (+0.041 vs +0.016).

---

## 6. F-E — small n is partly causal, and the rule's own two levels disagree

`experiments/exp10_forensics/fe_subsample/`. Train stars subsampled stratified by label, test untouched,
20 draws per level, 6 encoder seeds, GBM random_state paired between the two arms within a draw.

| task | family | full n | n ≈ 755 | n ≈ 160 |
|---|---|---|---|---|
| eb | linear | **+0.0329** | −0.0287 | −0.0184 |
| pulsating | linear | **+0.0571** | +0.0121 | −0.0209 |
| osc_giant | linear | −0.0028 | −0.0192 | −0.0410 |
| eb | gbm | +0.0164 | +0.0114 | −0.0149 |
| pulsating | gbm | −0.0050 | +0.0100 | +0.0244 |
| osc_giant | gbm | −0.0045 | −0.0002 | −0.0017 |

**R-E1 does not fire at n ≈ 755** (only `eb` goes from positive to ≤ 0, 1 of 3), so **R-E2 fires** as its
complement.

> **Mismatch recorded, twice.**
> 1. **The same rule fires at the other level.** At n ≈ 160, `eb` and `pulsating` both go negative under
>    `linear` — 2 of 3, R-E1's own threshold. The rule's answer depends on which of its two subsampled
>    levels is read, and it names only one.
> 2. **The clearest n-effect in the table is invisible to the rule's phrasing.** `osc_giant`'s linear
>    loss deepens monotonically −0.0028 → −0.0192 → −0.0410 with **0 of 20 draws positive** at both
>    small levels. "Positive at full n, ≤ 0 at n ≈ 755" cannot see it, because osc_giant was never
>    positive at full n — and osc_giant is one of the three tasks the losses were to be explained on.

**A separate estimator finding that outlives this forensic.** At the subsampled levels the **draw**
spread is 0.007–0.070 while the **encoder-seed** spread is 0.002–0.014 — resampling noise beats encoder
noise by 5–15×. Any small-n gate whose error bar comes from 6 encoder seeds alone is under-dispersed by
roughly an order of magnitude on exactly the rows that are hardest to call.

---

## 7. F-F — C3b is robust to fair tuning, and the GBM uses µ heavily without gaining from it

`experiments/exp10_forensics/ff_gbm_fairness/`. Grid `max_iter {200, 800} × max_depth {3, 6} ×
max_features {0.3, 1.0}` = 8 configs, applied with an identical budget to **both** arms, selected on a
25 % validation split carved from train (stratified, keyed on the encoder seed so both arms are selected
on the same rows), test scored once at the selected config.

| task | metric | default Δ | tuned Δ | tuned 2·SE | survives default → tuned | µ split-gain share |
|---|---|---|---|---|---|---|
| transit | pr_auc | +0.0413 | +0.0333 | 0.0092 | yes → yes | 0.707 |
| eb | pr_auc | +0.0164 | +0.0166 | 0.0063 | yes → yes | 0.371 |
| rotation_period | r2 | +0.0125 | +0.0309 | 0.0216 | yes → yes | 0.226 |
| rgb_vs_heb | roc_auc | +0.0142 | +0.0133 | 0.0166 | yes → **no** | 0.648 |
| numax_hon | r2 | +0.0001 | +0.0015 | 0.0009 | no → **yes** | 0.071 |
| rotation | pr_auc | −0.0294 | −0.0206 | 0.0058 | no → no | 0.463 |
| solar_like_osc | pr_auc | −0.0190 | −0.0124 | 0.0083 | no → no | 0.378 |
| flare | pr_auc | −0.0005 | −0.0165 | 0.0109 | no → no | 0.362 |
| ijspeert | pr_auc | −0.0085 | −0.0150 | 0.0214 | no → no | 0.430 |
| osc_giant | pr_auc | −0.0053 | −0.0031 | 0.0028 | no → no | 0.149 |
| pulsating | pr_auc | −0.0050 | −0.0132 | 0.0139 | no → no | **0.909** |

**4/11 → 4/11. R-F1 does not fire; R-F2 fires.** C3b's verdict is not a tuning artifact, and exp10's
gate stays on the published baseline. Worth naming anyway: the *membership* swaps, and the row swapped
in is `numax_hon` at **+0.0015 on an R² of 0.92** — another 2·SE pass with no effect size behind it, and
the row swapped out (`rgb_vs_heb`) was one of C3b's four.

**R-F3 fires, and its answer is the opposite of the redundancy story.** µ is 128 of 153 columns
(83.7 %). On the tasks with a flat delta the GBM still spends **36 % (flare)**, **38 %
(solar_like_osc)** and **91 % (pulsating)** of its total split gain on µ columns. Not one flat-delta
task has a µ gain share below 5 %. So the GBM is **using µ and gaining nothing from it**, not ignoring
it as redundant — µ's columns are informative enough to be split on and not complementary enough to
improve the held-out score. That is a different problem from redundancy and a different objective would
be needed to fix it.

---

## 8. Method notes and traps discovered here

- **`arm_parts` hardcodes the family for any arm named `untrained*`.**
  `analyze_exp08_menu_channel.arm_parts` returns `("untrained", seed)` for every name starting with
  that word, so six derived arms named `untrained_pca4_s0 … untrained_whiten128_s0` all collapsed into
  **one** family at seed 0 and silently overwrote each other in F1's summary — the surviving row was
  whichever arm was scored last. Caught by an all-NaN untrained column in F-C's gain table, not by any
  assertion. Fixed by naming the twin `untr_<transform>_s0`; the affected untrained arms were re-scored
  into `fa_pca_untrained/` and `fc_windowstats_untrained/`. **Trained rows were never affected.** Any
  future work that adds more than one untrained variant will hit this.
- **Derived caches carry one window row per star.** That makes them exact at readout `mean` and
  meaningless at `mean_std` (the std over one row is zero), so every F1 invocation against them passes
  `--readouts mean`. The builder refuses to write into `exp08_menu_channel/` and refuses to overwrite an
  existing cache.
- **Arm lists written by `Path.write_text` on Windows get CRLF**, and a `\r` inside a shell-expanded arm
  name fails as an unreadable family/seed rather than as a bad file. The builder now passes
  `newline="\n"`.
- **Nothing under `f1_fusion_scorecard/`, `f1_nonlinear_control/`, `c3_feature_controls/` or
  `exp09_bump_ablation.csv` was written to.** All output is under `experiments/exp10_forensics/`.
- **`src/swm/tests`: 153 passed**, the session baseline. No shipped module was modified.

## 9. Cost, measured

| step | jobs / cells | wall |
|---|---|---|
| derived cache builds (identity, pca, windowstats, ensemble) | 94 arms × 2 populations | 78 s total |
| footing F1 + gates | 14 jobs | 4.4 min |
| F-A | 84 jobs (+12 untrained re-score) | 71 min (+4) |
| F-B | 12 population×seed cells, 4 workers | 50 min |
| F-C window statistics | 42 jobs (+6 untrained re-score) | 77 min (+2) |
| F-C MIL | 14 arm×task cells, 4 workers | 2.7 min |
| F-D | 24 gbm jobs + 4 linear jobs | 49 min |
| F-E | 1,656 cells, 5 workers | 17 min |
| F-F | 132 cells (+12 smoke), 5 workers | 18 min (+3) |
| **total** | | **≈ 4 h 45 min**, 1.4 GB |

All timings are under 2–5 concurrent jobs on 16 cores; a serial run would differ.

## 10. Reproduce

```bash
# swm env, PYTHONPATH=src, repo root, CPU only
python experiments/exp10_build_derived_mu.py --family identity      # then pca / windowstats / ensemble
python experiments/analyze_f1_fusion_scorecard.py --arms $(cat experiments/exp10_forensics/identity/arms.txt) \
    --cache-dir experiments/exp10_forensics/identity/mu_cache \
    --subset-cache-dir experiments/exp10_forensics/identity/subset_mu_cache \
    --readouts mean --families linear gbm --out-dir experiments/exp10_forensics/footing
python experiments/exp10_footing_checks.py                          # must pass before anything is read
python experiments/analyze_exp10_fb_mu_predictability.py --jobs 12
python experiments/analyze_exp10_fc_mil.py --jobs 6
python experiments/analyze_exp10_fe_subsample.py --jobs 12
python experiments/analyze_exp10_ff_gbm_fairness.py --jobs 8
python experiments/exp10_forensics_verdict.py                       # scores all 14 rules
```
