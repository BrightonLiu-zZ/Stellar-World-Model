# C3 — GBM and MLP on the 25 engineered features, scorecard arm 7 (2026-08-25)

`experiments/analyze_c3_feature_controls.py` → `experiments/c3_feature_controls/c3_{probe,summary,repro}.csv`
(195 / 45 rows). Roadmap row **C3**, provenance Yue Ma **Y13a**, arm 7 of the D18 scorecard.

*(`docs/` and `*.csv` are gitignored in this repo, so this file is the committed record.)*

## Why this arm exists, and the risk it was pre-registered to carry

F1's headline is `features ⊕ µ` beating `features`, **both under a linear readout**. That comparison is
only interesting if the linear readout is not itself the bottleneck. The risk was written down in the
08-15 analysis file before anything ran: *if a nonlinear model on the same 25 features recovers what µ
was adding, the claim narrows from "SSL adds information the engineered features do not carry" to "SSL
adds to a **linear** readout on engineered features" — report it either way, don't soften it.*

**The risk fired.** See the verdict below.

## What already existed, and what this adds

`exp08_prechecks/ceiling_A1A2.csv` holds A1 (linear) and A2 (GBM) on the engineered features — but for
the **menu block only**, and at a **single fit**. C3 adds the four v1 tasks under both nonlinear
families, the menu block under MLP, and a **seed axis** on both.

**Why a seed axis at all, when the features are deterministic.** The µ arms carry a 6-seed spread from
encoder training. A single nonlinear fit has no error bar, and comparing a point estimate against a
6-seed mean is how a difference gets called significant that is not. So each nonlinear family is fit
under 6 `random_state`s. This is **estimator noise, not representation noise** — the CSV labels it as
such in a `spread_kind` column rather than letting a reader assume parity. `linear` is deterministic
and is fit **once**, carrying seed −1, so the summary cannot report a fake zero-width error bar as if
it had been measured.

Threading `random_state` through `fit_readout_scores` / `score_regression` and the five menu scorers
was done with **default 0**, the value previously hardcoded, so every published number riding those
functions is bit-identical. `window_score` (MIL) deliberately keeps its frozen `random_state=0`: those
numbers are published and must not gain a seed axis silently.

## Control — exact

| control | rows | max abs diff |
|---|---|---|
| `gbm` seed 0, menu block, vs `ceiling_A1A2.csv` **A2_feats** | 11 | **0.0** |
| `linear`, menu block, vs `ceiling_A1A2.csv` **A1_feats** | 11 | **0.0** |

C3's GBM seed 0 *is* the A2 fit, so exactness is the correct expectation and anything else would have
meant the scorers had drifted.

## Results — scores on the 25 engineered features alone

| block | task | linear | **gbm** | mlp |
|---|---|---|---|---|
| v1 | pulsating | 0.789 | 0.858 | **0.862** |
| v1 | eb | 0.742 | **0.795** | 0.766 |
| v1 | rotation | 0.540 | **0.614** | 0.567 |
| v1 | transit | 0.190 | 0.189 | 0.186 |
| menu | osc_giant | 0.920 | 0.924 | **0.927** |
| menu | solar_like_osc | 0.320 | **0.477** | 0.477 |
| menu | flare | 0.474 | **0.567** | 0.563 |
| menu | rgb_vs_heb (ROC) | **0.758** | 0.730 | 0.719 |
| menu | numax_hon (R²) | 0.831 | **0.921** | 0.846 |
| menu | rotation_period (R²) | 0.703 | **0.731** | 0.706 |
| menu | ijspeert | **0.508** | 0.498 | 0.397 |

GBM beats the linear baseline on **8 of 11**, ties `transit`, and loses on the two smallest probes
(`rgb_vs_heb` 161 stars, `ijspeert` 93 positives) — the same size-driven pattern F1 sees for fusion.

**Seed spread, measured rather than assumed:** `gbm` mean sd **0.0014** (max 0.0067) — near-deterministic
here, so its error bars are genuinely tiny and comparisons against it rest on the *other* arm's spread.
`mlp` mean sd **0.0182** (max 0.0449) — a real error bar, and wide enough that several MLP-vs-GBM
orderings in the table above are not separable.

## Verdict — the pre-registered risk fired, and it is reported as written

Against F1's **linear** fusion arm at readout `mean`, GBM on the engineered features alone wins on
**10 of 11 tasks**:

| task | features+linear | features ⊕ µ, linear | **features + GBM** | GBM − fusion |
|---|---|---|---|---|
| solar_like_osc | 0.320 | 0.390 | **0.477** | +0.087 |
| rotation (v1) | 0.540 | 0.555 | **0.614** | +0.058 |
| numax_hon (R²) | 0.831 | 0.867 | **0.921** | +0.055 |
| ijspeert | 0.508 | 0.454 | **0.498** | +0.044 |
| flare | 0.474 | 0.524 | **0.567** | +0.042 |
| rgb_vs_heb (ROC) | 0.758 | 0.708 | **0.730** | +0.022 |
| eb (v1) | 0.742 | 0.775 | **0.795** | +0.020 |
| rotation_period (R²) | 0.703 | 0.717 | **0.731** | +0.014 |
| pulsating (v1) | 0.789 | 0.846 | **0.858** | +0.013 |
| osc_giant | 0.920 | 0.918 | **0.924** | +0.007 |
| **transit (v1)** | 0.190 | **0.218** | 0.189 | **−0.029** |

The single exception is `transit` — the localized task, and the one where MIL/pooling results have
always behaved differently from the global tasks.

**What this does and does not license.** It does **not** show that µ carries nothing: that comparison
changes the readout *and* the input at once, so it confounds readout capacity with information content.
It does show that **the linear-fusion-vs-linear-features framing is not sufficient on its own**, and
that the paper cannot claim "SSL adds information engineered features do not carry" on the strength of
F1's linear table alone.

The comparison that separates the two explanations — `(features ⊕ µ)` under the **same** nonlinear
readout — is not in D18's arm list. It was approved as a **control** on 2026-08-25, run as roadmap row
**C3b**, and is reported in full in `f1_fusion_scorecard_README.md` §C3b.

**Its answer, in one line:** under GBM the fusion delta survives on **4 of 11** tasks (transit +0.041,
eb +0.016, rgb_vs_heb +0.014, rotation_period +0.013) and under MLP on **0 of 11** — so most of what µ
added to the linear readout was compensating for that readout. The exception is `transit`, the
localized task, where µ's contribution **grows** under the stronger readout (+0.027 → +0.041). All four
survivors beat their untrained control, so what remains is not generic column addition.

**Consistency between these two scripts is exact.** C3 and F1 compute `features_only` under GBM and MLP
independently; after the seed-pairing fix in F1 they agree to **0.0** on all 15 rows of both families.
That check is what caught the pairing flaw in the first place — F1 was differencing 6 fusion seeds
against a single features fit, an offset of up to 0.0062 against margins of the same size.
