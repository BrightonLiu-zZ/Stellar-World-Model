# C1 + C2 — the two supervised arms of the ML4PS scorecard (2026-08-28/29)

`swm.train.supervised` (66 runs) → `experiments/c1c2_supervised/c1c2_{probe,absolute,summary}.csv`
(66 / 22 / 66 rows), scored by `experiments/analyze_c1c2_supervised.py`. Manifest:
`experiments/configs/c1c2_supervised_baselines.yaml`. Roadmap rows **C1** (D20, provenance Yue Ma
**W6** merged with the Y13c supervised ceiling) and **C2** (**Y13b**), arms 4 and 6 of the D18
scorecard.

*(`docs/` and `*.csv` are gitignored in this repo, so this file is the committed record of the verdict.)*

Both arms are **external baselines** under ADR-0012 decision 3. They never touch the frozen
linear-probe protocol and are never reported as the probe.

---

## What did not exist before

`swm/eval/new_task_ceiling.py:12` says it in as many words:

> The B1 supervised ceiling is deliberately absent (it needs GPU training runs); so "beats engineered
> features" is claimable from this module and "closes the supervised gap" is not.

C1 is that hole filled. Until now the project could bound its representation against engineered
features (Ceiling A) but had **no** measurement of how much signal the data and labels contain at all.

## The design, and the one decision that mattered most

Full argument for every choice is in the manifest (D-C1.1 … D-C1.10). The load-bearing ones:

| | choice | why the alternative was rejected |
|---|---|---|
| code home | `swm.train.supervised`, consuming the **probe's own loaders** | split identity with F1 becomes a property of the code, not a claim to re-verify. `swm.train` cannot reach the pool population at all, and its 133-test suite stays untouched |
| architecture | `Encoder`'s conv stack **verbatim** + `fc_mu` (4096→128) + dropout + `Linear(128,1)` | a scalar head factors through 128 dims, so the bottleneck costs **no ceiling** while keeping a state_dict that mirrors the SSL encoder |
| **loss level** | **star bag**: mean of window logits *in-graph*, one class-weighted loss per star | label broadcast forces every window of a positive star to be called positive. On the localized tasks that is manufactured label noise, and it would have depressed the ceiling exactly where C3b left the fusion claim alive |
| bag scope | first segment (16–20 windows), the windows µ is pooled over | see **Limitation 1** |
| selection | best **val** metric, patience 10, cap 60; test scored once | both populations carry a `val` split no probe has ever touched. Selecting on val *loss* would have understated the ceiling at 6–12% prevalence |
| regularisation | one pre-registered setting for all 11 tasks (AdamW wd 1e-4, dropout 0.2) | per-task tuning stops being one protocol and has to be disclosed as 11 budgets |

C1 and C2 differ in the **trunk and nothing else** — same bottleneck, head, pooling, optimiser,
selection rule. That is what lets `C1 − C2` read as the convolutional inductive bias.

## Footing — the populations are F1's, checked rather than asserted

| control | result |
|---|---|
| v1 star set + per-star bag sizes vs `exp08_menu_channel/subset_mu_cache` | **exact**, train and test |
| pool star set + bag sizes vs `exp08_menu_channel/mu_cache` | **exact** (16,002 / 3,429 / 3,429; **zero** stars dropped) |
| all 11 keep masks vs F1's published `n_test` / `n_test_pos` | **exact on all 11** (eb 2021/196 · transit 2021/122 · ijspeert 2021/93 · rgb_vs_heb 161/113 · flare 3429/304 · numax_hon 1313 · rotation_period 150) |

All three are permanent tests (`src/swm/tests/test_supervised_baseline.py`, 20 tests; full suite
**153 passed**, was 133).

## The pilot gate, pre-registered before the wave ran

Seed-0 wave, all 11 tasks, `conv_supervised`. **PASS on all four automated clauses**: every task
strictly above its metric-native floor · `eb` 0.778 ≥ 0.742 (F1's features-linear) · `numax_hon`
R² 0.843 ≥ 0 · no non-`small_n` run selecting at epoch 0 or the cap.

## Results — 3 seeds, readout `mean`, all arms named (rule 5)

`features` / `µ` / `fusion` are F1's linear arms on `hann0p3_fbwd`; C1/C2 are end-to-end supervised.

| task | floor | features | µ | fusion | **C1 conv** | **C2 mlp** | fusion − C1 |
|---|---|---|---|---|---|---|---|
| eb | 0.097 | 0.742 | 0.771 | 0.775 | 0.774 ±0.005 | 0.720 ±0.011 | 0.000 |
| pulsating | 0.107 | 0.789 | 0.806 | **0.846** | 0.816 ±0.023 | 0.756 ±0.010 | +0.030 |
| rotation | 0.089 | 0.540 | 0.559 | 0.555 | 0.571 ±0.006 | 0.535 ±0.006 | −0.015 |
| transit | 0.060 | 0.190 | 0.144 | **0.218** | 0.133 ±0.023 | 0.108 ±0.007 | +0.085 |
| ijspeert | 0.046 | 0.508 | 0.440 | 0.454 | 0.457 ±0.003 | 0.456 ±0.005 | −0.002 |
| rotation_period (R²) | 0.0 | 0.703 | 0.677 | **0.717** | 0.625 ±0.022 | 0.700 ±0.017 | +0.092 |
| osc_giant | 0.383 | 0.920 | 0.854 | 0.918 | 0.873 ±0.007 | 0.885 ±0.003 | +0.045 |
| solar_like_osc | 0.122 | 0.320 | 0.336 | 0.390 | 0.395 ±0.003 | 0.278 ±0.008 | −0.005 |
| flare | 0.089 | 0.474 | 0.453 | 0.524 | 0.528 ±0.007 | 0.450 ±0.012 | −0.004 |
| numax_hon (R²) | 0.0 | 0.831 | 0.802 | **0.867** | 0.841 ±0.003 | 0.834 ±0.002 | +0.026 |
| rgb_vs_heb (ROC) | 0.5 | 0.758 | 0.660 | 0.708 | 0.522 ±0.003 | **0.761** ±0.006 | +0.185 |

Detection metric is PR-AUC; every delta's prevalence rides on the CSV rows (R8-F1). Error bars are
2·SE over 3 seeds; seed sd is **0.002–0.023**, so the arms are well separated relative to their noise.

### Verdict 1 — the supervised arm never beats the fusion probe

Counting at 2·SE, **unpaired** (a supervised seed is an init/shuffle seed; an encoder seed is a
pretraining seed — pairing them by index would manufacture a correlation that does not exist):

| against | F1 arm ahead | C1 ahead | tied |
|---|---|---|---|
| `features` linear | 5 / 11 | 6 / 11 | 0 |
| `µ` linear | 2 / 11 | **4 / 11** | 5 |
| **`features ⊕ µ` linear** | **6 / 11** | **0 / 11** | 5 |

C1 clears the engineered-feature baseline on 6 of 11 and the µ-only probe on 4, so it is not a
strawman — but it does not exceed the fusion readout on a single task.

### Verdict 2 — what that does and does not license

**Does:** on 5 of 11 tasks a supervised Conv1D of the same architecture, trained end-to-end on the
same labels and the same input, reaches the frozen fusion readout and no further; on the other 6 it
falls short of it. `recovery fraction` (probe − floor)/(C1 − floor) sits at **0.97–1.09** on eight
tasks — the labelled ceiling and the label-free readout are in the same place.

**Does not:** this is *not* "SSL matches supervision". The supervised arm has 669–16,002 labelled
stars, so the claim is **"SSL matches what this architecture reaches on the labels that exist"** —
which is the W6 semi-supervised argument, and must be worded that way in the paper. A supervised
model with an order of magnitude more labels is not bounded by this measurement.

### Verdict 3 — the convolution earns its keep, except where n is small

`C1 − C2`, the controlled architecture contrast: conv ahead on **8 of 11**, dense ahead on 3.

| dense wins | C1 conv | C2 mlp | n_train |
|---|---|---|---|
| rgb_vs_heb | 0.522 | **0.761** | 755 |
| rotation_period (R²) | 0.625 | **0.700** | 669 |
| osc_giant | 0.873 | **0.885** | 16,002 |

The two reversals with a margin are exactly the two smallest training sets, and the dense trunk is
**132 k parameters against the conv trunk's 1.1 M** — a capacity story, not an architecture-superiority
one. `osc_giant` flips by +0.012 at 16,002 stars and is the one that does not fit that reading.

## Honest reporting — everything flagged, nothing dropped

1. **Limitation, input scope (manifest D-C1.9).** Both arms see each star's **first segment only**
   (16–20 windows, ~5.7 d), because that is the probe's bag scope. Equal access is what makes
   `C1 − probe` isolate *supervision* rather than *supervision plus four times the data per star*,
   and an all-segment arm would re-open the bag-size confound the K-matched control exists to price.
   **Consequence, stated rather than argued if challenged: this bounds the labelled ceiling at the
   probe's input scope, not in general.** No all-segment arm is run; it is a v2/journal cell.
2. **One flagged run, reported not re-run.** `conv_supervised / pulsating / seed 2` selected at
   **epoch 0** and is flagged `selected_first_epoch`. Its score (0.794) sits below its siblings
   (0.832, 0.823) and is the reason `pulsating` carries the widest conv error bar (±0.023). It was
   **not** dropped or re-run: removing a seed selected post-hoc on an outcome-correlated criterion is
   the estimator swap this project's VOID rule forbids. `conv/pulsating` and `conv/transit` both
   select very early across all seeds (median epoch 2 and 3) — their val metric is near-flat from the
   first epoch.
3. **Five of eleven tasks overfit after their selected epoch**, so W13's "no overfit" holds **at the
   reported checkpoint** and early stopping is doing real work rather than decorating:
   `rotation_period` val R² 0.581 → −0.147, `transit` −42%, `numax_hon` −40%, `ijspeert` −23%,
   `solar_like_osc` −17%; the other six are flat within 5%. That the labelled sets are small enough
   to overfit in 10–20 epochs is itself evidence for the semi-supervised framing.
4. **`small_n` cells are reported, never used to support a claim.** `rotation_period` (669 train
   stars) and `rgb_vs_heb` (755) train ~1.1 M parameters on very little; they were flagged in the
   manifest *before* running and were exempt from the pilot gate.
5. **`rgb_vs_heb`'s recovery fraction is 9.26× and must not be quoted.** C1 stays the designated
   Ceiling B — swapping the denominator to whichever arm won, after seeing which arm won, is the
   post-hoc swap the VOID rule forbids — so the row carries two machine-generated caveats instead:
   `mlp_raw` reaches 0.761 on this task so the ratio understates the ceiling, and the denominator
   span (0.022) is under a quarter of the probe's own span, making the ratio numerically unstable.
6. **`flare` is run and is not printable.** L1's reporting rule (B) leaves it blocked on the manual
   visual pass. That governs the write-up, not whether the cell exists — a hole in the 11×7 figure is
   worse than a row with a stated status.
7. **`ijspeert_excl_villanova`, `numax_hatt`, `dnu_hatt`, `prot_kounkel` are not run** (ADR-0010: one
   probe per physical quantity). F1 scores them; they are not scorecard rows.

## Cost, measured

**45.0 min of GPU for all 66 runs** (mean 0.68 min/run) on the RTX 4060, plus ~13 min of one-off CPU
to cache the first-segment blocks for both populations. The handoff's "~2 GPU nights" was out by two
orders of magnitude and the manifest's own pre-pilot estimate by 4×; the cause of both is that early
stopping fires at epochs 15–31, not at the 60-epoch cap. Quote `mean_minutes` from
`c1c2_absolute.csv`, never a planning estimate.

## Feeds

**R5'** (the 11×7 figure) gets arms 4 and 6, with absolutes, unpaired 2·SE deltas against all three
F1 linear arms, and the recovery fraction. **R4'** must carry Verdict 2's wording: the sentence is
"SSL matches what this architecture reaches on the labels that exist", never "matches supervision".
