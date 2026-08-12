# Cross-experiment findings — method-level discoveries and problems

Findings that are **not about one experiment's hypothesis** but about how this project measures
things. They were each found while running a specific experiment, but they apply to every table
before and after it, which is why they are collected here rather than buried in one writeup.

This file records **what was discovered and what went wrong. It deliberately makes no
recommendations** about future experiments. The forward-looking counterpart — what is still unmeasured,
what would settle it, and what each answer would change — is [open_questions.md](open_questions.md).

Per-experiment verdicts stay in their own writeups: `expNN_*_README.md`,
[mil_pooling/README.md](mil_pooling/README.md), and the dated plans under `docs/plans/`.
Note that `docs/` is **gitignored**, so `docs/STATUS.md` and the plan docs are not part of the
tracked footprint; anything that must survive a fresh clone belongs under `experiments/`.

Last updated **2026-08-06** (exp08 pre-design checks: **F18 retracted**, F27 added. Previous: F24–F26,
exp07 curve forensics; F23, per-star edge measurement).

---

## F1 — The eval protocol scored ~26% of the available windows

`load_first_segment_blocks` (and `new_task_extract`) keep only each star's **first packed segment**.
Verified on disk: 96% of stored segments hold exactly 4 windows of 1024 (5 in the rest), so a bag is
**16–20 windows of 256, about 5.7 days**, while the packed corpus holds median **32** / mean **61.8** /
max **816** windows per star.

Every star-level table from exp00 through exp05 inherits this. Physically it means a planet with
period > 5.7 d can have **zero transits in its bag**.

*Where measured:* MIL/pooling sweep, 2026-07-26. *Detail:* [mil_pooling/README.md](mil_pooling/README.md).

## F2 — The bag-size confound is a detection artifact, not a property of larger bags

A `bagsize_only` control (logistic or ridge on log window-count alone, ignoring the light curve
entirely) scores at the base rate whenever bag size is fixed, and above it when bag size varies:

| scope | transit | eb | solar_like_osc | the four regression probes |
|---|---|---|---|---|
| first-segment (K≈16) | 1.03× base | 1.00× | 1.03× | R² ≈ 0 |
| K-matched (K=16) | **1.000×** | 1.000× | 1.000× | R² ≈ 0 |
| all-segment (K≈62) | **2.18× base** | 1.59× | **1.70×** | **R² −0.003 … −0.000** |

Detection labels leak: a star observed in more sectors has both a bigger bag and a higher chance of
being catalogued at all. Regression targets do not leak, because that population is already
restricted to catalogued stars and bag size says nothing about the *value* of ν_max. Same stars, same
bags, same scope; only the label type differs.

*Consequence recorded:* all-segment **detection** numbers carry a bag-size caveat; all-segment
**regression** numbers do not.

## F3 — What the K-matched control can and cannot separate

Subsampling every bag to K₀=16 drawn from across all segments is confound-free **by construction** —
every star has at least 16 windows, so at fixed K₀ the count carries exactly zero information, which
the control confirms to 6 decimals (ratio 1.000000).

Decomposition, consistent across both star pools: `kmatch − first ≈ 0` while `all − kmatch > 0`. So a
**longer baseline is not the mechanism**; having **more windows** is.

**Stated limitation.** For transit, the probability that a bag contains an in-transit window also
scales with the number of windows, so K-matching removes the genuine coverage benefit along with the
confound. It refutes temporal spread; it does **not** separate real coverage from the
observed-more-often selection effect. On the regression probes, where F2 shows the control is clean,
that ambiguity does not arise.

## F4 — The trained−untrained gap is not invariant to the readout or the pooling

The project's SSL win condition has been the trained-minus-untrained gap since the skyline suite.
Ranking *operators* by it systematically favours **whichever operator is worst for a random encoder**,
for any reason. Three independent mechanisms measured:

| mechanism | instance | gap says | absolute says |
|---|---|---|---|
| readout capacity | ABMIL on pulsating | **+0.288** (best) | 0.745 (worst of 3) |
| operator sharpness | `ws_max` on osc_giant | **+0.205** (best) | 0.805 vs 0.897 (worst) |
| feature-map expressiveness | `mean` vs `rff_meanmap` on νmax_hon | `mean` **+0.363** (best) | `mean` 0.793, `rff` **0.870** |

The third involves no capacity difference at all: both are fixed zero-parameter feature maps. The
driver is that the *baseline moves* — mean-pooled random μ is poor (0.430) while a random-Fourier map
is already good on random projections (0.762).

**Not a retraction of anything.** exp03, exp04 and exp05 all compared a *fixed* readout across
encoders, which is exactly the case where the gap is valid.

*Enforced in code:* `swm.eval.mil_report.winner_block` selects on absolute val score and excludes
learned heads; `src/swm/tests/test_mil_report.py` pins that behaviour.

## F5 — Per-task operator selection underperformed a single fixed operator

Across all 11 probes from both pools (first-segment, 4 seeds), gain over mean pooling:

| strategy | mean gain | median | probes improved |
|---|---|---|---|
| per-task, chosen on val | +0.0395 | +0.0402 | **8 / 11** |
| single fixed `mean_std` | +0.0353 | +0.0257 | **9 / 11** |

The flexible protocol improved *fewer* probes than the rigid one. On eb, pulsating and rotation, val
selected an operator that lost to plain mean pooling on test (pulsating `ws_max`: **−0.031**).
Validation splits of 120–420 positives cannot resolve differences of ~0.01.

*Where measured:* `src/notebooks/mil_pooling.ipynb`, cross-probe comparison cells.

## F6 — A val-declared winner flipped on a sub-noise difference

On transit, `moments`, `mean_std` and `mean_skew` score within **one standard deviation** of each
other on both val and test (val 0.2334 / 0.2413 / 0.2151; test 0.2308 / 0.2137 / 0.2353; seed SDs
0.012–0.025). Adding two of them to the candidate pool changed the reported winner on a **0.008 val
difference**. The headline operator label moved; the underlying finding (dispersion pooling beats mean
pooling on transit by +0.07 to +0.09) did not. Treat the three as interchangeable.

*Problem it caused:* the change was not flagged when it happened, which made a later reader
reasonably conclude a previous claim had been overturned.

## F7 — Two operators are provably redundant or degenerate under a frozen readout

- **Noisy-AND** (Kraus+2016) is a strictly increasing function of the bag mean, so under a frozen
  readout it produces the identical star ranking and therefore the identical PR-AUC as `ws_mean`. It
  differs only when trained jointly with the encoder. Now asserted in
  `test_pooling.py::test_noisy_and_is_rank_equivalent_to_mean` rather than run as a sweep cell.
- **`ws_topk`** at k=1 is exactly `ws_max`, and at k≥K exactly `ws_mean`. Both identities hold in the
  measured tables and are unit-tested.

## F8 — Max-like pooling degrades as bags grow

Transit, `ws_lse` at fixed temperature: β=10 scores **0.180** at K≈16 but **0.089** at K≈62; β=50
scores 0.182 → 0.148. Every dispersion operator improves over the same change. Reproduced on the
downstream regression probes (νmax_hatt −0.001, Δν_hatt −0.001, P_rot −0.016 under `max`).

This is the β_crit = O(log K) effect predicted by Maxsoft (NeurIPS 2025), observed directly on three
independent task families. A temperature tuned on short bags loses roughly half its performance when
applied to long ones.

## F9 — β\* measures witness rate only when the per-window readout can see the witness

The LSE temperature that maximises val score orders the v1 tasks exactly by how localized their
signal is (relative gain mean→max: transit +53.8%, eb +20.0%, rotation +6.6%, pulsating −1.0%).

It **failed on flare**, which is the most localized signal in the whole menu, returning −0.8%. Cause:
`ws_lse` aggregates the scores of a per-window *linear* classifier, and the 2026-07-22 Level-B result
had already shown trained μ makes flare windows *less* separable than untrained (0.053 vs 0.139)
because reconstruction smooths the spike away. With no detectable witness there is nothing for
max-like pooling to concentrate.

*Scope of the claim:* β\* is a witness-rate estimate conditional on the window readout being able to
detect the witness. True for transit (within-star window PR-AUC ≈ 0.50, Phase 1); false for flare.

## F10 — Learned pooling loses at this label budget

Gated ABMIL and DSMIL on frozen μ, run with the three small-data fixes the literature prescribes
(narrow attention width, ACMIL stochastic top-k instance masking, DTFD pseudo-bags), lose on all four
v1 tasks to zero-parameter operators: transit 0.123 / 0.197 vs **0.235**; eb 0.715 / 0.728 vs
**0.773**; pulsating 0.745 / 0.761 vs **0.807**; rotation 0.524 / 0.520 vs **0.569**. Test positives
per task: 122–216.

## F11 — Validation loss anti-selects across models; the selection metric is only valid within a run

Across the ten exp05 comb cells (same data, same loss functions, λ the only axis), the cell with the
lowest post-warmup val/recon is the dyn-off cell on **all four tasks**, while the probe winner is
always a dynamics-weighted cell. Spearman rank correlation between best val/recon and probe PR-AUC:
transit −0.58, eb −0.70, pulsating −0.84, rotation −0.59. Cost of trusting the loss: 0.021–0.049
PR-AUC per task. The λ-free recon+aux selection metric (`select_include_dyn=false`, both experiments)
is uncorrelated cross-cell in exp05 (−0.13…+0.14) but picks the fwd+bwd probe winner in exp06 at
w256/w512 — the cells where the dynamics dose was actually delivered — and fails at w1024 where the
dose ran at ~0.3× target.

This is the low-pass shortcut expressed in curve space: the dynamics term buys probe quality by
spending reconstruction. Checkpoint selection *within* a run is a separate case and was not shown
invalid (C1 replicated under it).

*Where measured:* `exp06_diagnostics.ipynb` section I (I1/I2), 2026-08-01. Curves:
`exp05_forensics/curves_exp05/` + `exp06_forensics/curves_exp06/`; per-seed probes:
per-cell `results/readout_sweep.csv` (exp05, logistic×mean) and `exp06_geometry_gap.csv`
(mean_resid, trained arm).

## F12 — No logged training metric ranks seeds within a cell

Merging per-seed curve summaries with per-seed probe scores over every (cell, task) with ≥4 seeds:
mean within-cell Spearman of probe score against best recon+aux, best recon, final n_active, and peak
n_active all sit within ±0.3 of zero (per-task means −0.27…+0.25, SDs ≈ 0.5). The seed that trains to
the lowest validation loss is not the seed whose μ probes best. Peak recruitment height during the
n_active transient is equally non-predictive, extending exp05-E4b/exp06-B3 from "do not select
checkpoints on n_active" to seeds.

*Where measured:* `exp06_diagnostics.ipynb` I3, 2026-08-01. 90 runs, 4–6 seeds per cell.

## F13 — Early-epoch pilot readouts miscalibrate steady-state loss ratios

The exp06 λ pilots read recon/dyn at epoch 7 of an 8-epoch run; the ratio keeps falling for 20+
epochs. Result: the w512/w1024 fwd+bwd cells, calibrated for contribution 1.0, delivered 0.275/0.342×
at steady state (still drifting down at epoch 100), while the two cells calibrated from long runs
held 0.99–1.34×. Back-out λ·target/dose_steady = 49/67/65/53 across the four fwd+bwd cells — a narrow
band the epoch-7 procedure missed by 3×. Any constant calibrated from an early-epoch ratio of two
non-stationary losses inherits this failure mode.

*Where measured:* `exp06_diagnostics.ipynb` I4, 2026-08-01; original shortfall exp06-A4.

## F14 — Curve-log semantics that misled a human reader (and would again)

Three properties of the training logs produced false anomaly reports when the W&B curves were read
by eye:

- **The epoch-0 row is not an initial state.** It is logged *after* a full epoch of training at β=0,
  so most losses have already crashed 10–200× by the epoch-1 row, and val/aux frequently sits *above*
  its epoch-0 value at epochs 1–3 (88% of exp06 runs; dyn-off cells too) because time-MSE wins the
  opening trade against the spectral aux term.
- **`free_bits` has no schedule.** It is a constant per-dim floor: 0.0 in every comb-recipe and exp06
  run (where kl_loss ≡ kl_total holds exactly, row for row) and 0.02 in the two lpsd cells, where the
  floor engages at *all* epochs. The 10-epoch knob is the linear β warmup; nothing in the config
  scopes free bits to early epochs.
- **The logged KL is unweighted**, so the end of β warmup at epoch 10 produces no step in any KL
  curve — the ramp is continuous and only its slope ends (measured epoch-9→10 step ≈ 2× a typical
  adjacent-epoch step). A reader expecting a jump at the warmup boundary is expecting the wrong
  quantity.

*Where measured:* `exp06_diagnostics.ipynb` I5, 2026-08-01, all 90 runs.

## F15 — A ratio of two signed responses has an uninformative sign, and it misled a reader

`analyze_exp06_edge.py` reports `aux_over_time_response = aux_response / time_response`, where each is
the *relative* change in that loss term when an impulse is injected. Both are signed, so the ratio's
sign says nothing about either term. The two negative entries in `exp06_edge_corner.csv` have opposite
causes:

| row | time_response | aux_response | ratio | why negative |
|---|---|---|---|---|
| `exp06_edge_comb_fb0p02` | **−3.4%** | **+68.8%** | −20.1 | aux rose; time-MSE fell |
| `untrained` | +2.9% | **−89.9%** | −31.4 | aux fell; time-MSE rose |

Read as one series they look like the same phenomenon at two magnitudes; they are two different
regimes. The negative denominator on the trained row is itself a finding rather than noise: the model's
own p0 error is ≈ −3, so a +3 injected impulse partially cancels it. That is recoverable in closed form
from the logged scalars — `time_response · time_base · 256 = 6·mean(e₀) + 9` gives mean(e₀) = −2.9485
against a direct measurement of −2.9483 — which is also the cheapest available audit of any such metric.

*Consequence:* report the numerator and denominator responses separately and only then a ratio, and
never compare ratios across rows without checking that both components share signs. Same class of
problem as F4 (a summary statistic that hides which side moved).

*Where measured:* exp07 pre-check C1, 2026-08-01.

## F16 — An ablation-based attribution needs a location control, not just a magnitude

Perturbing any part of a reconstruction changes a spectral loss, so "removing X raises the loss" cannot
by itself implicate X. In the edge case the discriminator was applying the **identical** operation (two
residuals forced to zero) at interior positions: the edges raised log-PSD 237–348% while the interior
raised it 6.5–10.3%, a 27–38x asymmetry that no artifact of the operation can explain. Without that
control the effect was equally consistent with "a 256-bin spectrum is sensitive to any two samples".

Two immune controls were also required and were nearly as informative: the dynamics-off arm and the
lpsd arm both show a *nonzero* rise (+35% and +61%), so an attribution resting on "the loss rises at
all" would have implicated arms that have no spike.

*Where measured:* exp07 pre-check C1, 2026-08-01.

## F17 — delta/SE is anticonservative when one side has a single seed

exp07 pre-check C3 compared a 1-seed cell against a 4- and a 6-seed reference. The natural table is
`delta / SE_reference`, and it produces values up to 6.7 on differences of 0.014 PR-AUC, because the
reference's seed spread is the *only* variance entering the denominator: the treatment's own seed
variance is unmeasured and silently set to zero. The same table showed deltas flipping sign across
pooling arms (rotation −0.034 on `mean`, +0.040 on `mean_std`), which is the signature of the missing
variance component.

This is the same failure that killed exp03's 1-seed pulsating headline under the exp04 3-seed confirm.
A 1-seed cell can support "no evidence of an effect" but cannot support a positive claim, and the
delta/SE column must be labelled as reference-only spread wherever it appears.

## F18 — **RETRACTED 2026-08-06.** A post-hoc statistic that reconstructs a selection decision must reproduce the selector's own constraints

*Original claim (wrong):* that `best_recon_aux` selects the pre-warmup epoch-0 checkpoint at aux weight
0.1 with dynamics off, leaving two exp07 cells at the untrained floor while wearing a trained label.

*What is actually true.* The training loop has always restricted checkpoint selection to post-warmup
epochs — `src/swm/train/loop.py`, `improved_select = track_select and epoch >= warmup and val_select <
best_select` — and `beta_warmup_epochs = 10` in every exp07 checkpoint's stored config. Reading the
epoch out of the saved `best_recon_aux.pt` files:

| cell | selected epoch, seeds 0–3 | active units at selection |
|---|---|---|
| `exp07_comb0p1_off` | 98, 97, 98, 98 | 5, 6, 5, 5 |
| `exp07_lpsd0p1_off` | 86, 91, 99, 85 | — |

against the epoch-0 row's 128. The `idxmin` in notebook A6/A6b was taken over the **raw** logged
history with no warmup filter, returned epoch 0 on every seed, and was reported as a live selection
bug. Both readings are reproducible side by side — argmin over all epochs 0; argmin over epochs ≥ 10,
97–98 — and the second is the one the selector computes.

*The generalisable finding, which is why the entry is kept rather than deleted.* Any analysis that
**re-derives a decision the code already made** must reproduce that code's constraints, or be checked
against the artifact the decision produced. Here the artifact was on disk the whole time and carried
its own `epoch` field. Same family as F14 (the epoch-0 row misleads a human reader) and F22 (align
curves on the `epoch` column, not on row position) — a curve statistic standing in for a thing the
curve does not fully determine.

*What survives unchanged.* The weight-0.1 dyn-off cells really do score near the untrained floor
(+0.003 and +0.012, the two lowest of exp07's ten cells). That measurement is independent of the
retracted mechanism, and its cause is now **unknown**. It is not the frozen recipe (weight 0.3), so it
is flagged and not chased before the Aug 15 freeze.

*Consequences.* Q8 closes — there is no guard to land, and the standing rule that any weight-sweeping
experiment must first fix the selector is withdrawn. The guard is now pinned by
`src/swm/tests/test_dual_checkpoint.py::test_select_never_picks_a_warmup_epoch` so it cannot be removed
silently. Active-units-at-selection remains worth printing in every eval fan, now as a health
confirmation rather than as a detector for a known bug.

*Where measured:* exp08 pre-design forensics notebook §1, 2026-08-06. *Original (incorrect) reading:*
exp07 diagnostics A6/A6b, 2026-08-03.

<details>
<summary>Original F18 text, retained so the retraction is auditable</summary>

`best_recon_aux` minimises `recon + w·aux` over all logged epochs, including epoch 0. Epoch 0 is logged
after one full epoch at β=0, so the model there is a near-deterministic autoencoder with **all 128 units
active** that reconstructs *better* than the converged model, whose posterior the KL warmup has since
collapsed to ~5 dimensions. Whether the metric prefers it is decided by an arithmetic race that depends
on `w`:

Seed-mean `val/recon` and `val/aux` at the first and last logged epoch, dyn-off arm:

| cell | recon ep0 → ep100 (penalty) | aux ep0 → ep100 | weighted aux credit | selected epoch |
|---|---|---|---|---|
| `exp07_comb0p1_off` | 0.766 → 0.875 (**+0.109**) | 4.278 → 4.218 | 0.1 × 0.060 = **+0.006** | **0**, all 4 seeds |
| `exp07_lpsd0p1_off` | 0.787 → 0.885 (**+0.097**) | 2.753 → 2.809 (worse) | 0.1 × −0.056 = **−0.006** | **0**, all 4 seeds |
| `exp07_comb0p3_off` | 0.930 → 0.939 (+0.010) | 3.845 → 3.506 | 0.3 × 0.339 = **+0.102** | 90–99 |
| `exp07_lpsd0p3_off` | 0.950 → 0.962 (+0.012) | 2.343 → 2.086 | 0.3 × 0.257 = **+0.077** | 89–99 |
| `exp07_hann0p3_off` | 0.998 → 0.861 (−0.137) | 4.028 → 3.417 | 0.3 × 0.611 = **+0.183** | 40–97 |

At weight 0.1 the credit is ±0.006, an order of magnitude below the ≈0.10 reconstruction penalty, so
epoch 0 wins on every seed; at weight 0.3 the credit is 0.077–0.183 and outbids a penalty that is either
tiny or absent.

The two affected cells then score at the untrained floor: averaged over all tasks and poolings they sit
**+0.003** and **+0.012** above the capacity-matched untrained encoder — the two lowest of exp07's ten
cells — against +0.020…+0.033 for the properly selected dyn-off cells, and `comb0p1_off` matches the
untrained value on pulsating to three decimals. Nothing distinguishes them in the results tables: they
carry a cell name, a seed, and a checkpoint like any other run.

The exposure is any experiment that **lowers or sweeps `recon_aux.weight`**, and it grows as the weight
falls. The fix is to restrict the argmin to epochs ≥ `beta_warmup_epochs`, which costs nothing and is
already the convention for every *other* curve statistic in this project (F14 records why the epoch-0 row
misleads a human reader; this is the same row misleading the selector). A cheap detector, worth printing
in any eval fan: **active units at the selected checkpoint** — 128 means the checkpoint predates the
collapse and is not a trained representation.

*Where measured:* exp07 diagnostics notebook A6/A6b, 2026-08-03. *Scope:* qualifies the weight-0.1
**dyn-off** arms of exp07's P2/P5 and the README's "every recipe's fbwd arm beats its off arm" clause; no
headline gate is affected, because `comb0p3` and `hann0p3` select epochs 40–99 on both arms.

</details>

## F19 — Comparing two window functions requires a referee neither model trained under

exp07 compared a rectangular-FFT aux term against a Hann-tapered one. Scoring the two models'
reconstruction spectra under either *training* window decides the comparison before any data is read, and
the margin is not subtle — mean squared log-PSD error on pulsator windows, 65–260 µHz:

| referee | `comb0p3` (rect-trained) | `hann0p3` (Hann-trained) |
|---|---|---|
| rectangular | **7.2** | 11.2 |
| Hann | 33.3 | **8.7** |
| DPSS multitaper (neutral) | 11.1 | 12.5 |

Each model wins under its own window by a factor of up to 4, and the two "results" point in opposite
directions. Only the third row is evidence. The generalisation: whenever a design choice is *also* a
measurement choice — a window, a normalisation, a resampling, a distance metric that a loss also uses —
the evaluation must be run through an instrument neither arm was fitted to, and the *disagreement between
referees* is itself the quantity of interest (here it is exactly the spectral leakage the taper removes).

*Where measured:* exp07 diagnostics notebook D1/D2, 2026-08-03.

## F20 — Reconstruction fidelity in a task's own frequency band does not predict that task's probe score

F11/I2 established that validation loss anti-predicts probe winners. The natural rescue is that a *better
resolved* loss would predict — that if a model reconstructs a task's characteristic frequencies well, it
should probe that task well. exp07 measures both sides of that and it does not hold. Under the neutral
referee `hann0p3` reconstructs the 65–260 µHz band (where this corpus's pulsators live) **13% worse** than
`comb0p3`, and it scores **+0.018 ± 0.009 better** on the pulsating probe. Two further readings agree:

- Decomposing the pulsating gain into per-star contributions and binning against the fractional-bin
  offset — the coordinate along which rectangular leakage is zero at integer cycles per window and
  maximal at half-bin — gives ρ = **−0.20** (p = 0.013), the *opposite* of the leakage prediction. The
  gain is spread across every offset bin.
- All five recipes retain ≈1% of the true peak power and place the dominant peak within one bin about
  25% of the time (0% above 300 µHz), so none of them reconstructs pulsations in a physical sense while
  all of them probe pulsating at 0.79–0.81 PR-AUC.

The rule generalises F11 rather than replacing it: no reconstruction-derived quantity, however
band-resolved, is a valid selection signal for a representation. Probe gates remain the only currency.

*Where measured:* exp07 diagnostics notebook D2/D3/G1, 2026-08-03.

## F21 — A correlation pooled across the dynamics arms can invert the within-arm relation

Asking whether the window-edge defect cost downstream score, over all 40 exp07 runs, returns a strong
*positive* association between edge severity and probe score (eb ρ = **+0.58**, rotation +0.52), which
reads as "the spike helps". Split by arm it vanishes and changes sign by task and arm (eb +0.07 / +0.32,
pulsating −0.28 / −0.02, rotation −0.01 / +0.44). The cause is that turning the dynamics term on raises
both quantities, so the pooled statistic measures the dyn axis and nothing else — a textbook Simpson
reversal, and the third instance in this project of a summary statistic pointing the wrong way (F15, F16).

Any correlation computed over a set of cells that spans a deliberate design axis must be computed
*within* the levels of that axis, or it is measuring the axis. Concretely for this project: `arm`
(dyn-off vs fwd_bwd) is never poolable, which is the same rule the exp06 verdict states for reporting
geometry numbers.

*Where measured:* exp07 diagnostics notebook B4, 2026-08-03.

## F22 — The exp07 W&B dump lost the pre-resume epochs of resumed runs

exp05 and exp06 preserved a killed run's earlier history alongside the resumed run as
`*.killedprefix.csv`, and both diagnostics notebooks stitch them. The exp07 dump wrote no such files, so
the two runs the sweep runner resumed (`exp07_comb0p3_off` seed 2, `exp07_hann0p3_fbwd` seed 3) have
histories that begin at epoch 30 and 37 with the earlier epochs unrecoverable.

Checkpoints are intact, so no probe number is affected, but three classes of statistic silently break:
anything anchored at a fixed early epoch (total post-warmup travel), any per-epoch average built by
stacking curves **by row position** rather than by the `epoch` column, and any convergence claim that
assumes a complete curve. The guard is cheap: assert `epoch.min() == 0` before computing an epoch-range
statistic and carry a `prefix_lost` flag through the run inventory.

*Where measured:* exp07 diagnostics notebook A1, 2026-08-03.

*Where measured:* exp07 pre-check C3, 2026-08-01; precedent exp03 → exp04.

## F23 — A ratio-of-means edge statistic hides which stars carry the defect, and can invert its sign

Every edge number in exp05–exp07 is `edge_max`: the worst endpoint MSE over a fixed window sample divided
by the interior level of that same sample. Reduced **per star** instead (median over each star's test
windows, 6 seeds per cell), the same quantity says something the pooled ratio cannot:

| cell | per-star median ratio | absolute excess | ρ(noise, ratio) | ρ(noise, absolute excess) | quietest → noisiest decile |
|---|---|---|---|---|---|
| `comb0p3_fbwd` | 28.2 ± 1.8 | +25.9 | +0.39 | **+0.90 ± 0.01** | 9.2 → 28.7 |
| `hann0p3_fbwd` | **0.71 ± 0.01** | **−0.28** | −0.64 | −0.71 | 3.1 → 0.62 |

Two readings follow. **(a)** The tapered recipe's published 1.15× is not the typical star's behaviour —
the median star's endpoints are reconstructed *better* than its interior (ratio below 1, excess negative),
and the residual is carried by a minority at the **quiet** end of the corpus. **(b)** The untapered
baseline's endpoint impulse scales with the star's own noise (ρ = +0.90 on the absolute excess), which is
the "decoder buys log-PSD power with an impulse" mechanism read off the star population rather than off
the loss surface: a noisier star has more broadband power to match, so the purchase is larger.

The two framings disagree about which stars look worst, because the ratio divides by an interior level
that itself scales with noise. Any edge statistic should report the absolute excess, or both; and any
"the defect is small now" claim should state whether it is small *everywhere* or small *on average*.

*Where measured:* `experiments/analyze_exp07_edge_noise.py` → `exp07_edge_noise_{stars,summary}.csv`;
exp07 diagnostics notebook C5, 2026-08-04.

## F24 — A window function refunded the entire reconstruction cost of tripling the aux weight

`hann0p3` and `comb0p3` differ only in `train.recon_aux.psd_window`; same weight 0.3, same λ 60, same
geometry, same seeds. At steady state their validation reconstruction differs by **0.089** (0.876 against
0.965), a **9.2%** refund, present in both dynamics arms (−0.084 dyn-off), opening by epoch 2 and largest
(−0.24) around epochs 5–10 while the KL warmup ramps.

The size places it: 0.876 sits on top of `comb0p1` (0.879) and `lpsd0p1` (0.885) — the cells that bought
their cheap reconstruction by running the aux term at **a third** of the weight. So the reconstruction
penalty normally attributed to aux weight 0.3 is not the cost of asking for spectral fidelity; it is the
cost of the *rectangular window's* demand for edge power, which the decoder had been paying reconstruction
error to supply (the C1–C3 purchase mechanism). Removing the purchase removes the penalty, at no
downstream cost (probe-tied, and better on pulsating).

Corollary for any future aux design: a term's apparent expensiveness in reconstruction may be an artefact
of its framing rather than a real trade-off, and is worth testing before the weight is lowered.

*Where measured:* exp07 diagnostics notebook K4 (W&B curve dumps, 6 seeds per cell), 2026-08-05.

## F25 — "Collapse reversal" was a learning-rate-horizon artefact; the dynamics benefit does not scale with capacity or with dose

Three measurements from the same curve corpus, all pointing the same way.

**(a) The exp05 mechanism number was set by the LR schedule, not by the recipe.** exp05 reported
dynamics-on recruiting 5 → ~25 active units. The *same recipe* at a 100-epoch budget selects with 7.0
(exp06) and 8.2 (exp07) against exp05's 27.5. The obvious explanation — exp05 stopped before the pruning
finished — is wrong: the runs already differ **at matched epochs**.

| epoch | exp05 (60 ep) lr / active | exp06 (100 ep) lr / active | exp07 (100 ep) lr / active |
|---|---|---|---|
| 20 | 2.3e-4 / 27 | 2.7e-4 / 35 | 2.7e-4 / 38 |
| 40 | **7.7e-5** / 30 | 2.0e-4 / 15 | 2.0e-4 / 11 |
| 59 | **3.2e-6** / 26 | 1.1e-4 / 7 | 1.1e-4 / 10 |

The cosine schedule is annealed over `max_epochs`, so the 60-epoch run is at a systematically lower rate
from about epoch 25 and both its active set (26 vs 7) and its reconstruction (1.011 vs 0.969 at epoch 59)
freeze there. **A short-budget run of a recipe is a different run, not a prefix of the long one**, and no
capacity number is comparable across experiments with different budgets. (This is also the mechanism behind
the empirical ep40 → ep100 dose drift ≈ 0.75 that exp07's λ pilots had to correct for.) The two ep100
sweeps, which share a schedule, replicate to <1% on every curve.

*Where measured:* exp07 diagnostics notebook K5/K6 matched-epoch table.

**(b) Capacity is available and unpaid.** `lpsd0p3_fbwd` holds **18.25** units at selection, three times the
winner's 5.8, and scores mid-table (0.491 mean PR-AUC against 0.498). Within an arm, more units go with
*worse* pulsating (ρ −0.73 over five cells).

**(c) The dose is a binary, not a dial.** Pooled over ten cells, λ·dyn/recon correlates with probe score at
ρ +0.68…+0.90 — but it is identically zero on every dyn-off cell, so the pooled figure is the arm itself
(F21 again; the correlation cannot even be computed in the off arm). Inside the fwd+bwd arm, where the dose
spans 0.79–1.07, it orders nothing. Every fwd+bwd cell beats every dyn-off cell (0.491–0.498 against
0.422–0.459, no overlap) regardless of which aux term produced it.

Together: at 256×16 the dynamics term buys a real and repeatable gain on a latent that recruits at most
three extra dimensions, and buying more of it — more units, more dose — does not buy more gain.

*Where measured:* exp07 diagnostics notebook K2/K5/K6 across the exp05, exp06 and exp07 curve dumps,
2026-08-05.

## F26 — Three readouts agree on the encoder ranking; the training loss is orthogonal to all of them

Four exp07 encoders (`{hann0p3, comb0p3} × {off, fbwd}`, 4 seeds) scored three ways on the v1 pool —
mean-pooled `mean_resid`, plain mean pooling under the MIL sweep, and the val-declared MIL winner:

| cell | MIL winner | mean pooling | `mean_resid` | val/recon |
|---|---|---|---|---|
| `hann0p3_fbwd` | **0.592** | **0.570** | **0.498** | 0.876 |
| `comb0p3_fbwd` | 0.579 | 0.569 | 0.498 | 0.965 |
| `hann0p3_off` | 0.561 | 0.555 | 0.459 | **0.863** |
| `comb0p3_off` | 0.559 | 0.543 | 0.444 | 0.939 |
| untrained | 0.531 | 0.523 | — | — |

The three readouts rank the cells identically (ρ = +1.00); Spearman between `val/recon` and either pooled
readout is **exactly 0.00** — the cheapest-loss cell places third and the most expensive places second.
The val-loss prohibition (F11) therefore holds under a pooled readout, not only under mean pooling.

Two further readings on the pooling operator itself. **The gain is task-selective and largely
representation-independent:** averaged over four tasks the MIL winner beats mean pooling by only
+0.006…+0.022, decomposing into transit +0.042…+0.080, eb ≈ 0, rotation/pulsating scattered about zero —
and the **untrained** arm gains +0.066 on transit from the same operator. **The winner is not a single
operator:** across four seeds of one cell the val-declared choice changes identity up to four times and
spans feature space (`rff_meanmap`, `mean_std`, `gmm_prototype`) and score space (`ws_ppv_lspv`,
`ws_topk`). Any table that adopts pooling must fix the operator per task and report the paired untrained
arm.

*Where measured:* `experiments/run_exp07_mil_sweep.ps1` → `mil_sweep_exp07_<cell>.csv`; exp07 diagnostics
notebook K3, 2026-08-05. Scope caveat: v1 pool, `first` bag scope (16–20 windows/star), not the
all-segment K-matched setting the 0.245 transit number came from.

## F27 — The taper relocated the reconstruction impulse rather than removing it, and a seed-averaged profile hides that

exp07 reported the window-edge defect closed (`hann0p3` edge_max 1.15× against `comb0p3`'s 31×) and F23
closed it as a probe cost. Reducing the per-position error profile **per seed** instead of over
seed-averaged profiles shows what the taper bought: a new impulse in the window **interior**.

Per-seed `max(mse, pos 112–144) / median(mse, pos 16–239)`, over all ten exp07 cells:

| cell | edge ratio | **centre ratio** | argmax position, per seed |
|---|---|---|---|
| `comb0p3_fbwd` | 31.7 | 1.15 ± 0.12 | 123, 136, 136, 136 |
| `comb0p3_off` | 10.7 | 1.19 ± 0.13 | 118, 124, 136, 141 |
| `lpsd0p3_fbwd` | 41.9 | 1.13 ± 0.06 | — |
| `lpsd0p3_off` | 2.4 | 2.08 ± 1.55 | — |
| **`hann0p3_fbwd`** | **1.31** | **11.8 ± 5.3** | 125, 131, 138, 143 |
| **`hann0p3_off`** | **1.37** | **8.8 ± 2.3** | 123, 129, 135, 143 |

The mechanism is immediate and was implicit in the fix's own docstring: a symmetric Hann window is
exactly zero at the endpoints and maximal at the centre, so the cheapest place for a decoder to buy
log-PSD power moves from p0 to the middle of the window. `comb0p3_fbwd` pays ~32× at the boundary;
`hann0p3_fbwd` pays ~12× in the interior. The defect changed location, not order of magnitude.

**Two measurement lessons, both instances of patterns already in this file.**

- **A seed-averaged profile understates a spike whose position moves.** The argmax wanders 125 → 143
  across hann seeds, so averaging profiles first smears the 11.8× to about 5×. Reduce per run, then
  aggregate — the same ratio-of-means pathology F23 records for the edge, one level down.
- **A fixed position band is the wrong estimator** for a defect that is not at a fixed position, and it
  cannot be compared across cells where no bump exists. Locate by argmax over the interior instead.

**Scope of the associated stitch-comb number.** Laying independently decoded windows end to end injects
a harmonic comb at the 256-cadence stitch rate (2.8125 c/d). Measured against controls, the **untrained**
arm combs *harder* than any trained cell while the input sits near 1.5×, so the comb is **architectural**
— a consequence of decoding windows independently — and training reduces it. It is also invariant to the
taper by construction: an impulse at position 0 and one at position 130 produce the same comb
frequencies and differ only in phase, so the published 26.2× → 24.0× is "unchanged", not an 8%
improvement. Probes read µ and never see a reconstruction, and no term in the objective spans a window
boundary, so the comb is **paper hygiene**, not a probe cost. Measured at 6 seeds × 200 strips:
untrained **31.9 ± 7.1**, trained cells 20.4–26.6, input **1.51**.

**Does the centre impulse cost probe score? No — and the statistic that says it does is a ratio
artifact.** Within the hann cells, ρ(centre_ratio, per-star pulsating score) = **−0.39 ± 0.06** over
~2,000 test stars, which reads as a real cost. The control kills it: the same correlation computed with
**`edge_ratio`, in the same cells, where there is no edge defect** (1.31×), returns **−0.39** as well. A
statistic that reports the same effect size for a defect that is present and one that is absent is not
measuring the defect. The mechanism is the shared denominator — `interior_mse` tracks the star's own
amplitude at ρ **0.92** and its noise at ρ **0.90**, so any ratio is a star property wearing an
artifact's name. The absolute form (`centre_excess`) disagrees with itself across arms (+0.21 fbwd,
−0.16 off), which is what no effect looks like.

This is F23's finding reproduced one level down, and the **third** time in this project a ratio statistic
has pointed the wrong way (F15, F23, here). Generalisation: before correlating any normalised severity
measure against an outcome, run the same correlation through a channel where the defect is known to be
**absent** — if it survives, it belongs to the normalisation, not to the defect.

One real parallel does survive: `centre_excess` correlates with star noise at **+0.85** in the hann
cells, just as `comb0p3`'s *edge* excess did at +0.90 (F23). The decoder's purchase scales with the
star's broadband power in both recipes — the same transaction, relocated.

*Scope:* the per-star join covers **eb and pulsating only**; `exp07_diag_star_scores.parquet` never
dumped rotation or transit per-star predictions.

**Consequence for the record:** F23's "the edge is closed as a lever" and `exp07_aux_README.md`'s "edge
fixed" are true as stated and misleading as read; both are qualified.

*Where measured:* `experiments/analyze_exp07_centre_artifact.py` → `exp07_centre_{cells,stars,summary}.csv`
and `experiments/analyze_exp07_stitch_spectrum.py` → `exp07_stitch_spectrum_summary.csv`; exp08
pre-design forensics notebook §3, 2026-08-06.

---

## Operational problems hit while producing the above

Recorded because each produced a *silently* wrong state rather than a loud failure.

- **`| tee` masks exit codes.** A background job that raised was reported as exit 0, because a shell
  pipeline returns the *last* command's status, not Python's. It was noticed only because an expected
  CSV was missing. Mitigation used since: `set -o pipefail` plus an explicit `${PIPESTATUS[0]}` echo,
  or verifying the output file exists.
- **Concurrent sweep runs clobber each other.** `swm.eval.mil_sweep` does read-existing → concat →
  write, so two invocations pointed at one `--out` silently drop one another's rows. Parts are now
  written separately and merged by `swm.eval.mil_report`.
- **Renaming an output column broke a notebook twice, silently.** Adding the R² regression probes
  renamed `pr_auc_test` to the metric-agnostic `score_test` in some blocks but not others; the
  failure only surfaced when a human re-ran the notebook. `src/swm/tests/test_mil_report.py` now pins
  the output schema.
- **Guards go stale.** An `assert` added to block an unsupported code path survived past the point
  where the path became supported, and killed two jobs instantly.
- **Caches without provenance are easy to mis-join.** Two star pools with different splits nearly
  shared one results CSV; they are now forced to separate default output files, and `cache_path`
  routes each (pool, scope) to its own directory.
- **Replaying per-star files is I/O-bound, not compute-bound.** Building all-segment caches for the
  downstream pool reads ~90k small npz files; the encoder itself runs at 93k windows/s. Restructuring
  so the read happens once for all 9 encoder arms, instead of once per arm, was worth ~9× on that step.

## Data-quality caveats that outlive individual experiments

- **v1 rotation labels are confounded.** `subset.py` defines quiet as "matched in NO catalog"
  *including rotation*, so 0 of 10,000 quiet stars are rotation-positive against a corpus rate of
  ~13%. Paired deltas survive (both arms share the confound) but rotation is partly a
  general-variability detector.
- **Engineered features beat the frozen SSL representation on all 11 downstream probes**
  (`new_task_exp05/`, 2026-07-25). "Beats engineered features" is measured **false**.
