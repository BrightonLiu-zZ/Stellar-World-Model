# exp07 — aux term first-class: form × weight factorization + Hann edge fix

Manifest: `experiments/configs/exp07_aux_factorization.yaml` (single source of truth) ·
Plan: `docs/plans/2026-08-01-exp07-aux-factorization.md` · Trained 2026-08-02 (40 runs, 0 failures,
seeds 0–3) · Eval: `analyze_exp06_geometry_gap.py` → `exp07_aux_gap.csv` / `exp07_aux_acf.csv`;
edge `analyze_exp07_c1_edge_sign.py` → `exp07_edge_profiles.{csv,parquet}` + `exp07_edge_ratios.csv`;
amplitude `analyze_exp07_c2_amplitude.py` → `exp07_amp_dominance.csv`; dose `exp07_dose_gate.csv`
(curves `exp07_forensics/curves_exp07/`). **6-seed extension of winner + baseline pending** (seeds
4–5 on {comb0p3, hann0p3} × {off, fbwd}); headline gates re-score at 6 seeds after it.

Grid: 2×2 `recon_aux` (type {combined, log_psd} × weight {0.1, 0.3}) + `hann0p3`
(combined@0.3 + `psd_window=hann`, taper-weighted demeaning), each × dyn {off, fwd_bwd},
256×16, ep100, fb=0, seeds 0–3. λ = 60 everywhere except lpsd0p1 (75), drift-corrected pilot
calibration (see manifest `constants`).

## Pre-registered verdicts (4 seeds)

**G-dose: PASS, all 20 fwd_bwd runs.** Achieved dose 0.75–1.08 (cell means 0.78–1.05) — the
drift-corrected λ calibration (pilot ep40 reading × 0.75 drift → ep100 target) landed as predicted.
The pilots also updated pre-check C4: **the 2.11× lpsd dyn inflation was free_bits-driven**; at fb=0
every recipe's dial is in the universal λ≈50–67 band.

**P1 (replication): PASS.** `comb0p3_fbwd` spikes (p0 22×, p255 30× interior; trailing edge worse,
matching the corrected dipole reading) and C1-rep passes on transit/eb/rotation with pulsating ns —
the exp05/exp06 pattern under the new λ60 dose.

**P2 (pressure): PASS.** Edge severity scales with aux weight for BOTH forms:
combined 34×→18×, log_psd 41×→5.5× going weight 0.3→0.1 (edge_max, seed-mean).

**P4 (discriminator): PRESSURE-DRIVEN.** `lpsd0p3_fbwd` spikes at **41×** — removing `hf_time` does
NOT confer immunity; the historical "lpsd is immune" was its 0.1 *weight* (plus free_bits keeping
dims alive), not its form. Consequence: **no untapered config combines full aux pressure with a
clean edge.**

**P3 (Hann fix): PASS, by a wide margin.** `hann0p3_fbwd` edge ratio **1.15×** (its own dyn-off arm:
1.20×) at full weight 0.3 and full dyn dose — the C1 "decoder buys spectral power at the edge"
mechanism, removed by construction, removes the spike entirely.

| edge_max (seed-mean) | fbwd | off |
|---|---|---|
| comb0p3 | 34.4 | 7.8 |
| lpsd0p3 | 41.5 | 2.0 |
| comb0p1 | 18.2 | 2.8 |
| lpsd0p1 | 5.5 | 2.5 |
| **hann0p3** | **1.15** | 1.20 |

**G-C1 (dynamics benefit generalizes): PASS in the winner.** hann0p3 fbwd−off (mean_resid):
eb +0.081±0.010, rotation +0.053±0.017 (>2·SE); transit +0.017 ns; pulsating ns. Every recipe's
fbwd arm beats its off arm on eb (>2·SE, both poolings) — the dynamics-weighting result is
aux-config-robust. **[Amended 2026-08-03 — see the diagnostics amendment below: for `comb0p1` and
`lpsd0p1` the off arm never trained a checkpoint, so those two recipes do not support the
aux-config-robustness clause. The winner's own G-C1 is unaffected.]**

**G-noregress → winner = `hann0p3`.** Only hann0p3 and lpsd0p1 show zero paired regressions vs
`comb0p3_fbwd` on any task/pooling. hann0p3 is probe-tied with the baseline everywhere (largest
paired delta ±0.02, all within 2·SE) and nominally best on pulsating (0.807 mean / 0.807
mean_resid). Tie-break over lpsd0p1 (edge 5.5×): hann keeps full spectral+hf pressure, is exactly
edge-clean, and costs one clause in the paper (tapered periodograms are the Welch-standard choice —
the untapered FFT was the nonstandard one). lpsd0p3 regresses rotation; comb0p1 regresses
transit+rotation (mean_resid).

**P5 (report-only, amplitude dominance).** PC1-on-amplitude R² tracks **weight**, not form
(0.3 cells 0.955–0.967; 0.1 cells 0.918–0.922) — the same pressure story as the edge. Dynamics-on
lowers concentration in every recipe (evr_pc1 0.93–0.94 → 0.81–0.83; participation ratio 1.13 →
1.43–1.53); nothing goes below the untrained floor (amp_var_frac 0.732; fbwd cells 0.77–0.84).

**Truncation (context):** 18/20 fbwd seeds still improving in their last 10 epochs at ep100 —
absolutes remain floors; paired gates unaffected (A2 pattern).

## 6-seed extension re-score (2026-08-02, seeds 0–5 on {comb0p3, hann0p3} × {off, fbwd})

All headline gates HOLD at the honest p≈0.05 footing; artifacts `exp07_aux_gap_6seed.csv`,
`exp07_edge_profiles_6seed.{csv,parquet}`.

- **G-noregress: PASS.** hann0p3_fbwd vs comb0p3_fbwd paired over 6 seeds: no regression on any
  task/pooling (largest negative −0.009±0.006); pulsating is **significantly better** on mean
  (+0.018±0.009 > 2·SE) and mean_std (+0.025±0.012 > 2·SE), tied on mean_resid (+0.011±0.013).
- **G-C1: PASS.** hann0p3 fbwd−off (mean_resid): eb **+0.080±0.007**, rotation **+0.046±0.013**
  (> 2·SE); transit +0.020 ns, pulsating ns (the pulsating lift lives in the hann aux itself:
  off arm 0.800 ≈ fbwd 0.801 on mean_resid).
- **Edge at 6 seeds:** hann0p3_fbwd **1.16 ± 0.03** vs comb0p3_fbwd **31.4 ± 4.8** (edge_max).
  **Qualified 2026-08-06 (F27): the edge number is correct and "the defect is fixed" is not what it
  means.** Reduced per seed, `hann0p3_fbwd` carries an *interior* impulse of **11.8 ± 5.3×** the
  interior level, at the position where the Hann weight is maximal — the same order as the ~32×
  boundary impulse it replaced. The taper relocated the decoder's log-PSD purchase from p0 to the
  window centre; it did not remove it. Every "edge-clean" / "edge exactly clean" claim below should be
  read as *edge*-clean specifically. See `experiments/analyze_exp07_centre_artifact.py` and exp08
  pre-design forensics §3.
- **G-dose:** extension seeds 0.83/0.83 (comb) and 1.00/0.92 (hann) — all pass.

## Consequence

**`hann0p3` (combined aux, weight 0.3, Hann-tapered log-PSD via taper-weighted demeaning,
fwd_bwd@λ60, fb0/β0.1, 256×16) is the ML4PS FINAL RECIPE** — confirmed at 6 seeds: probe-tied or
better vs the comb baseline (pulsating +0.018/+0.025 > 2·SE on mean/mean_std), dynamics gates pass
(eb +0.080, rotation +0.046), edge exactly clean (1.16× vs 31.4×), dose delivered. exp07 was the
last **recipe-touching** experiment — but not the last experiment: 2–3 further experiments follow
before the Aug 15 freeze, building on this frozen recipe without re-opening it. Remaining work on
exp07 itself: the edge ADR (mechanism + fix both measured) and the consolidation eval fan (7-probe
scorecard + v1 tables) on this recipe.

## Diagnostics amendment (2026-08-03, `src/notebooks/exp07_diagnostics.ipynb`)

The health-check notebook recomputed every pre-registered gate from the raw CSVs, independently of the
scripts that wrote this file. **All five published gate numbers reproduce to within 0.001** (G-noregress
pulsating +0.018/+0.025/+0.011, G-C1 eb +0.080 and rotation +0.046), G-dose passes on all 30 fwd_bwd runs
including the extension seeds, and a second independent probe implementation reproduces
`exp07_aux_gap_6seed.csv` **exactly** on all 300 rows. Nothing in the verdict changes: `hann0p3` remains
the frozen recipe. Three items amend or qualify the text above.

1. **Two cells never selected a trained checkpoint (notebook A6).** `comb0p1_off` and `lpsd0p1_off` pick
   **epoch 0** on all four seeds, with 128 active units and `mu_var` ≈ 22 at the selected step. The cause
   is the selection metric: `best_recon_aux` minimises `recon + w·aux`, and at weight 0.1 the weighted aux
   improvement over training (+0.006 for comb, −0.006 for lpsd, whose aux gets *worse*) cannot offset the
   reconstruction penalty the KL warmup imposes (0.77 → 0.88, i.e. +0.10), so the pre-collapse epoch-0
   model wins; at weight 0.3 the credit is +0.077 to +0.183 against a penalty of ≈0.01 or none, and every
   0.3 cell selects in the 40–99 range. Those two cells sit +0.003 and +0.012 above the
   untrained encoder averaged over all tasks and poolings — the two lowest of the ten — against +0.020 to
   +0.033 for the properly selected dyn-off cells. **Consequence:** the G-C1 sentence's "every recipe's
   fbwd arm beats its off arm" does not hold as a *dynamics* statement for `comb0p1` and `lpsd0p1`, whose
   off arms are near-untrained encoders. The winner and baseline are unaffected (both arms select epochs
   40–97), so G-noregress, G-C1 in `hann0p3`, G-dose and every 6-seed edge number stand. The dyn-on half
   of P2 (comb 34.4→18.2, lpsd 41.5→5.5) is also unaffected. The weight-0.1 **dyn-off** rows of P5 carry
   the same caveat. Recorded as F18.

2. **The edge fix is stronger than "spike removed", and it never cost probe score.** Decomposing the
   endpoint error separates the dipole from a symmetric offset: `comb0p3_fbwd` carries an antisymmetric
   component of 2.65 (seed spread large), `hann0p3_fbwd` 0.001 ± 0.011 — the taper *abolishes* the dipole
   rather than shrinking it, leaving a small symmetric, seed-repeatable taper shadow (1.15×). The
   amplitude sweep confirms unbuyability by construction: the untapered term has a minimum at the learned
   edge amplitude (4.29 → 2.07 → 2.66 across the sweep) while the tapered term is flat to four decimals.
   Separately, **within an arm the edge ratio does not predict probe score** (ρ ranges −0.28 to +0.44,
   sign varying by task and arm); the apparently strong pooled correlation (eb +0.58) is the dynamics
   axis. The case for the taper is correctness and standards, not score.

3. **The taper has a measured cost, and the pulsating gain is not spectral (notebook D2, G1).** Scored
   under a DPSS multitaper referee neither model trained against, `hann0p3` is **13% worse** than
   `comb0p3` at 65–260 µHz — the band this corpus's pulsators occupy — and 44–58% better above 1 mHz.
   Yet its pulsating gain does not track spectral leakage: decomposing the +0.018 into per-star
   contributions and binning against the fractional-bin offset gives ρ = −0.20 (p = 0.013), i.e. the gain
   is if anything larger where leakage is *zero*. So the taper reconstructs the pulsator band worse and
   scores better on the pulsating probe; whatever it bought, it did not buy through spectral fidelity.
   Recorded as F19 (referee choice) and F20 (fidelity does not predict probe score).

Also noted: two runs (`comb0p3_off` seed 2, `hann0p3_fbwd` seed 3) were resumed and the exp07 W&B dump
kept no `*.killedprefix.csv`, so their pre-resume epochs are unrecoverable (F22). Checkpoints are intact;
only epoch-range statistics are affected.

## Second diagnostics pass (2026-08-04): per-star panels and the per-star edge

Two additions to the notebook, and one measurement that qualifies how the edge result should be phrased.

**Section I — live light-curve inspection.** The nine detection probes of the downstream menu, fitted on
the winner's frozen mu (seed 0), with confusion matrices, per-quadrant star panels and a raw
positive-vs-negative viewer. Each quadrant panel draws one star per quadrant and shows **both** recipes'
reconstructions of it, one subplot each on a shared y-scale, which is where `comb0p3`'s edge spikes are
visible as excursions frequently larger than the star's entire flux range. These are display panels, not a
scorecard: one seed, one pooling, no untrained reference. Read them for *which* stars a probe gets right.

The panel readings, for orientation only (`hann0p3_fbwd` seed 0, mean pooling, PR-AUC):
`osc_giant` 0.858 · `rgb_vs_heb` 0.838 (n=161) · `pulsating` 0.808 · `eb` 0.769 · `rotation` 0.577 ·
`ijspeert` 0.444 · `solar_like_osc` 0.333 · `transit` 0.138 · `flare` 0.088.

Producing them required the first frozen-recipe mu over the new-task pool:
`experiments/exp07_forensics/new_task_mu_cache/hann0p3_fbwd_s0.npz`, written by
`python -m swm.eval.new_task_extract --arms hann0p3_fbwd_s0 --ckpt-dir experiments/exp07_hann0p3_fbwd/models
--out-dir experiments/exp07_forensics/new_task_mu_cache` (~12 min, GPU). The consolidation eval fan needs
the same pass for the remaining arms and seeds plus the untrained reference; the pool read is now proven.

**Section C5 — the edge, per star** (`analyze_exp07_edge_noise.py`, 4 cells x 6 seeds, all test windows).
`edge_max` is a ratio of means over a window sample, so it cannot say whether a residual is spread over
all stars or concentrated on a few. Reduced per star, the median test star under `hann0p3` has endpoint
error **below** its interior error (ratio 0.71 ± 0.01, absolute excess −0.28), and the residual that
produces the published 1.15x lives on the **quiet** end of the corpus (ratio 3.1 in the quietest noise
decile, 0.62 in the noisiest; ρ(noise, ratio) = −0.64). The impression from the section-I panels — that
noisy stars still spike under the taper — is wrong; what grows on those stars is the interior error.

The baseline moves the opposite way and confirms the C1 mechanism per star: `comb0p3_fbwd`'s **absolute**
edge excess correlates with the star's own noise at ρ = **+0.90 ± 0.01**, exactly as "the decoder buys
log-PSD power with an endpoint impulse" predicts — more broadband power to match, larger purchase.
Recorded as F23. Nothing in the verdict changes; the phrasing does: the taper leaves the typical star
cleaner than the interior, not uniformly 1.15x worse at the boundary.

## Third diagnostics pass (2026-08-05): cross-experiment curve forensics, section K

Section I of the exp06 notebook read the exp05/exp06 training curves for what they said about exp07.
Section K of this notebook does the same job one experiment later, over all three curve dumps
(`exp0{5,6,7}_forensics/curves_exp0{5,6,7}`, 138 runs / 29 cells), and adds the MIL axis the request
asked for. Comparability rule stated once and enforced throughout: `val/recon` is the same function of the
same data for every cell of a geometry and may be ranked; `val/aux` and `val/monitor_recon_aux` are
functions of the aux type, its weight and (for `hann0p3`) its integrand, and may not.

**The "hann's loss is worse" premise is half wrong (K4, F24).** Its *spectral* score is worse in one band
(13% at 65-260 uHz under the DPSS referee, D2). Its *reconstruction* is **9.2% better** - 0.876 against
`comb0p3`'s 0.965 at steady state, in both arms, opening by epoch 2. The size places it: `hann0p3` lands
on top of `comb0p1` (0.879) and `lpsd0p1` (0.885), the cells that got cheap reconstruction by running the
aux term at a third of the weight. **A window function refunded the entire reconstruction cost of tripling
the aux weight**, which is the loss-curve signature of the C1-C3 purchase mechanism: the penalty was not
the price of spectral fidelity, it was the price of edge power the rectangular framing demanded.

**"Collapse reversal" was a learning-rate-horizon artefact (K5/K6, F25).** exp05's 5 -> ~25 active units
is not the same recipe seen earlier: because the cosine schedule is annealed over `max_epochs`, the
60-epoch run is at lr 7.7e-5 by epoch 40 against the ep100 runs' 2.0e-4, and its active set freezes at ~26
while theirs prune on to 7-8 (at the *same* epoch: 30 vs 11-15). A short-budget run is a different run,
not a prefix - which is also the mechanism behind the ep40 -> ep100 dose drift the exp07 pilots corrected
for. At ep100 the winner holds **5.8** units, fewer than its runner-up, while `lpsd0p3_fbwd` holds
**18.25** and scores mid-table: capacity is available and unpaid. The two ep100 sweeps, which share a
schedule, replicate to <1% on every curve.

**The dose is a switch, not a dial (K2).** Pooled over ten cells lambda*dyn/recon correlates with probe
score at rho +0.68...+0.90 - but it is identically zero on every dyn-off cell, so the pooled figure is the
arm in disguise (F21 again). Inside the fwd+bwd arm, where it spans 0.79-1.07, it orders nothing, and
every fwd+bwd cell beats every dyn-off cell with no overlap whatever the aux term.

**MIL axis (K3, F26).** Four encoders were given bag caches (`swm.eval.mil_cache --scope first`) and swept
with `experiments/run_exp07_mil_sweep.ps1` (one invocation per cell -> `mil_sweep_exp07_<cell>.csv`, since
`mil_sweep` writes only at the end; the notebook merges the parts). MIL-winner, mean-pooling and
`mean_resid` return the **identical** ranking (rho +1.00) - `hann0p3_fbwd` 0.592 / `comb0p3_fbwd` 0.579 /
`hann0p3_off` 0.561 / `comb0p3_off` 0.559 - and `val/recon` is orthogonal to all three (rho **0.00**). The
pooling gain itself is transit-only (+0.042...+0.080) and largely not the encoder's: the untrained arm
gains +0.066 on transit from the same operator. The val-declared winner is unstable across seeds (up to
four distinct operators in four seeds), so any adoption must fix an operator per task.

**Epoch budget.** Median |slope| of `val/recon` over the final ten epochs is 1.3e-4 per epoch across every
ep100 cell, val-train gap -0.002, no overfitting anywhere. Another hundred epochs would buy ~0.013 in
reconstruction, a seventh of what the Hann window returned for free.

Curve-side consequences for exp08 are written up in notebook section K7 and tracked in
`open_questions.md` (Q3, Q4, Q6, Q11, Q12).
