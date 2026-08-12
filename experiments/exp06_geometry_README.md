# exp06 — geometry/coverage axis + training hygiene

Manifest: `experiments/configs/exp06_geometry_coverage.yaml` (single source of truth) ·
Plan: `docs/plans/2026-07-27-exp06-geometry-coverage.md` · Trained 2026-07-28 (43 runs, 0 failures) ·
Eval: `analyze_exp06_geometry_gap.py` → `exp06_geometry_gap.csv` / `exp06_geometry_acf.csv`,
gates `analyze_exp06_gates.py` → `exp06_gates.csv`, edge `exp06_edge_corner.csv`.

## Hypotheses and verdicts

**H1 (coverage — window length is the binding constraint for period-dominated tasks): NOT SUPPORTED.**
All four pre-registered gates fail on the gate arm (dyn-on, amplitude-residualized mean-pool):
Δ(eb)−Δ(pulsating) @512 −0.007±0.020 / @1024 +0.010±0.023; Δ(rotation)−Δ(pulsating) @512 −0.041±0.019
/ @1024 +0.014±0.020. The dyn-off arm shows three nominal differential passes (@1024 eb +0.064,
rotation +0.076; @512 rotation +0.037) but decomposition attributes them entirely to the **control
falling** (pulsating 0.758 → 0.690 from 256→1024) while eb (−0.003) and rotation (+0.008) stay flat —
a control-degradation artifact, not a coverage gain. The w2048 coverage probe (EB coverage >50%) makes
it decisive: every task *drops* at 2048 (eb mean-pool 0.736@256 → 0.680@2048).

**The mechanism the K-matched arm exposes:** per-window representation quality *does* improve with
window (kmatch4, bag fixed at 4, **dyn-off arm**: eb 0.628 → 0.641 → 0.716; pulsating 0.731 → 0.735 →
0.779) — but at fixed cadence budget the smaller-window geometries earn it back in bag size (16 vs 8
vs 4 windows per 4096 cadences). The two effects cancel almost exactly at the star level. Window
length is a granularity-neutral knob for eb/rotation and finer-is-better for pulsating/transit; it is
NOT the binding constraint.

**Every geometry number is arm-specific — always quote the arm.** The dyn-on (fwd_bwd) kmatch4 series
is eb 0.693 → 0.716 → 0.737, and a pooled average over both cells gives 0.660 → 0.679 → 0.726, which
matches neither arm. On the dyn-on arm (the configuration of record) the cancellation reads:

| eb, trained | w256 | w512 | w1024 |
|---|---|---|---|
| `mean`, all windows (16 / 8 / 4) | **0.779** | 0.769 | 0.737 |
| `kmatch4`, bag fixed at 4 | 0.693 | 0.716 | **0.737** |

Bag-size gain at w256 = **+0.086**; window-length gain at fixed bag = **+0.044**. Bag size is worth
about twice window length. The arms agree exactly at w1024 (0.737 = 0.737) because 4096/1024 = 4
windows, so "all windows" *is* four — an internal consistency check on the K-matched implementation.

**H2 (well-posedness pre-gate): rollout eval stays CLOSED at every geometry.** Lag-1 μ-trajectory ACF
on periodic stars: −0.006 (256), −0.083/−0.100 (512), −0.232 (1024) — the trajectory becomes *more*
anti-correlated as the window grows, never approaching the 0.3 threshold. The dynamics axis is
ill-posed across the whole v1 window family, not just at 256. Criterion-2-style rollout physics is
closed for v1.

**C1 replication (fwd_bwd − off, paired, 6 seeds): replicates at 256, weakens with window.**
At 256×16/ep100: eb +0.043±0.007, rotation +0.046±0.011, transit +0.027±0.007 all PASS, pulsating ns —
the exp05 result reproduced under longer training and more seeds. At 512 (λ=18): eb +0.051 and
(new) pulsating +0.034 PASS. At 1024 (λ=18): only eb +0.021 / transit +0.023 on raw mean, nothing on
residualized. The dynamics-term benefit is largest at fine granularity.

**Edge confirmation cell: prediction CONFIRMED.** `comb + free_bits=0.02 + fwd_bwd@66` still spikes
(edge ratio 15.3× vs 17.5× group mean; bias fraction 0.72; trained Jacobian inversion 2.37 vs 0.27
untrained). `free_bits` is exonerated → **`recon_aux` is the immunity knob**, closing the
`.scratch/exp05-window-edge-defect` issue's open question. Fix path for exp07: keep/log-PSD-only aux
or an edge-side decoder fix, with ADR.

## Supporting numbers

- Trained−untrained gap (residualized, dyn-on) is stable across geometries: eb +0.12…+0.13,
  rotation +0.10…+0.14, pulsating +0.04…+0.09 — SSL learning is real at every window; transit goes
  negative at 512/1024 (−0.011/−0.027).
- Capacity caveat: params grow 2.3M → 13.3M with window; every comparison above is against the
  per-geometry capacity-matched untrained reference.
- Standard per-cell rows (logistic+gbm, mean pooling) appended to each cell's
  `results/readout_sweep.csv` by the post-hoc scan.

## Post-meeting reinterpretation (2026-07-30)

Added after the Theissen meeting (record + decisions:
`docs/plans/2026-07-30-theissen-meeting-followups.md`). H1's verdict is unchanged; what changes is
the *mechanism* we attach to it and what remains open.

**The coverage paradox is a measured trade, not an anomaly.** Presented as "more coverage, worse
score" it reads as unexplained. With the K-matched arm above it is a budget constraint: at a fixed
4096-cadence horizon, doubling the window halves the bag, and bag size buys about twice what window
length does. Longer windows are not bad — **we cannot afford them.** Always present the two tables
together.

**Working mechanism: the model does not learn EB periodicity at all.** It appears to flag eclipses as
**rare high-contrast outlier windows** — individual windows unlike that star's own baseline. If it
were measuring a period, coverage should help, and it does not at any geometry. The dispersion-pooling
result corroborates: at w256 dyn-on, `mean_std` lifts transit 0.141 → 0.222, and the lift decays
monotonically with window (0.164 at 512, 0.151 at 1024) exactly as dilution predicts — a longer window
shrinks the eclipse's *share* of the window even as it raises its coverage of the period.

**Pre-registered test of that mechanism (written before looking), pending a period cut from Prof.
Theissen.** Contact binaries touch, so there is no flat stretch between eclipses and the signal is
present in every window. Prediction: on a contact-dominated bin, dispersion pooling should **not**
help and longer windows should **not** hurt — inverted on both levers relative to detached EBs.
Villanova ships `Per` but no morphology flag, so the split is a period proxy at **P < 0.5 d** (the
standard contact/W UMa regime) unless he names another, fixed before the test runs.
**Do not run this as a probe-score split:** only 19 of the 196 test-split EB positives have
P < 0.5 d. Run it as a per-star **window-score dispersion** statistic across all 1,936 EB positives
(182 below 0.5 d) — same mechanism, ~10× the sample, no train/test constraint.

**The `mean_resid` arm is the amplitude evidence, and its question is now closed.** Prof. Theissen
signed off on amplitude-dominance ("definitely okay to be amplitude-dominated") with the standing
condition that it be justified, which **retires the phase-aware/STFT auxiliary loss as an exp07
lever** and makes consolidation the default use of pre-freeze time. The justification owed in its
place is written up in the plan doc §2 D4.

**Window 128 is untested.** The retirement covers *longer* windows only. If it is run, register the
prediction first: per-window quality falls, bag size rises, net ≈ 0 or slightly negative.

## Consequence for exp07

256×16 stays the geometry of record for the star-level linear probe. The window lever is retired for
v1 star-level scores. Of the three live levers listed before the meeting, the objective
(phase-aware/STFT aux vs the amplitude-meter problem) is **withdrawn** — its gate closed against it —
leaving the edge fix (`recon_aux` confirmed) and consolidation per the ML4PS plan. The decoder
window-edge spike is a *window*-boundary decoder artifact and must not be conflated with the
sector-edge clipping suggested in the meeting, which is a separate data-quality change.

## Curve-forensics addendum (2026-08-01)

Added after the cross-experiment W&B curve analysis (`src/notebooks/exp06_diagnostics.ipynb` section I,
figs I2/I4/I5/I6 in `experiments/figs/exp06_diagnostics/`; exp05 curves `exp05_forensics/curves_exp05/`
joined to per-seed probe scores). Nothing above is amended; the "Consequence for exp07" section below
it is **superseded** where noted.

**The A4 dose shortfall is a calibration-timing artifact, not a large-window property.** The dose
trajectories (I4) show λ·dyn/recon still falling 20+ epochs after the epoch-7 pilot readout; the
epoch-7-calibrated cells (w512/w1024 @ λ=18) sink to 0.275/0.342× target while the
steady-state-calibrated ones hold 0.99–1.34×. Back-out λ_needed = λ·target/dose_steady lands at
**49 / 67 / 65 / 53** (exp05_fbwd_c1p0 / w256 / w512 / w1024) — **λ ≈ 50–67 at every geometry** despite
recon and dyn rescaling with window. So "fwd+bwd at target contribution was never tested at
w512/w1024" now has a concrete remedy: pilot ≥ 40 epochs, start fwd+bwd near λ ≈ 60, gate on achieved
dose at eval (lower-bound guide only — raising λ suppresses dyn, the documented undershoot).

**Validation loss anti-selects the probe winners.** Across the ten exp05 comb cells, min val/recon
crowns the dyn-off cell on all four tasks while the probe winner is always a dynamics cell
(Spearman −0.58…−0.84; cost 0.021–0.049 PR-AUC). The λ-free recon+aux selection metric is uncorrelated
cross-cell in exp05 but **does** pick the fwd+bwd winner in exp06 at w256/w512 — exactly the cells
where the dose was delivered — and fails at w1024 where it was not. Within a cell, no logged metric
(recon, recon+aux, n_active final or peak) ranks seeds (|ρ| < 0.3). Method-level statement filed as
F11–F13 in [cross_experiment_findings.md](cross_experiment_findings.md).

**Early-epoch anatomy (closes two curve puzzles).** The ep0→1 "increase" seen in the W&B viewer is
val/aux rising (exp06 median ×1.22, dyn-off cells too) while recon/dyn/KL crash 10–200× — time-MSE
wins the opening trade against the spectral aux; large-window off cells additionally show recon
creeping up ep2–10 under the β-warmup rate–distortion trade. Both benign. Free bits has **no
schedule**: 0.0 in every comb/exp06 run (kl_loss ≡ kl_total holds exactly), 0.02 in the lpsd hedge
where the floor engages at *all* epochs; the 10-epoch knob is the continuous β warmup, so no KL step
at epoch 10 exists to see.

**Replication is clean.** exp05 vs exp06 at 256×16 overlap through the shared 60 epochs on
recon/dyn/KL; exp05's n_active "25 ± 15" was the recruitment transient caught mid-decay (B3), and the
ep100 runs complete the decay to <10 with C1 intact.

### Consequence for exp07 (supersedes the 2026-07-30 paragraph, decision 2026-08-01)

The 2026-07-30 section retired the objective lever on the amplitude-dominance gate. **Per the
2026-08-01 decision the exp07 axis follows where the measured problems are, and every problem this
addendum and section F locate sits in the objective's aux term and the training protocol**: the aux
term is the component whose minimisation co-tracks probe quality (where dose is delivered), the
component that loses the early-training trade, and the confirmed edge-immunity knob; the dose was a
calibration-procedure failure with a concrete fix; and model/seed selection by val loss is measured
invalid. So exp07's live levers are: (1) **objective/aux work** — edge fix through the aux/decoder
side (with ADR), aux-weight and form treated as first-class knobs rather than frozen incidentals;
(2) **λ ≈ 60 steady-state dose calibration with an eval-time dose gate**; (3) **protocol** —
cross-model selection through probe gates only, seed spread reported never selected, epoch budget or
convergence stop for the truncation (A2). Geometry stays retired: 256×16 remains the geometry of
record.

## exp07 pre-check addendum (2026-08-01, eval-only)

Four checks on existing checkpoints and curves, requested by the exp07 design session before the
manifest was written. Nothing above is amended except the λ dial, noted below. Scripts:
`experiments/analyze_exp07_c{1,2,4}_*.py` plus a reuse of `analyze_exp06_geometry_gap.py`; artifacts
`experiments/exp07_c*`; full tables in `tmp/handoff/2026-08-01-exp07-precheck-request.md` §RESULTS.
Two protocol checks passed first: the C2 `comb_off` row reproduces forensics §2 to 4 decimals, and the
C3 untrained arm reproduces `exp06_geometry_gap.csv` exactly.

**C1 — the edge spike is the decoder buying log-PSD power, CONFIRMED.** Forcing the edge residual to
zero *raises* log-PSD by 237–348% on every `comb`+dyn-on arm (time-MSE falls 11–17%; `hf_time` moves
−0.5 to −4%, so the spectral sub-term is the whole effect), against 6.5–10.3% for the same operation at
interior positions — a 27–38x location asymmetry. The amplitude is *tuned*: rescaling the model's own
edge deviation puts the log-PSD minimum at ~1.5x the trained value, with the dyn-off control flat. The
spike is also **not a positive flux bias but an antisymmetric dipole** (−2.95 at p0, +5.11 at p255),
which the MSE-only profiles could not show. Consequence: a Hann-tapered log-PSD is a
mechanism-targeted fix — the purchase becomes unbuyable by construction, since a non-periodic Hann
window is exactly zero at both endpoints. Details and two corrections to the issue's text are filed in
`.scratch/exp05-window-edge-defect/issues/01-*.md` (2026-08-01 comment).

**C2 — the lpsd recipe lowers amplitude dominance but does not escape it.** R² of PC1 on the four
amplitude scalars is 0.964–0.967 under comb against 0.837–0.850 under lpsd, non-overlapping across all
8 seeds. But the variance-weighted amplitude fraction over all 128 dims reads untrained **0.726** →
lpsd 0.738–0.768 → comb-dyn 0.806 → comb-off 0.908: lpsd sits essentially at the random-projection
floor rather than below it. No exp05 recipe goes below that floor. Dynamics weighting also lowers
amplitude dominance within both recipes (comb 0.908 → 0.806), consistent with the collapse-reversal
mechanism. Attribution caveat: `base_lpsd.yaml` moves aux type, aux weight (0.3 → 0.1) **and**
`free_bits` (0.0 → 0.02) together — three knobs, not two.

**C3 — `free_bits` = 0.02 does not move the probe at w256 under comb.** On the `mean_resid` gate arm
nothing clears 2·SE against either the requested `exp06_w256_fbwd` reference or the epoch- and
λ-matched `exp05_comb_fbwd_c1p0` reference (against which `free_bits` is the *only* differing knob).
Deltas are ≤ 0.04 and flip sign across pooling arms. The cell is 1 seed at ep60, so this supports "no
evidence of an effect" and cannot support a positive one (F17).

**C4 — the λ ≈ 60 dial does NOT transfer to the log_psd recipe; this qualifies the addendum above.**
The estimator reproduces the published back-out (`comb_fbwd_c1p0` → 49.2 vs 49). The recon scale is not
the cause: `lpsd_off / comb_off` steady recon ratio is **0.94**. The dyn scale is: at λ = 0 the lpsd
arm's one-step dynamics MSE is **2.11x** the comb arm's, giving an implied dial ratio of 0.94 / 2.11 =
**0.445**, which the matched-multistep back-out independently corroborates at 39.7 / 83.5 = **0.475**.
So a log_psd fwd+bwd cell started at λ ≈ 60 would run at roughly **2x** its intended dose; **start the
log_psd arms near λ ≈ 25–30**. Everything else in the addendum's calibration rule is unchanged (pilot
≥ 40 epochs at steady state, dose gate at eval, expect one calibrate-verify iteration). Caveat: the
measured lpsd cells carry `free_bits` = 0.02, a plausible cause of the dyn inflation on its own, so a
fb = 0 log_psd cell must re-measure its dyn scale in the pilot rather than inherit 0.45x as a constant.

**New falsifiable prediction for the 2x2.** The immune arm's aux weight is 0.1 and every spiky arm's is
0.3. Since the spike is a tuned optimum of a *weighted* term, spike severity should scale with
`recon_aux.weight` — which is one of exp07's own axes. If the weight axis does not move the edge ratio
for both aux forms, immunity is about aux form rather than aux pressure, and the exp05 lpsd-vs-comb
comparison is confounded by weight as well as by `free_bits`.

---

*Cross-reference (added 2026-07-26, append-only — nothing above is amended).* The dispersion-pooling
numbers cited in the post-meeting section come from the MIL/pooling sweep,
[mil_pooling/README.md](mil_pooling/README.md). Method-level findings from that sweep which apply to
this writeup's measurements as well as to every earlier one — the first-segment protocol scoring ~26%
of available windows, the bag-size confound being specific to detection labels, what the K-matched
control can and cannot separate, and the trained−untrained gap not being invariant to the readout —
are collected in [cross_experiment_findings.md](cross_experiment_findings.md).
