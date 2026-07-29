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
window (kmatch4, bag fixed at 4: eb 0.628 → 0.641 → 0.716; pulsating 0.731 → 0.735 → 0.779) — but at
fixed cadence budget the smaller-window geometries earn it back in bag size (16 vs 8 vs 4 windows per
4096 cadences). The two effects cancel almost exactly at the star level. Window length is a
granularity-neutral knob for eb/rotation and finer-is-better for pulsating/transit; it is NOT the
binding constraint.

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

## Consequence for exp07

256×16 stays the geometry of record for the star-level linear probe. The window lever is retired for
v1 star-level scores; the live levers are the objective (phase-aware/STFT aux vs the amplitude-meter
problem), the edge fix (recon_aux confirmed), and consolidation per the ML4PS plan.
