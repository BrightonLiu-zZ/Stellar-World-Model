# exp08 — the dynamics ladder (Q4: is the dynamics term doing anything beyond smoothing?)

**Verdict (2026-08-08): yes — and the ladder localizes what.** The eb/rotation benefit and the
feature-complementary latent content both come from **prediction pressure the encoder cannot
trivially satisfy** — delivered equally by the learned GRU, a learned *linear* map, or a *frozen
random* recurrent function at sufficient dose — and **not** by temporal smoothness, which delivers
neither. "Is your world model just a smoothness prior?" is answered no, with a measured control:
smoothness at its maximum satisfiable pressure gains eb +0.019 ± 0.010 / rotation +0.015 ± 0.009
(both ns), while the GRU clears it by >2·SE on both. On the v1 tasks the strong "world model"
reading loses ground — a linear predictor is statistically indistinguishable from the GRU on eb and
rotation — but the ADR-0010 menu restores it: **on every transfer probe the GRU leads the entire
ladder** (numax_hon 0.802 vs best rung 0.755; see the menu section), and it is the only arm that
gains eb/rotation while holding pulsating at off-parity (linear and frozen pay −0.014…−0.018,
>2·SE). One-line mechanism: *unsatisfiable prediction pressure creates the content; the learned
recurrent predictor is what makes it transfer.*

Design + config: `experiments/configs/exp08_dynamics_ladder.yaml` (LOCKED DESIGN block; grilled
2026-08-07). Plan: `docs/plans/2026-08-07-exp08-dynamics-ladder.md`. 36 runs (6 cells × 6 seeds) on
the frozen `hann0p3` recipe; `exp07_hann0p3_{off,fbwd}` (6 seeds) reused as the ladder ends.

## The ladder (mean_resid pooling, 6 seeds, seed-paired; `exp08_ladder_gap.csv`)

| task | off | smooth_lo | linear | frozen_lo | frozen@22 | fwd_bwd | untrained |
|---|---|---|---|---|---|---|---|
| eb | 0.510 | 0.529 | **0.575** | 0.536 | **0.577** | 0.590 | 0.455 |
| rotation | 0.386 | 0.401 | **0.421** | 0.401 | **0.437** | 0.431 | 0.351 |
| pulsating | 0.800 | 0.799 | 0.786 | 0.775 | 0.783 | 0.801 | 0.705 |
| transit | 0.146 | 0.150 | 0.158 | 0.141 | 0.143 | 0.166 | 0.150 |

Key paired deltas (>2·SE marked \*): linear − off eb **+0.065 ± 0.015\***, rotation **+0.035 ±
0.014\***; linear − fbwd ns on both. frozen@22 − off eb **+0.067 ± 0.012\***, rotation **+0.051 ±
0.007\***; frozen@22 − fbwd ns. Collapsed smooth@270 actively hurts (eb −0.073, rotation −0.078 vs
off; at/below the untrained floor).

## Pre-registered gates

- **G-prior (smooth − off > 2·SE on eb AND rotation): FAIL** — +0.019 ± 0.010 / +0.015 ± 0.009 on
  the fair `smooth_lo` arm. Right direction, under the bar on both required tasks.
- **G-gru (fbwd − smooth > 2·SE on eb OR rotation): PASS** — +0.062 ± 0.010 and +0.030 ± 0.011.
- **G-dose:** `linear` PASS (0.978 ± 0.13). Every non-learned arm failed the gate, and that is a
  *finding*, not a calibration miss — see below.

## The dose gate became a result

- **The smoothness prior saturates.** λ 15 → 270 (×18) made the achieved dose *fall* 0.055 → 0.007,
  because the encoder complies by collapsing the latent to **1 active unit** (all 6 seeds; mu_var
  halves already at λ15). Dose parity with the GRU term is **unreachable**: a satisfiable objective
  cannot hold a loss floor. `smooth_lo`@15 = max pressure without collapse (dose 0.055, 4 units) is
  therefore the fair smoothness arm; `smooth`@270 and `smooth_half`@135 are kept as the
  saturation/collapse documentation.
- **The frozen-random-GRU term is bistable around the collapse transition.** λ9 → dose 0.401 with
  the latent collapsed to 4–5 units; λ22 → dose 2.37 with the latent held **wide open at 69–114
  units** (unattainable targets block pruning). No λ stably delivers dose 1.0. Deviation from the
  pre-registered recalibrate-rule, stated: instead of a third iteration onto a knife edge, frozen is
  reported at **two dose points bracketing the target (0.40 / 2.37)** — and they split: the
  collapsed arm probes at smooth level, the wide arm at fbwd level. Pressure-without-learning
  reproduces the benefit **only in the phase where it holds the latent wide**.
- Only the **learned** predictors (linear, GRU) sit stably at dose ≈ 1: a learnable map tracks the
  latent and maintains an unsatisfiable-but-stable error floor.

## P5 — the mechanism signature travels with the benefit (`exp08_signature_*.csv`)

CHK-4 protocol per arm vs `off` (CCA vs same-cell seed null, ⊥-residual asymmetry, fusion delta):

| arm | resid-var asymmetry (GRU ref ≈10×) | fusion deltas > 2·SE | eb residual probe (arm⊥off / off⊥arm) |
|---|---|---|---|
| smooth_lo | **0.24× (inverted)** | pulsating only | 0.213 / 0.205 — symmetric |
| frozen_lo | 1.06× (none) | eb, pulsating (off-like) | 0.212 / 0.185 |
| linear | **2.9×** | eb, pulsating, transit | 0.304 / 0.156 |
| frozen@22 | **12.8× (full GRU level)** | eb, pulsating, transit | 0.260 / 0.171 |

The signature ordering equals the probe ordering, arm for arm. The private, feature-complementary
directions Q11 attributed to "the dynamics term" are produced by any hard prediction target — the
learned-GRU is sufficient, not necessary. (smooth_lo's CCA tail does separate from the seed null,
but its private directions carry nothing: the residual probe is symmetric — CCA alone cannot
distinguish "different" from "differently useful"; the asymmetry can.)

## Predictions P1–P5 (pre-registered in the manifest), scored

| # | prediction | outcome |
|---|---|---|
| P1 | G-prior passes on eb | **FAIL** (+0.019 at 1.9 SE) |
| P2 | G-gru passes on ≥1 of eb/rotation | **PASS** (both) |
| P3 | eb ladder monotone off ≤ smooth ≤ linear ≤ fbwd | **PASS** on the fair arms (0.510 ≤ 0.529 ≤ 0.575 ≤ 0.590); the collapsed smooth arm breaks it below off |
| P4 | frozen tracks smooth, not fbwd | **SPLIT — phase-dependent**: collapsed frozen_lo tracks smooth; wide frozen@22 tracks fbwd. The discriminator returned a sharper answer than either branch: learning is unnecessary, *unsatisfiability at scale* is what matters |
| P5 | only learned arms reproduce the signature | **PARTIAL FAIL in the same direction as P4**: linear yes, but wide-frozen reproduces it fully (12.8×) — the signature follows the benefit, not the learning |

## Outcome-matrix application (fixed before any run)

"Smooth loses probes (G-gru passes, G-prior fails)" → **the world-model framing survives the
smoothness objection with a measured control** — amended by the ladder: the honest description of
the mechanism is *predictive-target pressure* rather than *learned recurrent dynamics*. Suggested
paper phrasing: the auxiliary prediction task is the active ingredient; the GRU is one (and the
best-rounded) implementation of it, uniquely holding pulsating at par.

## ADR-0010 downstream menu (reported, not gated) — the GRU is NOT replaceable out of distribution

`exp08_ladder_menu.csv` (mean pooling, seed mean ± sd; references from
`exp08_prechecks/new_task_scorecard.csv`):

| probe | off | smooth_lo | linear | frozen_lo | frozen@22 | **fwd_bwd** | untrained |
|---|---|---|---|---|---|---|---|
| numax_hon (R²) | 0.704 | 0.734 | 0.739 ± .087 | 0.755 | 0.739 | **0.802 ± .003** | 0.425 |
| rotation_period (R²) | 0.537 | 0.568 | 0.604 ± .123 | 0.559 | 0.523 | **0.677 ± .010** | 0.416 |
| osc_giant | 0.802 | 0.815 | 0.839 | 0.814 | 0.810 | **0.854** | 0.706 |
| solar_like_osc | 0.287 | 0.290 | 0.297 | 0.290 | 0.304 | **0.336** | 0.205 |
| rgb_vs_heb | 0.804 | 0.794 | 0.799 | 0.793 | 0.801 | **0.825** | 0.724 |
| ijspeert | 0.389 | 0.400 | 0.429 | 0.419 | 0.440 | **0.440** | 0.388 |
| flare (null control) | 0.441 | 0.452 | 0.441 | 0.452 | 0.448 | 0.453 | 0.418 |

Three readings:

1. **The v1 equivalence does not transfer.** On every asteroseismic/transfer probe `fwd_bwd` leads
   the whole ladder — numax_hon +0.047…+0.063 over the best rung (>2·SE vs both frozen arms and vs
   off; ~1.8·SE vs linear only because linear's seed spread is huge), rotation_period +0.073,
   solar_like_osc +0.032 — with a consistent sign across the block. The learned recurrent predictor
   is replaceable for in-distribution v1 probes and **not replaceable for transfer**.
2. **Linear is unstable out of distribution** (numax sd 0.087, rotation_period sd 0.123 — the same
   4–29 active-unit seed instability its dose-gate row showed). The GRU's sd on numax is 0.003.
3. **The collapsed smooth arms are catastrophic on the menu** — numax 0.287, *far below the
   untrained floor* (0.425): over-pressured slowness destroys transferable content, not just probe
   score.

This section amends the verdict's emphasis rather than its logic: the smoothness objection stays
answered (G-prior FAIL), the mechanism stays "unsatisfiable prediction pressure" (v1 + signature),
**and the learned recurrent implementation earns its keep on the downstream menu** — which is where
the ML4PS transfer story lives. The GRU is not merely best-rounded; it is the only arm whose gains
survive out of distribution.

## Deviations from the locked design, complete list

1. `smooth_half` lost its dose-0.5 meaning (both smooth cells sit at the saturation floor); kept as
   a saturation replicate.
2. Frozen arm reported at a dose bracket (0.40 / 2.37) instead of a third λ iteration (bistability
   measured; knife edge).
3. Two calibration cells (`smooth_lo`, `frozen_lo`) added after the dose gate, user-approved
   2026-08-08.

## Artifacts

- Curves: `exp08_forensics/curves_exp08/` (wave-1 pilots overwritten by wave-2; W&B holds both).
- Probe scores: `exp08_diag{,_lo}_probe_summary.csv`, per-star `exp08_diag{,_lo}_star_scores.parquet`
  (estimator identical to `exp07_aux_gap_6seed.csv`, shared mu cache `exp07_forensics/mu_cache/`).
- Ladder deltas: `exp08_ladder_gap.csv` (`analyze_exp08_ladder.py`).
- Signature: `exp08_signature_{channel_probe,dynunits_cca,dynunits_residual,dynunits_scaling}.csv`
  (`analyze_exp07_mu_channel.py --pairs`, extended this experiment with `--pairs`/`--no-feature-map`).
- Menu: `exp08_ladder_menu.csv`.
