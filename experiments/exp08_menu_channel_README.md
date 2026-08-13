# R1 — the fusion claim on the ADR-0010 downstream menu (2026-08-11)

`experiments/analyze_exp08_menu_channel.py` → `experiments/exp08_menu_channel/menu_channel_{probe,summary,repro}.csv`
+ `menu_channel_star_scores.parquet`. Roadmap row R1 (`docs/plans/2026-08-11-pre-freeze-roadmap.md`);
closes the fusion loose end in `experiments/open_questions.md`.

## What was measured

The ML4PS framing's central claim — `features ⊕ µ` beats `features` alone — existed only on the four
v1 tasks (`exp07_channel_probe.csv`, `exp08_signature_channel_probe.csv`) while being printed beside
the seven-probe downstream menu, which carried plain probes and no fusion readout at all. This runs the
fusion readout on that menu.

Five readouts per probe, mirroring the v1 `channel_probes()`: `mu`, `mu_resid_amp`, `mu_resid_full`,
`features_only`, `features_plus_mu`. Arms: `hann0p3_fbwd` and `hann0p3_off`, 6 seeds each, plus the
single-init `untrained` reference. Pooling is `mean` (the menu convention) on every row; no MIL.
Each readout is packed one row per star and pushed through `new_task_scorecard`'s own scorers, so every
keep-mask is byte-identical to the frozen-mu scorecard rather than re-derived.

Estimator: fusion delta paired per seed within an arm, never pooled across arms (F21); the arm contrast
`delta_fbwd − delta_off` carries both arms' spreads (F17); paired star-level bootstrap (2,000 resamples,
same resampled star index for every readout) on probes with `n_test < 400` or `n_pos < 100`. Because
`features_only` carries no seed index, the fusion delta's spread is the µ side's alone — stated, not
dressed up as a two-sided SE.

## Controls (all exact)

| control | rows | max abs diff |
|---|---|---|
| `mu` vs `exp08_prechecks/new_task_scorecard.csv` (pins the re-extraction to the right checkpoints) | 143 | 8.3e-17 |
| `features_only` vs `exp08_prechecks/ceiling_A1A2.csv` A1 rows (pins the cached feature table) | 11 | 5.6e-17 |
| v1 fusion rows, `analyze_exp07_mu_channel.py` hann0p3 seed 0 vs `exp07_channel_probe.csv` | 40 | 0.0 |

The v1 control is a seed-0, one-recipe rerun rather than the full 2×6 fan; it exercises the same code
path end to end.

## Verdict — the fusion claim does NOT generalise wholesale; it generalises where the engineered baseline is weak

`features ⊕ µ` − `features`, ADR-0010 menu, 6 seeds per arm, pooling `mean`:

| probe | n_test (pos) | features alone | hann0p3_fbwd | hann0p3_off | untrained | fbwd claimable |
|---|---|---|---|---|---|---|
| solar_like_osc | 3429 (417) | 0.320 | **+0.071 ± 0.006** | +0.024 ± 0.011 | −0.007 | yes |
| numax_hon | 1313 | 0.831 | **+0.036 ± 0.002** | +0.010 ± 0.004 | −0.002 | yes |
| flare | 3429 (304) | 0.474 | +0.051 ± 0.008 | +0.033 ± 0.007 | +0.006 | yes, but `flare` is the stated null and is reported, never claimed |
| rotation_period | 150 | 0.703 | +0.013 ± 0.010 | −0.005 | −0.057 | no, bootstrap CI spans 0 |
| osc_giant | 3429 (1313) | 0.920 | −0.003 ± 0.002 | −0.006 | −0.003 | no, negative |
| ijspeert | 2021 (93) | 0.508 | −0.054 ± 0.029 | −0.082 | −0.096 | no, negative |
| rgb_vs_heb | 161 (113) | 0.758 | −0.051 ± 0.019 | −0.052 | −0.038 | no, negative and the CI excludes 0 |

Two clean positives on the frozen recipe's dynamics arm (`solar_like_osc`, `numax_hon`), one on the
declared null (`flare`), one null, three negatives. Where fusion costs score, the feature baseline is
already strong (`osc_giant` 0.920) or the probe is small (`rgb_vs_heb` 161 stars, `ijspeert` 93
positives) — and the untrained arm loses there by the same or a larger margin, which is the signature of
128 extra collinear columns diluting a linear readout rather than of µ carrying misleading content.

**The dynamics-specificity DOES generalise.** `delta_fbwd − delta_off`, paired per seed (both spreads):

| probe | delta of deltas | > 2·SE |
|---|---|---|
| solar_like_osc | +0.047 ± 0.012 | yes |
| ijspeert | +0.028 ± 0.026 | yes |
| numax_hon | +0.025 ± 0.005 | yes |
| rotation_period | +0.019 ± 0.018 | yes |
| flare | +0.018 ± 0.013 | yes |
| osc_giant | +0.003 ± 0.004 | no |
| rgb_vs_heb | +0.002 ± 0.030 | no |

Five of seven, including two probes where the fusion level itself is negative: the dynamics term buys
complementarity to the engineered basis on the transfer menu exactly as it does on v1, even where that
complementarity is not enough to pay for the extra columns.

**µ is not a re-encoding of the engineered basis here either.** With all 25 features projected out
(fit on train only), `mu_resid_full` still scores, and it is graded by arm:

| probe | metric | fbwd | off | untrained |
|---|---|---|---|---|
| numax_hon | R² | 0.496 | 0.206 | 0.074 |
| osc_giant | PR-AUC | 0.535 | 0.479 | 0.433 |
| solar_like_osc | PR-AUC | 0.229 | 0.189 | 0.140 |
| flare | PR-AUC | 0.222 | 0.165 | 0.120 |
| ijspeert | PR-AUC | 0.163 | 0.129 | 0.068 |
| rotation_period | R² | 0.105 | 0.069 | 0.106 |
| rgb_vs_heb | ROC-AUC | 0.533 | 0.532 | 0.558 |

The `numax_hon` row is the strongest form of the claim anywhere in the project: after removing
everything 25 engineered features can linearly say about µ, the dynamics arm still predicts log νmax at
R² 0.50, the dyn-off arm at 0.21, an untrained encoder at 0.07.

Standing result unchanged: `features` alone still beat `µ` alone on 6 of 7 probes (the exception is
`solar_like_osc`, fbwd +0.016). "Engineered features beat SSL alone" and "SSL adds to engineered
features on some probes" remain simultaneously true.

## Consequence for the paper (flagged for R4)

The fusion claim must be **scoped in print**, not stated globally. Defensible wording: fusion helps on
v1 (all four tasks, dynamics-specific) and on the asteroseismic block of the transfer menu
(`solar_like_osc`, `numax_hon`); it is neutral-to-negative where the engineered baseline is already
strong or the probe is small, and that pattern is reproduced by the untrained control, so it reads as a
readout-capacity cost rather than a representation failure. The dynamics-specificity of the complementarity
generalises on 5 of 7 probes and is the more robust half of the result.

## Artifacts and cleanup

- Kept: `experiments/exp08_menu_features_{pool,subset}.parquet` (the A1-identical engineered tables; R16
  reads them), all CSVs and the per-star parquet under `experiments/exp08_menu_channel/`.
- Cleanup candidates: `experiments/exp08_menu_channel/{mu_cache,subset_mu_cache}` (3.5 GB, 13 arms) —
  keep until R16 (C2) has run, since it needs the same µ; the prechecks run's delete-then-need-again is
  why this extraction had to be repeated at all.
