# exp04 MIL (window_score x logistic) 3-seed confirm

Handoff task 3, grill 2026-07-22. Does the exp04 seed-0 leader-fan MIL advantage (eb +0.123 winner,
+0.097 comb) survive 3 seeds? Scored by `experiments/analyze_mil_confirm.py` on the cached
first-segment window-mu (`best_recon_aux`, v1 labels), reusing `readout_sweep.score_cells` verbatim; the
untrained MIL reference is the capacity-matched z128 `exp03_eval_cache` arm. Machine-readable:
`exp04_mil_confirm.csv` (aggregated) + `exp04_mil_confirm_perseed.csv` (raw per-seed). Feeds
ADR-0008-lite; the linear-probe mean-pooling probe stays the v1 headline.

## #1 MIL gap vs untrained (gap mean +/- SD over 3 seeds; ✓ = mean > 2*SE)

| cell | pulsating | eb | rotation | transit |
|---|---|---|---|---|
| `exp03_fb0_b0p1_comb` | -0.015 ± 0.018 ✗ | +0.077 ± 0.019 ✓ | +0.032 ± 0.010 ✓ | +0.072 ± 0.010 ✓ |
| `exp03_fb0p02_b0p1_lpsd` | +0.011 ± 0.007 ✓ | +0.119 ± 0.009 ✓ | +0.020 ± 0.010 ✓ | +0.054 ± 0.010 ✓ |

Headline: exp03_fb0_b0p1_comb eb +0.077 ± 0.019 (✓); exp03_fb0p02_b0p1_lpsd eb +0.119 ± 0.009 (✓).

## Mean-pooling reference (same arms/seeds, the v1 headline protocol)

| cell | pulsating | eb | rotation | transit |
|---|---|---|---|---|
| `exp03_fb0_b0p1_comb` | +0.045 ± 0.015 ✓ | +0.050 ± 0.018 ✓ | +0.017 ± 0.002 ✓ | +0.049 ± 0.014 ✓ |
| `exp03_fb0p02_b0p1_lpsd` | +0.016 ± 0.034 ✗ | +0.066 ± 0.006 ✓ | +0.027 ± 0.003 ✓ | +0.039 ± 0.003 ✓ |

## #2 MIL is the better eval shape? paired (window_score - mean) gap delta, mean +/- SD over 3 seeds

| cell | pulsating | eb | rotation | transit |
|---|---|---|---|---|
| `exp03_fb0_b0p1_comb` | -0.060 ± 0.004 | +0.027 ± 0.005 | +0.015 ± 0.011 | +0.023 ± 0.010 |
| `exp03_fb0p02_b0p1_lpsd` | -0.005 ± 0.033 | +0.053 ± 0.008 | -0.007 ± 0.008 | +0.015 ± 0.010 |

## Notes

- Per-seed raw gaps are in `exp04_mil_confirm.csv` (`gap_s0/s1/s2`); showing the spread is the point
  (per-seed transparency is what caught the exp04 pulsating fluke).
- Untrained MIL reference is seed-0 only (geometry-shared), so SD reflects trained-seed variance, exactly
  as the exp04 H-confirm mean-pooling table did.
- eb is skyline-closed (exp04); MIL is an eval-shape gain, not a new SSL win.
