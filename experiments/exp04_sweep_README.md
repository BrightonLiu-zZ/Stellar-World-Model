# exp04 — 3-seed confirm + encoder axis + KL-corner cross: results

**Trained overnight 2026-07-19→20** (39/39 runs incl. bonus, 0 failures); evaluated 2026-07-20.
Plan: [docs/plans/2026-07-19-exp04-confirm-encoder-kl.md](../docs/plans/2026-07-19-exp04-confirm-encoder-kl.md).
Protocol: `best_recon_aux` only, logistic+gbm × mean pooling (quick scan), v1 labels; each encoder
variant gapped against its own **capacity-matched** untrained reference
(`exp04_eval_cache/<variant>/`). Data files: `exp04_sweep_summary.csv` (seed-aggregated scan),
`exp04_transit_window_eval.csv` + `exp04_transit_probe_agg.csv` (KP+CP transit probe),
`exp04_forensics/` (per-dim KL), per-exp `results/readout_sweep.csv` and the winner's
`results/skyline_results.csv` (3 seeds, `ckpt=best_recon_aux`).

## H-confirm — the exp03 headline SPLITS (3 seeds, logistic × mean, gap mean ± SD)

| cell | pulsating | eb | rotation | transit |
|---|---|---|---|---|
| `exp03_fb0p02_b0p1_lpsd` (exp03 winner) | +0.016 ± 0.034 ✗ | **+0.066 ± 0.006 ✓** | +0.027 ± 0.003 ✓ | +0.039 ± 0.003 ✓ |
| `exp03_fb0_b0p1_comb` | **+0.045 ± 0.015 ✓** | +0.050 ± 0.019 ✓ | +0.017 ± 0.002 ✓ | +0.049 ± 0.014 ✓ |
| `exp03_fb0p05_b0p3_lpsd` | +0.026 ± 0.028 ✗ | +0.055 ± 0.006 ✓ | +0.040 ± 0.014 ✓ | +0.040 ± 0.007 ✓ |

✓ = mean > 2·SE. **The winner's pulsating +0.055 was a seed-0 fluke** (seeds 1/2: −0.008, +0.001);
its **eb win is real and tight** (0.061/0.064/0.073). `fb0_b0p1_comb` (free-bits **0**, β 0.1,
combined aux) is the only cell where **all four tasks** confirm — the most robust recipe overall,
and the only 3-seed-confirmed pulsating win.

## H-latent / encoder axis — smaller latent (or width) is a large eb win

Logistic × mean, gap mean across seeds vs capacity-matched untrained (2 seeds unless noted):

| variant | eb | pulsating | transit | note |
|---|---|---|---|---|
| `enc_whalf` (channels ÷2) | **+0.100** | +0.003 | +0.054 | no KL concentration (max dim 0.10 nats) — compresses via width |
| `enc_z32` (3 seeds) | **+0.099** | −0.031 | +0.015 | KL concentrates 3×: 0.067 nats/dim vs 0.023 at z128 |
| `enc_z64` (3 seeds) | +0.080 | +0.010 | +0.035 | intermediate (0.038 nats/dim) |
| `enc_d3` / `enc_w2x` | +0.077 / +0.074 | −0.035 / +0.016 | +0.033 / +0.036 | |
| `enc_d5` | +0.039 | −0.063 | **+0.071** | deeper/bigger receptive field helps transit (H-receptive, weak) |
| `enc_k9` / `enc_k15` / `enc_z64k9` | +0.046 / +0.067 / +0.042 | ≈0 | +0.033 / +0.033 / +0.027 | kernel size alone does little |

## H-corner / H-underrun

- eb gap positive in every corner cell (+0.04…+0.07); pulsating volatile everywhere (best corner
  seed-0 `fb0p02_b0p05` +0.062 → seed 1 +0.018). β=0.05 keeps the latent liveliest (KL 4.4 nats,
  max dim 1.09; `kl_dim_report`) but does not stabilize pulsating. **No corner beats the swept-grid
  edge decisively — the optimum is not beyond the corner.**
- `winner_ep100`: eb +0.074 vs +0.061 at 60 epochs (seed 0) — mild under-run confirmed for eb;
  pulsating did not benefit.

## Skyline gate (winner, 3 seeds, `best_recon_aux`, B1 reused from the exp03-era run)

- **eb: headroom CLOSED** — A1(engineered) − untrained = 0.029 < 2·SE 0.041, and GBM-on-μ no longer
  beats the linear probe (info_in_mu = False, all seeds). The +0.066 linear eb win sits at the
  engineered-feature skyline: **eb is done at this geometry.**
- **transit: headroom_real = True AND info_in_mu = True on all 3 seeds** (GBM-on-μ − linear
  = +0.08…+0.10) — μ holds transit signal the locked linear readout cannot express.
- pulsating / rotation: A1 itself barely beats untrained (no significant headroom to chase).

## KP+CP strict transit probe (the 217-star knob; `exp04_transit_window_eval.csv`)

Setup: window MIL readout on cached first-segment μ; fit modes broadcast / true-loose / true-strict;
strict = full-transit-contained, {CP,KP} only (1,751 windows / 217 stars; test = 32 positive stars,
90 non-strict transit stars quarantined at star level — grill 2026-07-20 Q2/Q3).

- **Strict star metric is far more separable than v1** (logistic, broadcast fit): trained arms reach
  `star_pr_auc_kpcp` 0.15–0.19 vs untrained ≈ 0.02–0.04 (gap +0.12…+0.16, ~12× the 1.4% base rate);
  the same arms gap only +0.04…+0.08 on `star_pr_auc_v1`. Confirmed/known-planet hosts are the
  separable core of the transit class; PC-only stars supply most of the v1 hardness.
- **Fitting ON strict labels hurts star-level detection everywhere** (`true_kpcp` fit ≤ broadcast/true
  fits on every star metric): 1,401 strict train windows are too few to train the window readout.
  The knob's value is as an **evaluation** target, not a training set — with n_test_pos = 32, treat
  per-arm differences cautiously.
- Strict-fit readouts do win the strict **window-level** metric they were trained for
  (`win_pr_auc_full_kpcp`, e.g. β=0.05 corner cells 0.11–0.14 vs untrained 0.02–0.04) — the
  in-transit-window signal is in μ, consistent with the skyline transit verdict.
- Best arms on the strict star metric (broadcast fit): `enc_w2x` 0.190, `enc_k9` 0.175,
  `fb0_b0p1_comb` 0.174, `fb0p01_b0p05` 0.174; the exp03 winner is mediocre (0.105).

## Leader fan (seed 0, full pooling × readout; diagnostic tier, ADR-0008 still pending)

- `window_score × logistic` **doubles** the winner's eb gap: +0.123 (and `fb0_b0p1_comb` +0.097,
  `enc_whalf` +0.101) vs +0.06…+0.10 at mean pooling — MIL pooling is the right eval shape for eb.
- No pooling/readout combination rescues pulsating (best remains `fb0_b0p1_comb` mean × logistic).
- Transit's best cells are nonlinear/max tiers (+0.08…+0.11), matching info_in_mu.

## Verdict / recommended next steps

1. **Promote `fb0_b0p1_comb` (fb 0, β 0.1, combined aux) to reference recipe** — the only
   all-four-task 3-seed confirm; the exp03 winner keeps the best single-task eb but its pulsating
   headline did not replicate. Needs an ADR before locking.
2. eb at this geometry is solved to the skyline; stop tuning for it. The cheap eb upgrades
   (`enc_whalf` / `enc_z32`, `window_score` pooling) are available if eb ever becomes the headline.
3. Transit is where trapped signal remains: μ beats the linear probe by +0.08–0.10 (all seeds) and
   the KP+CP core is highly separable. The lever is the readout/pooling protocol (ADR-0008
   decision), not more SSL tuning.
4. Pulsating: no KL/encoder knob in this grid stabilizes it; treat +0.05-class pulsating gaps as
   seed noise until a mechanism is found (candidate exp05 alongside fwd+bwd prediction).
