# exp03 — KL-schedule × objective sweep (36 combos, B/seed0) + readout×pooling eval fan

**Plan:** [`docs/plans/2026-07-13-exp03-loss-forensics-and-wide-sweep.md`](../docs/plans/2026-07-13-exp03-loss-forensics-and-wide-sweep.md)
**Aimed by:** [`exp03_forensics/README.md`](exp03_forensics/README.md) (H1–H5 all confirmed: checkpoint selection was 87–95 % clamp-saturated KL noise; every latent dim below the free-bits floor)
**Ran:** training 2026-07-14..16 (user terminal, overnight, two legs), eval fan 2026-07-16 (CC). Window 256 / seq_len 16, variant B / seed 0, junction-reuse of exp01 packed.

## Grid

`free_bits {0, 0.02, 0.05, 0.1} × beta_target {0.1, 0.3, 1.0} × objective {none, logpsd_amp w0.1, combined w0.3}`
= 36 runs, each with **dual checkpoints** (`best.pt` legacy monitor + `best_recon_aux.pt` KL-free `recon + w·aux + λ·dyn`),
max_epochs 60 / patience 10 on both bests. Eval fan (`swm.eval.readout_sweep`, first-segment protocol, untrained arm
capacity-matched per cell): quick scan = 36 combos × 2 ckpts × {logistic, gbm} × mean; full fan = 8 leaders × 2 ckpts ×
{logistic, gbm, mlp} × {mean, max, quantile, window_score}. Ranking `exp03_sweep_summary.csv`, upper-bound rows
`exp03_per_task_best.csv`; per-combo audit `exp03_*/results/readout_sweep.csv`.

## Headline — first linear-probe win of the project

**Winner (v1 protocol, logistic × mean pooling): `exp03_fb0p02_b0p1_lpsd`** (free_bits 0.02, β 0.1, log-PSD w0.1):

| task | trained | untrained | gap |
|---|---|---|---|
| pulsating | **0.822** | 0.767 | **+0.055** (2·SE 0.049 → passes) |
| eb | 0.774 | 0.713 | +0.061 (2·SE 0.041 → passes) |
| rotation | 0.561 | 0.532 | +0.029 (2·SE 0.061 → n.s.) |
| mean gated gap | | | **0.048** |

Paired star-bootstrap (skyline pass on the winner, `exp03_fb0p02_b0p1_lpsd/results/skyline_gate.csv`):
**pulsating and eb gaps individually clear the 2·SE gate** — a first. Additional skyline facts: trained-μ GBM 0.837
now *exceeds* the untrained-μ GBM ceiling 0.824 (SSL no longer degrades μ below a random projection); trained linear
0.822 beats the A1 engineered-features linear ceiling 0.789; only A2 (GBM-on-features 0.858) remains above.
`info_in_mu=False` in the *good* direction: GBM adds only +0.016 over linear on the same μ — the signal is linearly
stored.

- Every prior attempt had linear pulsating gap ≈ 0 or negative (exp01 +0.003, exp02 best −0.005..+0.023 w/ regressions).
  The trained **linear** 0.822 now sits at the untrained-μ-GBM ceiling (0.824) that exp02 could only reach nonlinearly.
- On the winner, GBM-on-μ adds ~nothing over linear (gbm/mean pulsating gap +0.013 vs linear +0.055 — GBM ≈ linear at
  ~0.82): the signal exp02 packed nonlinearly is now **linearly accessible**. The KL schedule, not the readout, was the
  final barrier for pulsating.
- In-grid exp01-replica control (`fb0p1_b1p0_none`, linear/mean): mean gap +0.016, pulsating −0.016 → the lift comes
  from the swept knobs, not the retrain.

## Axis findings (linear/`best_recon_aux`, quick scan)

- **β=1.0 with fb=0 is the worst cell block** (mean gap −0.012); the winning region is **low β (0.1) + small nonzero
  floor (0.02)**. With the floor below true KL, β finally exerts live gradient — and less prior-matching pressure
  preserves more discriminative structure.
- **log-PSD needs the KL fix to pay off linearly**: lpsd at fb0 scores 0.004; at fb 0.02–0.1 it is the best objective
  family (0.028–0.031). exp02 ran lpsd at fb0.1/β1.0 — right objective, wrong KL regime + selection.
- **Dual checkpoint**: net-neutral on average (16+/16−) but systematically positive exactly in the β=1.0 KL-noise
  regime the forensic indicted (e.g. `fb0p1_b1p0_lpsd` +0.028, where legacy best froze at epoch 14 vs 56). Cheap
  insurance; keep `track_recon_aux_best=true` for all future runs.
- 34/36 runs used all 60 epochs (dual-patience never truncated learning — exp02's stop-on-noise pathology gone).

## Readout × pooling (diagnostic until ADR-0008 signed)

- **mlp × {mean, window_score}** cells top the overall board (mean gap up to 0.075) — but mostly via rotation/eb;
  pulsating is often ≈0/negative there. **gbm × max** on the winner is the most balanced nonlinear cell
  (+0.039/+0.040/+0.073).
- **logistic × window_score** gives the largest eb gap anywhere (+0.123) — but note its untrained reference is much
  weaker (0.618 vs 0.713 at mean pooling); absolute trained eb is still ≤ linear/mean (0.741 vs 0.774). Gap and
  absolute rank different cells — report both.
- Per-task-best rows (upper bound, per-task encoder menu): eb +0.123 (winner, log/window_score), pulsating +0.101
  (`fb0p1_b0p1_lpsd`, gbm/window_score, abs 0.846), rotation +0.149, transit +0.104 (transit remains report-only;
  labels are pre-QC v1 — the 2026-07-16 coverage-filter labels v2 are not yet canonical).

## Caveats

- **1 seed.** The pulsating linear gap is volatile across adjacent cells (+0.055 next to −0.042) — the headline is
  hypothesis-grade until a 3-seed confirm of the winner (+ the two runner-up cells) lands. Winner CI: see
  `exp03_fb0p02_b0p1_lpsd/results/skyline_{results,gate}.csv` (paired star-bootstrap).
- Eval used the v1 labels (`processed/subset/subset_tics.parquet`); the Phase-0 QC label changes (transit 888→724)
  are non-canonical pending the roadmap Phase-2 delta and do not affect pulsating/eb/rotation.
- Nonlinear-readout numbers are diagnostic; the v1 headline remains the linear probe (ADR-0008 proposed, pending prof).

## Next

1. **3-seed confirm night**: winner + `fb0_b0p1_comb` + `fb0p05_b0p3_lpsd`, seeds 1–2 (seed 0 exists), same runner
   pattern → error bars on the headline.
2. ADR-0008 discussion with prof, now with a *linear* win in hand (the ask has shrunk: nonlinear readout is no longer
   needed for pulsating — it's an eb/rotation upside question).
3. If confirmed: promote the KL schedule (fb 0.02 / β 0.1 / lpsd w0.1) via ADR; it also motivates re-running the
   deferred cluster plan at these settings.
