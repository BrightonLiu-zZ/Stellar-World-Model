# exp02 — reconstruction-objective sweep (10 combos, B/seed0)

**Plan:** [`docs/plans/2026-07-12-exp02-recon-objective-sweep.md`](../docs/plans/2026-07-12-exp02-recon-objective-sweep.md)
**Ran:** 2026-07-12, variant B / seed 0, window 256 / seq_len 16 (reused exp01 packed via junctions).

## Hypothesis
The skyline suite showed the pulsating frequency signal is discriminative in the data (GBM on LS/PSD features 0.858, GBM on *random-init* μ 0.824) but **absent from the trained μ** (GBM-on-trained-μ 0.767, `info_in_mu`=False): MSE reconstruction squeezes the fast-oscillation variance out of μ. exp02 changes the reconstruction objective to force the encoder to retain that frequency structure.

## Variants (each folder `experiments/exp02*_.../`)
`recon_aux.type` ∈ {log_psd (amplitude / shape-norm), hf_time (high-pass), combined (time+log_psd+hf), masked (denoising)}. Weight bracket order-matched from smoke magnitudes: log_psd/combined {0.1, 0.3}, hf_time {0.3, 0.6}, masked mask_frac {0.15, 0.30}.

## Outcome — mechanism SUCCEEDS, linear probe does NOT

Pulsating bellwether (first-segment protocol; linear = frozen logistic probe, the v1 eval):

| combo | trained (lin) | gap trained−untrained | GBM-on-trained-μ | info_in_mu |
|---|---|---|---|---|
| exp01 baseline (MSE only) | 0.771 | +0.003 | 0.767 | **False** |
| exp02d_combined w0.3 | 0.763 | −0.005 | **0.820** | **True** |
| exp02a_logpsd_amp w0.1 | 0.741 | −0.027 | **0.816** | **True** |
| exp02b_logpsd_shape w0.3 | 0.791 | +0.023 | 0.798 | (lifted) |
| exp02e_masked f0.30 | 0.777 | +0.009 | 0.796 | (lifted) |
| exp02c_hf_time w0.3 | 0.769 | +0.001 | 0.757 | no lift |

Reference ceilings (fixed): untrained-μ→GBM **0.824**, features→GBM **0.858**, untrained linear 0.768.

- **Mechanism criterion MET.** The log-PSD / combined objectives lift GBM-on-trained-μ from 0.767 to **~0.82**, reaching the random-μ ceiling, and flip `info_in_mu` **False→True**: the frequency signal MSE discarded is now retained in μ. This confirms the skyline's Branch-α diagnosis — the objective *was* why μ lacked the signal.
- **Linear-probe criterion FAILED.** The frozen linear probe cannot exploit the now-present signal: the trained−untrained pulsating gap stays ≈0 or goes negative. The signal is packed into μ **nonlinearly**; the harder the objective packs it (combined w0.3, logpsd_amp w0.1 → highest GBM-μ), the *worse* the linear gap. Inverse relationship.
- **Pretrain-once aggregate (mean trained−untrained gap over pulsating/eb/rotation):** NO variant beats the exp01 baseline (0.028); most regress eb slightly, rotation is mixed. On the v1 linear metric there is no clean win. See `exp02_sweep_summary.csv`.
- **hf_time does not lift μ** (GBM-μ 0.757/0.749): a weak high-pass does not enforce spectral fidelity the way an explicit log-PSD term does.

## Verdict — the bottleneck moved from the OBJECTIVE to the READOUT

exp02 delivered its mechanistic goal and **reframes the problem**: the frequency signal is now demonstrably in μ (`info_in_mu`=True, GBM-μ 0.82 ≫ linear 0.77), so the objective is no longer the barrier — the **linear readout is**. The skyline chose Branch α because μ lacked the signal; exp02 fixed that, and Branch β (probe/pooling) resurfaces, now with *positive* evidence that a nonlinear-but-frozen readout would convert. Caveat: v1 CLAUDE.md locks the probe to linear, so this does not convert to a v1 win without an ADR. B/seed0 only; a 3-seed confirm of the `info_in_mu` flip is the natural next step before any writeup.

Per-combo audit: `experiments/exp02*/results/skyline_gate.csv` (has `trained_mu_gbm`, `untrained_mu_gbm`, `info_in_mu`); ranking `experiments/exp02_sweep_summary.csv`; all-window probe tables `experiments/exp02*/results/results_table.csv`.
