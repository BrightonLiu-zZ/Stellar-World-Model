# Dynamics term causes a ~17x reconstruction-error blowup at window edges (comb recipe only)

Status: ready-for-human
Found: 2026-07-26, exp05 diagnostics notebook (§C1b)
Affects: `exp05_comb_*` with `train.lambda_dyn > 0` — 36 of 48 exp05 runs
Blocks: exp06 planning (do not commit further to `comb` + dynamics until resolved)

## Summary

Enabling the latent-dynamics loss term makes the decoder's reconstruction fail catastrophically at
the **first position inside every 256-cadence window**: mean squared error at `p=0` is **~17x the
interior MSE**, versus **1.7x** when the dynamics term is off. The defect is inside `recon_loss`, so
it is a genuine training-side pathology — not the cosmetic cross-window seam discontinuity that
appears when the notebook concatenates decoded windows for display.

It consumes **~8-9% of the entire reconstruction loss** for the `c1p0` arms.

## Evidence

Measured over 2,000 random test windows per run, all 48 exp05 runs (notebook §C1b, reproducible via
`src/notebooks/exp05_diagnostics.ipynb`).

| group | runs | edge ratio (p0 MSE / interior MSE) |
|---|---|---|
| `comb`, dynamics **off** | 4 | **1.7x** |
| `comb`, dynamics **on** | 36 | **17.5x** (sd 11.0) |
| `lpsd`, dynamics off | 4 | 2.0x |
| `lpsd`, dynamics **on** | 4 | **2.1x** |

Per-position profile for `comb_fbwd_c1p0` seed 0: MSE 24.1 / 6.5 / 4.7 at p=0,1,2, settling to the
interior (~0.88) by p≈4. Both edges are affected — `comb_off` is worse at the *trailing* edge (9.5)
than the leading one (2.0), while dynamics-on arms are worse at the leading edge.

The decoder's first sample is **essentially input-independent**: `corr(recon, input)` = **0.12 at
p=0** versus **0.75 at mid-window**, while its spread (sd 1.9) exceeds the input's (1.16). It emits a
large value that is not reading the data.

## What it is NOT

Three candidate explanations were tested and rejected:

- **Not a lambda dose-response.** Within the 36 `comb`+dynamics runs, Spearman(lambda, edge ratio) =
  **-0.016, p = 0.93**. Even the smallest lambda (7, achieved contribution 0.13) triggers the full
  blowup. It is a switch, not a dial.
- **Not explained by latent collapse severity.** Spearman(n_active, edge ratio) within `comb`+dynamics
  = **-0.176, p = 0.31**.
- **Not architectural.** The untrained encoder's error profile is flat (~1.5) across the whole window
  with no edge structure, so the `ConvTranspose1d(kernel=4, stride=2, padding=1)` stack does not
  produce this on its own. The artifact is learned.

Statistical support: `comb` dynamics ON vs OFF, Mann-Whitney **p = 0.0012**; `comb` vs `lpsd` with
dynamics on, **p = 0.0021**.

## Root cause: unresolved, and exp05 cannot resolve it

`comb` and `lpsd` differ in **two** ways simultaneously:

| | `base_comb.yaml` | `base_lpsd.yaml` |
|---|---|---|
| `train.free_bits` | 0.0 | 0.02 |
| `train.recon_aux.type` | comb | log_psd |

`lpsd` is immune. No reinterpretation of the existing 48 runs can say which of the two knobs confers
immunity — it needs one targeted run.

## Proposed experiment (2 cells x 2 seeds, eval-only afterwards)

Disambiguate with a 2x2 corner, holding `model.dyn_mode=fwd_bwd` and the lambda that hits contribution
1.0 fixed:

1. `recon_aux=comb` + `free_bits=0.02` + dynamics on — does the KL floor alone fix it?
2. `recon_aux=log_psd` + `free_bits=0.0` + dynamics on — does the aux term alone fix it?

Cells 3 and 4 of the corner already exist (`exp05_comb_fbwd_c1p0` = broken, `exp05_lpsd_multi_c1p0` =
clean), though note cell 4 is `multistep` not `fwd_bwd`, so a matched `lpsd_fbwd_c1p0` would make the
corner exact.

**This is a training run — hand the command to the user's terminal per the repo rule; do not
background it in Claude Code.**

## Impact assessment (already done — does not block reading exp05)

Criterion 1 stands and is if anything **conservative**:

- The probe reads mu from the **encoder**; the defect is in the **decoder** output.
- Checkpoint selection is not visibly distorted — §A6 gives Spearman(lambda, selected epoch) =
  +0.155, p = 0.29.
- But every dynamics-on `comb` arm carries an ~8-9% reconstruction-loss handicap that the lambda=0
  baseline does not, so the paired criterion-1 gain is won *despite* the handicap.

## Related

- Notebook §C1b — the live diagnostic panel and the 48-run table.
- Notebook §C1 — where the symptom first became visible (spikes at multiples of 256).
- Incidental: the notebook loads checkpoints with `strict=False`, which silently drops six
  `dynamics_bwd.*` keys for fwd+bwd arms. Harmless here (`missing_keys` is empty), but it would also
  silently swallow a genuine key mismatch. Worth tightening separately.

## Comments

### 2026-07-27 - root cause largely identified without the 2x2; corner reduced to one cell

`experiments/analyze_exp06_edge.py` + `src/notebooks/exp06_design_forensics.ipynb` section 1 add four
measurements over all 48 runs. The group edge ratios reproduce this issue's table exactly
(1.72 / 17.54 / 2.02 / 2.11), so the new script is measuring the same object.

**1. The spike is mostly a learned constant.** Splitting the p=0 error into a part identical for every
star and a part that varies (`mean(err)^2 / mean(err^2)`): **0.66** for `comb`+dynamics-on, vs 0.19
(`comb` off), 0.31 (`lpsd` on), 0.06 (untrained). Two thirds of it is a fixed offset the decoder emits
regardless of input, which is why it has no lambda dose-response - a bias term does not need a dose.

**2. It IS architectural in origin, contrary to this issue's "not architectural" line.** That verdict
rested on untrained MSE being flat, which is a weak instrument (random weights are small). The decoder
Jacobian by position, `||d recon[p] / dz||`, measures it directly:

| group | p0 / interior |
|---|---|
| untrained (architecture alone) | **0.27** |
| `lpsd`, dyn on | 0.73 |
| `comb`, dyn off | 0.90 |
| `comb`, dyn on | **1.76** |

At random init the first output sample is **3.6x less** steerable from the latent than an interior one -
the `ConvTranspose1d(k=4, s=2, p=1)` boundary is assembled from fewer taps. The architecture supplies a
weak spot; the dynamics term then overshoots it by 6.5x and parks a bias there. So: architecture supplies
the location, training supplies the spike.

**3. `recon_aux`, not `free_bits`, is the immunity knob.** Injecting a controlled impulse (amplitude 3) at
position 0 of a model's own reconstruction, length-preserving, and measuring each loss term's response:

| group | time-MSE rises | log-PSD rises | ratio |
|---|---|---|---|
| `comb`, dyn off | 2.5% | 68% | **31x** |
| `lpsd`, dyn off | 5.5% | 54% | **11x** |
| `lpsd`, dyn on | 5.4% | 48% | **11x** |

An impulse is broadband and the log-PSD term scores every bin, so it punishes a spike 10-31x harder than
plain MSE, which sees one sample in 256. This is a property of the loss functions, not of any trained
model. `free_bits` has no mechanism by which it would penalise a broadband impulse at all.

**4. It does not reach the encoder.** Perturbing the input's first sample moves mu LESS than perturbing an
interior sample in every group (ratio 0.51-0.89). This issue's impact assessment assumed that; it now
holds by measurement, so criterion 1 is unaffected.

**5. The cost is about double what was recorded here.** Counting both edges, the 8 samples at the ends
(3.1% of the window) carry **17.6%** of the total reconstruction loss for `comb`+dynamics-on, vs 5.5%
(`comb` off), 4.7% (`lpsd` on), 3.2% (untrained). The "~8-9%" above was position 0 only, one arm.

**Consequence for the proposed experiment.** The 2x2 collapses to **one cell**: `recon_aux=comb` +
`free_bits=0.02` + dynamics on. Prediction: the spike SURVIVES. If it does, `recon_aux` is confirmed and
the second corner is unnecessary. Config resolution verified via the Hydra compose API (the training
command itself is a user-terminal handoff per the repo rule):

```
python -m swm.train +experiment=exp05/base_comb \
    exp_name=exp06_edge_comb_fb0p02 variant=B seed=0 \
    model.dyn_mode=fwd_bwd train.lambda_dyn=66 train.free_bits=0.02 \
    paths.packed_dir=C:/git_repo/Stellar-World-Model/experiments/exp01_window256_seq16/packed
```
