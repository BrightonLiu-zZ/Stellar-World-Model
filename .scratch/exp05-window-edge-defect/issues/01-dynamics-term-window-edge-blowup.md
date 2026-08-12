# Dynamics term causes a ~17x reconstruction-error blowup at window edges (comb recipe only)

Status: root cause identified — `recon_aux` confirmed by experiment, `free_bits` exonerated;
**mechanism identified 2026-08-01 (the decoder buys log-PSD power at the window edge)**
Found: 2026-07-26, exp05 diagnostics notebook (§C1b)
Diagnosed: 2026-07-27, exp06 design forensics notebook (§1)
Confirmed: 2026-07-28, exp06 edge-confirmation cell (`exp06_edge_comb_fb0p02`)
Mechanism: 2026-08-01, exp07 pre-check C1 (`experiments/analyze_exp07_c1_edge_sign.py`)
Affects: `exp05_comb_*` with `train.lambda_dyn > 0` — 36 of 48 exp05 runs
No longer blocks exp06 planning (exp06 ran; the confirmation cell was part of it)

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

### 2026-07-28 - the confirmation cell ran; the prediction held

The cell above was carried inside exp06 and scored with `analyze_exp06_edge.py`
(`experiments/exp06_edge_corner.csv`):

| run | free_bits | edge ratio | bias fraction | Jacobian ratio | edge share of recon |
|---|---|---|---|---|---|
| `exp06_edge_comb_fb0p02` | **0.02** | **15.3x** | 0.72 | 2.37 | 22.5% |
| group mean, `comb` + dyn on (48-run table) | 0.0 | 17.5x | 0.66 | 1.76 | 17.6% |
| untrained reference | - | 1.09x | 0.06 | 0.27 | 3.2% |

**The spike survives the KL floor essentially intact** (15.3x vs 17.5x), exactly as §1.4 of the
forensics notebook predicted from the loss-sensitivity measurement. Every accompanying signature
survives with it: the bias fraction is if anything higher (0.72), the decoder Jacobian is still
inverted at the boundary (2.37 against the untrained 0.27), and the edge still carries ~22% of the
reconstruction loss on 3.1% of the window.

**Conclusion: `recon_aux` is the immunity knob; `free_bits` is exonerated.** The second corner of the
originally proposed 2x2 is unnecessary and was not run. This closes the root-cause question this issue
was opened for. The remaining work is the fix itself, which is tracked outside this issue in
`experiments/exp06_geometry_README.md`.

### 2026-08-01 - the MECHANISM, plus two corrections to this issue

exp07 pre-check C1 (`experiments/analyze_exp07_c1_edge_sign.py`, 2,000 test windows, 9 runs across 5
cells; full tables in `tmp/handoff/2026-08-01-exp07-precheck-request.md` §RESULTS). Everything above
said *which knob* confers immunity. This says *what the decoder is buying*.

**The decoder is purchasing high-frequency spectral power at the window edge.** Replacing positions
{0, 255} with the input's own values (edge residual forced to exactly zero) *raises* the log-PSD term
by **+237% to +348%** on every `comb`+dynamics-on arm, while time-MSE falls 11-17% and the `hf_time`
sub-term moves only −0.5 to −4%. The entire effect is the spectral sub-term. An under-powered
reconstruction pays a large log-PSD penalty; two edge samples are the cheapest place to inject
broadband power, because they cost only 2/256 of the time-MSE.

**The amplitude is tuned, not incidental.** Rescaling the model's own edge deviation by a factor
(0 = interior-extrapolated, 1 = as trained), log-PSD has a clear minimum at scale **~1.5** for every
dynamics-on arm — the trained value sits at ~67% of the pure-log-PSD optimum, undershooting exactly as a
weighted trade against time-MSE and hf_time predicts. The dynamics-off control is flat (2.09-2.29 across
the whole sweep). This also resolves what looked like a contradiction with comment 3 above: *adding* a
+3 impulse raises log-PSD 68.8% AND *removing* the edge raises it 250%, because the learned value sits
near a minimum and both directions climb out of it.

**Location control.** The same operation (two residuals forced to zero) applied at interior positions
{96, 160} raises log-PSD only 6.5-10.3%, i.e. **27-38x less** than at the edges. The rise is a property
of the edge, not of perturbing a 256-bin spectrum.

**Correction 1 — the spike is not a positive flux bias; it is an antisymmetric dipole.** Mean *signed*
error (this issue and the profile parquet stored MSE only, which is sign-blind): **−2.95 at p0 and
+5.11 at p255** on `exp06_edge_comb_fb0p02`, −3.22 / +4.69 on `exp06_w256_fbwd`, with the interior at
zero. Polarity is seed-dependent (`exp05_comb_fbwd_c1p0` seed 0 runs +4.52 / −2.21) but the
leading-versus-trailing *opposition* holds in all 9 runs measured.

**Correction 2 — "dynamics-on arms are worse at the leading edge" was a single-seed reading.** Averaged
over seeds the **trailing** edge is worse in every group: `exp06_w256_fbwd` 16.5 (p0) vs 30.1 (p255),
`exp05_comb_fbwd_c1p0` 15.2 vs 19.2, `exp05_comb_off` 2.4 vs 11.5.

**The fix follows from the mechanism.** Under a Hann-tapered log-PSD the purchase becomes literally
unbuyable — a non-periodic Hann window is exactly zero at both endpoints, so `Δ log-PSD` for every edge
ablation is 0.0000 while the interior control still moves (−0.58 to −0.69). That is an analytic
consequence of the taper, not an empirical finding. Two caveats for whoever implements it: a taper also
blinds the aux term to genuine flux structure near window boundaries, which time-MSE alone would then
carry; and it does not touch `hf_time`, which was never the culprit.

**Open, and testable on exp07's own axes:** the immune arm's aux *weight* is 0.1 while every spiky arm's
is 0.3. Since the spike is a tuned optimum of a weighted term, spike severity should scale with
`recon_aux.weight`. If exp07's weight axis {0.1, 0.3} does not move the edge ratio for both aux forms,
then immunity is about aux *form* rather than aux *pressure*, and the exp05 lpsd-vs-comb comparison is
confounded by weight as well as by `free_bits`.
