# exp10 — fusion-spine rescue: the spine cannot be widened from the encoder side

**Manifest (single source of truth):** `experiments/configs/exp10_fusion_spine.yaml` (decisions D-E10.1–12,
predictions P-E10-1–4, gates). **Plan:** `docs/plans/2026-08-30-exp10-fusion-spine.md`.
**Run:** 18 GPU-hours-equivalent, 2026-08-30, 6 seeds per cell, ~9 h wall on the RTX 4060.
**Every verdict below is PROVISIONAL-AT-6-SEEDS** (D-E10.5) and provisional pending Yue Ma's review
(D-E10.2).

## The one-line result

Three encoder-side levers aimed at making µ carry content the 25 engineered features cannot express.
**None widened the fusion spine; the two valid ones narrowed it.** The published C3b count of 4/11 was
not beaten by any cell, and the cell that most successfully did what it was designed to do —
`decorr`, which stripped 70 % of µ's linear feature-correlation — came in *lowest* of the two.
**P-E10-4 fires: at this architecture and this data the spine cannot be widened, and the scoped D19
claim is the final claim.**

## Gate table (VOID order: validity → noregress → spine → mechanism)

| cell | G10-valid | G10-noregress | **G10-spine (primary)** | verdict |
|---|---|---|---|---|
| `exp10_cond_dec` | PASS (dose 0.884) | **FAIL** (3 rows regress) | **FAIL** 3/11 | negative |
| `exp10_decorr` | PASS (dose 1.008) | PASS | **FAIL** 2/11 | negative |
| `exp10_multistep` | **FAIL** (dose 0.236) | — | — | **VOID on dose** |
| `exp07_hann0p3_fbwd` (reference) | PASS (dose 0.954) | — | 4/11 (footing ✓) | incumbent |

## Footing, before anything new is read

The reference arm scored through the same fan on the same caches returns **exactly 4/11 survivors**
(`eb`, `rgb_vs_heb`, `rotation_period`, `transit`) — the published C3b count. The fan is measuring what
the gate was written about.

## G10-valid — and the one design miss

`experiments/exp10_valid_gate.csv`, `exp10_valid_per_seed.csv`.

| cell | dose mean (min–max) | val_recon max | KL min | median selected epoch | verdict |
|---|---|---|---|---|---|
| `exp10_cond_dec` | 0.884 (0.768–0.989) | 0.915 | 1.098 | 92 | PASS |
| `exp10_decorr` | 1.008 (0.935–1.077) | 0.878 | 2.656 | 82.5 | PASS |
| `exp10_multistep` | **0.236 (0.233–0.243)** | 0.864 | 2.428 | 76.5 | **FAIL, all 6 seeds under the 0.3 floor** |

**The multistep λ guess was wrong, and wrong in the direction opposite to its stated reasoning.** The
manifest dropped λ from 60 to 20 because "rollout MSE compounds over 15 free-running steps, so
train/dyn is several times the one-step value". Measured, `dyn/recon` is **0.0118** for the free-running
rollout against **0.0159** for the fwd+bwd pair — and since fwd_bwd sums two directional terms, the
rollout is roughly **1.5× a single one-step term, not several times it**. Errors compound in the
rollout, but the latent is smooth enough over 15 steps that the compounded error stays small relative
to reconstruction.

Nothing else was wrong with the cell: no collapse (KL 2.43–2.48, val_recon 0.855–0.864, all healthy),
selection well past warmup. It is void purely on under-dosed dynamics pressure, which means any probe
number it produces confounds "multistep" with "less dynamics pressure" — exactly what the gate exists
to catch. Pre-registered repair is λ_new = λ·(1/dose) = **20 × 4.23 ≈ 85**. **User decision 2026-08-30:
do not rerun.** exp10 closes on the two content-creation cells; the multistep rows below are printed
under a VOID heading and are counted in nothing.

## G10-noregress — what the levers cost on the v1 block

`experiments/exp10_noregress.csv`. Paired by seed vs `exp07_hann0p3_fbwd` s0–5, 4 tasks × {mean,
mean_std}; FAIL if any row regresses beyond 2·SE at either readout.

| cell | regressing rows | verdict |
|---|---|---|
| `exp10_cond_dec` | pulsating@mean −0.0267, eb@mean_std −0.0145, rotation@mean_std −0.0093 | **FAIL** |
| `exp10_decorr` | none (worst −0.0196 on rotation@mean, inside 2·SE) | PASS |
| `exp10_multistep` | eb, rotation, transit @mean; eb, transit @mean_std | VOID |

`cond_dec`'s losses are **6× the measured A3 checkpoint confound** (mean |d| 0.0006–0.0008, max 0.0047),
so they are the lever, not the selector. Feeding the star's engineered features to the decoder makes the
encoder stop paying for content the v1 probes were using — `pulsating` most of all.

## G10-spine (PRIMARY) — nobody widened it

`experiments/exp10_spine_gate.csv`, `exp10_spine_verdict.csv`. Fusion-minus-features under **GBM at
readout `mean`**, 6 seeds paired including GBM random_states; survive = >2·SE **and** |Δ| ≥ 0.01;
PASS needs **more than 4 of 11** with `transit` retained.

| arm | survivors | count |
|---|---|---|
| `hann0p3_fbwd` (ref) | eb +0.0164, transit +0.0413, rgb_vs_heb +0.0142, rotation_period +0.0125 | **4** |
| `exp10_cond_dec` | eb +0.0148, transit +0.0300, rgb_vs_heb +0.0188 | **3** |
| `exp10_decorr` | transit +0.0207, rgb_vs_heb +0.0190 | **2** |
| `exp10_multistep` (VOID) | transit +0.0230 | 1 |

**The draw-CI clause cannot change any verdict.** `rgb_vs_heb` is the only small-n row surviving
anywhere, and the extra CI can only *remove* survivors, never add them; every count is already at or
below the bar of 4, so the gate outcome is robust to it. This is recorded rather than the full draw
machinery being built, because the machinery could not have moved the answer.

The clearest single number: `eb`'s fusion delta falls **+0.0164 → +0.0084** under `decorr`, dropping
below the 0.01 effect floor. The lever that best achieved its stated mechanism did the most damage to
the claim it was meant to widen.

## G10-mech (secondary, reported never gated) — why it failed

Three independent instruments, all agreeing.

**1. F-B predictability of µ from the 25 features** (`experiments/exp10_mech/fb_*`, subset population,
6 seeds; the manifest's named primary mech read). Footing: the reference returns **0.6509** against the
published 0.6491.

| arm | var-weighted GBM R² | probe-coefficient-weighted GBM R² |
|---|---|---|
| `exp07_hann0p3_fbwd` | 0.9354 | **0.6509** (footing ✓) |
| `exp10_cond_dec` | **0.7427** (−0.193) | **0.6994** (+0.049, *rose*) |
| `exp10_decorr` | 0.9275 (−0.008, flat) | **0.2668** (−0.384) |

**Both levers moved predictability, on different weightings, and neither move helped.** `cond_dec` took
feature-predictable content out of the *high-variance* dims (var-weighted 0.935 → 0.743) while the dims
the probe actually uses became **more** feature-redundant — P-E10-1 branch (b) exactly, "content was
created but is not probe-relevant", and the direct explanation of its noregress FAIL. `decorr` did the
mirror image: the variance-weighted number is untouched, but the **probe-used** dims went from 0.651 to
0.267 predictable. It did precisely what it was designed to do, on precisely the dims that matter, and
the spine got worse.

**2. Achieved µ-vs-features corr²** (`experiments/exp10_decorr_anchor.csv`; the same statistic
`losses.decorr_loss` optimizes, on val, at each arm's own primary checkpoint). The reference number did
not exist before this wave — the exp07 seeds predate the channel — so it was measured here, which is
what made the decorr pilot rule scorable at all.

| arm | corr² (sd over 6 seeds) | vs reference |
|---|---|---|
| `exp07_hann0p3_fbwd` | 0.01718 (0.00099) | — |
| `exp10_cond_dec` | 0.01837 (0.00389) | +6.9 %, inside the seed spread |
| `exp10_decorr` | **0.00523 (0.00036)** | **−69.6 %** |
| `exp10_multistep` (VOID) | 0.01409 (0.00450) | −18.0 % |

**Read this statistic beside F-B, never instead of it.** corr² weights every latent dim equally — that
is deliberate (D-E10.11, so the 84 %-variance dim cannot absorb the penalty) — so it is blind to
exactly the move `cond_dec` made, which was concentrated in the high-variance dims. On corr² alone
`cond_dec` looks unmoved; the variance-weighted R² shows it moved a lot. Two instruments, two
weightings, one representation.

**3. µ⊥features residual over untrained**, mean over 15 task-readout rows, from the same F1 fan:

| arm | µ_perp_full over untrained |
|---|---|
| `hann0p3_fbwd` | **+0.1245** |
| `exp10_decorr` | +0.1111 |
| `exp10_cond_dec` | +0.1025 |
| `exp10_multistep` (VOID) | +0.0640 |

**Every cell shrank the feature-orthogonal residual.** This is the finding, and it is what ties the
other two instruments together. Both levers succeeded at making µ less predictable from the features —
`cond_dec` on the high-variance dims, `decorr` on the probe-used dims — and in both cases the content
that the features *cannot* explain went **down**, not up. Neither lever **relocated** content off the
feature manifold; both **destroyed** content. That is the difference between reorganising a
representation and damaging it, and it is visible only because all three instruments were read
together: any one of them alone tells a misleading story.

## Predictions vs outcome (recorded, never snapped)

- **P-E10-1 (cond_dec).** **Branch (b), cleanly.** "Predictability falls, spine count stays ≤ 4 →
  content was created but is not probe-relevant, the R-B2 warning arriving on schedule." Var-weighted
  predictability fell 0.935 → 0.743 and the spine count went 4 → 3. The branch even understated it: the
  probe-weighted number *rose* (0.651 → 0.699), so the content the conditioning freed up was moved
  **away from** the dims the probes read, not toward them.
- **P-E10-2 (decorr).** **Branch (b)**, and prediction (d) — the conflict signature — is **refuted**.
  (d) expected "val_recon rises while achieved corr² falls: the penalty is fighting reconstruction for
  the amplitude channel". Measured: corr² fell 69.6 % **and val_recon fell 0.7 %** (0.866 against the
  reference's 0.872). There was no reconstruction/penalty conflict at all. The cost landed on the
  *probe* — probe-weighted predictability 0.651 → 0.267, spine 4 → 2 — a channel (d) did not consider.
- **P-E10-3 (multistep).** Unanswered. VOID on dose.
- **P-E10-4 (cross-cell).** **Fires, antecedent fully satisfied.** It required that *both* E1 and E2
  move predictability while neither passes G10-spine, and both did move it — E1 on the variance
  weighting, E2 on the probe weighting. The conclusion stands on stronger evidence than the prediction
  assumed: not only did neither lever widen the spine, both *narrowed* it, and the mechanism read shows
  why — the feature-correlated content is load-bearing, not redundant ballast.

## What this settles

F-F measured the GBM as **using µ heavily without gaining from it** (36–91 % of split gain on
flat-delta tasks) and concluded the problem was missing residual content rather than redundancy. exp10
tested the two obvious ways to create that residual, and both failed in the same direction: **the
linearly feature-correlated part of µ is where the fusion gain lives.** Removing it (E2) or making it
unnecessary (E1) removes the gain with it. Combined with the forensics result that no readout-side
lever moves the count above 4/11, the honest statement is that **the spine is 4/11 at this
architecture and this data, from both sides**, and that negative closes the question for the journal
version.

The ML4PS paper is unaffected by construction (D-E10.1): it keeps the scoped D19 claim and `hann0p3`,
and D17 was never reopened.

## Not run, and why

- **The post-hoc dyn-off twin** for a G10-spine passer (D-E10.6): there is no passer.
- **The full draw/bootstrap CI** on small-n rows: cannot change any verdict (see above).
- **The exp08 signature suite** (residual asymmetry, rollout-vs-persistence) for E4: its cell is VOID on
  dose, so a mechanism read on it would describe an under-dosed run, not multistep.
- **The multistep relaunch at λ≈85**: user decision, 2026-08-30.

## Artifacts

| path | what |
|---|---|
| `experiments/exp10_features/subset_features25.parquet` | 13,470 stars × 25 train-standardized features, 0 missing |
| `experiments/exp10_eval/{mu_cache,subset_mu_cache}` | 31 arms × 2 populations, `best_recon_only` for exp10 cells |
| `experiments/exp10_forensics/curves_exp10/` | per-epoch W&B histories, 18 runs |
| `experiments/exp10_valid_{gate,per_seed}.csv` | G10-valid |
| `experiments/exp10_noregress{,_gate}.csv` | G10-noregress |
| `experiments/exp10_spine/`, `exp10_spine_{gate,verdict}.csv` | F1 fan + G10-spine |
| `experiments/exp10_decorr_anchor.csv` | achieved corr² per arm |
| `experiments/exp10_mech/fb_*` | F-B predictability on the exp10 arms |
| `experiments/analyze_exp10_{gates,noregress,spine}.py` | the three scorers |

## Reproduce

```bash
# swm env, PYTHONPATH=src, repo root
python experiments/exp10_build_features.py
python -m swm.exp.gen_sweep experiments/configs/exp10_fusion_spine.yaml
.\experiments\run_exp10_fusion_spine.ps1 -MaxHours 12.0        # user terminal, W&B online
python -m swm.eval.dump_wandb_history --groups exp10 --out experiments/exp10_forensics/curves_exp10
python experiments/analyze_exp10_gates.py                       # G10-valid, BEFORE any probe
# extract 18 arms x 2 populations into experiments/exp10_eval/ (one invocation per cell)
python experiments/analyze_exp07_diagnostics.py --stages mu stars --cells exp10_cond_dec exp10_decorr \
    exp10_multistep --seeds 0 1 2 3 4 5 --ckpt best_recon_only --out-prefix experiments/exp10_diag
python experiments/analyze_exp10_noregress.py
python experiments/analyze_f1_fusion_scorecard.py --families linear gbm --readouts mean \
    --cache-dir experiments/exp10_eval/mu_cache --subset-cache-dir experiments/exp10_eval/subset_mu_cache \
    --out-dir experiments/exp10_spine --arms <25 arms>
python experiments/analyze_exp10_spine.py
python experiments/exp10_decorr_anchor.py --models-dir experiments/exp10_<cell>/models \
    --checkpoint best_recon_only --label exp10_<cell>
```
