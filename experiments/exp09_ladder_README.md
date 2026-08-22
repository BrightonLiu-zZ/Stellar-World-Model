# exp09 — loss-exploit ladder: does closing the channel close the model-selection void?

**Status:** complete through wave 4, 2026-08-21. 60 runs over 10 cells on two axes (mechanism, then amount). `aug_hfnoise` is the one designed cell never run.
**Manifest (single source of truth):** `experiments/configs/exp09_loss_exploit_ladder.yaml`
**Roadmap:** `docs/roadmap/2026-08-15-post-yue-ma-roadmap.md` (lane M) ·
**Provenance:** `docs/roadmap/2026-08-15-yue-ma-suggestions-analysis.md` ·
**ADR:** `docs/adr/0012-recipe-unfreeze-and-probe-boundary.md` ·
**Diagnostics:** `src/notebooks/exp09_diagnostics.ipynb` (88 cells, executes clean) · **Collaborator pack:** `tmp/exp09_yue_ma_report.ipynb` (44 cells, interactive)

---

## Identity

Yue Ma (2026-08-13) made a **causal claim** about the model-selection void:

```
  the objective contains an exploitable channel
      -->  the loss can be driven down WITHOUT learning
      -->  therefore the loss cannot rank representations
      -->  close the channel  ==>  the loss may become usable
```

exp09 is that claim's experiment. It is **not** "fix the artifact" — exp07 already measured the
artifact as probe-harmless (`numax_hon` +0.0001 ± 0.0037).

**What the exploit actually is** (ADR-0012, Decision 2): every objective term is computed inside one
256-cadence window (`losses.py:41-57`), so the *comb* is a display artifact of stitching and the real
purchase is a **within-window impulse**. Two of the five cells are honest translations of suggestions
that targeted the comb instead.

## What ran

3 cells × 6 seeds, `fbwd`-only, on the un-frozen `hann0p3` base. 18 runs, ~7.5 h, 0 failures.
exp07's `hann0p3_{off,fbwd}` are reused as reference arms and were **never retrained**.

| cell | intervention | provenance |
|---|---|---|
| `exp09_aux_none` | `recon_aux.weight = 0.0` (term still computed and logged) | Y6 / Y11 "baseline 1" |
| `exp09_aux_dpss` | log-PSD averaged over a DPSS family (NW 4, K 7) | Y4, translated |
| `exp09_aux_impulse_pen` | + residual excess-kurtosis penalty, weight 0.1 | Y9, translated |
| `exp09_aux_clip` | `spectral_floor = 5.23` (Y9-F impulse-ablation floor); ran 2026-08-17 | Y7 |
| `exp09_aux_dpss_impulse` | DPSS **and** kurtosis penalty together; ran 2026-08-18 | post-hoc (P6) |
| `exp09_dpss_impulse_w{0p05,0p10,0p20}` | the wave-2 recipe at lower `recon_aux.weight`; ran 2026-08-21 | post-hoc (P7) |
| `exp09_dpss_impulse_w{0p025,0p0125}` | bisects `0 < w < 0.05` — **wave 4, queued 2026-08-21** | post-hoc (P8) |
| `exp09_aug_hfnoise` | **not run** — awaits its σ pilot | Y3 |

`aux_none` is `type=combined, weight=0.0`, **not** `type=none`: the loop computes the aux term
unconditionally, so weight-0 yields a *measured* no-pressure reference (`val/aux` 57–67 against a
pressured model's 3.4). `type=none` would leave nothing to compare against.

## Gate outcomes

| gate | `aux_none` | `aux_clip` | `aux_dpss` | `aux_impulse_pen` | `aux_dpss_impulse` |
|---|---|---|---|---|---|
| **G9-dose** `[0.6, 1.4]` | PASS 0.691 ± 0.007 | PASS 0.826 ± 0.010 | PASS 0.888 ± 0.016 | PASS 1.005 ± 0.035 | PASS 0.900 ± 0.041 |
| **G9-artifact** `≤ 1.5×` | **PASS 1.17 ± 0.01** | fail **4.76 ± 2.60** | fail 5.28 ± 1.39 | fail 3.08 ± 0.33 | fail **2.91 ± 0.48** |
| **G9-noregress** | **FAIL** — rotation −0.0154 ± 0.0075 at `mean`; eb, pulsating, rotation at `mean_std` | **FAIL** — transit −0.0135 at `mean`; eb −0.0066 at `mean_std` | PASS both readouts | PASS both readouts | PASS both readouts |
| **G9-select** | — **FAIL** overall — | | | | |

**`aux_clip` (wave 2, 2026-08-17).** Fails on transit while posting the **best `eb` at `mean` of any
cell including the reference** (0.7840, +0.0129 ± 0.0064) and the best `rotation` at both readouts
(+0.0055, +0.0188). A reshuffle, not `aux_none`'s uniform regression — consistent with its collapsed
latent (4–6 units) costing the localized task. It is also the **most seed-unstable** cell on the
artifact metric (per-seed 2.00–8.68, a 4.3× spread; every other cell's sd ≤ 1.4), and that instability
is **not** explained by when the clamp engaged (engage epochs cluster at 81–86; ρ = −0.41 on n=5, wrong
sign, not significant). Cause unattributed.

**Provenance note:** `aux_clip` seed 2 is a resumed run — its W&B curve starts at epoch 40 and the
earlier fragment is `state killed`. The run itself completed (last.pt at epoch 99, shipped checkpoint at
89, inside the curve) and `argmin val/recon` lands at epoch 71–95 for every seed, so nothing used here
is contaminated. Do not compute epoch-range statistics over 0–39 for that seed.

**λ = 60 transferred without recalibration**, so no cell was re-run. Selection sanity: median selected
epoch 78–95, minimum 45 — the degenerate recon-only selection pre-registered as decision A3's risk did
not occur.

**`aux_dpss_impulse` (wave 3, 2026-08-18) — the post-hoc combined cell, scored against P6.** Lands at
**2.91 ± 0.48**, the lowest severity of any probe-preserving cell, but outside the P6a band, so **P6c is
the scored outcome: the two interventions are not independent.** The interaction index is **2.87 ± 0.92**
— the child kept nearly three times as much of the artifact as independence predicts — and the marginal
effects name the redundancy asymmetrically: adding DPSS to `hann0p3` alone multiplies severity by
**0.45**, adding it on top of the kurtosis penalty by **0.95**. The penalty was already removing most of
what DPSS removes.

Three things the combined cell nevertheless bought, all in notebook section J:

- **Purchase composed even though severity did not.** Ablation asymmetry at budget 2 falls to **1.94×**,
  below both parents (3.47× / 3.87×) and close to the ~1.6× an independence model predicts on that axis.
  The leftover bump got *more honest* without getting much *smaller*.
- **Residual kurtosis reaches the no-pressure floor**: 2.943 ± 0.023 against `aux_none`'s 2.941 ± 0.007,
  where `comb0p3` sits at 7.32 and `hann0p3` at 3.64. Consistent with a 2.91× severity, not contrary to
  it — a 2.9× excess at one of 256 positions shifts whole-window kurtosis by only ~0.04, so **the gate
  and the kurtosis instrument do not measure the same thing** and peakiness-suppression alone can never
  pass G9-artifact.
- **Best `mean` v1 delta in the ladder** (+0.0005 averaged over the four tasks, the only non-negative
  one), against **−0.0109** at `mean_std` — free on the headline readout, mildly costly on the second.

Per-seed note: seed 4 trips the health flag at **4.1** active units against a cell median of 20.0 (the
first flag anywhere in exp09). It is a width flag, not a failure — that seed posts the cell's *highest*
`eb` and `rotation` and its lowest `pulsating`. Retained; dropping it would be selecting on an outcome.

> **No cell passes both `G9-artifact` and `G9-noregress`** — now across **eight** attempts on two
> orthogonal axes (six mechanisms at `w = 0.3`, then five weights of the best mechanism).

## Findings

**0. A floor on the loss is not a floor on the achieved value — P3 falsified.** `aux_clip`'s measured
log-PSD sub-term is **4.217 ± 0.109**, i.e. **0.806× the floor**, *below* it rather than at it.
`clamp(spectral, min=floor)` caps what the loss reports and zeroes the gradient underneath, so the
moment a run dips below the floor the spectral term **switches itself off permanently** and nothing
restores it; improving reconstruction keeps incidentally improving the spectrum, so it coasts down. The
design reasoning — "it descends to the floor and stops" — was wrong: descent does not stop at the floor,
only *pressure* does. `aux_clip` is therefore a **partially-pressured** model, sitting between
`hann0p3`'s 1.96 and `aux_none`'s 64.8.

> If the clip is revisited, use a one-sided **penalty** — `relu(floor − spectral)` — which keeps a
> restoring gradient and cannot self-deactivate. Yue Ma's **Y8** ("regularise instead of clip") was the
> better of her two suggestions here, and we ran the weaker one first because it was cheaper to read.

**1. The impulse and the probe benefit share one source, and the ablation prices it.**
`analyze_exp09_bump_ablation.py` generalises exp07 pre-check C1 (force the residual to zero at the
top-error positions, recompute the cell's own objective; asymmetry = rise there ÷ rise at random
positions):

```
  purchase strength   34.4x (rect)  ->  12.3x (Hann)  ->  3.5x (DPSS)  ->  0.01x (no aux term)
  probe outcome       ------------- unharmed ------------------------>  ->  REGRESSES
```

The positive control reproduces C1 (**34.4×** vs published 27–38×; **+291%** vs +237–348%) and
`aux_none` defines the honest-residual floor at **0.01**. DPSS cuts the purchase ~3.5× below Hann —
the largest reduction achieved *while keeping the probe* — but the only arm reaching the floor is the
one that fails the probe.

**2. A label-free failure detector exists; it does not restore selectability.** A threshold on
`val/recon` flags **exactly** the independently-collapsed cells (`exp08_smooth`, `smooth_half`), no
overlap. But among non-collapsed cells the anti-correlation *returns* — the collapsed runs were
**masking** the problem, not causing it. Adding the three exploit-closed cells makes ρ more negative in
6 of 8 task×readout combinations. **The void survives its most obvious explanation** (pre-registered P4).

**3. The void is scale-free.** The loss ranks neither recipes nor **seeds**: pooled mean |ρ| **0.310**
against an exact n=6 null of **0.371**, 1 of 24 significant where chance predicts 1.2. Latent width
does no better (0.256). Not for want of variation — within `aux_dpss`, ρ(`n_active`, `val/recon`) =
**+0.94**. Real seed-level structure exists; it is orthogonal to probe quality. Replicates F12/I3 at 6
seeds on a new experiment.

**4. DPSS moved the artifact onto the decoder's granularity.** The visual sign-off found ~8 wide, low
bumps rather than one spike. Not taper lobes (the aggregate DPSS weight has 3 maxima; ρ = +0.195,
below a control cell's +0.240). It is **period 16** — one bottleneck position under
`4× ConvTranspose1d(k=4, s=2)` — and `aux_dpss` is the only cell where that dominates (power 0.046,
3–12× every other). The wandering argmax (24–227) is the argmax hopping between repeating patch-scale
bumps, not the absence of structure.

**5. F27 sharpened.** The full-segment view shows rectangular and Hann both produce a **regular train,
one spike per 256 cadences**, differing only in phase (on the seam vs mid-window). The taper changed
the comb's phase, not its periodicity — which is why 6-seed comb contrast is Hann 26.6 vs rectangular
20.4, slightly *worse*.

**6. The A3 checkpoint confound is real to check and does not bite.** `best_recon_only` vs
`best_recon_aux`: mean |Δ| 0.0006, max 0.0047 over 24 comparisons; `aux_none` exactly 0.0000 (the two
metrics are algebraically identical at weight 0 — a free wiring check).

## Consequence

`hann0p3` **remains the recipe**. The pre-registered spine-switch rule does not fire (G9-select fails;
best margins vs `hann0p3_fbwd` are +0.003 to +0.005, inside noise), so the Aug 29 submission keeps its
default spine and exp09 goes to the journal version.

The void section is nevertheless **stronger** than before: it now has a mechanism, a working failure
detector, a refuted explanation, and a priced trade-off, rather than four failed attempts.

## Deviations and traps hit

- **`scipy.find_peaks` cannot return endpoint maxima.** An unpadded call skipped the rectangular
  recipe's edge impulse entirely, making the project's strongest known purchase score **0.73** —
  below its own random control. Fixed by sentinel padding; `edge_selected` is now asserted.
- **Ablation budget dilutes asymmetry.** At 40 of 256 cadences the random control alone moves log-PSD
  +63–88%, compressing every ratio toward 1 (Hann read 1.53 vs 12.3 at budget 2). The budget is swept
  and the verdict read where the control is small.
- **The µ cache is keyed `{cell}_seed{seed}.npz` with no checkpoint in the key**, and short-circuits on
  `exists()`. Scoring two checkpoints requires separate `--cache-dir`s or the second silently reuses
  the first.
- **`untrained_ref_ckpt` was hardcoded** to read its architecture from `EXTENSION_CELLS[0]` using the
  caller's `--ckpt` name, so any new checkpoint name crashed it. Now searches the requested cells first
  and falls back; verified to resolve to the identical historical path for exp07/exp08 invocations.
- **`impulse_penalty_weight` briefly sat under `recon_aux`** in the manifest while the loop reads it at
  `train` level — the cell would have trained with no penalty and nothing would have failed loudly.
  Caught by a `run_epoch` test, which is now a regression test.

## Reproduce

```bash
# training (USER TERMINAL ONLY — GPU + W&B online)
.\experiments\run_exp09_loss_exploit_ladder.ps1 -MaxHours 13.0

# eval fan (CC-runnable; two checkpoint arms need SEPARATE cache dirs)
python -m swm.eval.dump_wandb_history --groups exp09 --out experiments/exp09_forensics/curves_exp09
python experiments/analyze_exp07_diagnostics.py --stages mu stars --cells exp09_aux_none exp09_aux_dpss \
    exp09_aux_impulse_pen --seeds 0 1 2 3 4 5 --ckpt best_recon_only \
    --cache-dir experiments/exp09_forensics/mu_cache_ro --out-prefix experiments/exp09_diag_ro
# aux_clip trained a day later, so it has its OWN probe files (the mu cache short-circuits on
# exists() and is keyed without the checkpoint name, hence a separate --cache-dir):
python experiments/analyze_exp07_diagnostics.py --stages mu stars --cells exp09_aux_clip \
    --seeds 0 1 2 3 4 5 --ckpt best_recon_only \
    --cache-dir experiments/exp09_forensics/mu_cache_clip_ro --out-prefix experiments/exp09_diag_clip_ro
# same again for the wave-2 combined cell (own cache dir, own probe file, tag "di"):
python experiments/analyze_exp07_diagnostics.py --stages mu stars --cells exp09_aux_dpss_impulse \
    --seeds 0 1 2 3 4 5 --ckpt best_recon_only \
    --cache-dir experiments/exp09_forensics/mu_cache_di_ro --out-prefix experiments/exp09_diag_di_ro
python experiments/analyze_exp09_artifact.py          # G9-artifact, max-over-position
python experiments/analyze_exp09_bump_ablation.py     # purchase vs honest residual, ALL 7 cells
#   ^ run it with no --cells: a Y9-F-scoped invocation overwrites the table with hann0p3 alone and
#     silently breaks notebook section D11 (this happened 2026-08-17; repaired 2026-08-18)
python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=swm src/notebooks/exp09_diagnostics.ipynb
```

## Outputs

`exp09_diag_{ro,ax}_probe_summary.csv` · `exp09_diag_{clip,di,w3,w4}_ro_probe_summary.csv` ·
`exp09_impulse_{runs,summary}.csv` · `exp09_impulse_profile.parquet` · `exp09_bump_ablation.csv` ·
`exp09_forensics/curves_exp09/` (60 runs) · `exp09_forensics/mu_cache_{ro,ax,clip_ro,di_ro,w3_ro,w4_ro}/`

## Waves 3–4 — the aux-weight ladder (2026-08-21, post-hoc, 30 runs, ~12.5 h)

Six mechanisms had been tried and none cleared `G9-artifact`. Scoring **all 42 runs** of the mechanism
ladder against the gate showed the mechanism was never the operative variable: only 6 of 42 runs sat
under 1.5×, and **all six were `aux_none`** (spectral term off); ρ(n_active, severity) = +0.14 and
ρ(val_recon, severity) = +0.36 — neither explains anything. So waves 3–4 swept the **amount**:
`train.recon_aux.weight` on the wave-2 recipe, one knob, `impulse_penalty_weight` fixed at 0.1.

| w | severity | G9-artifact | G9-dose | probe `mean` | probe `mean_std` | spectral | purchase | worst-seed `n_active` |
|---|---|---|---|---|---|---|---|---|
| 0.0000 | 1.172 ± 0.012 | **PASS** | PASS | FAIL (rot) | FAIL (eb,puls,rot) | 63.97 | 0.01 | 3.0 |
| 0.0125 | **1.345 ± 0.096** | **PASS** | **FAIL** | FAIL (puls) | FAIL (puls,tran) | 7.31 | 2.05 | **0.0** |
| 0.0250 | 1.553 ± 0.054 | fail | PASS | **PASS** | PASS | 5.33 | 2.17 | 4.0 |
| 0.0500 | 1.803 ± 0.141 | fail | PASS | PASS | PASS | 3.71 | 2.08 | 4.9 |
| 0.1000 | 2.203 ± 0.108 | fail | PASS | PASS | PASS | 2.97 | 2.28 | 4.0 |
| 0.2000 | 2.864 ± 0.385 | fail | PASS | PASS | FAIL (puls) | 2.65 | 2.49 | 5.2 |
| 0.3000 | 2.913 ± 0.482 | fail | PASS | PASS | PASS | 2.54 | 1.94 | 4.1 |

**F-exp09-12 — the gate is sound; severity is a dose.** Monotone in the weight at all seven points
(asserted in-cell), so `G9-artifact` measures spectral pressure and not decoder architecture. **P7c is
dead.**

**F-exp09-13 — it is a threshold, not an exchange rate. This is the finding, and it contradicts the
pre-registration.** P7b predicted the probe would degrade in lockstep with severity. It does not: the
probe is **flat from severity 1.55 to 2.91** and then cliffs. Above the threshold the artifact is
*free*. Every other quantity steps at the same place — achieved spectral rises 2.9× across the whole
pressured range then jumps **8.8×** in the last step, and the purchase fraction sits flat at 1.94–2.49
before dropping to the honest floor of 0.01. **Lowering the weight buys a smaller artifact, not a less
dishonest one.**

**F-exp09-14 — the cliff is latent collapse, and the separator is the worst seed.** Cell-mean
`n_active` divides nothing (`w = 0.05` passes at 5.65 while `w = 0` fails at 4.33); the **minimum over
seeds** divides it perfectly — every failing cell has a seed ≤ 3 active units, every passing cell has
all six ≥ 4. At `w = 0.0125` one seed dies outright at **0.0** units, which drives its dose ratio to
0.0000 and **fails `G9-dose`**, so that row is void by the pre-registered rule. Its probe is
near-untrained (eb 0.7248 vs an untrained encoder's 0.7124). So the spectral term's job is to hold the
latent open; the impulse is a **side effect** of that, not a co-purchase.

**F-exp09-15 — `w = 0.025` is the best cell the project has produced, and still misses.** First cell to
improve **all four** v1 tasks over `hann0p3` simultaneously at `mean` (+0.0137 eb, +0.0015 pulsating,
+0.0229 rotation, +0.0051 transit); best absolute `eb` (0.7847), `rotation` (0.5821) and `transit`
(0.1492) in the ladder (`pulsating` stays with `aux_dpss`, 0.8099); artifact **7.6× smaller** than the
frozen recipe. But severity **1.553 ± 0.054 is 2.4 SE above the 1.5 gate** — a real miss — and the
spine-switch rule needs ≥2 tasks beating `hann0p3` by >2·SE where **only rotation** does, which is also
the task flagged as confounded on the v1 subset. At `mean_std` it is 0 of 4.

**Scoring P8 honestly — the outcome matches no pre-registered branch.** P8a needed severity ≤ 1.5 *with*
the probe (`w = 0.025` has the probe, misses the gate); P8b's demonstration cell (`w = 0.0125`) reaches
1.345 and fails the probe exactly as P8b describes, but is **void on `G9-dose`**; P8c is simply wrong
(the crossing is bracketed at `w ≈ 0.021`, not below 0.0125). A dose failure arising from latent
collapse was not anticipated. Recorded as the mismatch it is rather than assigned to the nearest branch.

**Limitation this exposed in the F detector.** It thresholds the *cell mean* `val/recon`, and
`w0p0125`'s cell mean is 0.854 — comfortably under 1.10. It would **not** flag that cell despite one of
its six seeds being completely dead. It separates catastrophic *cell-level* collapse from everything
else; section H's per-seed table is what catches a single failed seed.

**G9-select is unchanged in verdict but moves in one direction.** Folding the five weight cells into F's
population (11 → 16 cells) makes every task *less* anti-correlated — eb +0.145→+0.265, pulsating
−0.345→−0.306, rotation −0.273→**+0.047** (flips sign), transit −0.436→−0.268. Only rotation clears the
"0.3 less negative" bar, so `G9-select` still fails, but the movement belongs on the record.

## Not done

`aug_hfnoise` (needs its σ pilot — and see below) · an `off` arm for the winning cell · a
`relu(floor − spectral)` rebuild of the clip (P3 falsified the hard clamp — it self-deactivates, so it
must not be stacked with anything until rebuilt) · **`w = 0.02`**, the one cell that would settle
whether *any* weight clears both gates, since the crossing is bracketed at ≈ 0.021 · more seeds at
`w = 0.025` (eb sits at +0.93× the 2·SE line; 12 seeds would settle whether the spine-switch rule fires
on a second task).

`aug_hfnoise` is now **deprioritised on evidence, not just cost**: it noises input and target together,
which makes it a *third* floor experiment, and both floor results so far (the clip, and the weight
ladder's low end) point the same way. Its σ pilot also needs redesign — the acceptance rule as written
is vacuous, since the augmentation is band-limited above 1009 µHz by construction and the pulsator band
(bins 2–7) therefore reads a 0.0% change at **every** σ. At the manifest's σ = 0.10 the HF noise floor
rises only **3%**, because TESS light curves are already noise-dominated up there (the HF band carries
48% of the clean per-bin power). A meaningful dose needs σ ≈ 0.4, which moves total window power 17%.

**Ran 2026-08-18 — `exp09_aux_dpss_impulse` (post-hoc, 6 seeds, 2.5 h): P6c.** Full result in the gate
table and notebook section J. The pre-registration held that if DPSS and the kurtosis penalty acted
independently, severity would land at ≈ **1.91×** (per-seed form: **2.23×**); it landed at **2.91×**, so
the two are **not** independent — they were largely buying the same reduction. Reported as a **post-hoc
wave-2 addendum, never folded into the pre-registered ladder.** Two results worth carrying forward
anyway: the purchase measure *did* compose (1.94× at budget 2, against ~1.6× for independence), and the
residual kurtosis reached the no-pressure floor while severity stayed at 2.91× — which is what
established that **G9-artifact and peakiness are different quantities**.
