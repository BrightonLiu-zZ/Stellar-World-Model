# exp09 — loss-exploit ladder: does closing the channel close the model-selection void?

**Status:** complete through **wave 7, 2026-08-28**. 138 runs over 20 cells on FOUR axes (mechanism, amount, level, KL floor) plus a 12-seed encoder-decision wave. `aug_hfnoise` is the one designed cell never run.
**Encoder decision (roadmap D17): `hann0p3` SHIPS** — the switch rule did not fire, branch **P11-C**. See the wave-6 section. Wave 7 does not reopen it.
**Headline, unchanged and now broader: no cell passes both `G9-artifact` and `G9-noregress` across ELEVEN interventions on four axes.** Wave 7 did fix the posterior collapse (`free_bits = 0.01`, 12/12 clean) — the cell still fails both gates, in the way P12-D pre-registered.
**Manifest (single source of truth):** `experiments/configs/exp09_loss_exploit_ladder.yaml`
**Roadmap:** `docs/roadmap/2026-08-15-post-yue-ma-roadmap.md` (lane M) ·
**Provenance:** `docs/roadmap/2026-08-15-yue-ma-suggestions-analysis.md` ·
**ADR:** `docs/adr/0012-recipe-unfreeze-and-probe-boundary.md` ·
**Diagnostics:** `src/notebooks/exp09_diagnostics.ipynb` (102 cells, executes clean; section L = the level axis) · **Collaborator pack:** `tmp/exp09_yue_ma_report.ipynb` (44 cells, interactive)

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

> **No cell passes both `G9-artifact` and `G9-noregress`** — now across **ten** attempts on three
> orthogonal axes: six mechanisms at `w = 0.3`, then six weights of the best mechanism, then
> three floors of the rebuilt clip at fixed weight.

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

## Wave 5 — the last weight point, and the clip rebuilt (2026-08-22, post-hoc, 24 runs, ~10 h)

**Verdict: P9b and P10b. The weight axis is exhausted; the floor axis works as a mechanism and prices
the trade-off in a new currency; still no cell passes both gates — ten attempts on three axes.**

| | severity | G9-art | achieved | dose (min) | n_active (min) | KL | noregress `mean` / `mean_std` |
|---|---|---|---|---|---|---|---|
| `w0p02` | 1.426 | **PASS** | 6.10 | 0.595 (**0.000**) | 4.0 (**0.0**) | 1.19 | **VOID** |
| `aux_clip` (clamp, 5.23) | 4.763 | fail | 4.199 | 0.826 (0.807) | 5.0 (4) | 1.64 | — |
| `clip_hinge_f3p23` | 9.962 | fail | 2.815 | 1.009 (0.934) | 11 (4) | 2.05 | fail puls / fail puls |
| `clip_hinge_f5p23` | 5.629 | fail | 4.883 | 1.095 (1.045) | 35 (19) | 1.77 | **PASS / PASS** |
| `clip_hinge_f7p54` | **2.712** | fail | 7.079 | 1.082 (1.032) | 53 (37) | 1.58 | fail puls / fail puls |

### P9 — the weight axis is exhausted and *un-settleable*

`w = 0.02` reached severity **1.426**, passing `G9-artifact` exactly as the interpolation predicted. It
is nonetheless **void**: one seed ended at 0.0 active units, dose 0.0000. Same fate as `w0p0125`.

So across all eight weight points, exactly **one** cell clears `G9-artifact` with a valid dose —
`aux_none` (1.172, dose 0.691) — and that one regresses on the probe. Every weight strictly between
`aux_none` and `w0p025` that reaches the gate does so only by killing a seed, which voids it. **There is
no weight of this objective that is simultaneously clean, useful and valid.**

*Do not quote w0p02's probe row either way.* At `mean` it technically passes `G9-noregress` while
improving 0 of 4 and posting eb 0.743 against 0.773–0.785 for every non-void cell — it passes by being
too **noisy** to fail, since the dead seed inflates every paired SE 3–8× versus the hinge cells. The
2·SE rule is permissive exactly where a cell is collapsing.

### P10 — the hinge works; the pre-registered mechanism gate was the wrong *shape*

Scored as written, **P10-mech fails**: the 10% band holds at 5.23 (0.934) and 7.54 (0.939) but not at
3.23 (0.871).

I proposed a rescue — that the *offsets* are constant (0.415 / 0.347 / 0.461), the fixed-slope L1
equilibrium signature — and then **withdrew it**, which turned out to be the more useful result. That
reading came from the median; the mean disagrees (0.441 / 0.327 / 0.072), and the per-seed table
settles it:

```
offset below the floor, per seed
  3.23   0.403  0.598  0.548  0.332  0.428  0.340    sd 0.109   6/6 below
  5.23   0.602  0.768  0.637 -0.132 -0.003  0.092    sd 0.385   4/6 below
  7.54   0.506  0.370  2.359  1.129 -4.346  0.416    sd 2.292   5/6 below

within-cell sd 0.929   vs   between-cell sd 0.057   ->   ratio 16x
```

`f7p54` has a seed that finished at achieved **11.886** against a floor of 7.54. A quantity whose
scatter *within* a cell is sixteen times its variation *between* cells is not a constant, and the
median was hiding it.

**What is true is narrower:** the hinge's grip is tight at the low floor and loosens monotonically —
the floor controls the level only while it sits near what the model reaches unaided (`hann0p3` gets to
1.974). Note severity moves the *other* way: its seed spread **tightens** as the floor rises
(4.15 → 3.02 → 0.76), so the floor controls the artifact better than it controls the achieved value.

The clamp control is untouched and still decisive: 1.031 under its own floor as a **runaway**, since a
clamp has no restoring force once it engages. **P3's self-deactivation is fixed** — the hinge is a
working mechanism with a measured operating range, not a working mechanism full stop.

**P10b confirmed.** Severity falls monotonically as the floor rises, 9.96 → 5.63 → 2.71, and the purchase
measured at the bumps falls with it, +139.9% → +45.5% → +17.4% (`hann0p3` +167.5%). Severity's seed
spread collapses too, 4.15 → 3.02 → 0.76. The artifact is now priced in **achieved-spectral units at
constant pressure** — the measurement waves 3–4 could not make, because there the weight moved pressure
and level together.

**P10a not reached, and the two ends fail differently.** `f5p23` passes `G9-noregress` at *both* readouts
(eb +0.0103 ± 0.0048, rotation +0.0040 ± 0.0098) but reads 5.629 on artifact. `f7p54` gets artifact to
2.712 and improves 3 of 4 at `mean` — but fails noregress on **pulsating** (−0.0284 ± 0.0119). Pulsating
is the single blocker at the clean end, which is mechanically coherent: holding the spectrum *above* its
free level is exactly the intervention a spectral task should pay for.

### The second pre-registration is refuted, and it narrows the wave-3/4 story

`n_active` does reverse — 5.0 under the clamp → 11 / 35 / 53 — far past the ~10.5 predicted. **But the
extra units carry nothing.** Total KL is 2.05 / 1.77 / 1.58, flat and even *below* `aux_dpss_impulse`'s
2.56, so KL-per-active-unit falls to 0.058 and 0.031 against 0.20–0.38 for every other cell. The width is
**dilution, not capacity**. Across all 14 exp09 cells: ρ(eb, n_active) = +0.213 (p 0.46), ρ(eb, KL total)
= −0.095 (p 0.75), ρ(eb, KL/unit) = +0.011 (p 0.97).

⇒ "the spectral term's value is that it **holds the latent open**" must be narrowed to "**avoiding total
collapse** matters; width above that does not." The wave-4 separator was always a statement about
catastrophic seeds (≤ 3 units), never a continuous driver — and with 14 cells it is now measured not to
be one. **`n_active` must not be reported as a quality metric.**

### Method debt found by the data

The ablation script's `asymmetry` column (bump % ÷ random %) **breaks** on floored cells: it reads −4.07
and −1.16 because the *random control* goes negative (−11.2%, −15.0%), not because the bump term does
(it stays +45.5% and +17.4%). Nulling ordinary residual *improves* the spectral term in a model held
above its free spectral level — honest-residual behaviour, and a sign the ratio was never designed for.
**Report the two components separately for any floored cell**; the ratio is interpretable only while the
control is positive.

### Y9-G, the pre-check that reshaped the design (no GPU, no retraining)

Before spending a night on a rebuilt clip, the floor it would be pinned to was re-read for every cell
that trains under the **same Hann taper**, from the ablation CSV already on disk:

| cell | impulse-free spectral | seed spread | achieved |
|---|---|---|---|
| `exp09_aux_impulse_pen` | **3.23** | 11% | 1.979 |
| `exp07_hann0p3_fbwd` | **5.23** ← Y9-F's published value | 41% | 1.974 |
| `exp09_aux_clip` | **7.54** | 85% | 4.199 |
| `exp09_aux_none` | 63.04 | — | (no spectral term) |

**The floor is not a task constant — 82.5% spread across cells.** The diagnostic pair is
`aux_impulse_pen` vs `hann0p3`: they train to the *same* achieved spectral (1.979 vs 1.974) yet their
impulse-free values differ 1.6×, because the kurtosis penalty left less impulse to remove. So
`log_psd_ablated` measures "what **this** model scores without **its** impulse", not a universal honest
level — and seed spread tracks the same thing, since the less a model cheats the tighter its estimate.

This matters *only* under the rebuild. Under `clamp` a wrong floor is harmless (the term merely engages
early or never). Under a hinge the floor is an **attractor**, so a floor above the honest level *forces*
the model to reconstruct worse than it could. A quantity known to within a factor of 2.3 cannot be
hard-coded into an attractor — hence a three-point ladder over the measured values, not one cell.

### What runs

`spectral_floor_mode: hinge` = `floor + |spectral − floor|`, a symmetric V. Above the floor it is the
unmodified spectral term; below it the gradient mirrors and pulls back up. It cannot self-deactivate,
which is the P3 failure. **Not** the bare `relu(floor − spectral)`: that is inert above the floor, and
untrained spectral is ~64 against a floor of ~5, so the spectral term would contribute nothing for most
of training — surrendering exactly what waves 3–4 measured as its real job (holding the latent open).

| cell | knob vs its reference | why |
|---|---|---|
| `exp09_dpss_impulse_w0p02` | `w0p025` + `weight 0.02` | the crossing is bracketed at ≈0.021; last point worth spending |
| `exp09_clip_hinge_f5p23` | `aux_clip` + `mode: hinge` | clean one-knob contrast: isolates self-deactivation from the floor's value |
| `exp09_clip_hinge_f3p23` | `f5p23` + `floor 3.23` | the least-cheating model's estimate — best guess at the honest level |
| `exp09_clip_hinge_f7p54` | `f5p23` + `floor 7.54` | deliberately over-high: prices what a too-high attractor costs |

**Why this is the measurement waves 3–4 could not make.** The V pins the *achieved* spectral value while
the aux weight stays at 0.3. The weight ladder could only move the achieved value by lowering the weight,
which simultaneously lowered the pressure and collapsed the latent — three things at once, which is why
the probe cliff could not be attributed. Here pressure is constant and only the target level moves.

**A correction to this repo's own reasoning, found while setting the priors.** "Holding the weight at 0.3
keeps the latent open" is *not* automatic: `aux_clip` ran at 0.3 and still ended at n_active **5.0**
(min 4), against `aux_impulse_pen`'s **10.5** (min 6) on the same base at the same weight, and
`aux_none`'s 4.33. The difference is that `aux_clip`'s term went dead at epoch ~81–86. So if the hinge
keeps the term live for all 100 epochs, that narrowing should **reverse** toward ~10 — a falsifiable
mechanism check independent of both gates. λ = 60 is carried unchanged, now bracketed by measured doses
on the same Hann base (`aux_clip` 0.826, `aux_impulse_pen` 1.005) rather than extrapolated.

**Pre-handoff verification.** `long_run_guard` forbids training as a Claude Code job, so the exact-command
smoke was not possible; instead all four cells were composed exactly as the runner composes them and
driven through `run_epoch` on synthetic data, covering both hinge branches (above-floor identity across
the three floors; below-floor value, live gradient, and the restoring branch reaching the loop). 8 new
unit tests, 102 in the suite, all green.

## Not done

`aug_hfnoise` (needs its σ pilot — and see below) · an `off` arm for the winning cell · more seeds at
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

---

## Wave 6 — the encoder decision (2026-08-26, post-hoc, 30 runs, ~13 h, 0 runner failures)

**Verdict: P11-C on the encoder question and P11-F on the mechanism multiplier. `hann0p3` ships.**

Wave 6 does not ask exp09's question — that one is answered and stays answered. It asks the downstream
question roadmap D17 needed settled: which encoder does the ML4PS paper ship? The switch candidate was
`exp09_dpss_impulse_w0p025`, the one cell that had reached the encoder-selection gate with a valid dose.

**Two gates, both named, neither substituted for the other.** `G9-artifact <= 1.5` is pre-registered
(2026-08-15) and governs exp09's *mechanism* verdicts; it is untouched and nothing above is rescored.
`G17-artifact <= 2.0` was declared 2026-08-25 **before any wave-6 run** and governs the *encoder choice*
only. The published claim — no cell passes both gates across ten interventions on three axes — stands
exactly as written.

### What ran

| cell | seeds | why |
|---|---|---|
| `exp09_dpss_impulse_w0p025` | **6-11** (to 12) | the switch candidate, extended to the pre-registered final n |
| `exp07_hann0p3_fbwd` | **6-11** (to 12) | reference extension: every exp09 delta is paired by seed, so the reference needed 12 too |
| `exp09_hann_w0p025` | 0-5 | plain-Hann twin of `w0p025` — one-factor mechanism contrast at constant pressure |
| `exp09_hann_w0p10` | 0-5 | plain-Hann twin of `w0p10` — the second point that measures how the multiplier moves |
| `exp09_dpss_impulse_w0p025_off` | 0-5 | dyn-off insurance for the paper's arm contrast; never an input to the switch rule |

### Footing — both reproductions exact before anything new was read

```
severity, w0p025 seeds 0-5 re-run vs published          : max abs diff 0.0
probe,    hann0p3_fbwd seeds 0-5 vs exp07_aux_gap_6seed : 72 rows, max abs diff 0.0
6-seed switch ratios reproduce the manifest's pre-registered values exactly:
  eb 0.93 . pulsating 0.07 . rotation 1.52 . transit 0.39
```

### Severity — G9 1.5 (mechanism) and G17 2.0 (encoder), side by side

| cell | n | severity | sd | `max_pos` span | G9 (1.5) | G17 (2.0) |
|---|---|---|---|---|---|---|
| `dpss_impulse_w0p025` | 12 | **1.525** | 0.115 | 19-234 | fail | **PASS** |
| `dpss_impulse_w0p025_off` | 6 | 1.596 | 0.075 | 93-233 | fail | **PASS** |
| `hann_w0p025` | 6 | 7.487 | 3.115 | 119-130 | fail | fail |
| `hann_w0p10` | 6 | 10.853 | 1.988 | 113-131 | fail | fail |
| `exp07_hann0p3_fbwd` | 12 | 11.998 | 4.070 | 122-150 | fail | fail |

**`max_pos` span is a second, independent read on the same question, and it is free.** A planted impulse
sits at one address every seed; noise wanders. The candidate's maximum lands anywhere in 19-234 across
its twelve seeds, while every plain-Hann cell locks into 113-131. The ratio and the address agree.

### The result nobody predicted: the candidate is VOID at 12 seeds

Seed 6 finished at **0 active units, KL 0.000, dose 0.000, `val/recon` 1.201** against ~0.816 for every
healthy seed — a total posterior collapse. G9-dose is applied **before** the probe is read, so the row
is void and its probe verdict is not evidence in either direction.

This is the wave-5 P9 finding arriving at the one cell that had escaped it. Wave 5 established that
every weight strictly between `aux_none` and `w0p025` which reaches the artifact gate does so only by
killing a seed. `w0p025` was the exception — clean, useful, valid at 6 seeds. **At 12 seeds it is not an
exception; 6 seeds was simply too few to see a 1-in-12 collapse mode.** The honest generalisation is now
stronger and simpler than the one wave 5 could support: *there is no weight of this objective that is
simultaneously clean, useful and reliably valid* — the qualifier that moved is "reliably".

Note which direction the collapse pushes the gate: a collapsed decoder reconstructs a constant, so its
error profile is flat and its severity (1.200) sits near the no-pressure floor. It drags the 12-seed
mean **down**, from 1.555 to 1.525. The void seed *flatters* G17; it does not threaten it.

### The verdict does not turn on the void call

Dropping a seed selected post-hoc on a probe-correlated outcome is precisely the estimator swap the
VOID rule exists to forbid, so it is not done. It was nonetheless computed, for one reason: to find out
whether a reader has to adjudicate the void question at all. They do not.

| clause | as written (12 seeds, VOID) | seed 6 dropped (11 seeds, **not a decision input**) |
|---|---|---|
| (1) G17-artifact <= 2.0 | 1.525 **PASS** | 1.555 **PASS** |
| (2) visual sign-off | **clean** (see below) | clean |
| (3) G9-noregress `mean` | PASS | PASS |
| (3) G9-noregress `mean_std` | **FAIL: pulsating** (-1.40) | **FAIL: pulsating** (-1.24) |
| (4) >=2 of 4 v1 above 2*SE at `mean` | 0/4 | 2/4 (eb 1.01, rotation 1.13) |

**`G9-noregress` fails on `pulsating` at `mean_std` under both estimators**, and clause (3) requires
both readouts. The switch cannot fire either way. The six new seeds moved `pulsating` against the
candidate (`mean_std` ratio -0.62 at 6 seeds -> -1.24 at 11 -> -1.40 at 12); the 6-seed footing that made
this wave look decisive was the optimistic tail of a noisier distribution than it appeared to be.

**12 was the pre-registered final n. There is no further extension** — that would be optional stopping,
and the whole point of writing the number down in advance was to make this moment non-negotiable.

### The visual clause reads the other way, and that is worth saying out loud

`experiments/exp09_wave6_signoff.png` (fresh unseeded sample, re-drawn every run). The incumbent plants
a violent spike at position ~125-145 in **every sampled window**, independent of what the flux does
there; the candidate has none, and tracks high-frequency structure the incumbent smooths away. On the
6-seed mean position profile the incumbent peaks at **5.31x** the interior median at position 125, the
candidate at **1.19x** at position 30.

So the picture prefers the candidate and the probe does not. That disagreement is the honest result and
it is *why* the rule was pre-registered with four conjunctive clauses rather than one: a clean picture
was never sufficient, in either direction. It also re-states the project's oldest finding in a new
place — **the artifact is real, visible, reproducible, and probe-harmless** — which is exactly what
exp07 measured (`numax_hon` +0.0001 +/- 0.0037) and what this wave now confirms at 12 seeds on four v1
tasks.

### P11-F: the mechanism multiplier does not decay — it *grows* as pressure falls

Pre-registered branches: **P11-D** (multiplier ~ weight-independent -> hann ~ 3.5 / 7.5) or **P11-E**
(multiplier decays toward 1 -> hann_w0p025 ~ 1.7-2.0 and becomes a live, simpler candidate). Measured,
against the weight-matched dpss+kurtosis twins, on both the raw ratio and the floor-subtracted excess
(the metric's null is `aux_none` 1.172, not 1.0, so a raw ratio silently credits the mechanism with a
floor both cells share):

| weight | hann only | dpss+kurtosis | mult (raw) | mult (excess over floor) |
|---|---|---|---|---|
| 0.025 | 7.487 | 1.525 | 4.91 | **17.88** |
| 0.10 | 10.853 | 2.203 | 4.93 | **9.40** |
| 0.30 | 11.998 | 2.913 | 4.12 | **6.22** |

Neither branch. **P11-F**, recorded as a mismatch and not snapped to the nearest branch (the P8
precedent). The excess multiplier nearly triples as pressure drops 12x, and `hann_w0p10` at 10.85 is the
literal example P11-F names ("far above 7.5").

The mechanism is not a fixed discount on the artifact — **it is the only thing that makes the weight
knob work at all.** With the machinery, severity falls 2.913 -> 2.203 -> 1.525 as weight falls 0.30 ->
0.10 -> 0.025. Without it, plain Hann barely moves: 11.998 -> 10.853 -> 7.487 over the same 12x range,
still at the incumbent's level. This retires the strong form of F12 from the other side: F12 said the
gate measures pressure rather than mechanism, and the honest statement is now that **pressure is a lever
only in the presence of the mechanism**.

**Independently corroborated by the bump ablation** (`exp09_bump_ablation_w6.csv`, budget 2, purchase
asymmetry = bump% / rand%), which is a different instrument on the same claim:

| cell | weight | spectral | asymmetry |
|---|---|---|---|
| `aux_none` (aux term off) | 0 | 63.97 | **0.01** — the honest-residual floor |
| `dpss_impulse_w0p025` | 0.025 | 5.33 | 2.17 |
| `dpss_impulse_w0p025_off` | 0.025 | 5.02 | 2.29 |
| `dpss_impulse_w0p10` | 0.10 | 2.97 | 2.28 |
| `hann_w0p025` | 0.025 | 3.68 | **17.33** |
| `hann_w0p10` | 0.10 | 2.20 | 15.43 |
| `exp07_hann0p3_fbwd` | 0.30 | 1.98 | 12.31 |

Plain Hann's purchase **rises** as pressure falls (12.3 -> 15.4 -> 17.3); the machinery arm sits flat at
~2.2 at every weight. Note also that `hann_w0p025` reaches a *better* aux loss (3.68) than its twin
(5.33) — at one twelfth of `hann0p3`'s weight it is still buying the term with an impulse rather than
earning it.

### The off arm — insurance that was not needed, and one number worth keeping

`exp09_dpss_impulse_w0p025_off` was added in case the switch fired and stranded R1's dynamics-
specificity contrast on a retired encoder. The switch did not fire, so the insurance goes unused. Its
one carry-forward: severity 1.596 (dyn-off) against 1.525 (dyn-on) — **artifact suppression at this
recipe is not dynamics-dependent**, which is a small but free control on the wave-3/4 weight story.
G9-dose is inapplicable to it by construction (lambda = 0, so dose is identically 0) and the VOID rule
does not apply, as was already the case for exp07's off arms.

### Reproduce

```bash
# training (USER TERMINAL ONLY -- GPU + W&B online); resumable, DONE-marker skipping
.\experiments\run_exp09_loss_exploit_ladder.ps1 -MaxHours 13.0

# eval fan (CC-runnable). THREE invocations: the checkpoint arm differs, and the mu cache is keyed
# {cell}_seed{seed}.npz with NO checkpoint in the key, so a shared cache dir silently reuses the
# wrong arm. exp07_hann0p3_fbwd seeds 0-5 predate best_recon_only.pt and must be read at
# best_recon_aux; all 12 are therefore scored there.
python -m swm.eval.dump_wandb_history --groups exp09 exp07_hann0p3_fbwd \
    --out experiments/exp09_forensics/curves_exp09
python experiments/analyze_exp09_artifact.py --cells exp09_dpss_impulse_w0p025 \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --out-prefix experiments/exp09_impulse_w6_cand
python experiments/analyze_exp09_artifact.py --cells exp07_hann0p3_fbwd \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --out-prefix experiments/exp09_impulse_w6_ref
python experiments/analyze_exp09_artifact.py --cells exp09_hann_w0p025 exp09_hann_w0p10 \
    exp09_dpss_impulse_w0p025_off --seeds 0 1 2 3 4 5 --out-prefix experiments/exp09_impulse_w6_new
python experiments/analyze_exp07_diagnostics.py --stages mu stars --cells exp09_dpss_impulse_w0p025 \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --ckpt best_recon_only \
    --cache-dir experiments/exp09_forensics/mu_cache_w6_ro --out-prefix experiments/exp09_diag_w6_ro
python experiments/analyze_exp07_diagnostics.py --stages mu stars --cells exp07_hann0p3_fbwd \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --ckpt best_recon_aux \
    --cache-dir experiments/exp09_forensics/mu_cache_w6_ref --out-prefix experiments/exp09_diag_w6_ref
python experiments/analyze_exp07_diagnostics.py --stages mu stars \
    --cells exp09_dpss_impulse_w0p025_off exp09_hann_w0p025 exp09_hann_w0p10 --seeds 0 1 2 3 4 5 \
    --ckpt best_recon_only --cache-dir experiments/exp09_forensics/mu_cache_w6_new \
    --out-prefix experiments/exp09_diag_w6_new
# the bump ablation writes to its OWN wave-6 file. Do NOT point a --cells-scoped run at the default
# --out: that overwrites the 14-cell table and silently breaks notebook section D11 (happened
# 2026-08-17, repaired 2026-08-18).
python experiments/analyze_exp09_bump_ablation.py --cells exp09_hann_w0p025 exp09_hann_w0p10 \
    exp09_dpss_impulse_w0p025_off --seeds 0 1 2 3 4 5 --out experiments/exp09_bump_ablation_w6.csv
python experiments/analyze_exp09_wave6_gates.py     # both gates, VOID before probe, P11 branches
python experiments/plot_wave6_signoff.py            # the visual clause, fresh unseeded sample
```

### Wave-6 outputs

`exp09_impulse_w6_{cand,ref,new}_{runs,summary}.csv` · `exp09_impulse_w6_*_profile.parquet` ·
`exp09_diag_w6_{ro,ref,new}_{probe_summary.csv,star_scores.parquet}` · `exp09_bump_ablation_w6.csv` ·
`exp09_wave6_signoff.png` · `exp09_forensics/mu_cache_w6_{ro,ref,new}/` ·
`exp09_forensics/curves_exp09/` (now 120 runs)

### What wave 6 changes downstream

- **Encoder is settled: `hann0p3`.** D17 closes ahead of its Sept 1 deadline. F1/C3/C3b need no re-run —
  they were already computed on `hann0p3` mu, and the "encoder-agnostic code" hedge goes unused.
- **No further encoder tuning**, per D17 as written ("if nothing passes both gates -> `hann0p3` ships").
- The 12-seed reference (`exp07_hann0p3_fbwd` seeds 0-11) is now available for any later paired test
  that wants more than 6 seeds on the shipping encoder.

---

## Wave 7 — the collapse wave: is the posterior collapse fixable? (2026-08-28, 24 runs, ~10 h)

**Verdict: the collapse IS fixed, and the cell still fails both gates. P12-D fired as pre-registered.**
Eleven interventions on FOUR axes now — mechanism, amount, level, and the KL floor — and no cell passes
`G9-artifact` and `G9-noregress` together.

Wave 6 lost a seed of `w0p025` to total posterior collapse (KL 0.000, 0 active units) and voided the
project's best-probing cell. Wave 7 asks the narrow follow-up: is that a fixable stability bug, and does
the cell survive 12 seeds once it is fixed? **Scope: this does NOT reopen D17** — `hann0p3` still ships,
and this is a Lane-M / journal cell (rule 6 bars new mechanism experiments pre-submission).

### The design: a 2x2 whose fourth corner was already paid for

```
                free_bits 0.00                     free_bits 0.01
  w = 0.025     wave 6: 12 seeds, 1 collapse       NEW: 6 -> 12 seeds (stage 2)
  w = 0.030     NEW: 6 seeds                       NEW: 6 seeds
```

`free_bits = 0.01` was **derived, not copied**. exp03's winning floor worked because it sat *below*
true KL — "with the floor below true KL, beta finally exerts live gradient" (`exp03_sweep_README.md`),
geometry 2.56 nats against KL ~3.0, ratio 0.853. exp09's measured KL is 1.588 at w=0.025 and 1.638 at
w=0.03, so the same ratio gives fb 0.0106 and 0.0109 -- both round to 0.01, which is what makes this a
clean factorial. Copying exp03's literal 0.02 would have put the floor at **1.56x above** KL, binding on
every seed and destroying the property that made it win.

### Stage 1 (18 runs, 6 seeds each): every arm clean, and the levers separate on the probe

| cell | collapses | dose (min) | KL | `n_active` (min) | severity | probe at `mean` |
|---|---|---|---|---|---|---|
| `w0p025` fb=0.00 (wave 6) | **1 / 12** | 0.676 (0.000) | 1.456 | 6.2 (0.0) | 1.525 | VOID |
| `w0p025` fb=0.01 | 0 / 6 | 1.097 (0.984) | 1.918 | 94.5 (63.4) | 1.471 | 2/4 past 2*SE |
| `w0p03` fb=0.00 | 0 / 6 | 0.708 (0.680) | 1.671 | 6.2 (5.0) | 1.606 | **0/4** |
| `w0p03` fb=0.01 | 0 / 6 | 1.112 (1.090) | 2.079 | 78.2 (65.7) | 1.674 | 1/4 |

**Raising the weight fixes the collapse and costs the probe.** `w0p03` fb=0.00 is the one arm where the
collapse metric is still honestly measured (unfloored, `n_active` 5-8), and it is clean at 6 seeds -- but
rotation falls 0.5821 -> 0.5587 and `mean4` to 0.5679, *below the incumbent's* 0.5699. Pressure buys
stability by degrading the representation. Only the floor at the LOW weight bought stability without
that cost.

### `n_active` is NOT diagnostic under a floor -- and per-dim KL says the floor CONCENTRATED the latent

`active_unit_kl_threshold` is 0.01 and `free_bits` is 0.01: **the floor sits exactly at the counting
threshold**, so the 6 -> 94 jump in "active units" is largely mechanical. `swm.eval.kl_dim_report` on
`best_recon_only` settles what is really there:

| cell | KL_total | dims >0.01 | dims >0.02 | dims >0.05 | frac KL in top-6 dims |
|---|---|---|---|---|---|
| `w0p025` fb=0.00 | 1.491 | 7.3 | 4.6 | **3.4** | **79%** |
| `w0p025` fb=0.01 | 1.951 | **109.2** | 2.8 | 1.7 | 34% |
| `w0p03` fb=0.00 | 1.721 | 6.0 | 5.3 | **4.0** | **85%** |
| `w0p03` fb=0.01 | 2.099 | **95.0** | 1.3 | 1.2 | 40% |

The ~100 extra "active" dims sit pinned in the band [0.01, 0.02]. The floor did not add capacity -- the
floored cells carry **fewer** strongly-informative dims (1.7 vs 3.4 above 0.05). This is wave 5's
dilution finding reproduced through a different knob, and it retires `n_active` as a collapse indicator
for any floored cell, exactly as wave 5 retired the ablation `asymmetry` ratio for them.

**A consequence for the VOID rule itself, recorded here rather than assumed away.** `free_bits` removes
the gradient below the floor, so `KL_total` can never reach 0 and a floored cell **cannot register a
collapse in the terms G9-dose names**. "12/12 clean" is therefore evidenced by `val_recon` (0.810-0.865,
against the collapsed seed's 1.201) and by the probe, NOT by the metric the rule is written on. Any
future floored cell must be judged the same way.

### Stage 2 (6 runs): the 6-seed pass did not replicate -- for the third time

Pre-registered before stage 1: any arm clean at 6 seeds gets seeds 6-11 before any claim. All three
qualified; the user narrowed the extension to `w0p025_fb0p01` alone. **Recorded as a deviation** -- see
the manifest -- because extending only the best-looking arm is selection on the outcome. What it costs
is the cross-weight attribution: both `w0p03` arms stay at 6 seeds, so "the floor works, the weight does
not" remains a 6-seed statement. What it buys is the most informative 6 runs available: a matched-n
12v12 one-factor test on the `free_bits` axis, against a 12-seed reference.

| | 6 seeds (stage 1) | **12 seeds (final)** |
|---|---|---|
| collapses | 0/6 | **0/12** (dose 0.958-1.177, `val_recon` 0.810-0.865) |
| **G9-artifact <= 1.5** | **1.471 PASS** | **1.546 FAIL** (seeds <=1.5: 5/12) |
| G17-artifact <= 2.0 | PASS | PASS |
| G9-noregress `mean` | PASS | **FAIL: pulsating** (-0.0163, -1.09) |
| G9-noregress `mean_std` | PASS | **FAIL: pulsating** (-0.0180, -1.03) |
| >=2 of 4 past 2*SE at `mean` | 2/4 | **2/4** (eb +0.0151 = 1.83, rotation +0.0206 = 2.06) |

```
severity, seeds 0-5 : 1.399 1.493 1.424 1.549 1.578 1.384   mean 1.471   PASS
severity, seeds 6-11: 1.529 1.456 1.714 1.778 1.646 1.601   mean 1.621
all 12                                                       1.546   FAIL
```

The four highest severities in the whole set are stage-2 seeds. **A 6-seed gate pass in this project has
now failed to replicate three times** (wave 6's dose, wave 6's probe, wave 7's artifact); treat any
6-seed pass as provisional by default.

What DID hold: eb and rotation barely moved (1.91 -> 1.83, 2.29 -> 2.06). The gains are real; they are
simply not enough, because the gate is a conjunction.

### P12-D, pre-registered 2026-08-26, FIRED -- with its predicted mechanism

> "P12-D (free_bits COSTS the probe): fixes the collapse but regresses on G9-noregress where its
> unfloored twin does not -> the floor bought stability with representation quality. Watch this:
> floor/KL is 0.81 at w=0.025 and 0.78 at w=0.03, close enough to bind on the weakest seeds."

At 12 seeds the unfloored twin regresses on pulsating at ONE readout; the floored cell regresses at
**both**. The floor bound, and it bound on pulsating -- the spectral task, which is what a latent
diluted across ~109 floor-pinned dimensions should be expected to lose.

**The bistability gives that mechanism a face.** The 12 seeds split 6/6 into two basins, and the
pulsating cost lives almost entirely in one of them:

| cluster | seeds | KL | units | pulsating | eb | rotation | `mean4` |
|---|---|---|---|---|---|---|---|
| A | 0,2,3,8,9,11 | ~2.05 | 60-78 | **-0.0041** | +0.0152 | +0.0288 | 0.5797 |
| B | 1,4,5,6,7,10 | ~1.70 | 111-124 | **-0.0285** | +0.0149 | +0.0124 | 0.5696 |

More dims pinned at the floor -> worse pulsating; `eb` is identical across basins (+0.0152 / +0.0149).
**This does NOT rescue the cell.** Splitting on a training quantity after seeing the result is still
post-hoc, and cluster A is 6 seeds -- the exact n that has now misled this project three times. It is
recorded as hypothesis-generating: if a floor can be made to land in basin A reliably (a lower fb, or a
floor with a cap on pinned units), the trade P12-D describes might be avoidable. That is a new
experiment under a new pre-registration, not a reading of this one.

### Absolute PR-AUC at `mean`, each cell at its own final n

| cell | pulsating | eb | rotation | transit | `mean4` |
|---|---|---|---|---|---|
| `hann0p3_fbwd` (ref, 12) | **0.8078** | 0.7741 | 0.5569 | 0.1458 | 0.5712 |
| `w0p025` fb=0.00 (12, VOID) | 0.7971 | 0.7503 | 0.5596 | 0.1413 | 0.5621 |
| `w0p025` fb=0.01 (12) | 0.7915 | **0.7892** | **0.5775** | 0.1404 | 0.5746 |
| `w0p03` fb=0.00 (6) | 0.7908 | 0.7775 | 0.5587 | 0.1445 | 0.5679 |
| `w0p03` fb=0.01 (6) | 0.8015 | 0.7873 | 0.5730 | **0.1517** | **0.5783** |
| untrained | 0.7683 | 0.7124 | 0.5315 | 0.0805 | 0.5232 |

`mean4` is a REPORTED SECONDARY, never a gate (decision 2026-08-26): PR-AUC is prevalence-scaled, so an
unweighted mean is dominated by the high-prevalence tasks. Note it disagrees with the gates -- the best
`mean4` belongs to a 6-seed cell that clears only 1 of 4 tasks -- which is precisely why it is not one.

### What wave 7 changes

- **The collapse has a known fix.** `free_bits = 0.01` at w=0.025 gives 12/12 healthy seeds. Anyone
  re-running the low-weight recipe should carry it.
- **`n_active` is retired for floored cells**, and the VOID rule's metric is undefined for them --
  judge on `val_recon` and the probe instead.
- **The exp09 headline is unchanged and now broader:** eleven interventions, four axes, still no cell
  passes both gates. The floor was the most promising untried lever and it failed in a *pre-registered*
  way, which is a stronger result than an unexplained failure.
- **`hann0p3` still ships.** D17 stays closed; nothing here reopens it.

### Reproduce

```bash
# training (USER TERMINAL ONLY -- GPU + W&B online); resumable, DONE-marker skipping
.\experiments\run_exp09_loss_exploit_ladder.ps1 -MaxHours 8.0    # stage 1, 18 runs
.\experiments\run_exp09_loss_exploit_ladder.ps1 -MaxHours 3.0    # stage 2, 6 runs

# eval fan (CC-runnable). All wave-7 cells read best_recon_only -- decision A3.
python -m swm.eval.dump_wandb_history --groups exp09 exp07_hann0p3_fbwd \
    --out experiments/exp09_forensics/curves_exp09
python experiments/analyze_exp09_artifact.py --cells exp09_dpss_impulse_w0p03 \
    exp09_dpss_impulse_w0p03_fb0p01 exp09_dpss_impulse_w0p025_fb0p01 --seeds 0 1 2 3 4 5 \
    --out-prefix experiments/exp09_impulse_w7
python experiments/analyze_exp09_artifact.py --cells exp09_dpss_impulse_w0p025_fb0p01 \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --out-prefix experiments/exp09_impulse_w7s2
python experiments/analyze_exp07_diagnostics.py --stages mu stars --cells exp09_dpss_impulse_w0p03 \
    exp09_dpss_impulse_w0p03_fb0p01 exp09_dpss_impulse_w0p025_fb0p01 --seeds 0 1 2 3 4 5 \
    --ckpt best_recon_only --cache-dir experiments/exp09_forensics/mu_cache_w7_ro \
    --out-prefix experiments/exp09_diag_w7_ro
python experiments/analyze_exp07_diagnostics.py --stages mu stars \
    --cells exp09_dpss_impulse_w0p025_fb0p01 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 \
    --ckpt best_recon_only --cache-dir experiments/exp09_forensics/mu_cache_w7s2_ro \
    --out-prefix experiments/exp09_diag_w7s2_ro
# the per-dim KL evidence for the dilution finding (n_active is NOT diagnostic under a floor)
python -m swm.eval.kl_dim_report --ckpt-glob \
    "experiments/exp09_dpss_impulse_w0p0*/models/B_seed*/best_recon_only.pt" \
    --out experiments/exp09_forensics/kl_dim_w7
```

### Wave-7 outputs

`exp09_impulse_w7{,s2}_{runs,summary}.csv` · `exp09_impulse_w7*_profile.parquet` ·
`exp09_diag_w7{,s2}_ro_{probe_summary.csv,star_scores.parquet}` ·
`exp09_forensics/kl_dim_w7/{kl_dim_long,ckpt_summary}.csv` ·
`exp09_forensics/mu_cache_w7{,s2}_ro/` · `exp09_forensics/curves_exp09/` (now 144 runs)
