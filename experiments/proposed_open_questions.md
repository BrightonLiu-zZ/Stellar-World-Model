# Proposed open questions — staging file

**Status:** proposed 2026-08-09, none yet entered in [open_questions.md](open_questions.md).

Companion to [open_questions.md](open_questions.md) (what is unmeasured) and
[cross_experiment_findings.md](cross_experiment_findings.md) (what was measured). This file is the
**staging area**: candidates land here with their full evidence trail, and move to the ledger as
one-paragraph entries once accepted. It exists because these were surfaced by reading code and
READMEs against each other rather than by an experiment — the gap between two documents is the
finding, and it is expensive to re-derive.

Append new candidates at the next free ID. Do not renumber; ledger IDs are permanent.

| ID | candidate | class | cost |
|---|---|---|---|
| **Q13** | v1 rotation probe ignores ADR-0004's P ≤ 5 d scope | label scope, report-level | zero-GPU, one probe re-fit |
| **Q14** | `eb` has no in-window coverage filter (no ADR-0009 analogue) | label hygiene | zero-GPU, one CPU fold pass |
| **Q15** | encoder-capacity arms (`z32`/`whalf`) parked at exp04, never re-tested on the frozen recipe | architecture, parked lever | **one overnight GPU wave + a pilot** |
| **Q16** | can *"the latent learns the physics"* be saved as a claim, and what would it take? | framing / claim-level | zero-GPU for 4 of 5 tests |

Q13 and Q14 share a shape: **a documented rule the probe path does not implement.** Neither changes
any *paired* delta (both arms of every ablation share the defect) and neither is a bug in the model —
they change what the reported number **means**. Q15 is a different animal: a lever that was measured,
won, and was parked for reasons that have all since expired. Q16 is different again: the evidence
already exists on both sides, nobody has assembled it, and two of the measurements that killed the
framing may have been mis-specified.

---

## Q13 — the v1 rotation probe does not implement ADR-0004's P ≤ 5 d scope limit

**Status:** open · report-level · no training, no extraction · **cost: one probe re-fit on cached µ**

### What is documented

[ADR-0004](../docs/adr/0004-v1-rotation-scope-limit.md) (Accepted, 2026-05-24) is unambiguous:

> **Primary rotation classification (v1):** positive class is `rotation = 1 AND rotation_period ≤ 5
> days`. Stars with `rotation = 1 AND rotation_period > 5 days` are excluded from primary F1 and
> ROC-AUC but reported separately in the stratified-by-P_rot evaluation. The negative class
> (`rotation = 0`) is unchanged.

The physical argument behind it is not in dispute: one packed segment spans 4096 cadences ≈ **5.7 d**,
SpinSpotter (Holcomb+2022) puts reliable period recovery at ~⅓ of the baseline (1.9 d strictly), and
TARS periods in the corpus run 0.5–25 d. A 15-day rotator's cycle is not in any tensor the encoder
ever sees. `architecture.md` carries the same rule plus the four pre-registered strata
(≤2 d / 2–5 d / 5–10 d / >10 d) and calls their monotonic degradation "a paper finding in its own
right".

### What is implemented

The period never enters the v1 classification path:

| file:line | what it does |
|---|---|
| `src/swm/data/subset.py:123` | `keep_cols = [...] + ["rotation", "flare_ever"]` — bakes the **raw TARS 0/1 column** into `processed/subset/subset_tics.parquet`. `rotation_period` is not carried into the subset at all. |
| `src/swm/eval/readout_sweep.py:204` | `label_of[task] = dict(zip(subset["tic_id"], subset[task]))` — reads that column straight; `task = "rotation"` is one of `tasks_default`. |
| `src/swm/eval/{skyline,mil_sweep,mil_learned,rollout_eval}.py` | same `rotation` column, same absence of a period filter. |

Grepping `rotation_period` across `src/swm/eval/` returns matches **only** in
`new_task_scorecard.py`, where the downstream *regression* probe does apply it
(`new_task_scorecard.py:280-281`, `canon["rotation_period"] <= 5`). So the cut exists exactly once,
on the task ADR-0004 discusses second, and is absent from the task ADR-0004 discusses first.

### Why it matters

1. **Every reported v1 rotation number is on a different population than the one the ADR defines.**
   That includes the headline dynamics results: exp07 `hann0p3` fbwd−off rotation **+0.046 ± 0.013**
   (6 seeds, `mean_resid`), exp08 G-gru rotation **+0.030 ± 0.011**, the exp05 criterion-1 rotation
   entry, and every rotation cell in `readout_sweep.csv` back to exp02.

2. **It compounds with the confound already tracked as Q9.** `subset.py:90` defines quiet as matched
   in **no** catalogue *including rotation*, so 0 of 10,000 quiet negatives are rotation-positive
   against a corpus rate of ~13% (25,369 / 195,883). Q9 says the negatives are wrong; Q13 says the
   positives are wrong too, in the opposite direction — out-of-scope positives are stars whose
   *periodicity* is invisible but whose *activity* (elevated within-window variance, slope,
   granulation-scale power) is not. Given that the latent is amplitude-dominated
   (`evr_pc1` 0.926, `r2_amp_pc1` 0.967) and that residualising amplitude out drops rotation
   0.558 → 0.424, the live hypothesis is that a chunk of the rotation signal is an activity
   detector reading through an amplitude channel, on stars whose labelled quantity is unresolvable.

3. **The stratified table was pre-registered and appears never to have been produced.** No
   `exp05`–`exp08` artifact carries a P_rot-bucketed rotation score. If the monotonic-degradation
   result is claimed in the paper as "the encoder respects the physical limit of its input", it has
   to be measured, and Q13's test produces it as a by-product.

4. **The first number is unknown.** Nobody has printed how many of the v1 subset's rotation
   positives have P > 5 d. If it is 5%, this is a footnote. If it is 40%, the rotation row means
   something different from what the paper will say it means.

### Cheapest test

No GPU, no re-extraction, no re-subset. µ is already cached
(`experiments/exp07_forensics/mu_cache/`, `experiments/exp08_*`), and the estimator is the one that
already reproduces `exp07_aux_gap_6seed.csv` exactly on all 300 rows.

1. **Count first (5 minutes, pure pandas).** Join `processed/subset/subset_tics.parquet` to
   `labels/variability_labels_star.csv` on `tic_id`; report, per split, `rotation = 1` broken into
   the four ADR-0004 strata (≤2, 2–5, 5–10, >10 d) plus `period NaN`. This alone decides whether the
   rest is worth running.
2. **Re-fit, paired, on the same cached µ.** For the frozen recipe's arms (`exp07_hann0p3_{off,fbwd}`,
   6 seeds, `mean_resid`), fit the rotation probe three ways:
   - `all` — current behaviour, the reproduction control (must match `exp07_aux_gap_6seed.csv`);
   - `inscope` — ADR-0004 as written: drop `rotation = 1 AND P > 5 d` stars from **both** train and
     test, negatives unchanged;
   - `strata` — score the four buckets separately against the shared negative pool.
3. **Report** trained, untrained-ref, and the fbwd−off delta for each, with the 6-seed SE.

Suggested home: a `--rotation-scope` stage on `experiments/analyze_exp07_diagnostics.py` (it already
owns the µ cache and the star-level probe), writing `exp09_rotation_scope.csv`. Keep the `all` arm in
the output permanently so the two populations are never confused again.

### Decision rule (pre-register before looking)

- **`inscope` delta within 2·SE of `all`** → the scope limit is cosmetic on this corpus. Report the
  in-scope number as the headline (it is the one the ADR promises), footnote the agreement, close.
- **`inscope` delta materially larger** → out-of-scope positives were diluting a real result; the
  paper's rotation row improves and the physical-limit story gains its evidence.
- **`inscope` delta materially smaller, or collapses toward zero** → the rotation result was being
  carried by unresolvable stars, i.e. by activity rather than rotation. That is a **reporting
  change**, not a retraction: rotation is already v1-supplementary, and the honest framing
  ("the probe reads activity, not periodicity") is compatible with everything else measured. It
  would also want a sentence in the exp07/exp08 READMEs, since both quote a rotation gate.
- **Strata non-monotonic** → the pre-registered "encoder respects the physical limit" claim does not
  survive and should be dropped from the paper rather than argued around.

### What it does *not* change

Paired ablation conclusions. G-C1, G-gru, G-prior and the exp05 criterion-1 verdict all compare two
arms scored on the identical population, so the defect cancels. Q13 is about what the rotation row
means, not about whether dynamics helps.

### Risks / why it might be a non-issue

- The v1 subset is enriched for transit/eb/pulsating positives, and rotation positives ride in as a
  side-effect of that selection (Q9). The in-scope subset could be small enough that the re-fit is
  noise-dominated — the step-1 count is exactly the guard against spending time on that.
- ADR-0004 also specifies a 2 d and a 10 d sensitivity check. Both fall out of the `strata` arm for
  free; do not add them as separate cuts.

---

## Q14 — `eb` has no in-window coverage filter, and nobody has measured whether it needs one

**Status:** open · label-hygiene · no training · **cost: one CPU pass over ~1,288 stars' `.npz`**

### The asymmetry

`transit` is the only task in the project whose positives are checked against the data:

- `src/qc/transit_window_coverage.py` folds each whitelisted TOI's ephemeris (`pl_orbper`,
  `pl_tranmid` → BTJD, `pl_trandurh`) onto the **real cadence timestamps stored in
  `processed/sequences/*.npz`** and marks a window covered iff it holds ≥1 in-transit cadence.
- [ADR-0009](../docs/adr/0009-transit-observed-coverage-filter.md) turns that into a label rule and
  finds **89 stars (`DROP_NO_TRANSIT`) that are windowed, foldable, and contain zero transits in our
  cadences** — every window of such a star carries `transit = 1` while no transit is ever in frame.
  Pure label noise, identified only because someone folded.

`eb` gets no equivalent. A star is an EB positive iff it is in Villanova/Prša+2022 and has ≥1 packed
window (`docs/data-sources-and-features.md`: 1,936 star-level → **1,541 window-grounded** → **1,288**
after the 5a/5b outlier cleanup). Nothing asks whether an eclipse is in frame.

### Why this is not merely symmetric bookkeeping

Four measured facts make EB the *more* likely place for the defect, not the less:

1. **The corpus and the catalogue do not share epochs.** Villanova is vetted on Sectors 1–26; the
   corpus holds **zero** `.npz` in S1–26 — all windows are S27+. The label transfers on the argument
   that *EB-ness is an epoch-stable stellar property*, which is correct for the star and says nothing
   about whether an eclipse lands in the particular S27+ cadences we kept.

2. **Duty cycle is small and period-dependent.** A P = 10 d detached EB with 3 h eclipses puts an
   eclipse in ~1% of its cadences. Only **19 of 196 test-split EB positives have P < 0.5 d** (47 at
   P < 1.0 d), so the contact-binary regime — the one where every window contains signal — is a small
   minority of the label set.

3. **The 20-MAD `absmax` guard preferentially deletes the windows that would have counted.** Of 677
   deleted windows, **332 (49%) sit on 145 EB stars = 37% of affected stars against a ~1.3% base
   rate, ≈28× enrichment** (`data.max_absmax = 20.0`, applied per window at stored granularity in
   `swm/data/pack.py:54-55`). The guard is symmetric on |flux| and drops whole windows, so the
   deepest eclipses are exactly what it removes. Windowed EBs fell 1,541 → 1,288 through it. Some
   stars may have lost the only eclipse they had.

4. **The mechanism the paper reports depends on eclipses being in frame.** The pre-registered
   coverage-paradox reading is that the model does not learn EB periodicity at all — it flags
   eclipses as **rare high-contrast outlier windows**, which is why dispersion pooling (`mean_std`,
   `window_score`/MIL) lifts `eb` and `transit` and does nothing for `pulsating`. Under that
   mechanism a zero-eclipse EB positive is not a weak positive, it is an unlearnable one, and it
   depresses the task the paper leans on hardest (`eb` is the strongest v1 result: exp07 fbwd−off
   **+0.080 ± 0.007**, exp08 ladder off 0.510 → fbwd 0.590).

### The epoch problem, and the way around it

`labels/variability_labels_star.csv` carries `eb_period` but **no epoch** — there is no `BJD0` column,
so the transit recipe (predict event times from P and T₀) is not directly available. Two options:

- **Preferred — epoch-free, fully offline.** `src/qc/eb_villanova_audit.py` already folds each star's
  corpus cadences at the Villanova period into a phase-binned median curve without an epoch
  (`eb_villanova_audit.py:64-93`; "epoch unknown → eclipse may sit at any phase"). Recover the
  eclipse phase empirically as the minimum of that curve, and the duration from the contiguous run of
  bins below baseline. Then map every cadence's phase back to its window and mark in-eclipse. Reuse
  its existing guards verbatim: `phase_filled < 0.8` → **unverifiable, never suspect** (the eclipse
  could hide in unsampled phase), missing period → unverifiable. Those guards are what keep this from
  becoming a purity cut applied to noise.
- **Fallback / cross-check.** Re-query Vizier `J/ApJS/258/16` for `BJD0` and run the true
  transit-style fold on the subset where it is available. Network-dependent → `/astroquery-resilience`
  rules apply (tenacity, `status="error"`, `--resume`). Worth it only as a validation of the
  epoch-free estimator on a few hundred stars.

Both paths reuse `index_npz_by_tic()` from `transit_window_coverage.py:42-53` (one `os.scandir` over
the 400k-file directory instead of re-globbing per TIC — the whole reason that pass is fast).

### Cheapest test

Run in the `astro` env (data-pipeline stage, not `swm`), CPU only, backgrounded with
`PYTHONUNBUFFERED=1`, tqdm over the star loop with `desc=` and `total=`.

Output `labels/qc/eb_window_coverage.csv`, one row per windowed EB positive, mirroring the transit
diagnostic's three granularities:

| column | meaning |
|---|---|
| `tic_id`, `eb_period`, `n_npz`, `n_windows_256` | provenance |
| `phase_filled`, `eclipse_phase`, `eclipse_width_phase` | epoch-free fold products |
| `cov256`, `cov1024`, `cov_segment` | fraction of windows / segments holding ≥1 in-eclipse cadence |
| `n_eclipse_windows` | absolute count — the number that decides `DROP_NO_ECLIPSE` |
| `verdict` | `COVERED` / `NO_ECLIPSE_IN_DATA` / `UNVERIFIABLE` (phase_filled < 0.8 or period NaN) |
| `absmax_deleted_windows` | join to the 5a/5b cleanup manifest — did the guard delete this star's only eclipse? |

Report, in this order:
1. **The headline count**: how many of the 1,288 windowed EB positives are `NO_ECLIPSE_IN_DATA`, and
   how many are `UNVERIFIABLE`. This is the direct analogue of ADR-0009's 89.
2. **The distribution of `cov256`**, with the transit gate's 10% line drawn on it, split by
   `eb_period` bucket (<0.5 d, 0.5–1, 1–5, 5–10, ≥10 d). Prediction from the duty-cycle argument:
   coverage falls monotonically with period, and the ≥5 d bucket is where the zero-coverage stars
   live.
3. **The `absmax` interaction**: among `NO_ECLIPSE_IN_DATA` stars, how many had windows deleted by
   the 20-MAD guard. If that fraction is high, the guard is manufacturing label noise, which is a
   different (and more interesting) finding than catalogue-vs-cadence bad luck.
4. **A probe re-fit, only if step 1 is non-trivial** — drop `NO_ECLIPSE_IN_DATA` stars from train and
   test, re-fit `eb` on the same cached µ, paired over 6 seeds, `mean_resid`. Same estimator as Q13.

### Decision rule (pre-register before looking)

- **`NO_ECLIPSE_IN_DATA` < ~2% of windowed positives** → EB label hygiene is fine; record the number,
  cite it as the answer to the obvious reviewer question ("your EB catalogue is from sectors you do
  not use"), close. This is a genuinely useful outcome even though it changes nothing.
- **Between ~2% and ~10%** → report-level. State the count in the paper's label section next to
  ADR-0009's 89, and fold the drop into the **next** label regeneration alongside ADR-0011's 33
  contaminants — not mid-freeze, same reasoning as [Q10](open_questions.md).
- **> ~10%, or the probe re-fit moves `eb` beyond 2·SE** → this is a label defect large enough that
  the v1 `eb` headline needs the filtered number, and it should be applied before the ML4PS tables
  are finalised.
- **`UNVERIFIABLE` dominates** (poor phase sampling on most stars) → the epoch-free estimator is not
  strong enough on this corpus; escalate to the `BJD0` fallback or drop the question, but do **not**
  report unverifiable stars as clean.

### Interaction with ADR-0011

[ADR-0011](../docs/adr/0011-eb-planet-contamination-drops.md) already removes 33 flagged contaminants
(eb 1,936 → 1,903; window-grounded 1,288 → ~1,285) at the next regeneration, with the v1→v2 delta
measured at **+0.004 ± 0.014 PR-AUC over 304 paired cells**. Q14's drops are a *different* population
— ADR-0011 removes stars that are not EBs, Q14 would remove stars that are EBs whose eclipses we
never observed — but both land in the same regeneration. If Q14 fires, batch the two so the label
file is rebuilt once.

### Risks / why it might be a non-issue

- **The eclipse need not be the only signal.** Ellipsoidal modulation, reflection and out-of-eclipse
  variability are present in every window of a close binary, so a "zero-eclipse" EB may still be
  learnable — unlike a zero-transit TOI, where out-of-transit flux genuinely carries nothing about
  the planet. This asymmetry is the strongest argument that Q14 comes back clean, and it should be
  stated in the write-up either way.
- **Epoch-free phase recovery fails on shallow eclipses**, which biases the diagnostic toward calling
  faint-but-real EBs `NO_ECLIPSE_IN_DATA`. The `phase_filled ≥ 0.8` guard plus an SNR floor borrowed
  from the pass-1 audit (`eb_villanova_audit.py`) mitigate it; report the SNR distribution of the
  drop set so the failure mode is visible rather than assumed away.
- Both Q14 and Q13 are pre-freeze *reporting* work, not experiments. Neither should displace the
  consolidation fan if time is short — but both are cheap enough that "no time" is unlikely to be the
  real reason.

---

## Q15 — the encoder-capacity arms won at exp04, were parked, and every reason for parking them has expired

**Status:** open · architecture lever · **cost: one λ pilot + ~12 runs (one overnight wave)** · collides
with a CLAUDE.md hard constraint, so it needs an explicit decision before it can run

### What was measured

exp04 (2026-07-19/20, 39 runs, 0 failures) swept the encoder axis with each variant gapped against
its **own capacity-matched untrained reference** (`exp04_eval_cache/<variant>/`). Logistic × mean,
gap vs untrained, 2 seeds unless noted:

| variant | change | eb | pulsating | transit |
|---|---|---|---|---|
| `enc_whalf` | channels ÷2: `[32,64,128,256]` → `[16,32,64,128]` | **+0.100** | +0.003 | +0.054 |
| `enc_z32` (3 seeds) | z 128 → 32 | **+0.099** | −0.031 | +0.015 |
| `enc_z64` (3 seeds) | z 128 → 64 | +0.080 | +0.010 | +0.035 |
| `enc_d3` / `enc_w2x` | 3 stages / channels ×2 | +0.077 / +0.074 | −0.035 / +0.016 | +0.033 / +0.036 |
| `enc_d5` | 5 stages | +0.039 | −0.063 | **+0.071** |
| `enc_k9` / `enc_k15` / `enc_z64k9` | kernel 9 / 15 / (64, 9) | +0.046 / +0.067 / +0.042 | ≈0 | +0.033 |
| *baseline z128 recipe* | — | *+0.066* | — | *+0.039* |

Two arms — `enc_z32` and `enc_whalf` — beat the recipe of record on `eb` by **~+0.033 gap**, roughly
half again the baseline's whole eb win. The exp04 README parked them in one clause: *"available if eb
ever becomes the headline."*

### Why they were parked, and why each reason has since expired

| reason at exp04 (2026-07-20) | status now |
|---|---|
| **eb was skyline-closed** — engineered-features − untrained = 0.029 < 2·SE, and `info_in_mu = False` on all seeds, so there was nothing left to buy | Still true *for the v1 eb probe*, but the frozen recipe's whole case now rests on `eb` and `rotation` (exp07 G-C1: eb +0.080 ± 0.007; exp08 ladder off 0.510 → fbwd 0.590). eb is no longer a closed side-task, it is the headline the paper leans on. |
| **architecture locked for the A/B/C ablation** (CLAUDE.md: *"Conv1D-VAE architecture is final; do not change encoder/decoder"*) | Still a live hard constraint — see *Risks*. This is the one reason that has **not** expired, and it is a decision, not a measurement. |
| **z_dim ↔ KL entanglement made the sweep unclean**: `free_bits` is per-dim, so the total floor is 0.02 × z_dim — **2.56 nats at z=128 vs 0.64 at z=32**. The z arms varied two knobs. | **Expired, and this is the strongest argument for re-running.** The frozen `hann0p3` recipe runs **free_bits = 0** (exp07 C3: fb0.02 moves nothing at w256; fb=0 everywhere in exp07/08). At fb=0 the confound is gone — a z sweep on the frozen recipe varies exactly one knob. exp04 could not have measured this cleanly; the current recipe can. |

### Why the exp04 numbers cannot simply be carried forward

Everything the arms were measured under has been replaced:

| axis | exp04 arms | frozen `hann0p3` recipe |
|---|---|---|
| aux | `log_psd` @ weight 0.1, `psd_normalize=false` | **combined** @ 0.3, **Hann-tapered** log-PSD |
| free_bits | 0.02 | **0** |
| dynamics | `variant=B`, `dyn_mode=fwd`, **λ = 1.0** | `fwd_bwd`, **λ = 60**, achieved dose ≈ 1.0 |
| epochs | 60 | **100** |

The dynamics dose alone is a different regime — and F25 says a short-budget run is **not a prefix** of
a long one (at ep40 the ep60 run sits at lr 7.7e-5 vs the ep100 runs' 2.0e-4, and its active set
*freezes* at ~26 while theirs prune to 7–8). An exp04 gap measured at λ=1/ep60/fb0.02 predicts
approximately nothing about the same knob at λ=60/ep100/fb0.

### Why the question is more interesting now than it was then

exp08 turned latent width into a **measured, objective-driven quantity rather than a hyperparameter**:

- the `frozen_fbwd` arm holds the latent **wide open at 69–114 active units** on the identical z=128
  trunk (unattainable random-function targets block pruning), while the winner holds **5.8**;
- `smooth` collapses to **1 active unit** on all 6 seeds and dose parity is unreachable;
- so **z_dim interacts with `dyn_mode`**, and "how many dims are live" is set by what the loss can and
  cannot satisfy — not by the layer size.

That reframes Q15. It is no longer "is 128 too many"; it is **"does the prediction pressure that
creates the content need room to put it, and does squeezing the bottleneck concentrate it or destroy
it?"** — which is the mechanism sentence the paper is currently missing (the loose end flagged in
`open_questions.md`: *what the 1–3 extra active units buy*).

### And the axis has never been scored where it now matters

The downstream ADR-0010 menu has **never** seen an encoder-capacity arm. Two facts make that a real
gap rather than a completeness itch:

1. **Gate 0 (2026-08-07)** was the first time the frozen recipe was scored on the menu at all
   (+0.376 ± 0.025 on `numax_hon` vs six untrained inits). Every asteroseismic number before it was
   an exp05-arm, ep60 number.
2. **exp08's menu run showed the v1 ordering does not transfer.** On the ladder, `linear`, `frozen`
   and `fwd_bwd` are statistically tied on v1 `eb`/`rotation`, yet **`fwd_bwd` leads the entire ladder
   on every ADR-0010 transfer probe** (`numax_hon` 0.802 ± 0.003 vs best rung 0.755; `rotation_period`
   0.677 vs 0.604), with collapsed arms falling **below the untrained floor** (0.287 vs 0.425).

So a v1-eb-only reading of `enc_z32` is exactly the kind of reading exp08 just proved unsafe. The arm
that wins v1 eb could be at or under the untrained floor on transfer — and if the paper's spine is
global-vs-localized, transfer is the axis that matters.

### Cheapest test

**Not zero-GPU** — this is the first candidate in this file that needs training.

1. **λ pilot (required, ~2 × 10 min user-terminal).** λ=60 was calibrated for z=128 at fb=0. Shrinking
   the latent changes the dyn/recon scale ratio, exactly as the log_psd arms did in exp07 pre-check
   C4 (dial ratio 0.445 there). Run the seed-0 wave at **full ep100** per F13/F25a — no short pilots,
   no borrowed drift — and read the last-10-epoch dose. Gate [0.6, 1.4], target 1.0.
2. **Two cells × 6 seeds × ep100 at 256×16**, on the frozen `hann0p3` recipe with only the capacity
   knob changed: `hann0p3_z32` and `hann0p3_whalf`, dyn `fwd_bwd` at calibrated λ. Reuse the existing
   6-seed `exp07_hann0p3_fbwd` as the control — **never retrain it**. ≈12 runs; at exp07's ~0.42 h/run
   and smaller trunks, comfortably one overnight.
3. **Per-variant capacity-matched untrained references** are mandatory (params fall, so absolutes are
   not comparable without them). exp04 already established the pattern.
4. **Score on both axes, not one:** v1 gates (`mean_resid`, 6 seeds, paired) **and** the full ADR-0010
   menu, plus active-unit counts and the exp08 signature suite's residual-variance asymmetry so the
   "does squeezing concentrate or destroy" question gets an answer beyond the probe score.
5. Generated from a single manifest per the one-manifest rule; the cells are a two-line delta on
   `exp07_aux_factorization.yaml`.

### Decision rule (pre-register before looking)

- **z32/whalf beat `hann0p3_fbwd` on v1 eb AND hold on the ADR-0010 menu** → the capacity lever is
  real on the frozen recipe. This is a recipe change after the freeze; it becomes a v2 headline with
  the v1 number reported as-is, not a late swap.
- **Win on v1 eb, lose on the menu** → the most likely outcome given exp08, and a *result*: it
  sharpens the global-vs-localized spine by showing the bottleneck trades transfer for one localized
  task. Report as an ablation row, change nothing.
- **No significant movement either way** → the exp04 eb win was an artifact of the fb0.02 / λ=1 / ep60
  regime (most plausibly the per-dim free-bits floor scaling with z_dim). Closes a parked lever with a
  measurement and removes a "why didn't you shrink the latent?" reviewer question. Cheap and clean.
- **Active units at selection do *not* fall when z_dim falls** (i.e. z32 also holds ~6) → strong
  evidence that the active set is set entirely by the objective, which is a mechanism sentence worth
  more than the probe delta.

### Risks

- **It violates a CLAUDE.md hard constraint** (*"Conv1D-VAE architecture is final; do not change
  encoder/decoder"*). Running it needs an explicit decision and, if adopted, an ADR — this is not a
  quiet re-parameterisation. The constraint exists to protect exp00–08 comparability, and running two
  extra cells does not break that as long as the frozen recipe stays the control and nothing is
  retrained.
- **ML4PS freeze is Aug 15.** Q15 is a v2 lever unless it comes back overwhelming. It should not
  displace the consolidation fan; the honest framing is "measured, parked, revisited post-freeze".
- **The λ pilot is a real dependency, not a formality** — a mis-dosed z32 arm measures dose, not
  capacity, which is exactly the trap A4 (exp06's dose shortfall) set the first time.
- **`whalf` and `z32` are not the same hypothesis.** `whalf` compresses via *width* with no KL
  concentration (max dim 0.10 nats); `z32` concentrates KL 3× (0.067 nats/dim vs 0.023 at z128). They
  scored the same on eb by two different routes. Keep both or the result is uninterpretable.

---

## Q16 — can *"the dynamics model's latent is learning the physics"* be saved, and what would it take?

**Status:** open · framing / claim-level · **cost: zero-GPU for 4 of the 5 tests** · gates a sentence
the paper would like to write and currently cannot

### The framing under review

The hypothesis the project was built on (`architecture.md`): *"the dynamics-consistency objective
produces better representations because variability types are defined by temporal structure — transits
are periodic dips, eclipses are periodic deep/shallow alternations, pulsations are persistent
oscillations — all temporal structures the dynamics term directly pressures the encoder to capture."*

The strong reading of that — **the latent tracks the state of an oscillator, i.e. it learned the
physics** — is **not currently supported**. The weak reading — the dynamics term deposits usable
content that reconstruction alone does not — *is* supported, repeatedly. This entry is about whether
the strong reading can be recovered, weakened into something defensible, or must be abandoned outright.

This is not idle: the paper's mechanism sentence is still missing, and *"the encoder learned temporal
structure"* is the most natural thing a reader will assume the title claims.

### (1) Evidence AGAINST the framing

| # | evidence | where |
|---|---|---|
| A1 | **The rollout ordering inverted.** Free-running rollout beats copy-last persistence on *every* class (gain up to 2.7×) — but the pre-registered *periodic > quiet* ordering did not appear: gain is **larger on quiet** stars (multistep@48: periodic 2.17 vs quiet **2.71**), and decoded periodic rollouts relax toward flat. Read as: the rollout learned **"smooth is predictable"**, not periodic physics. | exp05 criterion 2, `exp05_rollout_summary.csv` |
| A2 | **The latent trajectory has no memory.** Lag-1 μ-ACF over consecutive windows sits at **−0.014…+0.012** at 256 cadences and goes *more negative* with window length (**−0.232** at w1024). The H2 pre-gate (open rollout eval only where lag-1 ACF ≥ 0.3) **never opened at any geometry**. If μ carried an oscillator's phase, consecutive-window latents would correlate. | exp06 H2, `exp06_geometry_acf.csv`; exp07 diagnostics |
| A3 | **Learning is not required for the v1 benefit.** A **frozen random** GRU (eb +0.067 ± 0.012 vs off) and a **learned linear map** (+0.065 ± 0.015) both reproduce it, neither separable from `fwd_bwd`. The active ingredient is prediction *pressure*, not learned dynamics. | exp08 ladder, `exp08_ladder_gap.csv` |
| A4 | **The latent is an amplitude summary.** `evr_pc1` **0.926**, `r2_amp_pc1` **0.967**, participation ratio **1.16 / 128** (random init 1.96 — training *narrows* it). Residualising four amplitude scalars out collapses eb 0.764 → 0.563 and rotation 0.558 → 0.424. A phase/state code would not behave like this. | amplitude subsection; exp07 C2 |
| A5 | **The decoder does not reproduce the oscillation.** Every recipe retains **≈1% of true peak power** and places the dominant spectral peak correctly **~25%** of the time — **0%** above 300 µHz. | exp07 diagnostics |
| A6 | **The detection mechanism that *is* measured is not periodicity.** EBs are flagged as **rare high-contrast outlier windows**, which is why dispersion pooling (`mean_std`, `window_score`) lifts eb/transit and does nothing for pulsating, and why the window-length lever was retired. | exp06 coverage-paradox reading |

### (2) Evidence SOMEWHAT IN FAVOR (real, but short of the claim)

| # | evidence | why it is not enough |
|---|---|---|
| B1 | **The GRU genuinely predicts.** Pre-check: it beats persistence **2.6–6.7×**; rollout gain_ratio > 1 on every class. | Predictive signal ≠ *physical* signal. Beating persistence is consistent with "smooth is predictable" (A1). |
| B2 | **Only the learned recurrent predictor transfers.** `fwd_bwd` leads the **entire** ADR-0010 menu (`numax_hon` **0.802 ± 0.003** vs best rung 0.755; `rotation_period` 0.677 vs 0.604), `linear` is seed-unstable out of distribution (`numax` sd **0.087** vs the GRU's 0.003), and collapsed `smooth` arms fall **below the untrained floor** (0.287 vs 0.425). | Strongest item on this side — the *learned recurrence* is replaceable in-distribution and not replaceable out of it. But the menu was **reported, not gated**, and per-probe SEs are not recorded for all seven. |
| B3 | **`fwd_bwd` is the only arm that holds `pulsating` at off-parity** while gaining eb/rotation (`linear` −0.014, `frozen` −0.018, both >2·SE **below** off). Pulsating is the persistent-oscillation task. | An absence of damage, not a gain. Consistent with the framing; does not require it. |
| B4 | **"ns" is underpowered, not equal.** `linear − fbwd` eb −0.015 and `frozen@22 − fbwd` −0.013 against paired SE ≈ 0.012–0.015 — both ≈1 SE, same sign. `fwd_bwd` is *nominally* on top in **10 of 11** scored tasks (3/4 v1 + 7/7 downstream). | Sign consistency across non-independent tasks is an observation, never a p-value. Resolving it needs ~24 seeds/arm (~36 runs). |
| B5 | **Dynamics units carry new content.** µ is **not** a re-encoding of the dyn-off latent, fusion gains are dynamics-specific, and residual-variance asymmetry runs **12.8×** on the GRU arms vs 0.24× (inverted) on the collapsed smoothness arm. | Says the content is new and dynamics-specific. Says nothing about it being *physical*. |
| B6 | **Frequency-domain physics does transfer.** νmax is a physical oscillation frequency and the frozen recipe beats six untrained inits by **+0.376 ± 0.025** on `numax_hon`. | Plausibly bought by the log-PSD auxiliary, not the dynamics term. Never disentangled — the aux and dyn axes have never been crossed on the downstream menu. |

### Two reasons the case against may be weaker than it looks

Both are **specification defects in the measurements that killed the framing**, not new data:

1. **A1's "periodic" class includes stars whose period is unresolvable.** `rollout_eval.py:36` sets
   `PERIODIC_TASKS = ("eb", "pulsating", "rotation")` and exp06 defines periodic as
   `pulsating|eb|rotation = 1`. Rotation positives run to **25 d** against a **5.7 d** sequence, and
   most EBs are long-period (only 19 of 196 test EBs have P < 0.5 d). So the pre-registered
   *periodic > quiet* test was run on a "periodic" class **mostly composed of stars with no resolvable
   cycle in the sequence** — the same defect as [Q13](#q13), in a different file. The test as run
   cannot falsify what it was meant to falsify.
2. **A2's lag-1 μ-ACF is the wrong statistic for half the population.** Consecutive window centres are
   **8.53 h** apart at the 256 geometry. A δ Scuti at P < 0.2 d (4.8 h) or a contact EB at P < 0.5 d is
   **aliased** — lag-1 ACF is structurally blind to it, and a near-zero reading is the *expected*
   result whether or not phase is encoded. The right statistic is a **period-matched lag** or a
   phase-folded ACF. Honest caveat: this does **not** rescue rotation (P = 3 d gives ~8 samples/cycle,
   where lag-1 ACF *should* be strongly positive and is not), so A2 survives for the slow half. The
   negative ACF at long windows (−0.232) has no explanation on record at all.

### Cheapest tests, in order of information per hour

1. **Eigen-decompose the learned linear map** (zero-GPU, checkpoints exist, ~1 h). `linear_fbwd`
   learns a literal matrix `A` with `z_{t+1} ≈ A z_t` (`dynamics.py:48-62`), 6 seeds, forward and
   backward. Take its eigenvalues: **complex pairs with |λ| ≈ 1 and argument θ = 2π·Δt/P** are exactly
   what latent-space oscillator transport looks like. Compare the implied periods `2π·Δt/θ` against the
   corpus period distribution. This is the single most direct test of "the latent transports phase"
   that the project can run, it costs nothing, and it has never been done. Null result is equally
   informative (`A` is a contraction with real spectrum → the map is a smoother, full stop).
2. **Re-score exp05 criterion 2 on resolvable periods only** (zero-GPU, cached rollouts). Redefine
   "periodic" as *catalogued period < sequence span* rather than *carries any periodic label*, and
   re-run the periodic-vs-quiet rollout comparison. If the ordering flips, A1 was a label-scope
   artifact and the strongest anti-evidence weakens. Batch with Q13's re-fit — same defect, same fix.
3. **Period-matched μ-ACF** (zero-GPU, cached μ). Replace lag-1 with a per-star lag chosen from the
   catalogued period, or fold the μ trajectory on the period. Report separately for aliased
   (P < 2 × window) and sampled (P > 2 × window) populations. Also: explain the −0.232.
4. **Probe the GRU hidden state — never done.** Every measurement in this project is on μ. Two cheap
   readouts on existing checkpoints: (a) regress `eb_period` / `pulsating_period` from the GRU hidden
   state `h` and ask whether it beats the same regression from μ alone; (b) test whether one-step
   prediction error is lower on stars whose period is resolvable. A win here is direct evidence that
   the *predictor* holds temporal structure the static code does not — much closer to "learned
   dynamics" than any probe-score delta.
5. **Only if 1–4 encourage it — period-aware rollout** (GPU). exp05's own listed levers: longer
   horizon, periodic-only rollout loss, larger latent scale. Do not spend a wave here on current
   evidence.

### Decision rule (pre-register before looking)

- **Test 1 finds oscillatory eigenvalues whose implied periods track the population** → the strong
  framing is partially recoverable and gets a figure, stated for the *linear* arm with the GRU's
  behaviour argued by analogy (not asserted).
- **Test 2 flips the periodic/quiet ordering** → A1 retires as a label-scope artifact; criterion 2 goes
  from "partial, flagged" to open, and the exp05 README needs an amendment.
- **Test 4 shows `h` beats μ on period regression** → the defensible claim becomes *"the predictor
  learns temporal structure; the latent it is fitted to does not have to carry it"*, which is a
  genuinely interesting and honest mechanism sentence.
- **All four come back null** → **abandon the strong framing explicitly in the paper**, and fall back
  on the sentence that is already measured and survives a reviewer: *unsatisfiable prediction pressure
  creates the content; the learned recurrent predictor is what makes it transfer* — a claim about
  representation transfer, not about physics being learned. Write it as a stated limitation rather
  than letting a reader infer the stronger claim from the title.

### Risks

- **Confirmation pressure is the real hazard here.** This entry exists because a framing we like is
  under-supported, which is exactly the setting where post-hoc reinterpretation is tempting. Every test
  above must be pre-registered with its null, and tests 2 and 3 must report the *original* statistic
  alongside the corrected one so the change is visible rather than substituted.
- **B6 is confounded and should not be quoted in support until it is not.** The aux and dynamics axes
  have never been crossed on the downstream menu, so "νmax transfers" cannot presently be attributed
  to the dynamics term.
- **A5 is not addressed by any of these tests.** Whatever the latent does, the decoder demonstrably
  does not reproduce the oscillation — any recovered framing must be about the *latent*, never about
  reconstruction fidelity.

---

## Suggested ledger entries

If accepted, add to [open_questions.md](open_questions.md) under the existing status vocabulary and
delete nothing from this file (it is the evidence trail):

```
## Q13 — v1 rotation probe ignores ADR-0004's P <= 5 d scope · open · report-level, no experiment
## Q14 — eb has no in-window coverage filter (transit's ADR-0009 has no eb analogue) · open · label hygiene
## Q15 — encoder-capacity arms (z32/whalf) parked at exp04, never re-tested on the frozen recipe · open · needs a GPU wave + an architecture-lock decision
## Q16 — can "the latent learns the physics" be saved? evidence assembled both ways; 4 zero-GPU tests · open · framing
```
