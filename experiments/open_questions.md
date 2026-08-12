# Open questions — what is unmeasured, and what would settle it

The companion to [cross_experiment_findings.md](cross_experiment_findings.md). That file records **what
was measured**, by charter without recommendations. This one records **what is not yet measured**: the
open mysteries, the levers nobody has pulled, and the decisions waiting on a number. It exists so that
choosing the next experiment is a matter of reading one tracked file rather than reconstructing the
state from a notebook, a Slack thread and a gitignored plan.

Scope note: `docs/` is gitignored, so the dated plans under `docs/plans/` and `docs/STATUS.md` do not
survive a fresh clone. Anything that must outlive this machine belongs here or in an `expNN_*_README.md`.

**How to use it.** Each entry states what is measured, what is not, the cheapest test that would settle
it, and what the answer would change. An entry is closed by editing it in place with the verdict and a
pointer to where it was measured — a closed entry stays in the file, because "we already asked that and
this is what came back" is the part that is expensive to rediscover. Entry IDs are permanent; the F-
numbers referenced are findings in `cross_experiment_findings.md`.

Status vocabulary: **open** (nobody has measured it) · **partially measured** (a reading exists but does
not settle it) · **closed** (settled; verdict recorded here) · **blocked** (waiting on a decision or on
another entry).

Last updated **2026-08-07**. Current frozen training recipe: `hann0p3` (exp07). ML4PS freeze Aug 15.

**Pre-exp08 check suite COMPLETE** (`experiments/run_exp08_prechecks.ps1` →
`src/notebooks/exp08_design_forensics.ipynb`, 2026-08-07). It closed **Q1, Q2, Q3, Q8, Q11 and Q12**,
qualified Q5 (F27), and selected **Q4 as exp08** by the pre-registered branch rule. Six of twelve
entries are now closed. **Q4 has since closed too** (2026-08-08, exp08 — see its entry below and
`exp08_ladder_README.md`). What remains genuinely open: **Q6** (blocked on a reporting decision, not a
measurement), **Q9** (report-level label confound), **Q10** (blocked on the freeze), and two loose ends —
the weight-0.1 dyn-off cells sit near the untrained floor with **no explanation** since F18's was
retracted, and the **fusion claim has never been measured on the ADR-0010 menu** (`features ⊕ µ` exists
only for the four v1 tasks in `exp07_channel_probe.csv`; `exp08_prechecks/menu.csv` carries the seven
downstream probes with no fusion readout at all).

---

## Q1 — Does the taper's pulsator-band cost show up in the downstream probes? · **CLOSED 2026-08-07** — no

*Verdict.* Paired `hann0p3 − comb0p3` on the frozen recipe's arm, 6 seeds, primary probe `numax_hon`:
**+0.0001 ± 0.0037** R². The three replication probes agree (`osc_giant` −0.0035, `rgb_vs_heb` +0.0018,
`solar_like_osc` +0.0124, none beyond 2·SE). The taper's measured 13% reconstruction penalty at
65–260 µHz (F19) **does not reach the probes that read that band**. This is the sharpest instance of F20
yet: band-resolved reconstruction fidelity fails to predict probe score even when measured in the
probe's own band, with the probe that reads it, on the recipe that pays the cost.

*And the question nobody had asked, which the same caches answered.* **Gate 0 (transfer): PASS by 15
SE** — `hann0p3_fbwd` beats six untrained inits on `numax_hon` by **+0.376 ± 0.025**. Every asteroseismic
number in this project had been an exp05-arm, ep60 number, and F25(a) says a short-budget run is not a
prefix of a long one; the frozen recipe had never been scored on this menu. It transfers.

*Consequence:* **branch A is dead.** Q12 (multitaper / frequency-weighted aux) would buy back a cost the
downstream menu cannot detect. *Where measured:* exp08 pre-design forensics §2;
`experiments/exp08_prechecks/gates.csv`.

<details><summary>Original entry</summary>

## Q1 (original) — Does the taper's pulsator-band cost show up in the downstream probes?

*Measured.* Under a DPSS referee neither model trained against, `hann0p3` reconstructs 65–260 µHz
**13% worse** than `comb0p3` and 44–58% better above 1 mHz (F19, exp07 D2). On the four v1 tasks the two
recipes are probe-tied, with `hann0p3` *better* on pulsating (+0.018 ± 0.009).

*Not measured.* The 7-probe downstream menu (ADR-0010) has never been scored on the frozen recipe. Every
asteroseismic number in the project — `numax_hon` R² +0.286, `osc_giant` +0.098, `solar_like_osc`,
`rgb_vs_heb` — was measured on **exp05** arms. Those probes read exactly the band the taper degrades.

*Cheapest test.* The consolidation eval fan: `new_task_extract` + `new_task_scorecard` on
`{hann0p3, comb0p3} × {off, fbwd} × seeds 0–5` plus the capacity-matched untrained arm, paired by seed.
One seed of `hann0p3_fbwd` is already extracted (`exp07_forensics/new_task_mu_cache/hann0p3_fbwd_s0.npz`,
2026-08-04), so the pool read is proven and the remaining cost is ~20 encoder passes, a few GPU-minutes
each.

*What it changes.* If the asteroseismic probes lose under `hann0p3`, the frozen recipe is wrong for the
downstream menu and exp08's first job is a frequency-weighted or multitaper aux term that keeps the
sidelobe suppression without paying in the pulsator band. If they hold, the taper is free where it
matters and exp08 can spend its budget elsewhere. This is also a **hard dependency of the ML4PS tables**,
which currently quote exp05-arm numbers alongside an exp07 recipe.

</details>

## Q2 — What channel do the probes actually read? · **CLOSED 2026-08-07** — not a re-encoding, and the fusion claim now has a number

*Verdict.* µ is **not** a re-encoding of the engineered basis. With all 25 T'DA-style features projected
out (fit on train only), the residual latent still probes at **2.1–3.4× base rate** on the fbwd arm
(eb 0.315 vs base 0.097, pulsating 0.360 vs 0.107, rotation 0.249 vs 0.089, transit 0.125 vs 0.060).

*The fusion cell is the headline, and it is dynamics-specific.* `features ⊕ µ` beats `features` alone on
all four v1 tasks in the fbwd arm, every one beyond 2·SE (paired over 2 recipes × 6 seeds): eb
**+0.036 ± 0.002**, pulsating **+0.047 ± 0.004**, rotation **+0.013 ± 0.005**, transit **+0.025 ± 0.004**.
In the **dyn-off** arm only pulsating (+0.042) and eb (+0.009) survive. So the complementarity to
engineered features is **bought by the dynamics term** — which is also Q11's answer from the other side.

The ML4PS fusion framing stops being an assertion: "SSL carries something engineered features do not" is
now measured, paired and seed-resolved on the frozen recipe. It does **not** overturn the standing result
that engineered features *alone* beat SSL *alone* on the downstream probes; both remain true.

*Secondary, consistent with exp06:* residualizing against the amplitude-only basis barely touches
pulsating (0.796 → 0.795) while costing eb (0.775 → 0.594) and rotation (0.561 → 0.436).

*Consequence:* the pre-registered tie-break (CHK-3 outranks CHK-1 if µ were a re-encoding) **did not
fire**. Aux/geometry are not shown dead by this route. *Where measured:* exp08 forensics §4;
`exp07_channel_{probe,feature_map}.csv`.

<details><summary>Original entry</summary>

## Q2 (original) — What channel do the probes actually read?

*Measured, repeatedly, as a negative.* Reconstruction fidelity does not predict probe score at any
resolution: not val loss (F11, ρ −0.58…−0.84), not band-resolved spectral accuracy (F20), not peak
fidelity (every recipe keeps ~1% of true peak power and finds the dominant peak ~25% of the time, yet all
probe pulsating at 0.79–0.81). The latent is amplitude-dominated (PC1-on-amplitude R² 0.92–0.97) and
collapsed to 5–8 of 128 dimensions. Two recipes that differ in their spectral treatment span the *same*
subspace (F1/F2 of exp07).

*Not measured.* What the surviving 5–8 dimensions encode, in physical terms. Nobody has regressed µ
against a basis of engineered quantities (amplitude percentiles, ACF structure, period, skew, gap
statistics) to see how much of the probe-carrying signal is a re-encoding of features we already have —
which is the natural follow-up to the standing result that engineered features beat the SSL
representation on all 11 probes.

*Cheapest test.* No training. Ridge/GBM from µ onto each engineered feature and back (both directions),
on caches already on disk, plus the same probes fitted on µ-with-amplitude-projected-out (the exp06
`mu_probe` machinery already does the last part).

*What it changes.* Everything about what exp08 should optimise. If µ is mostly an amplitude-and-scatter
summary, no aux-term or geometry change will move the probes, and the lever is the *readout* (MIL, ADR-
0008-lite) or the objective class (contrastive/masked rather than reconstructive). If µ carries something
the features do not, the fusion claim in the ML4PS framing gets a mechanism instead of an assertion.

</details>

## Q3 — Is the collapse to 5–8 active units a binding constraint? · **CLOSED 2026-08-07** — no, capacity is not a lever

*Verdict.* `lpsd0p3` (18.25 active units at selection, 4 seeds — the cell was never extended to 6) was
carried into the downstream fan and **ties the winner** (5.8 units) on the primary: `numax_hon` 0.8029 vs
0.8018. It loses on `rotation_period` (0.664 vs 0.677) and is otherwise indistinguishable across the
ADR-0010 menu. A latent holding three times the capacity buys nothing on the probes that would most
plausibly have used it. F25(b) inferred this from the v1 tasks; it is now measured on the downstream menu.

*Consequence:* capacity is closed as a lever and the "representational richness" caveat can be dropped
from the writeup. *Where measured:* exp08 forensics §2 (`lpsd0p3` columns).

<details><summary>Original entry</summary>

## Q3 (original) — Is the collapse to 5–8 active units a binding constraint?

*Measured.* Active units at the selected checkpoint sit at 5–8 of 128 under every healthy exp07 recipe;
dynamics-on recruits marginally (exp05 mechanism: 5 → 25 transiently, exp06 B3: recruitment is a warmup
transient that decays). `free_bits=0` is the frozen setting and `fb=0.02` moved nothing measurable in the
exp07 pre-checks. **Strengthened 2026-08-05 (F25, notebook K2/K5):** a wide latent is not merely untested,
it is *available and unpaid* — `lpsd0p3_fbwd` holds **18.25** units at selection, three times the winner's
5.8, and scores mid-table; within an arm more units go with worse pulsating (ρ −0.73 over five cells).

*Not measured.* Whether a latent forced to stay wide (free bits scheduled rather than fixed, or a
KL-target controller) scores *differently* — every previous free-bits reading was confounded with a λ or
selection change, and F13/F14 show unscheduled free bits is not the same knob as β warmup.

*Cheapest test.* Two cells on the frozen recipe: scheduled free bits vs the frozen fb=0, 4 seeds, paired.
~4 h at 256×16/ep100. Cheaper still, and worth doing first: the `lpsd0p3_fbwd` checkpoints already on disk
are a wide-latent arm, so probing them against their own dyn-off arm is a zero-training partial answer.

*What it changes.* If a wide latent probes the same, capacity is closed as a lever for good and the
"representational richness" caveat can be dropped from the writeup. If it probes better, that is exp08.
The 2026-08-05 reading has moved the prior firmly toward "closed".

</details>

## Q4 — Is the dynamics term doing anything beyond smoothing? · **CLOSED 2026-08-08 (exp08)** — yes; the active ingredient is unsatisfiable prediction pressure

*Verdict.* Ran as the exp08 **dynamics ladder** (off → smooth → learned-linear → frozen-random-GRU →
fwd_bwd, 6 seeds each on the frozen recipe; `experiments/exp08_ladder_README.md`). **G-prior FAILED**
(smoothness at max satisfiable pressure: eb +0.019 ± 0.010, rotation +0.015 ± 0.009, both ns) and
**G-gru PASSED** (fbwd − smooth eb +0.062, rotation +0.030, >2·SE) — the term is not a smoothness
prior, with a measured control. But the ladder sharpened the claim: a **learned linear map** is
statistically indistinguishable from the GRU on eb/rotation (linear − off eb +0.065 ± 0.015\*), and a
**frozen random GRU at dose 2.4** matches it too while holding the latent wide open (69–114 units) and
reproducing the full Q11 mechanism signature (residual asymmetry 12.8×, fusion deltas). The smoothness
prior *saturates* (λ×18 → dose ×0.13, latent collapses to 1 unit — dose parity is unreachable for a
satisfiable objective), and the frozen term is bistable around the collapse transition (dose 0.40
collapsed / 2.37 wide; the collapsed phase probes at smooth level). So: the benefit and the
feature-complementary content come from **prediction targets the encoder cannot trivially satisfy**;
learned *recurrent* dynamics are sufficient, not necessary. The GRU's unique property: only arm that
gains eb/rotation while holding pulsating at off-parity (linear/frozen pay −0.014…−0.018, >2·SE).

<details><summary>Original entry</summary>

## Q4 (original) — Is the dynamics term doing anything beyond smoothing? · **partially measured** · **THIS IS exp08**

**Selected as exp08 on 2026-08-07** by the pre-registered branch rule: Gate 0 passed, Gate 1 held, and
the CHK-3 tie-break did not fire, so the aux term is not exp08's problem and the budget goes here.

Q11's closure raises the stakes rather than lowering them. The dynamics term is now the thing carrying
content the engineered features do not have (Q2: only the fbwd arm's µ adds to the feature basis on all
four tasks) and holding latent directions its own dyn-off arm does not (Q11: CCA tail below the seed
null, 10× residual-variance asymmetry). So "what is that term actually doing" has moved from a caveat to
the paper's central question, and the smoothness ablation is the test that answers it.

*Design, unchanged from below:* one ablation cell — a two-step smoothness penalty on µ with no learned
dynamics — against `fwd_bwd`, 4 seeds, paired, on the frozen recipe.

*Measured.* Dynamics-weighting beats dynamics-off on eb (+0.080) and rotation (+0.046) in the frozen
recipe, and the GRU beats persistence 2.6–6.7× in latent space. But the rollout axis is **closed** at
every geometry and every aux configuration: lag-1 µ-ACF −0.014…+0.012 against a 0.3 threshold, and the
pre-registered "periodic > quiet" ordering fails — quiet stars have the more predictable trajectories
(exp05 criterion 2, exp06 H2, exp07 H1).

*Not measured.* Whether the benefit survives replacing the GRU rollout with a much weaker temporal
prior (e.g. a two-step smoothness penalty on µ with no learned dynamics at all). If it does, "world
model" is the wrong description of what earned the +0.080.

*Sharpened 2026-08-05 (F25c).* The term behaves like a **switch, not a dial**: inside the fwd+bwd arm the
dose spans 0.79–1.07 and orders nothing, while every fwd+bwd cell beats every dyn-off cell with no overlap
(0.491–0.498 vs 0.422–0.459) whatever the aux term. A benefit that is insensitive to *how much* dynamics
pressure is applied is exactly what a weak generic temporal prior would also produce — which raises the
prior on this ablation rather than lowering it.

*Cheapest test.* One ablation cell: smoothness-penalty-only arm vs `fwd_bwd`, 4 seeds, paired on the
frozen recipe. Directly answers the sharpest objection a reviewer can raise about the framing.

*What it changes.* The paper's central claim. Worth doing before Aug 15 if anything is.

</details>

## Q5 — Is the residual taper shadow a per-star effect? · **CLOSED 2026-08-04** — no, and the sign inverts

*Verdict.* On the median test star `hann0p3` reconstructs the two endpoints **better** than the interior
(per-star median ratio 0.71 ± 0.01 over 6 seeds; absolute excess −0.28). The published 1.15× `edge_max` is
a ratio of means carried by a minority, and that minority is the **quiet** end of the corpus: the ratio
runs 3.1 in the quietest noise decile to 0.62 in the noisiest, ρ(noise, ratio) = −0.64.

The untapered baseline moves the other way — `comb0p3_fbwd`'s *absolute* edge excess correlates with star
noise at ρ = **+0.90 ± 0.01**, which is the C1 purchase mechanism showing up in the star population: a
noisier star has more broadband power to match, so the impulse that buys it is larger.

*Where measured.* `analyze_exp07_edge_noise.py` → `exp07_edge_noise_{stars,summary}.csv`; notebook C5.
Recorded as F23. **Consequence:** the edge is closed as a lever (it was already closed as a probe cost,
B4/F21); no exp08 work follows from it.

*Qualified 2026-08-06 (F27).* "The edge is closed" is true and is not the same as "the defect is
closed". Reduced per seed, `hann0p3_fbwd` carries an **interior** impulse of 11.8 ± 5.3× the interior
level, at the position where the Hann weight is maximal — the same order as the 32× boundary impulse it
replaced. The taper moved the purchase, it did not remove it. Whether *that* costs downstream score is
the per-star correlation of exp08 CHK-2; the associated stitch-comb number is architectural (the
untrained arm combs harder than any trained cell) and is paper hygiene, not a probe cost.

## Q6 — Does MIL pooling get adopted for the frozen recipe's tables? · **blocked** on a decision · medium

*Measured.* `window_score × logistic` beats mean pooling by >2·SE on **eb** (+0.119 winner / +0.077 comb)
and on transit, and *kills* pulsating — MIL is task-selective, which is ADR-0008-lite. The transit gain is
dispersion, not location (F: quantile3 reaches 0.178 with the same dimensionality), and the all-segment
0.354 is confounded by bag size (F2), so the claimable number is the K-matched 0.245.

*Measured on the frozen recipe 2026-08-05* (notebook K3; `mil_cache` + `run_exp07_mil_sweep.ps1`, v1 pool,
`first` bag scope, `{hann0p3, comb0p3} × {off, fbwd} × 4 seeds` + untrained). Three results:
- **Pooling does not change which encoder wins.** MIL-winner, mean-pooling and `mean_resid` return the
  identical cell ranking (ρ = +1.00), and `val/recon` is orthogonal to all three (ρ = 0.00).
- **The gain is a transit gain and mostly not the encoder's.** Over four tasks the val-declared MIL winner
  beats mean pooling by only +0.006…+0.022; per task it is transit +0.042…+0.080, eb ≈ 0, rotation and
  pulsating scattered around zero. The **untrained** arm gains **+0.066** on transit from the same
  operator, so most of the localized-task lift is the operator, not the representation.
- **The winning operator is not stable.** Across four seeds of one cell the val-declared winner changes
  identity up to four times, spanning feature space (`rff_meanmap`, `mean_std`, `gmm_prototype`) and score
  space (`ws_ppv_lspv`, `ws_topk`).

*Still not measured.* The all-segment / K-matched setting on the frozen recipe (the 0.245 transit claim is
an exp05-arm, `all`-scope, K-matched number), and the new-task pool.

*What it changes.* Whether the ML4PS tables report one pooling or two, and whether transit is reported at
0.145 (mean) or 0.245 (MIL, K-matched). Still blocked, and the 2026-08-05 reading sharpens why: the rule
has to name a **fixed** operator per task, because "the best operator" is val-selection noise at this
sample size, and the paired untrained arm has to be reported alongside or the transit number will read as
a representation result when it is largely a pooling result.

## Q7 — Does the epoch budget still bind? · **CLOSED 2026-08-03** — no, not usefully

*Verdict.* 22 of 26 fwd_bwd runs still select inside their last ten epochs at ep100, but they made only
1.3–3.9% of their post-warmup travel over the final 20 epochs. Absolutes remain floors and paired gates
are unaffected. More epochs is the worst available purchase at this geometry; headroom has to come from
the objective or the readout. *Where measured:* exp07 diagnostics A2.

## Q8 — Selection metric needs a warmup-aware floor · **CLOSED 2026-08-06** — the floor was always there

*Verdict.* There is no guard to land. `loop.py` has always restricted checkpoint selection to
`epoch >= beta_warmup_epochs`, and the saved `best_recon_aux.pt` files for the two cells F18 named
select epochs **85–99** with **5–6** active units, not epoch 0 with 128. F18 was a post-hoc `idxmin`
over the raw logged history that did not reproduce the selector's own constraint; it is **retracted**
and rewritten as a method finding about re-deriving decisions the code already made.

The standing rule this entry imposed — that any experiment sweeping or lowering `recon_aux.weight` must
first fix the selector — is **withdrawn**. The guard is now pinned by
`src/swm/tests/test_dual_checkpoint.py::test_select_never_picks_a_warmup_epoch`.

*Left open, as a separate and much smaller thing.* The weight-0.1 dyn-off cells really do sit at
+0.003 / +0.012 above untrained, the two lowest of exp07's ten cells. That is measured and its cause is
now unknown. It concerns a recipe two steps from the frozen one, so it is flagged rather than chased.

*Where measured:* exp08 pre-design forensics notebook §1.

## Q9 — v1 rotation labels are confounded · **open** · report-level, no experiment

`subset.py` defines quiet as "matched in NO catalogue" *including* rotation, so 0 of 10,000 quiet stars are
rotation-positive against a corpus rate of ~13%. Paired deltas survive (both arms share the confound) but
the v1 rotation probe is partly a general-variability detector, and the standing scorecard entry
`rotation +0.046` inherits that. Settled only by rebuilding the subset with rotation excluded from the
quiet definition, which re-opens every v1 number — hence nobody has.

## Q10 — ADR-0011 label regeneration timing · **blocked** on the freeze · low risk

33 Villanova EB contaminants are signed off for removal (eb 1,936 → 1,903), to be applied at the next label
regeneration and not mid-freeze. ML4PS reports **v1** numbers with the measured v1→v2 delta (+0.004 ± 0.014
PR-AUC over 304 cells) cited. Open only in the sense that the regeneration has not happened; the number that
would change is known and negligible.

## Q11 — What do the one-to-three extra active units buy? · **CLOSED 2026-08-07** — (a) new content

*Verdict.* All three independent measurements agree on hypothesis **(a)**: the dynamics arm holds
genuinely new content, not a re-encoding (b) and not a scale change the probe's `StandardScaler`
exploits (c).

- **CCA cross-arm, read against the same-cell cross-seed null** (without which a principal angle means
  nothing). The cross-arm curve tracks the null for four components then separates: component 5
  **0.772** vs 0.970/0.924, component 6 **0.489** vs 0.949/0.844, component 7 **0.235** vs 0.873/0.712.
  Two seeds of the *same* cell agree to 0.87 where the two arms agree to 0.24.
- **Asymmetric residual probe.** `fbwd ⊥ off` retains **2.2–2.4%** of its variance; `off ⊥ fbwd` retains
  **0.23–0.25%** — a **10×** asymmetry. The residual probes accordingly (eb 0.30 vs 0.15, pulsating 0.285
  vs 0.18), and the paired asymmetry clears 2·SE on every task: eb +0.146 ± 0.010, pulsating
  +0.107 ± 0.009, rotation +0.075 ± 0.010, transit +0.008 ± 0.002.
- **Scaling control — (c) refuted in the strong direction.** Removing the probe's `StandardScaler` does
  not shrink the fbwd−off gap, it **inflates** it (pulsating 0.003 → 0.204, eb 0.040 → 0.139, rotation
  0.026 → 0.102), because the *dyn-off* arm is the one that depends on the scaler. The scaler compresses
  the dynamics advantage rather than manufacturing it, so every gap this project reports is conservative.

*Consequence:* the paper can now say **what** the dynamics term added, not only that it helped — and Q2
says the same thing from the other side. This is the mechanism sentence for the central result. It also
sharpens Q4 (now exp08): a term carrying private, probe-relevant directions is exactly what a weak
generic temporal prior would *not* obviously produce, so the smoothness ablation is now a real test
rather than a formality. *Where measured:* exp08 forensics §5;
`exp07_dynunits_{cca,residual,scaling}.csv`.

<details><summary>Original entry</summary>

## Q11 (original) — What do the one-to-three extra active units buy?

*Measured.* At ep100 the dynamics arm holds barely more latent capacity than its own dyn-off arm
(`comb0p3` 8.2 vs 5.2, `hann0p3` 5.8 vs 5.0) and still wins every task; a cell that holds **18** units
(`lpsd0p3_fbwd`) scores mid-table (F25). So the win is not capacity, and the exp05 "collapse reversal"
framing is retired.

*Not measured.* What those marginal dimensions carry. Nobody has asked whether the dynamics arm's extra
units are (a) new physical content, (b) a re-encoding of content the dyn-off arm already has but in a more
linearly-readable direction, or (c) a scale/whitening change the probe's `StandardScaler` exploits.

*Cheapest test.* No training. Per-dimension KL and probe-weight mass in the paired arms; project the
dyn-off subspace out of the dyn-on µ and re-probe (the exp07 F2 residual machinery does this already,
cross-*recipe*; here it would run cross-*arm*, which nobody has done). Add per-dimension ridge from µ onto
the engineered-feature basis of Q2 to name them.

*What it changes.* This is the mechanism sentence for the paper's central result. Right now the project
can say the dynamics term helps and cannot say what it added. If the answer is (b) or (c), the finding is
about *readability* rather than content, and the readout lever (Q6) becomes the main story.

</details>

## Q12 — Is the aux term's reconstruction cost real, or an artefact of its framing? · **CLOSED 2026-08-07** — moot

*Verdict.* This entry was explicitly conditional on Q1 ("Depends on Q1. If the downstream menu is
unharmed by the taper, this is a nice-to-have"). Q1 closed: the taper costs the downstream menu nothing
(+0.0001 ± 0.0037 on the primary). A multitaper or frequency-weighted aux term would buy back a
reconstruction cost that no probe can detect, so it is not worth an experiment before the freeze. F24
stands on its own as a finding about aux framing; no exp08 work follows from it.

<details><summary>Original entry</summary>

## Q12 (original) — Is the aux term's reconstruction cost real, or an artefact of its framing?

*Measured (F24).* Switching the log-PSD window from rectangular to Hann, with weight and λ unchanged,
refunded **9.2%** of validation reconstruction — landing the weight-0.3 recipe on top of the weight-0.1
cells. The penalty normally attributed to "asking for spectral fidelity" was mostly the cost of supplying
edge power the rectangular framing demanded.

*Not measured.* Whether the remaining reconstruction cost of the aux term is similarly recoverable, and
what a *frequency-weighted* or multitaper aux term does to the one measured cost of the frozen recipe
(13% worse at 65–260 µHz, the pulsator band, F19/Q1). A DPSS multitaper is the natural next window: it
suppresses sidelobes like Hann without Hann's variance penalty at low frequency.

*Cheapest test.* One `spectral_recon_loss(window_fn)` variant plus a band-weight vector — the plumbing and
the neutral referee both exist (`src/swm/tests/test_spectral_window.py` pins the interface). Two cells
× 4 seeds against the frozen recipe, scored under the DPSS referee *and* on probes.

*What it changes.* Depends on Q1. If the downstream menu is unharmed by the taper, this is a nice-to-have;
if the asteroseismic probes lose, this is exp08's first job.

</details>

---

## Standing constraints any exp08 design has to respect

Not questions — settled rules that have each been violated at least once, with the cost recorded.

- **Never select on val loss, or on reconstruction fidelity in any band** (F11, F20). Probe gates are the
  only currency.
- **Never pool across the dynamics arm.** A statistic computed over cells spanning `dyn` measures `dyn`
  (F21, Simpson reversal; third instance in this project).
- **Never select on the trained-minus-untrained gap when the readout has free capacity** — a large head
  inflates the gap by collapsing on random features. Select on absolute score, report the gap.
- **A window comparison needs a referee neither arm trained against** (F19): scoring under either
  training window flips the exp07 verdict by up to 4×.
- **One experiment = one manifest.** All config in `experiments/configs/<expNN>_<slug>.yaml`; per-cell
  Hydra groups and the runner are generated from it.
- **Assert `epoch.min() == 0`** before any epoch-range statistic, and align curves on the `epoch` column,
  never on row position (F22 — the exp07 dump lost two runs' pre-resume prefixes).
- **Only `val/recon` may be compared across cells of a geometry.** `val/aux` and `val/monitor_recon_aux`
  are functions of the aux type, its weight and (for `hann0p3`) its integrand, so they order cells only
  inside a recipe; dimensionless ratios to recon travel further. Across geometries nothing but the ratios
  and the curve *shapes* survive.
- **A short-budget run is not a prefix of a long one, so never compare capacity (or pilot a dose) across
  budgets.** The cosine LR is annealed over `max_epochs`: at epoch 40 a 60-epoch run sits at lr 7.7e-5
  against a 100-epoch run's 2.0e-4, and its active set freezes there (26 units vs 11–15 at the *same*
  epoch). exp05's "5 → 25 collapse reversal" is that freeze, not a property of the recipe (F25a); the same
  mechanism is behind the ep40 → ep100 dose drift ≈ 0.75 the exp07 λ pilots had to correct for.
