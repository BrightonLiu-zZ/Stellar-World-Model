# MIL pooling sweep — how to aggregate a star's bag of window mu (results + how to run)

**Status: RUN + ANALYSED 2026-07-26 (v1 subset only; new-task pool = Phase 2, not yet run).**
Plan: [docs/plans/2026-07-25-mil-pooling-sweep.md](../../docs/plans/2026-07-25-mil-pooling-sweep.md).
Design fixed by grill-with-docs 2026-07-25/26. Eval-only: no training, no GPU terminal handoff.

## What this tests

Every table through exp05 reduced a star's windows to one vector with **mean pooling** (plus a single
MIL alternative, `window_score` = max over per-window logistic scores). This sweep asks whether that
choice was leaving performance on the table, across ~18 operators, 3 bag arms, 4 tasks, 4 seeds.

**Machine-readable:** `mil_pooling_results.csv` (headline blocks) + `mil_sweep.csv` (6,516-row long
table) + `mil_learned.csv` (Tier-3). Notebook: `src/notebooks/mil_pooling.ipynb`.

## Headline

**Pooling choice matters enormously for transit and almost not at all for anything else.**

| task | mean pooling | best operator (val-declared) | gain | gap vs untrained |
|---|---|---|---|---|
| **transit** | 0.1450 | **0.2137** `mean_std` | **+0.069** | +0.067 ✓ |
| eb | 0.7638 | 0.7634 `moments` | −0.000 | +0.044 ✓ |
| pulsating | 0.8037 | 0.7723 `ws_topk`(1) | −0.031 | +0.038 ✓ |
| rotation | 0.5580 | 0.5575 `ws_ppv_lspv`(0.95) | −0.000 | +0.032 ✓ |

(first-segment bag, test PR-AUC, 4 seeds. `mean_skew` reaches 0.2353 on transit but val ranked
`mean_std` first, and the protocol reports the val-declared choice.)

Transit had been stuck near base rate for the whole project. **Changing only the readout aggregation
moves it from 2.4x to 3.5-3.9x base rate, with no retraining.**

## The mechanism: dispersion, not location

Moments ablation (transit, first-segment): the win is not extra feature count.

| operator | dims | test |
|---|---|---|
| `mean` | 128 | 0.1450 |
| `quantile3` | **384** | 0.1784 |
| `max` | 128 | 0.1968 |
| `mean_std` | 256 | 0.2137 |
| `moments` | 384 | 0.2308 |
| `mean_skew` | 256 | **0.2353** |

`quantile3` has the same 384 features as `moments` and reaches only 0.178. The second and third
central moments *across a star's windows* each independently carry the signal: what identifies a
transiting star is **how far its windows scatter from its own baseline, and whether that scatter is
one-sided**. Mean pooling integrates exactly that away. Skew is the physically motivated term, since
transits and eclipses are one-sided dips.

## Witness rate is measurable

`ws_lse` spans mean pooling (beta -> 0) and max pooling (beta -> inf) with one scalar. Relative test
PR-AUC change across that range, first-segment bags:

| task | rel. gain mean -> max |
|---|---|
| transit | **+53.8%** |
| eb | +20.0% |
| rotation | +6.6% |
| pulsating | **−1.0%** |

The ordering is the pre-registered criterion, and it is the standard-MI vs collective distinction
made quantitative: pulsating is the only task where max-like pooling actively hurts, because its
signal is in every window.

## Bag scope: all-segment is a large gain but NOT confound-free

All-segment bags (median 32 / mean 62 / max 816 windows vs 16) give the biggest numbers of the whole
sweep: **transit 0.3538**, eb 0.8088, pulsating 0.8191 (all `moments`). But the `bagsize_only`
control fires:

| task | bagsize_only / base rate, `first` | `kmatch16` | `all` |
|---|---|---|---|
| transit | 1.03 | **1.00** | **2.18** |
| eb | 1.00 | 1.00 | 1.59 |
| rotation | 0.99 | 1.00 | 1.33 |
| pulsating | 1.02 | 1.00 | 1.23 |

At all-segment scope, a logistic on log(window count) **alone** scores transit at 2.2x base rate,
because stars observed in more sectors are both bigger bags and likelier to be catalogued.

**K-matched-16** (16 windows drawn from across ALL segments, 3 draws) is confound-free by
construction: every star has at least 16 windows, so at fixed K0 the count carries exactly zero
information, which the control confirms (ratio 1.000 to 6 decimals).

| transit | first (K=16) | **kmatch16** | all (K~62) |
|---|---|---|---|
| `mean` | 0.1450 | 0.1580 | 0.1742 |
| `moments` | 0.2308 | 0.2169 | 0.3538 |
| `mean_skew` | 0.2353 | 0.2247 | 0.3306 |
| `ws_ppv_lspv`(0.5) | 0.2089 | **0.2448** | 0.2281 |

**Read:** holding bag size fixed erases the all-segment gain, so *temporal spread is not the
mechanism*. **Limitation, stated plainly:** for transit the probability that a bag contains an
in-transit window also scales with the number of windows, so K-matching removes the genuine coverage
benefit along with the confound. This control rules out temporal spread; it does **not** cleanly
separate "more windows means more chance of catching a transit" (real) from "well-observed stars are
likelier to be catalogued" (artifact). The artifact is demonstrably present at some magnitude.

**Defensible claim:** at fixed bag size, pooling lifts transit **0.158 -> 0.245** (`ws_ppv_lspv`,
gain over mean **+0.087**, gap vs untrained +0.065 > 2*SE). The 0.354 figure is an upper bound and
must carry the bag-size caveat.

## Max-like pooling degrades as bags grow

Transit, `ws_lse`: beta=10 gives 0.1803 at K=16 but **0.0885** at K~62; beta=50 gives 0.1818 -> 0.1477.
`moments` improves over the same change. This is the Maxsoft (NeurIPS 2025) beta_crit = O(log K)
prediction observed directly: a temperature tuned on short bags loses half its performance when
deployed on long ones. Relevant to any LSST-scale "pre-train once, deploy on longer baselines" claim.

## Tier-3 learned pooling loses on every task

Gated ABMIL and DSMIL on frozen mu, with the three prescribed small-data fixes (narrow attention
width, ACMIL stochastic top-k instance masking, DTFD pseudo-bags), first-segment, 4 seeds:

| task | best zero-parameter operator | ABMIL | DSMIL |
|---|---|---|---|
| transit | **0.2353** `mean_skew` | 0.1233 | 0.1972 |
| eb | **0.7726** `mean_std` | 0.7148 | 0.7283 |
| pulsating | **0.8069** `rff_meanmap` | 0.7449 | 0.7608 |
| rotation | **0.5690** `ws_lse` | 0.5240 | 0.5204 |

At 122-216 test positives, a ~16k-parameter attention head is beaten on all four tasks by a
zero-parameter concatenation of moments. Tier-3 is diagnostic-only (needs the unsigned ADR-0008
exception) and is excluded from winner selection in `mil_report.winner_block`.

## Methodological finding: the gap is not invariant to readout capacity

Ranking by trained-minus-untrained gap instead of absolute PR-AUC **reverses** the Tier-3 conclusion:

| task | ABMIL gap | best-simple gap | ABMIL untrained |
|---|---|---|---|
| pulsating | **+0.288** | +0.016 | **0.457** |
| rotation | **+0.244** | +0.050 | **0.280** |
| eb | +0.136 | +0.034 | 0.578 |

ABMIL would win 3 of 4 tasks on gap while being worst on 3 of 4 in absolute terms: a high-capacity
head inflates its gap by **collapsing on random features**, not by reading the trained encoder
better. Milder instances: `moments` on rotation (gap +0.067, absolute −0.049 vs mean),
`gmm_prototype`(4) on pulsating (gap +0.104, absolute −0.003).

**Resolution (grill 2026-07-26):** select operators on **absolute** val PR-AUC, report the gap only
for the selected operator, and publish this pathology as a finding. Nothing in exp03/04/05 is
invalidated: those all compared a *fixed* readout across encoders, and the pathology only bites when
the gap ranks readouts of *differing capacity*.

## exp05 survives the readout change

Paired `comb_fbwd_c1p0` − `comb_off`, 4 seeds, under each task's winning operator: transit **+0.035**,
eb **+0.028**, rotation **+0.024** all > 2*SE; pulsating +0.016 null (exactly as in exp05). The
dynamics result is not an artifact of mean pooling.

## Cut for a proven reason

**Noisy-AND** (Kraus+2016) is a strictly increasing function of the bag mean, so under a *frozen*
readout it produces the identical star ranking and therefore the identical PR-AUC as `ws_mean`. It
differs only when trained jointly with the encoder. It survives as a unit test asserting that
rank-equivalence (`test_pooling.py::test_noisy_and_is_rank_equivalent_to_mean`), not as a sweep cell.

`gmm_prototype` at C=16 emits 4,112 features for ~200 positive bags and loses badly on every task
(transit 0.145); it is the main cost driver and should be dropped or capped at C=4 before Phase 2.

## Phase 2 — the new-task pool (run 2026-07-26)

Three **detection** probes on the separate 22,860-star new-task pool (`osc_giant`, `solar_like_osc`,
`flare`), same 9 arms, reusing the mu caches `new_task_extract` had already written, so **no GPU pass**.
The pool's four regression probes score with R2 and stay out of this PR-AUC sweep by design.
Results: `mil_pooling_results_new_task.csv`, long table `mil_sweep_new_task.csv`.

**1. Dispersion pooling generalizes, including to GLOBAL tasks.** It now beats `mean` on 7 of 7
probes across both pools in absolute terms:

| probe | n_test_pos | `mean` | best | gain |
|---|---|---|---|---|
| osc_giant | 1,313 | 0.8450 | **0.8975** `mean_std` | +0.053 |
| solar_like_osc | 417 | 0.3134 | **0.3545** `rff_meanmap`(1024) | +0.041 |
| flare | 304 | 0.4388 | **0.4940** `moments` | +0.055 |

This was **not** predicted. On the v1 subset `moments` beat `mean` only on transit and tied or lost on
the three others; here it wins on two *global* asteroseismic probes as well. Leading hypothesis
(**untested**): support. The new-task pool trains on 16,002 stars with 304-1,313 test positives,
versus 9,428 stars and 122-216 positives on v1, and a 256-384-dim feature map needs positives to pay
off. A learning curve subsampling train stars on `osc_giant` would settle it.

**2. The witness-rate prediction holds for the asteroseismic probes and FAILS for flare.**
`ws_lse` relative gain mean -> max: osc_giant **−1.7%**, solar_like_osc **−3.2%** (global, as
predicted), flare **−0.8%** (predicted positive, since a flare is the most localized signal in the
whole menu).

**Why, and what it bounds:** `ws_lse` aggregates the scores of a per-window *linear classifier*. The
2026-07-22 Level-B result already showed the trained encoder makes flare windows *less* separable
than an untrained one (window PR-AUC 0.053 vs 0.139) because the reconstruction smooths the spike
away. The window readout cannot see the witness, so max-like pooling has nothing to concentrate.
**beta\* estimates witness rate only when the per-window readout can detect the witness.** For transit
it can (Phase-1: within-star window PR-AUC ~0.50); for flare it cannot. State this whenever beta\* is
presented as a physical measurement.

**3. flare is an honest null on the SSL gap.** `moments` 0.4940 vs untrained 0.4983 = **−0.004, not
confirmed**. Pooling lifts flare's absolute score substantially (+0.055 over mean) but helps the
untrained encoder just as much, consistent with the standing "training hurts flare localization"
finding. osc_giant (+0.035) and solar_like_osc (+0.080) both confirm at > 2*SE.

**4. The capacity pathology reproduces on a different pool.** On osc_giant, `ws_max` has the LARGEST
gap of any operator (**+0.205**) while being the WORST in absolute terms (0.8046 vs 0.8975), because
it drops the untrained arm to 0.599. Independent confirmation that D9 (select on absolute) is not a
Tier-3 artifact.

**5. Controls clean.** `bagsize_only` ratio to base rate = 1.01-1.03 on all three probes, as required
at the fixed-K first-segment scope.

### Phase 2b — the four REGRESSION probes (R2)

Continuous whole-star targets, scored by R2 with Ridge, feature-space poolings only. Score-space MIL
operators aggregate per-window scores under the standard-MI assumption ("the bag is positive if ANY
window is"), which has no counterpart for a continuous property like nu_max; every target here is
global, so the collective assumption holds by construction. Population masks mirror
`new_task_scorecard.score_regression_task` exactly, including the `prot_kounkel_gt57` cut.

| probe | n_test | `mean` | best | gain | gap | confirmed |
|---|---|---|---|---|---|---|
| numax_hon | 1,313 | 0.7928 | **0.8695** `rff_meanmap`(1024) | +0.077 | +0.108 | ✓ |
| numax_hatt | 417 | 0.8299 | **0.8619** `rff_meanmap`(1024) | +0.032 | +0.075 | ✓ |
| dnu_hatt | 417 | 0.8217 | **0.8566** `rff_meanmap`(1024) | +0.035 | +0.076 | ✓ |
| prot_kounkel | 138 | 0.3472 | **0.5158** `max` (val-declared `gmm_prototype`(4) 0.4538) | +0.169 | +0.040 | ✓ |

**Richer pooling beats `mean` on all four**, so across both pools the count is now **11 of 11 probes**.
`bagsize_only` gives R2 = −0.002 to +0.011, the correct null for a regression baseline.

**This generalizes the gap pathology (D9) beyond capacity.** On `numax_hon`, plain `mean` has the
LARGEST gap of any operator (**+0.363**) and nearly the lowest absolute R2 (0.793), while
`rff_meanmap` has the best absolute (**0.870**) and one of the smallest gaps (+0.108). Neither is a
capacity difference; both are fixed zero-parameter feature maps. The cause is that mean-pooled
*random* mu is poor (untrained 0.430) whereas a random-Fourier map is already good on random
projections (untrained 0.762). **Ranking by gap systematically favours whichever operator is worst
for a random encoder**, whatever the reason. Same inversion on `prot_kounkel` (`mean` gap +0.228,
absolute 0.347 = worst). Three independent instances now: ABMIL (capacity), `ws_max` on osc_giant
(operator sharpness), `mean` vs `rff_meanmap` (feature-map expressiveness).

### Phase 2c — all-segment scope on the new-task pool, and the confound RESOLVED

All-segment caches for this pool did not exist (`new_task_extract` only ever read each star's first
segment), so they are built here by replaying `processed/sequences` with pack.py's recipe; restricted
to the first segment the replay reproduces the existing cache exactly (same 3,429 stars, identical
order and counts, mu to 1e-3). 9 caches, mean **67 windows/star** vs 16.

**The bag-size confound is a DETECTION artifact, not a property of larger bags.** `bagsize_only`
(logistic/ridge on log window count alone) at all-segment scope:

| probe | metric | first | **all** | null |
|---|---|---|---|---|
| numax_hon / numax_hatt / dnu_hatt / prot_kounkel | R2 | +0.011 … −0.002 | **−0.003 … −0.000** | 0 |
| osc_giant | PR-AUC | 0.387 | 0.420 | 0.383 (1.10x) |
| flare | PR-AUC | 0.091 | 0.098 | 0.089 (1.10x) |
| solar_like_osc | PR-AUC | 0.125 | **0.206** | 0.122 (**1.70x**) |

Every regression probe sits at **R2 ~ 0** even at all-segment scope, on the same stars and the same
bags where v1 transit reached 2.18x base rate. The reason: for *detection*, window count correlates
with catalogue membership (a star observed more is likelier to be catalogued at all), so bag size
leaks the label; for *regression* the population is already restricted to catalogued stars and within
it bag size says nothing about the *value* of nu_max.

**Consequence: the regression all-segment gains need no bag-size caveat, unlike v1 transit.**

Coverage-vs-bag-size decomposition (same three-arm structure as v1):

| probe | first | kmatch16 | all | kmatch − first | all − kmatch |
|---|---|---|---|---|---|
| numax_hon `rff_meanmap` | 0.8695 | 0.8770 | **0.9169** | +0.008 | **+0.040** |
| numax_hatt `rff_meanmap` | 0.8619 | 0.8438 | **0.8898** | −0.018 | **+0.046** |
| dnu_hatt `rff_meanmap` | 0.8566 | 0.8375 | **0.8833** | −0.019 | **+0.046** |
| prot_kounkel `moments` | 0.5101 | 0.4960 | **0.5351** | −0.014 | **+0.039** |
| osc_giant `mean_std` | 0.8975 | 0.9022 | **0.9105** | +0.005 | +0.008 |

Same shape as v1: **K-matching erases the gain (temporal spread is not the mechanism), while more
windows genuinely helps.** But here the null `bagsize_only` licenses the causal reading v1 could not
support: for a global stellar property, more windows means a better estimate, full stop.

**Best confound-free numbers, both levers compounding:**

| probe | `mean` @ first | best @ all | total gain |
|---|---|---|---|
| numax_hon | 0.7928 | **0.9169** `rff_meanmap` | **+0.124 R2** |
| prot_kounkel | 0.3472 | **0.5351** `moments` | **+0.188 R2** |
| dnu_hatt | 0.8217 | **0.8833** `rff_meanmap` | +0.062 |
| numax_hatt | 0.8299 | **0.8898** `rff_meanmap` | +0.060 |
| osc_giant | 0.8450 | **0.9105** `mean_std` | +0.066 |

**Two honest nulls appear only at all-segment scope**, where the untrained arm catches up:
`flare` gap **−0.032** (untrained 0.537 > trained 0.506, deepening the standing "training hurts flare
localization" finding) and `prot_kounkel` gap **−0.003** (confirmed at first +0.040 and kmatch16
+0.063, null at `all`). More windows help the random encoder at least as much on these two.

`max` pooling degrades or stalls at all-segment on every regression probe (numax_hatt −0.001,
dnu_hatt −0.001, prot_kounkel −0.016) while every dispersion operator improves: the same
beta_crit = O(log K) effect seen on v1 transit, now on continuous targets.

## How to run

```powershell
$env:PYTHONPATH="src"; $py="C:\Users\user1\miniconda3\envs\swm\python.exe"
# 1. bag caches (9 arms x 2 scopes; ~11 s/arm first, ~30 s/arm all; fp16, segment offsets kept)
& $py -m swm.eval.mil_cache --cells exp05_comb_fbwd_c1p0 exp05_comb_off --seeds 0 1 2 3 --scope first
& $py -m swm.eval.mil_cache --cells exp05_comb_fbwd_c1p0 exp05_comb_off --seeds 0 1 2 3 --scope all
# 2. sweeps. CONCURRENT RUNS MUST USE DIFFERENT --out: each does read -> concat -> write and would
#    otherwise silently clobber the other. mil_report merges the parts back.
& $py -m swm.eval.mil_sweep --scope first
& $py -m swm.eval.mil_sweep --scope all    --out experiments/mil_pooling/mil_sweep_all.csv
& $py -m swm.eval.mil_sweep --scope all --kmatch 16 --kmatch-draws 3 `
      --poolings mean max moments mean_std mean_skew quantile3 ws_mean ws_max ws_lse ws_topk `
                 ws_ppv_lspv ws_linsoftmax ws_smooth --out experiments/mil_pooling/mil_sweep_kmatch.csv
& $py -m swm.eval.mil_learned --scope first          # Tier-3 diagnostic, ~80 min
# 2b. Phase 2 - new-task pool. First-segment reuses experiments/new_task_exp05/mu_cache (no GPU
#     pass); the two pools are different star populations and MUST NOT share a CSV (defaults enforce).
$T="osc_giant solar_like_osc flare numax_hon numax_hatt dnu_hatt prot_kounkel" -split ' '
$P="mean max quantile3 moments mean_std mean_skew rff_meanmap" -split ' '
& $py -m swm.eval.mil_sweep --pool new_task                       # detection, first segment
& $py -m swm.eval.mil_sweep --pool new_task --tasks numax_hon numax_hatt dnu_hatt prot_kounkel `
      --poolings @P --out experiments/mil_pooling/mil_sweep_new_task_reg.csv    # regression, R2
# all-segment for this pool needs its own caches (new_task_extract has no all-segment path):
& $py -m swm.eval.mil_cache --pool new_task --scope all --cells exp05_comb_fbwd_c1p0 exp05_comb_off `
      --seeds 0 1 2 3                                             # ~55 min, one shared I/O pass
& $py -m swm.eval.mil_sweep --pool new_task --scope all --tasks @T --poolings @P `
      --out experiments/mil_pooling/mil_sweep_new_task_all.csv
& $py -m swm.eval.mil_sweep --pool new_task --scope all --kmatch 16 --kmatch-draws 2 --tasks @T `
      --poolings mean moments mean_std rff_meanmap --out experiments/mil_pooling/mil_sweep_new_task_kmatch.csv
# 3. consolidate + headline tables, then the notebook
& $py -m swm.eval.mil_report
& $py -m swm.eval.mil_report --pool new_task
conda.exe run -n swm jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.kernel_name=swm src/notebooks/mil_pooling.ipynb
```

## Code

`src/swm/eval/pooling.py` (operators; imported by `readout_sweep` and available to
`new_task_scorecard`) · `mil_cache.py` (bag caches with the val split and segment offsets;
batched encoding at 93k win/s vs 8.2k) · `mil_sweep.py` (orchestrator) · `mil_learned.py` (Tier 3) ·
`mil_report.py` (merge + headline blocks). Tests: `src/swm/tests/test_pooling.py` (17),
`test_mil_learned.py` (5), all green.

## Next

- Decide whether the confound-free K-matched claim or the all-segment upper bound leads the paper.
- A within-star fixed-K stratum test (same stars, first 16 vs first 32 windows) would separate
  coverage from selection where K-matching cannot.
- **Test the support hypothesis** for Phase-2 finding 1: a learning curve over training-star count on
  `osc_giant`, checking whether `mean_std`'s advantage over `mean` shrinks as positives are removed.
  If it does, "dispersion pooling needs positives" is the rule and v1's null on eb/pulsating/rotation
  is a sample-size effect rather than a task-type effect.
- Score-space operators were not run at the new-task all-segment scope (a per-window logistic over
  1.07M rows for a question already answered: the witness-rate gains were negative on every new-task
  probe at first-segment scope). Deliberate cut, not an oversight.
- `gmm_prototype` was excluded from the all-segment and K-matched arms for cost; it loses everywhere
  it has been run.
