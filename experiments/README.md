# Experiments

Each ablation is a self-contained folder `expNN_<slug>/` holding its own `packed/`, `models/`, `results/`, and `figs/`. Window-independent inputs are **shared** at repo root: `processed/subset/` (TIC train/val/test split), `processed/sequences/`, `labels/`.

An ablation is expressed as one Hydra experiment-group YAML in `src/swm/configs/experiment/`; the v1-locked defaults (`data/default.yaml`, `train/default.yaml`) are never edited. Run any stage with `+experiment=<name>`.

| exp | window × seq_len | variants | plan | status | headline (pulsating trained − untrained) |
|---|---|---|---|---|---|
| exp00_window1024_seq4 | 1024 × 4 | A, B, C · seed 0 | (baseline) | reference | +0.008 (0.744 vs 0.736) |
| exp01_window256_seq16 | 256 × 16 | B · seed 0 | [2026-07-09](../docs/plans/2026-07-09-window-shrink-ablation-exp01.md) | done | +0.003 (0.771 vs 0.768) — mechanism fixed, SSL≈untrained |
| exp02\* recon-objective sweep (10 combos) | 256 × 16 | B · seed 0 | [2026-07-12](../docs/plans/2026-07-12-exp02-recon-objective-sweep.md) | done | linear gap still ≈0, but **GBM-on-trained-μ 0.767→0.82** (`info_in_mu` False→True) — objective fixed, linear readout is the new barrier. See [exp02_sweep_README.md](exp02_sweep_README.md) |
| exp03_forensics (no training) | — | — | [2026-07-13](../docs/plans/2026-07-13-exp03-loss-forensics-and-wide-sweep.md) | done | H1–H5 all confirmed: checkpoint selection was 87–95 % clamp-saturated KL noise; every latent dim BELOW the free-bits floor (dead KL gradient). See [exp03_forensics/README.md](exp03_forensics/README.md) |
| exp03\* KL-schedule × objective sweep (36 combos) | 256 × 16 | B · seed 0 | [2026-07-13](../docs/plans/2026-07-13-exp03-loss-forensics-and-wide-sweep.md) | done | **first linear-probe win**: `fb0p02_b0p1_lpsd` pulsating **+0.055** (0.822 vs 0.767, >2·SE) & eb +0.061 (>2·SE); winning region = low β + small nonzero floor; 1-seed, 3-seed confirm pending. See [exp03_sweep_README.md](exp03_sweep_README.md) |
| exp04\* 3-seed confirm + encoder axis + KL corner (39 runs) | 256 × 16 | B · seeds 0–2 | [2026-07-19](../docs/plans/2026-07-19-exp04-confirm-encoder-kl.md) | done | headline **splits**: winner's pulsating +0.055 was seed noise (+0.016 ± 0.034), **eb +0.066 ± 0.006 confirmed** and at the engineered-feature skyline (headroom closed); `fb0_b0p1_comb` = only all-task 3-seed confirm; `enc_whalf`/`enc_z32` eb ≈ +0.10; transit info_in_mu=True all seeds; KP+CP(217) star set highly separable (+0.12…+0.16 vs untrained). See [exp04_sweep_README.md](exp04_sweep_README.md) |
| exp05\* dynamics axis: multistep rollout × λ (12 cells × 4 seeds = 48) | 256 × 16 | B · seeds 0–3 | [2026-07-22](../docs/plans/2026-07-22-exp05-dynamics-axis.md) | done | **criterion 1 MET**: dynamics-weighting > dynamics-off (rotation fwd+bwd **+0.049**, eb +0.029, transit fwd@λ130 +0.035, all >2·SE; pulsating null under comb, lpsd multistep +0.037) — first "SSL > untrained-off" win; **fwd+bwd best, higher λ better**; mechanism = **collapse-reversal** (comb 5→17–25 active dims). **criterion 2 PARTIAL**: rollout beats persistence on every class (all >1) but NOT more on periodic (quiet ≥ periodic; decoded periodic rollouts flat) → "smooth is predictable," not periodic physics — exp06 target. See [exp05_sweep_README.md](exp05_sweep_README.md) — **and its [Superseded / later corrections](exp05_sweep_README.md#superseded--later-corrections-2026-07-27-forensics--2026-07-28-exp06) before quoting**: criterion 2's "real predictive signal" is retracted (an oracle constant beats persistence by more), "fwd+bwd best / higher λ better" does not survive dose-matching, and collapse-reversal is an accompaniment rather than the mechanism. Criterion 1 itself strengthened. |
| exp06\* geometry/coverage axis + hygiene (43 runs) | {256×16, 512×8, 1024×4} at a fixed 4096-cadence horizon (+ 2048×2 EB-coverage probe) | B · 6 seeds · ep100 | [2026-07-27](../docs/plans/2026-07-27-exp06-geometry-coverage.md) · manifest [exp06_geometry_coverage.yaml](configs/exp06_geometry_coverage.yaml) | done | **H1 NOT SUPPORTED — window is not the binding constraint; the window lever is retired and 256×16 stays the geometry of record.** All four pre-registered gates fail on the dyn-on arm; the dyn-off "passes" are a **control-degradation artifact** (pulsating 0.758→0.690 from 256→1024 while eb −0.003 / rotation +0.008 stay flat), and the w2048 probe is decisive — every task *drops* (eb 0.736@256 → 0.680@2048). Mechanism, from the K-matched arm: per-window quality **does** rise with window (eb 0.628→0.716 at bag=4) but at a fixed cadence budget bag size compensates almost exactly, and is worth **~2× window length** (+0.086 vs +0.044). **H2**: rollout stays closed at *every* geometry — lag-1 μ-ACF −0.006@256 → −0.232@1024, growing more anti-correlated and never approaching the 0.3 pre-gate. **C1 replicates** at 256×16/ep100/6 seeds (eb +0.043, rotation +0.046, transit +0.027; pulsating ns at 256 but +0.034 passes at 512). **Edge**: the spike survives `free_bits=0.02` at 15.3× → **`recon_aux` confirmed as the immunity knob, `free_bits` exonerated**. ⚠ **Every geometry number is arm-specific — always quote the arm; never average across cells** (the pooled kmatch4 series matches neither arm). See [exp06_geometry_README.md](exp06_geometry_README.md), including its 2026-07-30 post-meeting reinterpretation (the coverage paradox is a measured budget trade, and the working mechanism is that the model flags eclipses as rare high-contrast outlier windows rather than learning EB periodicity). || exp07\* aux factorization + Hann edge fix (10 cells × 4 seeds = 40 runs) | 256 × 16 | B · seeds 0–3 (winner + baseline → 6) · ep100 | [2026-08-01](../docs/plans/2026-08-01-exp07-aux-factorization.md) · manifest [exp07_aux_factorization.yaml](configs/exp07_aux_factorization.yaml) | **designed — pilots pending** | The aux term becomes first-class: 2×2 `recon_aux` (type {combined, log_psd} × weight {0.1, 0.3}) + a Hann-tapered cell (`psd_window=hann`, taper-weighted demeaning — makes the C1-confirmed "decoder buys spectral power at the window edge" purchase impossible by construction), each × dyn {off, fwd_bwd@full dose}. λ dials recipe-specific (comb 60, log_psd 27 per pre-check C4), steady-state pilots + eval dose gate [0.6, 1.4]. Pre-registered P1–P5 + gates in the manifest; winner (absolute probe score, F4) = ML4PS final recipe; **last recipe-touching experiment before the Aug 15 freeze**. |
| exp08\* dynamics ladder — Q4 smoothness ablation (4 new cells × 6 seeds = 24 runs; off/fbwd arms reused from exp07) | 256 × 16 | B · seeds 0–5 · ep100 | [2026-08-07](../docs/plans/2026-08-07-exp08-dynamics-ladder.md) · manifest [exp08_dynamics_ladder.yaml](configs/exp08_dynamics_ladder.yaml) | **done 2026-08-08** | **Q4 CLOSED: not a smoothness prior — the active ingredient is *unsatisfiable prediction pressure*, and learned recurrent dynamics are sufficient, not necessary.** G-prior FAIL (smoothness at max satisfiable pressure: eb +0.019 ± 0.010, rotation +0.015 ± 0.009, ns) · G-gru PASS (fbwd − smooth eb +0.062, rotation +0.030, >2·SE). A **learned linear map matches the GRU** on eb/rotation (linear − off eb +0.065\*, rotation +0.035\*; vs fbwd ns) and a **frozen random GRU at dose 2.4** matches too, holding the latent wide (69–114 units) and reproducing the full Q11 signature (residual asymmetry **12.8×**, fusion deltas). Dose gate became a result: the smoothness prior **saturates** (λ×18 → dose ×0.13, latent → 1 unit; parity unreachable for a satisfiable objective) and the frozen term is **bistable** around the collapse transition (0.40 collapsed → smooth-level probes / 2.37 wide → fbwd-level). GRU's unique property: only arm gaining eb/rotation while holding pulsating at off-parity. P1 FAIL · P2 PASS · P3 PASS (fair arms) · P4/P5 split in the same direction — the signature follows the benefit, not the learning. See [exp08_ladder_README.md](exp08_ladder_README.md). |

## Downstream probe re-scores (eval-only, no training)

| set | arms | plan | status | headline |
|---|---|---|---|---|
| `new_task/` | exp03 leader `fb0_b0p1_comb` ×3 seeds + untrained | [2026-07-22](../docs/plans/2026-07-22-task2-new-task-label-integration.md) | done | global-vs-localized dichotomy: SSL > untrained on every asteroseismic probe; MIL only helps localized transients; training *hurts* flare localization. **Frozen — do not overwrite** (the comparison baseline). |
| `new_task_exp05/` | `exp05_comb_off` (λ0) ×4 + `exp05_comb_fbwd_c1p0` (λ66) ×4 + reused untrained | [2026-07-25](../docs/plans/2026-07-25-exp05-downstream-and-notebook.md) | done | **criterion 1 transfers: 8/11 probes** (rotation_period +0.193 R², numax_hon +0.140, rgb_vs_heb +0.069 → *flips the old null*). **But the A1/A2 engineered-feature ceilings beat SSL on all 11** — "beats features" is measured false; report `frac_gap_to_A1_closed` (34–86%) instead. Runner: `run_new_task_exp05.ps1`; analysis: `analyze_new_task_exp05.py`. |

Diagnostics + results notebook for exp05: `src/notebooks/exp05_diagnostics.ipynb` (sections 0/A–H —
run integrity, collapse-reversal, recon/lightcurves, UMAP + controls, criterion 1, criterion 2,
downstream transfer, MIL placeholder).

## Readout / pooling studies (eval-only, no training)

These vary **how a star's bag of window-level μ is aggregated**, holding the encoder fixed — the
opposite axis to the ablations above, which vary the encoder and hold the readout fixed.

| set | what varied | plan | status | headline |
|---|---|---|---|---|
| `mil_pooling/` | ~18 pooling operators × 3 bag scopes (first-segment / all-segment / K-matched-16) × 4 seeds, on `exp05_comb_fbwd_c1p0` + `exp05_comb_off` + capacity-matched untrained; **both** star pools (v1 subset and the new-task pool), detection scored by PR-AUC and 4 downstream probes by R² | [2026-07-25](../docs/plans/2026-07-25-mil-pooling-sweep.md) | done | **Pooling is a lever for localized signals only.** transit **0.145 → 0.21–0.24** at fixed bag size from the readout alone, no retraining (dispersion operators: mean⊕std, mean⊕skew); eb / pulsating / rotation move ≈0. The LSE temperature orders tasks by how localized their signal is (relative gain mean→max: transit +54%, eb +20%, rotation +7%, pulsating −1%). Learned pooling (**ABMIL, DSMIL**) loses on all four v1 tasks to zero-parameter operators at 122–216 positives. Also surfaced: the eval protocol had been scoring only **~26%** of available windows, and the trained−untrained **gap is not invariant to the readout**. See [mil_pooling/README.md](mil_pooling/README.md); notebook `src/notebooks/mil_pooling.ipynb`. |

Machine-readable: `mil_pooling/mil_pooling_results{,_new_task}.csv` (headline blocks) and
`mil_pooling/mil_sweep*.csv` (long tables). Code: `src/swm/eval/{pooling,mil_cache,mil_sweep,mil_learned,mil_report}.py`,
tests `src/swm/tests/test_{pooling,mil_cache,mil_learned,mil_report}.py`.

## Findings that span experiments

Method-level discoveries and problems that apply to every table rather than to one hypothesis — the
first-segment protocol's coverage, the bag-size confound being specific to detection labels, what the
K-matched control can and cannot separate, the gap metric not being readout-invariant, and the
operational traps that produced silently wrong states — are collected in
[cross_experiment_findings.md](cross_experiment_findings.md). Read it before quoting any single
experiment's numbers.

Its forward-looking counterpart is [open_questions.md](open_questions.md): what is still **unmeasured**,
the cheapest test that would settle each item, and what the answer would change. Entries carry permanent
IDs (Q1–Q10) and are closed in place with their verdict. Read it before choosing the next experiment —
`docs/` is gitignored, so the dated plans and `STATUS.md` do not survive a fresh clone and this does.

Follow-on diagnosis: `src/notebooks/exp06_design_forensics.ipynb` (§1 window-edge spike · §2 the
amplitude-meter test · §3 the AR(1) skyline that closed criterion 2 · §4 frequency reach and pooling ·
§5 what predicts the criterion-1 gain). It explains the problems the exp05 notebook only documented;
its measurements are the source of most rows in exp05's corrections table. Precompute:
`analyze_exp06_{edge,mu_structure,temporal}.py`. Method-level findings that outlive single experiments
are collected in [cross_experiment_findings.md](cross_experiment_findings.md).

## Pre-exp08 check suite (eval-only, no training) — **COMPLETE 2026-08-07 → exp08 is Q4**

**Outcome.** Gate 0 (transfer) **PASS** by 15 SE (`numax_hon` +0.376 ± 0.025 vs six untrained inits — the
frozen recipe had never been scored on this menu). Gate 1 (taper cost) **HOLD** (+0.0001 ± 0.0037), so the
taper's 13% pulsator-band reconstruction penalty never reaches the probes and **branch A / Q12 is dead**.
The CHK-3 tie-break did not fire: µ is *not* a re-encoding of the engineered basis (it probes at 2.1–3.4×
base rate with all 25 features projected out) and `features ⊕ µ` beats `features` on all four v1 tasks
> 2·SE **in the fbwd arm only** — the complementarity is bought by the dynamics term. CHK-4 says the same
from the other side: hypothesis **(a) new content** on all three tests. Closed **Q1, Q2, Q3, Q8, Q11, Q12**;
selected **Q4 (smoothness ablation) as exp08**.

`src/notebooks/exp08_design_forensics.ipynb` does for exp07 what `exp06_design_forensics.ipynb` did for
exp05: it checks what the last experiment left standing before the next one is designed. Five checks,
all zero-training. Runner `run_exp08_prechecks.ps1` (`-Smoke` for a shakedown into a throwaway
namespace, `-KeepCaches` to skip the ~12 GB cleanup); precompute
`analyze_exp07_{centre_artifact,stitch_spectrum,mu_channel}.py`, `analyze_exp08_gates.py`,
`build_subset_mu_cache.py`.

| check | question | why it exists |
|---|---|---|
| **CHK-1** | the ADR-0010 7-probe menu on the frozen recipe, under two pre-registered gates | **Gate 0 (transfer)** — every asteroseismic number in this project is an *exp05-arm, ep60* number, and F25(a) says a short-budget run is not a prefix of a long one, so the frozen recipe has never been scored on this menu at all. **Gate 1 (taper cost)** — paired `hann0p3 − comb0p3`, the question Q1 asks. Primary probe `numax_hon` R²; `fbwd` decides, `off` replicates, never pooled. Untrained reference carries **six inits**, not one (F17). |
| **CHK-2** | the centre-of-window artifact the taper created | F27: the taper *relocated* the decoder's log-PSD purchase to the window centre rather than removing it. Verdict rests on the per-star correlation; the cell-level version measures `recipe` (F21 one level down). The stitch comb is architectural — untrained combs harder than trained — hence paper hygiene, not a probe cost. |
| **CHK-3** | what channel the probes read (Q2) | `probe(µ ⊥ B_full)` against the full engineered basis is the number the ML4PS fusion claim rests on and had never been run; exp06 only ever residualized against a periodicity-free amplitude basis. **Outranks CHK-1 in the branch rule**: Gate 1 ranks aux terms, CHK-3 says whether ranking them can matter. |
| **CHK-4** | what the dynamics arm's extra active units carry (Q11) | CCA against a same-cell cross-seed null, the *asymmetric* residual probe, and a direct scaling control for "the probe's StandardScaler exploits it". |
| **CHK-5** | the checkpoint-selection guard (Q8) | **Resolved before the suite ran: F18 is retracted.** The loop always guarded selection; the reported pathology was an `idxmin` that did not reproduce the selector's constraint. Now pinned by `test_dual_checkpoint.py::test_select_never_picks_a_warmup_epoch`. |

## How to run an ablation

```bash
# from repo root, swm CUDA env, PYTHONPATH=src
python -m swm.data.pack    +experiment=exp01_window256_seq16                     # pack (subdivides 1024->256 at pack time)
python -m swm.train        +experiment=exp01_window256_seq16 variant=B seed=0    # pretrain
python -m swm.eval.extract +experiment=exp01_window256_seq16 variant=B seed=0    # frozen-encoder mu
python -m swm.eval.probe   +experiment=exp01_window256_seq16 variant=B seed=0    # linear probe -> results/
# diagnostics: set EXP_NAME in src/notebooks/ablation_diagnostics.ipynb, then nbconvert --execute
```

`processed/subset/` is built once (`python -m swm.data.subset`) and reused by every experiment so comparisons use identical stars.
