# exp05 — latent-dynamics axis (multistep rollout × λ): recipes + how to run

**Status: trained + evaluated 2026-07-24 (48/48 runs, 0 failures).** Plan:
[docs/plans/2026-07-22-exp05-dynamics-axis.md](../docs/plans/2026-07-22-exp05-dynamics-axis.md).
Design fixed by grill-with-docs 2026-07-22 (memory `project_exp05_config_decision.md`).

> **Read the corrections before quoting this file.** Three of the headline claims below were revised or
> retracted by later work (§ [Superseded / later corrections](#superseded--later-corrections-2026-07-27-forensics--2026-07-28-exp06)).
> The 2026-07-24 results text is kept verbatim as the record of what was believed at the time.

## What exp05 tests

Does weighting the GRU latent-dynamics term up (and switching it to a hard **free-running 15-step
rollout**) improve the frozen representation, and does the rollout learn "physics" (beat copy-last
persistence on periodic stars, not quiet)? exp04 showed the dynamics term is tiny in the loss (<1–3%)
but a persistence-gap pre-check showed the GRU genuinely beats persistence 2.6–6.7× — dyn is small only
because the latent scale is tiny. So exp05 sweeps λ high (target dyn-contribution up to 1.0× recon) and
adds fwd+bwd and multistep modes, watching for latent collapse.

## Grid — 12 cells × 4 seeds = 48 runs (~12.5 h, seed-major, -MaxHours 12)

- comb `fb0_b0p1_comb`: `off`(λ0) · fwd×{0.1,0.3,1.0} · fwd+bwd×{0.1,0.3,1.0} · multistep×{0.1,0.3,1.0}
- lpsd `fb0p02_b0p1_lpsd` hedge: `off`(λ0) · multistep@1.0
- λ = target dyn-contribution {0.1,0.3,1.0} × recon / dyn_mode_steady, **calibrated 2026-07-23**:
  comb fwd {13,40,130} · comb fwd+bwd {7,20,66} · comb multistep {5,14,48} · lpsd multistep {32}.
- Measured: fwd ~12.4 min/run, fwd+bwd ~15, multistep ~20 → ~187 min/seed-pass. At `-MaxHours 12`
  seeds 0–2 finish (~9.4 h) and seed 3 covers ~11/12 cells ⇒ **every cell ≥3 seeds, most 4**.
- Collapse baseline: comb (fb0) falls to **5–6 active units** even at λ=1 fwd (exp04) — recipe baseline,
  not a multistep effect; lpsd (fb0.02) holds 128. The recipes bracket the collapse axis.
- Geometry window=256 / seq_len=16 (exp01 packed junction). Cosine LR; μ selected λ-independently
  (`recon+aux`); per-epoch `μ_var` logged (collapse monitor).

## How to run

```powershell
# 0. Pre-launch calibration smoke (each mode; reads steady dyn -> fill $lam in the runner)
#    NOTE: packed_dir is derived from exp_name, and only the SWEEP RUNNER creates each cell's packed
#    junction. A stand-alone smoke must point paths.packed_dir at the shared exp01 packed dir itself,
#    else SeqWindowDataset asserts "missing .../packed/train_index.parquet".
#    max_epochs must exceed beta_warmup_epochs (10) or dyn is read mid-warmup and is NOT steady.
#    (data.limit is a PACK-time knob only - inert here since packed is reused.)
$env:PYTHONPATH="src"; $py="C:\Users\user1\miniconda3\envs\swm\python.exe"
$packed="C:/git_repo/Stellar-World-Model/experiments/exp01_window256_seq16/packed"
& $py -m swm.train +experiment=exp05/base_comb model.dyn_mode=multistep train.lambda_dyn=1 `
    paths.packed_dir=$packed train.max_epochs=14 train.resume=false train.wandb.mode=disabled `
    exp_name=exp05_smoke_multi                                            # repeat for fwd, fwd_bwd
Remove-Item -Recurse -Force experiments\exp05_smoke_*                     # cleanup after reading dyn
# 1. Full sweep — USER TERMINAL (GPU + W&B online), after filling $lam:
cd C:\git_repo\Stellar-World-Model ; .\experiments\run_exp05_train_sweep.ps1            # -DryRun to preview
# 2. Morning-after eval (Claude Code, backgrounded):
.\experiments\run_exp05_eval_scan.ps1
python -m swm.eval.rollout_eval --exp-glob "exp05_*" --seeds 0 1 2 3
```

## Outputs

- `experiments/exp05_*/results/readout_sweep.csv` — per-cell probe PR-AUC (+ gbm diagnostic) vs the shared
  exp03 untrained reference; aggregate λ0-vs-λ>0 gap (>2·SE @4 seeds) = success criterion 1.
- `experiments/exp05_*/results/rollout_vs_persistence.csv` — per-class (periodic/quiet) rollout vs
  persistence latent MSE = success criterion 2; `figs/rollout_*` = the decoded "learned physics" examples.
- Collapse curve: `μ_var` / `n_active` vs λ from the W&B/stdout curves.

## Results (trained + evaluated 2026-07-24; 48/48 runs, 0 failures)

**Criterion 1 (representation) = MET.** Δ = pr_auc(λ>0 cell) − pr_auc(`off`), paired 4 seeds, logistic×mean,
>2·SE (`experiments/exp05_gap.csv`, `analyze_exp05_gap.py`): rotation fwd+bwd **+0.043…+0.049**, fwd@130
+0.038, multistep +0.023; eb fwd+bwd@0.3 **+0.029**, fwd@130 +0.024, multistep@48 +0.024; transit fwd@130
**+0.035**, fwd+bwd@66 +0.025, multistep +0.013. pulsating null under comb (all ns) but **lpsd multistep@32
+0.037**. → "dynamics-weighted SSL > dynamics-off SSL". **fwd+bwd best mode; higher λ → bigger gain.**

**Mechanism = collapse-reversal** (training W&B summaries): comb (fb0) 5 active units at λ0 → **17 (fwd λ130) /
25 (fwd+bwd λ66)**; μ_var rises. Multistep does NOT recruit dims (~5) and its dyn never scaled (contribution
capped ~0.5). lpsd holds 128 active.

**Criterion 2 (learned physics) = PARTIAL** (`exp05_rollout_summary.csv`, `analyze_exp05_rollout.py`):
rollout beats copy-last persistence on **every** class (gain_ratio > 1 everywhere, to 2.7×) — the dynamics
carry real predictive signal — but the pre-registered *periodic > quiet* ordering did NOT appear: gain is
slightly larger on **quiet** than periodic (periodic−quiet ratio negative for all trained cells; multistep@48
periodic 2.17 vs quiet 2.71), and decoded periodic rollouts relax toward flat (`figs/rollout_*`). → rollout
learns **"smooth is predictable," not periodic physics** — necessary but not sufficient. Neither win nor null;
**open improvement target for exp06** (levers: larger latent scale / period-aware or longer-horizon rollout /
periodic-only rollout loss). (fwd/fwd_bwd rollout is off-distribution — trained teacher-forced one-step; judge
rollout on multistep cells only.)

**Verdict:** first positive "dynamics half earns its keep" result (criterion 1) + a partial rollout finding
(criterion 2) flagged for exp06. Candidate headline recipe = comb + **fwd+bwd at high λ**. See memory
`project_exp05_verdict.md`, `docs/STATUS.md`.

## Revisited later — what happened to these claims (added 2026-07-26)

Forward-links only; nothing above is amended. Cross-cutting method findings from the same period are
collected in [cross_experiment_findings.md](cross_experiment_findings.md).

- **Criterion 1 survives a change of readout.** Both criteria above were measured under one pooling
  choice (logistic × mean). The MIL/pooling sweep re-ran the paired `comb_fbwd_c1p0 − comb_off`
  contrast under ~16 different pooling operators, 4 seeds, first-segment bags: **transit +0.035,
  eb +0.028, rotation +0.024 (all > 2·SE), pulsating +0.016 null** — the same pattern reported here.
  So criterion 1 is not an artifact of mean pooling. Evidence:
  [mil_pooling/README.md](mil_pooling/README.md), `src/notebooks/mil_pooling.ipynb` (final cell).
- **The λ0-vs-λ>0 gap used here is unaffected by the gap-metric pathology found later.** That
  pathology only bites when the gap is used to rank readouts of *differing* capacity or
  expressiveness; every number in this file holds the readout and pooling fixed and varies only the
  encoder, which is the case where the gap is valid. See finding F4 in the cross-cutting doc.
- **Criterion 2 did not survive to exp06 as an improvement target** — the 2026-07-27 forensics
  closed it first (correction #3 below). exp06 instead tested the geometry/coverage axis; verdict
  there: the coverage hypothesis was not supported and the window lever was retired for v1
  star-level scores; 256×16 stays the geometry of record. See
  [exp06_geometry_README.md](exp06_geometry_README.md), including its post-meeting reinterpretation.
- **The eval protocol used here scores only each star's first packed segment** (16 windows ≈ 5.7 d).
  That was later measured to discard about **74%** of available packed windows (median 32, mean 62,
  max 816 per star). It is a property of every table in this file, not a defect specific to exp05.
  See finding F1.

## Superseded / later corrections (2026-07-27 forensics · 2026-07-28 exp06)

Added 2026-08-01. Nothing above is amended — the Results section stays as the record of what was
believed on 2026-07-24. Sources: `src/notebooks/exp05_diagnostics.ipynb` (§§A–G, the section-by-section
audit), `src/notebooks/exp06_design_forensics.ipynb` (§§1–5, the diagnosis),
[exp06_geometry_README.md](exp06_geometry_README.md) (the geometry test that closed several of these).

### Claims that changed

| # | Original claim | Status | What measured it |
|---|---|---|---|
| 1 | "rollout beats persistence on every class **→ the dynamics carry real predictive signal**" | **inference RETRACTED**, observation stands | An **oracle constant** (predict the target latents' own mean) scores 2.32–2.70 — at or above *every* cell in the F1 bar chart. The λ=0 cells also clear ratio > 1 while losing to copy-last on **64–70% of individual stars**. Ratio > 1 is therefore not evidence of learned dynamics. The multistep cells do survive both readings (92–99% per-star wins). `analyze_exp05_rollout_floor.py` → `exp05_rollout_floor.csv`, notebook §F1b/§F2b |
| 2 | "the *periodic > quiet* ordering did not appear" | **confirmed, but overstated ~4×** | The metric's own floor is class-dependent (2.32 periodic vs 2.70 quiet), which accounts for **76%** of the −0.496 gap plotted. Floor-corrected residual is **−0.056 ± 0.005** — still the wrong sign, still gated, but a small effect. §F2b |
| 3 | "open improvement target for exp06 (levers: larger latent scale / period-aware or longer-horizon rollout / periodic-only rollout loss)" | **CLOSED, not inherited** | Lag-1 μ-trajectory autocorrelation is ≈ 0 at w256 (+0.017 periodic) and a **closed-form AR(1) fitted on train** also loses to the constant (1.02–1.11 vs the GRU's 1.01–1.07) — the GRU is at the linear ceiling and the ceiling is below a flat line, so this was never a modelling shortfall. exp06 then measured ACF going **more negative** with longer windows (−0.006 @256 → −0.232 @1024), closing criterion-2 physics for the whole v1 window family. Forensics §3; exp06 H2 |
| 4 | "**fwd+bwd best mode; higher λ → bigger gain**" | **REVISED** | λ was back-solved *per mode* to hit common contribution rungs, so raw λ is not comparable across modes. At matched dose fwd+bwd beats plain fwd only on `rotation`, and only at the two lower rungs, while `fwd` is significantly better on `transit` at the top rung. Defensible ranking: **fwd+bwd ≈ fwd > multistep**. Dose-response on the achieved axis (n=9) is real for **transit** (ρ=+0.85, p=0.004) and **eb** (+0.68, p=0.042) only; rotation and pulsating are p≈0.10. §E2b |
| 5 | "**Mechanism = collapse-reversal**" | **REVISED to accompaniment** | Demeaning within each cell — where mode and λ are fixed but seeds still recruit 9 to 41 units — collapses every correlation (\|ρ\| ≤ 0.18, all p > 0.28). exp06's curve audit then showed recruitment is a **transient**: fwd_bwd@256 re-recruits to ~40 active units at ep15–35 and re-collapses below 10 by ep50, while checkpoint selection sits at median ep89+ — yet C1 still replicates. So `n_active` at selection is **not** the carrier. §E4b; exp06 notebook audit |
| 6 | "multistep … contribution capped ~0.5" | **refined — a calibration miss, not a ceiling** | The shortfall is a *flat* 0.63× of target at every rung: raw `dyn` moves 2% (0.01244→0.01269) while λ rises 10×. So the cell labelled "multistep @ 1.0" trained at 0.63, confounding "rollout is a weaker objective" with "rollout got a smaller dose". exp06 hit the same failure mode across geometries (achieved 0.98 @w256 but ~0.28/~0.34 @w512/w1024) — the shared cause is calibrating λ from a short ep-7 pilot rather than steady state. §A4; exp06 notebook audit |
| 7 | "Criterion 1 = MET" | **STRENGTHENED** | Survives regressing out four amplitude scalars and *grows* on 3 of 4 tasks (rotation +0.043→**+0.054**, pulsating +0.016→**+0.038**, eb +0.020→+0.027; transit +0.025→+0.014), so it is not an amplitude-calibration artifact. Independently replicated in exp06 at **ep100 / 6 seeds**: eb +0.043, rotation +0.046, transit +0.027 (>2·SE), pulsating ns at 256 but **+0.034 passes at 512**. Forensics §2.3; exp06 C1 |

### Problems this file did not record

* **A decoder window-edge spike affects 36 of these 48 runs.** Reconstruction MSE at position 0 is
  **17.5×** the interior for `comb` + dynamics-on, against 1.7× with dynamics off; `lpsd` is immune
  (2.1×). Counting both ends it consumes **17.6%** of the total reconstruction loss on 3.1% of the
  window. It is **two-thirds a learned constant** (bias fraction 0.66) sitting where the decoder is
  structurally weakest — at random init the first output sample is **3.6× less** steerable from z than
  an interior one (Jacobian ratio 0.27), which dynamics-on training then inverts to 1.76. exp06's
  confirmation cell settled the cause: the spike **survives** `free_bits=0.02` at 15.3×, so
  **`recon_aux` is the immunity knob and `free_bits` is exonerated**. It does **not** reach the encoder
  (‖Δμ‖ from perturbing x[0] is 0.51–0.89× that of an interior sample), so criterion 1 above stands —
  but every dynamics-on `comb` arm carries a handicap the λ=0 baseline does not.
  `analyze_exp06_edge.py`, `.scratch/exp05-window-edge-defect/`
* **The `> 2·SE` gate is weaker than it reads.** With n=4, `2·SE = sd`, i.e. `t > 2` on 3 df — a
  one-sided **p ≈ 0.070**, not 0.05 — and it is applied **40 times** with no multiplicity correction,
  so ≈**2.8** cells would confirm by chance. 18 confirmed, so the aggregate is safe; individual
  marginal cells are not. §E1
* **`lpsd` is one cell against `comb`'s nine.** It carries the whole pulsating result with no λ ladder
  behind it, and supplies three of the five thinnest confirms. §E1
* **One run diverged and was averaged in silently.** `comb_fbwd_c1p0` seed 1 has a 45× rollout-MSE
  blow-up and lost to copy-last on **all 431** periodic stars; 14 of 144 cell×seed×class rows lose to
  persistence overall. F1 rated this notebook's designated winner cell at 1.55 purely because of it —
  its healthy seeds average **2.06**. §F1
* **On `rotation` the λ=0 baseline sits *below* the untrained encoder** (0.5155 vs 0.5317), so the
  first 0.016 of every rotation gain recovers a deficit training itself created. The
  `gap_vs_untrained` column is also never gated — no seed sd is stored for it. §E3
* **The representation is ~89% an amplitude meter.** PC1 holds **92.6%** of star-level variance and is
  **96.6%** linearly explained by four scalars carrying no frequency information. Training *reduces*
  effective dimensionality below a random encoder (participation ratio 1.96 untrained → **1.16** at
  λ=0 → 1.57 at λ=66). Per task, residualising amplitude leaves `pulsating` at 96% of its PR-AUC but
  `eb`/`rotation` at 72% — with the four scalars *alone* scoring at or above the residual — while
  `transit` **improves** (0.120→0.144). Forensics §2.
  *Status of the finding:* not treated as a defect. Prof. Theissen signed off on amplitude-dominance on
  2026-07-30 with a standing condition that it be justified, which closed the question; see the
  post-meeting section of [exp06_geometry_README.md](exp06_geometry_README.md) and
  `docs/plans/2026-07-30-theissen-meeting-followups.md`.
* **Nothing measurable predicts the criterion-1 gain within a cell.** 20 predictor×task within-cell
  tests (`n_active`, edge ratio, edge share, Jacobian ratio, interior recon), 1.0 expected false
  positive, exactly 1 survivor with a sign no mechanism explains. Forensics §5
* **44/48 runs were still improving at the freeze** (`max_epochs=60`). exp06 raised it to 100 and
  truncation **persisted** — 5–6 of 6 seeds still select within the last 10 epochs. Paired results are
  unaffected; absolute numbers are ceilings-in-progress. §A2; exp06 notebook audit
* **The `rotation` headline is label-confounded.** In the frozen v1 subset every rotation positive is
  also a transit/eb/pulsating star and no quiet star is rotation-positive, because `subset.py` defines
  quiet as matched in *no* catalog, rotation included. The paired Δ is valid (both arms share it); what
  "rotation" means is not. §D1b

## Curve-forensics addendum (2026-08-01, append-only)

The exp05 W&B histories (`exp05_forensics/curves_exp05/`, 48 runs) were re-analysed jointly with
exp06's in `src/notebooks/exp06_diagnostics.ipynb` section I. exp05-specific findings:

* **Best val/recon anti-selects the probe winners across the 12 cells.** Min post-warmup val/recon
  crowns `comb_off` on all four tasks; the probe winner is always a dynamics cell; Spearman(best
  recon, probe PR-AUC) = −0.58…−0.84. The λ-free recon+aux metric is uncorrelated cross-cell here
  (it crowns `fwd_c0p1`, never the probe winner). Filed as F11 in
  [cross_experiment_findings.md](cross_experiment_findings.md); seed-level version (no metric ranks
  seeds) is F12.
* **`comb_fbwd_c1p0` overshot its dose target at steady state** (achieved contribution 1.34× vs the
  1.0 target read at calibration) — the same non-stationarity that made exp06's epoch-7 pilots
  undershoot 3× (F13). Steady-state back-out puts λ_needed ≈ 49 for this cell.
* **The lpsd cells run `free_bits=0.02` and the floor engages at every epoch** (kl_loss ≠ kl_total
  row-for-row), unlike the comb cells (fb 0, identity exact). Free bits has no schedule; the 10-epoch
  knob is β warmup. Curve-reading traps filed as F14.
* **exp05 ↔ exp06 replication at 256×16 is clean**: recon/dyn/KL overlap through the shared 60
  epochs; the "25 ± 15 active units at freeze" above is the recruitment transient caught mid-decay
  (exp06 B3), completing to <10 by epoch 100 with criterion 1 intact.

**Addendum 2026-08-05 (exp07 notebook K5/K6, F25a) — the collapse-reversal number is a schedule
artefact, not a truncation.** The "25 ± 15 active units" is not the ep100 runs seen earlier: the cosine
LR is annealed over `max_epochs`, so at **matched epoch 40** this 60-epoch cell sits at lr 7.7e-5 with 30
active units while the ep100 runs of the same recipe are at 2.0e-4 with 11–15 and prune on to 7–8. The
active set freezes when the rate collapses. Criterion 1 is unaffected (it is a paired probe result), but
**the mechanism sentence "dynamics wins by reversing collapse" does not survive**: at ep100 the dynamics
arm holds only one to three units more than its own dyn-off arm and still wins every task, and a cell that
holds 18 units (`exp07_lpsd0p3_fbwd`) scores mid-table. A short-budget run is a different run, not a
prefix — the same mechanism behind the ep40 → ep100 dose drift ≈ 0.75.
