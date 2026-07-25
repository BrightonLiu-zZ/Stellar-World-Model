# exp05 — latent-dynamics axis (multistep rollout × λ): recipes + how to run

**Status: configs/code/runners staged; NOT yet trained.** Plan:
[docs/plans/2026-07-22-exp05-dynamics-axis.md](../docs/plans/2026-07-22-exp05-dynamics-axis.md).
Design fixed by grill-with-docs 2026-07-22 (memory `project_exp05_config_decision.md`).

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
