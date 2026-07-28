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
| exp05\* dynamics axis: multistep rollout × λ (12 cells × 4 seeds = 48) | 256 × 16 | B · seeds 0–3 | [2026-07-22](../docs/plans/2026-07-22-exp05-dynamics-axis.md) | done | **criterion 1 MET**: dynamics-weighting > dynamics-off (rotation fwd+bwd **+0.049**, eb +0.029, transit fwd@λ130 +0.035, all >2·SE; pulsating null under comb, lpsd multistep +0.037) — first "SSL > untrained-off" win; **fwd+bwd best, higher λ better**; mechanism = **collapse-reversal** (comb 5→17–25 active dims). **criterion 2 PARTIAL**: rollout beats persistence on every class (all >1) but NOT more on periodic (quiet ≥ periodic; decoded periodic rollouts flat) → "smooth is predictable," not periodic physics — exp06 target. See [exp05_sweep_README.md](exp05_sweep_README.md) |

## Downstream probe re-scores (eval-only, no training)

| set | arms | plan | status | headline |
|---|---|---|---|---|
| `new_task/` | exp03 leader `fb0_b0p1_comb` ×3 seeds + untrained | [2026-07-22](../docs/plans/2026-07-22-task2-new-task-label-integration.md) | done | global-vs-localized dichotomy: SSL > untrained on every asteroseismic probe; MIL only helps localized transients; training *hurts* flare localization. **Frozen — do not overwrite** (the comparison baseline). |
| `new_task_exp05/` | `exp05_comb_off` (λ0) ×4 + `exp05_comb_fbwd_c1p0` (λ66) ×4 + reused untrained | [2026-07-25](../docs/plans/2026-07-25-exp05-downstream-and-notebook.md) | done | **criterion 1 transfers: 8/11 probes** (rotation_period +0.193 R², numax_hon +0.140, rgb_vs_heb +0.069 → *flips the old null*). **But the A1/A2 engineered-feature ceilings beat SSL on all 11** — "beats features" is measured false; report `frac_gap_to_A1_closed` (34–86%) instead. Runner: `run_new_task_exp05.ps1`; analysis: `analyze_new_task_exp05.py`. |

Diagnostics + results notebook for exp05: `src/notebooks/exp05_diagnostics.ipynb` (sections 0/A–H —
run integrity, collapse-reversal, recon/lightcurves, UMAP + controls, criterion 1, criterion 2,
downstream transfer, MIL placeholder).

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
