# Experiments

Each ablation is a self-contained folder `expNN_<slug>/` holding its own `packed/`, `models/`, `results/`, and `figs/`. Window-independent inputs are **shared** at repo root: `processed/subset/` (TIC train/val/test split), `processed/sequences/`, `labels/`.

An ablation is expressed as one Hydra experiment-group YAML in `src/swm/configs/experiment/`; the v1-locked defaults (`data/default.yaml`, `train/default.yaml`) are never edited. Run any stage with `+experiment=<name>`.

| exp | window × seq_len | variants | plan | status | headline (pulsating trained − untrained) |
|---|---|---|---|---|---|
| exp00_window1024_seq4 | 1024 × 4 | A, B, C · seed 0 | (baseline) | reference | +0.008 (0.744 vs 0.736) |
| exp01_window256_seq16 | 256 × 16 | B · seed 0 | [2026-07-09](../docs/plans/2026-07-09-window-shrink-ablation-exp01.md) | done | +0.003 (0.771 vs 0.768) — mechanism fixed, SSL≈untrained |

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
