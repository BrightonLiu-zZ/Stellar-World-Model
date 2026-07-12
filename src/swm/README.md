# swm — Stage 1 (pretrain) + Stage 2 (linear probe)

Self-supervised Conv1D-VAE + GRU latent-dynamics world model on TESS light curves, plus the frozen
linear probe. Config-driven (Hydra) for the A/B/C ablation; a variant is one CLI override.
See `docs/plans/0008-stage1-pretrain-stage2-probe.md` for the full design and locked decisions.

## Environment

Runs in the dedicated **`swm`** conda env (CUDA), NOT `astro`:
`C:\Users\user1\miniconda3\envs\swm\python.exe` (torch 2.5.1+cu121, hydra, wandb, sklearn).
Set `PYTHONPATH` to the repo `src/` so `swm` is importable.

```powershell
$py = "C:\Users\user1\miniconda3\envs\swm\python.exe"
$env:PYTHONPATH = "C:\git_repo\Stellar-World-Model\src"
```

## Pipeline

```
swm/
  configs/   Hydra: config.yaml + groups data/model/train/variant (+ paths). A variant = one override.
  data/      subset.py (subset + frozen split), pack.py (memmap + index), dataset.py (SeqWindowDataset)
  models/    encoder, decoder, dynamics (GRU), world_model (wraps all)
  train/     losses.py, loop.py, __main__ (Hydra entrypoint)
  eval/      extract.py (frozen-encoder mu -> per-star vectors), probe.py (logistic probe + results table)
  tests/     test_stage1.py
```

## Run order

```powershell
$py -m swm.data.subset                               # processed/subset/{subset_tics,split}.parquet
$py -m swm.data.pack                                 # processed/packed/{split}_windows.dat + index + manifest
$py -m swm.train variant=A seed=0                     # models/A_seed0/{best,last}.pt   (needs `wandb login` for online)
$py -m swm.train variant=B seed=0                     # models/B_seed0/...
$py -m swm.eval.extract variant=A seed=0; $py -m swm.eval.extract variant=B seed=0
$py -m swm.eval.probe   variant=A seed=0; $py -m swm.eval.probe   variant=B seed=0   # results/results_table.csv
$py -m pytest src/swm/tests -q
```

Smoke (no GPU pressure, no W&B, tiny data):
```powershell
$py -m swm.data.pack data.limit=50
$py -m swm.train variant=B seed=0 data.limit=50 train.max_epochs=2 data.num_workers=0 train.wandb.mode=disabled
```

## Key locked knobs (override on the CLI)

- `data.batch_size` (default 32; raise after the OOM check), `data.num_workers`
- `train.beta_target` / `train.beta_warmup_epochs` / `train.free_bits` (anti posterior-collapse)
- `train.lambda_dyn` (variant: A=0, B=1.0, C=5) — the one knob that differs A vs B
- `train.wandb.mode` (online | offline | disabled), `train.wandb.entity`
