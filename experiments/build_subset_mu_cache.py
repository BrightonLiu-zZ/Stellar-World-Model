"""Build the v1-packed-subset mu cache for one cell's arms, using THAT CELL's checkpoints.

`new_task_scorecard` takes a single `--ckpt-dir` and uses it for every arm whose subset cache is
missing, because historically those caches already existed on disk (the exp05 cells carry them under
models/B_seed<N>/extracted/ and the caller hardlinked them in). The exp07 cells have no such caches, so
running the scorecard over a multi-cell fan with one --ckpt-dir would silently build every arm's
`rotation_period` / `ijspeert` mu from ONE cell's weights -- an arm label pointing at the wrong network,
which nothing downstream could detect.

This prebuilds each cell's subset cache with the right checkpoints, so by the time the scorecard runs
every `<arm>.npz` is already on disk and its ckpt-dir is never consulted. Same code path as the
scorecard's own builder (`new_task_scorecard.subset_mu`), so the cache is byte-identical to what it
would have written -- only the weights are the correct ones.

Run (repo root, swm env, PYTHONPATH=src):
    python experiments/build_subset_mu_cache.py --ckpt-dir experiments/exp07_hann0p3_fbwd/models \
        --arms hann0p3_fbwd_s0 hann0p3_fbwd_s1 --cache-dir experiments/exp08_prechecks/subset_mu_cache
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from swm.eval.new_task_scorecard import subset_mu

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("build_subset_mu_cache")


def main() -> int:
    ap = argparse.ArgumentParser(description="Prebuild v1-subset mu caches with the correct checkpoints.")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="Arm names ending seed<N> / _s<N>, or untrained / untrained_s<N>.")
    ap.add_argument("--ckpt-dir", required=True, help="models/ dir holding B_seed<N>/best_recon_aux.pt.")
    ap.add_argument("--cache-dir", required=True, help="Destination subset mu cache directory.")
    ap.add_argument("--ckpt", default="best_recon_aux",
                    choices=["best_recon_aux", "best_recon_only", "best"],
                    help="Which tracked best to encode. Default best_recon_aux, the value formerly "
                         "hardcoded, so every published cache stays bit-identical. exp09/exp10 cells "
                         "must pass best_recon_only (ADR-0012 A3): their sweep axis touches the "
                         "objective, so an aux-bearing selector would compare cells at checkpoints "
                         "chosen by different rules.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        target = cache_dir / f"{arm}.npz"
        if target.exists():
            log.info(f"{arm}: cache already present, skipped")
            continue
        subset_mu(arm, args.device, cache_dir, Path(args.ckpt_dir), ckpt=args.ckpt)
        assert target.exists(), f"subset_mu did not write {target}"
        log.info(f"{arm}: wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
