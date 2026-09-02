"""exp10 pilot anchor: what `val/decorr` reads on the REFERENCE encoder, before any penalty is applied.

The manifest's decorr-weight pilot rule (constants.decorr_weight_guess) is stated as a RELATIVE target:
after the seed-0 run, achieved mean corr2 must fall at least 50% "vs the reference" while val_recon
rises under 5%. That rule is unreadable without the reference number, and the exp07 hann0p3 runs
predate the channel, so nothing logged it. This script measures it: the same statistic
`swm.train.losses.decorr_loss` computes during training, evaluated on the same val split, the same
loader geometry and the same feature table -- so the pilot compares like with like rather than against
a star-level number from a mu cache.

Reported per seed and pooled, because a single-seed anchor would put the 50% rule at the mercy of one
draw. Read the pooled mean; the spread is printed beside it (P10's median hid a 16x sd ratio once).

Run (repo root, swm env, PYTHONPATH=src; needs the GPU for the encoder pass, ~1 min for 6 seeds):
    PYTHONUNBUFFERED=1 python experiments/exp10_decorr_anchor.py
    python experiments/exp10_decorr_anchor.py --seeds 0 --split val
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from swm.data.dataset import SeqWindowDataset  # noqa: E402
from swm.eval.readout_sweep import build_model_from_ckpt  # noqa: E402
from swm.train.losses import decorr_loss  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("exp10_decorr_anchor")

packed_dir = repo_root / "experiments" / "exp01_window256_seq16" / "packed"
features_path = repo_root / "experiments" / "exp10_features" / "subset_features25.parquet"
reference_dir = repo_root / "experiments" / "exp07_hann0p3_fbwd" / "models"


@torch.no_grad()
def anchor_for_seed(seed: int, checkpoint: str, split: str, device: str,
                    models_dir: Path | None = None) -> float:
    """
    Mean squared mu-feature correlation for one seed, averaged over the split's batches.
    Encodes with `model.encoder` directly rather than the full forward: decorr_loss reads mu only, and
    skipping the sampled decode makes the pass both cheaper and deterministic.
    Pointed at any cell's models/ dir, this is also the only way to read E1's achieved corr2: cond_dec
    runs at decorr_weight 0, so its logged `val/decorr` is an inert zero and not a measurement.
    """
    path = (models_dir or reference_dir) / f"B_seed{seed}" / f"{checkpoint}.pt"
    assert path.exists(), f"missing reference checkpoint {path}"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model, cfg = build_model_from_ckpt(ckpt, device)
    dataset = SeqWindowDataset(packed_dir, split, int(cfg["data"]["seq_len"]), int(cfg["data"]["window"]),
                               randomize=False, features_path=features_path)
    assert dataset.n_missing_features == 0, f"{dataset.n_missing_features} segments have no feature row"
    loader = DataLoader(dataset, batch_size=int(cfg["data"]["batch_size"]), shuffle=False, num_workers=0)
    total = 0.0
    n_batches = 0
    for x, feats in tqdm(loader, desc=f"seed{seed}[{split}]", total=len(loader), leave=False):
        x = x.to(device)
        feats = feats.to(device)
        bsz, seq_len = x.shape[0], x.shape[1]
        flat = x.reshape(bsz * seq_len, x.shape[2], x.shape[3]) # (B*S, window, 1)
        mu, _ = model.encoder(flat) # (B*S, z)
        mu_seq = mu.reshape(bsz, seq_len, mu.shape[1]) # (B, S, z)
        total += float(decorr_loss(mu_seq, feats))
        n_batches += 1
    return total / max(1, n_batches)


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the exp07 reference's achieved mu-feature corr2.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                    help="Reference seeds; 0-5 is the set exp10's cells are paired against.")
    ap.add_argument("--checkpoint", default="best_recon_aux",
                    help="Reference checkpoint (the exp07 seeds predate best_recon_only.pt).")
    ap.add_argument("--split", default="val", help="Split to average over; val matches the logged channel.")
    ap.add_argument("--models-dir", default=None,
                    help="Any cell's models/ dir. Default: the exp07 hann0p3_fbwd reference.")
    ap.add_argument("--label", default=None, help="Name written to the `arm` column; default the dir's cell.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="Default: experiments/exp10_decorr_anchor.csv")
    args = ap.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else reference_dir
    label = args.label or models_dir.parent.name
    rows = []
    for seed in args.seeds:
        value = anchor_for_seed(seed, args.checkpoint, args.split, args.device, models_dir)
        log.info(f"{label} seed {seed}: mean corr2 {value}")
        rows.append({"arm": label, "seed": seed, "checkpoint": args.checkpoint, "split": args.split,
                     "mean_corr2": value})
    result = pd.DataFrame(rows)
    log.info(f"{label} over {len(rows)} seeds: mean {result['mean_corr2'].mean()}, "
             f"sd {result['mean_corr2'].std(ddof=1)}, min {result['mean_corr2'].min()}, "
             f"max {result['mean_corr2'].max()}")
    if models_dir == reference_dir: # the >=50% clause is defined against the reference, nothing else
        log.info(f"pilot target (>=50% reduction): val/decorr <= {0.5 * result['mean_corr2'].mean()}")
    out_path = Path(args.out) if args.out else repo_root / "experiments" / "exp10_decorr_anchor.csv"
    if out_path.exists() and "arm" in pd.read_csv(out_path).columns:
        previous = pd.read_csv(out_path)
        keep = previous[~((previous["arm"] == label) & (previous["checkpoint"] == args.checkpoint)
                          & (previous["split"] == args.split))]
        result = pd.concat([keep, result], ignore_index=True)
    result.to_csv(out_path, index=False)
    log.info(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
