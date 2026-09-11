"""Replay the C1/C2 supervised checkpoints on their test stars and keep the per-star predictions.

Why this exists (Yue Ma review 2026-09-10, item 2): `swm.train.supervised` scores test once and writes
only the scalar into `result.json`; the paired star-bootstrap needs the predictions themselves. Each
run's `best.pt` holds the selected weights, and `data_cache/` holds the exact test bags, so this is
132 forward passes and NO training.

Footing check: the metric recomputed from the replayed predictions must reproduce the run's own
`result.json` score. Tolerance defaults to 1e-4 because inference runs under autocast (fp16) and the
GPU kernel order is not bit-reproducible; a larger gap means a stale checkpoint or a changed cache
and fails the script.

Run (swm env, repo root, PYTHONPATH=src, GPU, a few minutes):
    python experiments/replay_c1c2_scores.py
    python experiments/replay_c1c2_scores.py --tasks eb --seeds 0 --device cpu     # smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from swm.data.labelled import load_bags, star_label_frame, task_targets  # noqa: E402
from swm.train.supervised import DEFAULT_MANIFEST, build_net, load_manifest, predict_split, score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("replay_c1c2")


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay C1/C2 checkpoints and dump per-star test predictions.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--arms", nargs="+", default=["conv_supervised", "mlp_raw"])
    ap.add_argument("--tasks", nargs="+", default=None, help="default: every task in the manifest")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="default: the manifest's seeds")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="experiments/paper_bootstrap")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    base = manifest["base"]
    root = repo_root / manifest["paths"]["root"]
    cache_dir = root / "data_cache"
    tasks = [t for t in manifest["tasks"] if args.tasks is None or t["name"] in args.tasks]
    assert tasks, f"no manifest task matches {args.tasks}"
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available; pass --device cpu"
    window, batch_stars, amp = int(base["window"]), int(base["batch_stars"]), bool(base["amp"])
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = star_label_frame()
    frames = []
    worst = 0.0
    # seeds are declared per arm in the manifest (`arms.<arm>.seeds`), as the run queue reads them
    jobs = [(arm, task, seed) for arm in args.arms for task in tasks
            for seed in (args.seeds or [int(s) for s in manifest["arms"][arm]["seeds"]])]
    for arm_name, task, seed in tqdm(jobs, desc="arm x task x seed", total=len(jobs)):
        run_dir = root / "runs" / arm_name / task["name"] / f"seed{seed}"
        try:
            payload = torch.load(run_dir / "best.pt", map_location=args.device, weights_only=False)
            with open(run_dir / "result.json", "r", encoding="utf-8") as handle:
                result = json.load(handle)
        except OSError as err:
            log.error(f"{run_dir}: cannot read run artifacts: {err}")
            raise
        test_bags = load_bags(task["population"], "test", window, cache_dir)
        y_test, keep = task_targets(task, test_bags.tics, labels)
        test_bags, y_test = test_bags.subset(keep), y_test[keep]
        assert len(y_test) == int(result["n_test"]), (
            f"{run_dir}: test population {len(y_test)} != recorded {result['n_test']}")

        # Regression targets were standardized on TRAIN ONLY inside the trainer; the inverse needs the
        # same two numbers, recomputed from the same cached train split.
        y_mean, y_sd = 0.0, 1.0
        if task["shape"] == "regression":
            train_bags = load_bags(task["population"], "train", window, cache_dir)
            y_train, keep_train = task_targets(task, train_bags.tics, labels)
            y_train = y_train[keep_train]
            y_mean, y_sd = float(y_train.mean()), float(y_train.std()) or 1.0

        net = build_net(manifest["arms"][arm_name], base, args.device)
        net.load_state_dict(payload["model"])
        pred = predict_split(net, test_bags, batch_stars, args.device, amp)
        if task["shape"] == "regression":
            pred = pred * y_sd + y_mean
        replayed = score(task["shape"], y_test, pred)
        gap = abs(replayed - float(result["score"]))
        worst = max(worst, gap)
        if gap > args.tol:
            raise AssertionError(f"{run_dir}: replay gives {replayed:.6f}, result.json says "
                                 f"{result['score']:.6f} (gap {gap:.2e} > {args.tol})")
        frames.append(pd.DataFrame({"arm": arm_name, "family": arm_name, "seed": seed,
                                    "arm_set": arm_name, "task": task["name"], "shape": task["shape"],
                                    "tic_id": np.asarray(test_bags.tics).astype(np.int64),
                                    "y": np.asarray(y_test).astype(float),
                                    "score": np.asarray(pred).astype(float)}))

    dumps = pd.concat(frames, ignore_index=True)
    log.info(f"footing check: {len(jobs)} runs reproduce result.json, worst gap {worst:.2e}")
    out_path = out_dir / "c1c2_star_scores.parquet"
    dumps.to_parquet(out_path, index=False)
    log.info(f"wrote {out_path} ({len(dumps)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
