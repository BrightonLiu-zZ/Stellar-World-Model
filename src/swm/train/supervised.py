"""C1/C2 supervised baselines: train one (arm, task, seed) end to end and score it once on test.

Roadmap rows C1 (D20, provenance Yue Ma W6 merged with the Y13c supervised ceiling) and C2 (Y13b).
Every design choice is recorded in the manifest, `experiments/configs/c1c2_supervised_baselines.yaml`,
which this module READS at run time -- there are no generated per-cell configs, so the manifest is
literally the single source of truth rather than the source a generator copied from.

WHAT MAKES THIS A FAIR BASELINE, in one place:
  populations   swm.data.labelled takes the star sets, splits and keep masks from the SAME functions
                the F1 probe scorers call. Split identity is a property of the code.
  input         first-segment 256-cadence windows -- the exact windows mu is pooled over.
  star score    mean over the star's window outputs, in-graph, for BOTH the loss and the metric.
  selection     best val metric, patience, cap. val is a split no probe has ever touched.
  protocol      one pre-registered optimiser/regularisation setting for all 11 tasks; nothing tuned
                per task, so no cell can be accused of being tuned up or down.

The remaining asymmetry is stated, not hidden: the SSL encoder was pretrained without labels on all
segments of the v1 train stars, while this arm sees first segments only. That asymmetry is what
self-supervision IS.

These are EXTERNAL BASELINES (ADR-0012 decision 3). They never touch the frozen linear-probe protocol
and are never reported as the probe.

Run (swm env, repo root, PYTHONPATH=src) -- one run per invocation, driven by the generated queue:
    python -m swm.train.supervised --arm conv_supervised --task eb --seed 0
    python -m swm.train.supervised --arm mlp_raw --task numax_hon --seed 2 --wandb disabled
    python -m swm.train.supervised --arm conv_supervised --task eb --seed 0 --limit 200 \
        --wandb disabled --out-root experiments/_smoke_c1c2      # smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

import wandb
from swm.data.labelled import load_bags, star_label_frame, task_targets
from swm.models.supervised import SupervisedNet, pool_bags
from swm.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("supervised")

repo_root = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = repo_root / "experiments" / "configs" / "c1c2_supervised_baselines.yaml"
SPLITS = ("train", "val", "test")


def load_manifest(path: Path) -> dict:
    """Read the one manifest that defines both arms, all 11 tasks and every knob."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except OSError as err:
        log.error(f"cannot read manifest {path}: {err}")
        raise


def score(shape: str, y: np.ndarray, pred: np.ndarray) -> float:
    """The task's headline metric, matching what the F1 scorers report for the same task shape."""
    if shape == "detection":
        return float(average_precision_score(y, pred))
    if shape == "contrastive":
        return float(roc_auc_score(y, pred))
    if shape == "regression":
        return float(r2_score(y, pred))
    raise ValueError(f"unknown task shape {shape!r}")


def metric_floor(shape: str, y_test: np.ndarray) -> float:
    """Metric-native floor for the `recovery fraction` denominator (manifest D-C1.7).

    Prevalence is a random ranker's expected PR-AUC, 0.5 a random ROC-AUC, 0.0 the R2 of predicting the
    mean. Capacity-free by construction, which the untrained-mu floor is not: that floor is itself a
    frozen linear readout while this ceiling is end-to-end.
    """
    if shape == "detection":
        return float(np.mean(y_test))
    if shape == "contrastive":
        return 0.5
    return 0.0


def build_net(arm: dict, base: dict, device: str) -> SupervisedNet:
    """Instantiate the arm's network: conv trunk (C1) or dense trunk (C2), same bottleneck and head."""
    net = SupervisedNet(trunk=arm["trunk"], window=int(base["window"]), z_dim=int(base["z_dim"]),
                        dropout=float(base["dropout"]),
                        enc_channels=arm.get("enc_channels"), kernel_size=int(arm.get("kernel_size", 5)),
                        hidden=arm.get("hidden"))
    return net.to(device)


def forward_stars(net: SupervisedNet, bags, rows: np.ndarray, device: str, amp: bool) -> torch.Tensor:
    """Star scores for the given star rows: encode every window once, then bag-mean inside the graph."""
    windows, bag_ids = bags.gather(rows)
    x = torch.from_numpy(windows).unsqueeze(-1).to(device, non_blocking=True) # (n_win, window, 1)
    index = torch.from_numpy(bag_ids).to(device, non_blocking=True)
    with autocast("cuda", enabled=amp and device == "cuda"):
        window_out = net(x)
    return pool_bags(window_out.float(), index, len(rows))


@torch.no_grad()
def predict_split(net: SupervisedNet, bags, batch_stars: int, device: str, amp: bool) -> np.ndarray:
    """Star-level predictions over a whole split, in the split's own star order."""
    net.eval() # switch off dropout / batchnorm train-mode behaviour
    out = []
    for start in range(0, len(bags), batch_stars):
        rows = np.arange(start, min(start + batch_stars, len(bags)))
        out.append(forward_stars(net, bags, rows, device, amp).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_one(manifest: dict, arm_name: str, task_name: str, seed: int, out_root: Path,
              device: str, limit: int | None, wandb_mode: str) -> dict:
    """One cell of the C1/C2 queue: train, select on val, score test once, write the run's artifacts."""
    base = manifest["base"]
    arm = manifest["arms"][arm_name]
    task = None
    for candidate in manifest["tasks"]:
        if candidate["name"] == task_name:
            task = candidate
    assert task is not None, f"task {task_name!r} is not in the manifest"

    set_seed(seed)
    window = int(base["window"])
    batch_stars = int(base["batch_stars"])
    max_epochs = int(base["max_epochs"])
    amp = bool(base["amp"])
    run_dir = out_root / "runs" / arm_name / task_name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "data_cache"

    labels = star_label_frame()
    data = {}
    for split in SPLITS:
        bags = load_bags(task["population"], split, window, cache_dir)
        y, keep = task_targets(task, bags.tics, labels)
        if limit is not None:
            # smoke only: keep the first `limit` KEPT stars, never a slice of the raw population (a
            # blind head would drop every positive on the low-prevalence tasks and score nothing).
            allowed = np.flatnonzero(keep)[:limit]
            trimmed = np.zeros(len(keep), dtype=bool)
            trimmed[allowed] = True
            keep = trimmed
        data[split] = (bags.subset(keep), y[keep])
        log.info(f"{split}: {len(data[split][0])} stars, {len(data[split][0].windows)} windows")

    y_train = data["train"][1]
    shape = task["shape"]
    # Regression targets are standardized on TRAIN ONLY and inverted before scoring, so R2 is
    # unaffected; it is an optimisation aid for a linear head, not a change of target.
    y_mean, y_sd = 0.0, 1.0
    if shape == "regression":
        y_mean = float(y_train.mean())
        y_sd = float(y_train.std()) or 1.0
    if shape == "regression":
        criterion = nn.MSELoss()
    else:
        pos = float(y_train.sum())
        neg = float(len(y_train) - pos)
        assert pos > 0, f"task {task_name} split train has no positives"
        pos_weight = torch.tensor(neg / pos, device=device) if bool(base["class_weighted"]) else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    net = build_net(arm, base, device)
    n_params = sum(p.numel() for p in net.parameters())
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(base["lr"]),
                                  weight_decay=float(base["weight_decay"]))
    scaler = GradScaler("cuda", enabled=amp and device == "cuda")

    wandb.init(project=manifest["wandb"]["project"], group=manifest["wandb"]["group"],
               name=f"{arm_name}_{task_name}_s{seed}", mode=wandb_mode,
               config={"arm": arm_name, "task": task_name, "seed": seed, "n_params": n_params,
                       "n_train": len(data["train"][0]), **base})

    history = []
    best_metric = -np.inf
    best_epoch = -1
    best_state = None
    patience_ctr = 0
    started = time.time()
    for epoch in range(max_epochs):
        net.train() # switch dropout / batchnorm back to training mode
        order = np.random.permutation(len(data["train"][0]))
        total_loss = 0.0
        n_batches = 0
        for start in tqdm(range(0, len(order), batch_stars), desc=f"{task_name} s{seed} ep{epoch}",
                          total=int(np.ceil(len(order) / batch_stars)), leave=False):
            rows = order[start:start + batch_stars]
            target = torch.from_numpy(y_train[rows].astype(np.float32)).to(device)
            if shape == "regression":
                target = (target - y_mean) / y_sd
            star_out = forward_stars(net, data["train"][0], rows, device, amp)
            loss = criterion(star_out, target)
            optimizer.zero_grad() # clear gradients from the previous batch before backprop
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(base["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss)
            n_batches += 1

        val_pred = predict_split(net, data["val"][0], batch_stars, device, amp)
        if shape == "regression":
            val_pred = val_pred * y_sd + y_mean
        val_metric = score(shape, data["val"][1], val_pred)
        train_loss = total_loss / max(1, n_batches)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_metric": val_metric})
        wandb.log({"epoch": epoch, "train/loss": train_loss, "val/metric": val_metric}, step=epoch)
        log.info(f"[{arm_name}/{task_name}/s{seed}] ep {epoch} train_loss {train_loss} val_metric {val_metric}")

        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= int(base["patience"]):
            log.info(f"[{arm_name}/{task_name}/s{seed}] early stop at epoch {epoch} "
                     f"(no val improvement for {patience_ctr})")
            break

    assert best_state is not None, "no epoch was ever selected; the run produced no model"
    net.load_state_dict(best_state)
    test_pred = predict_split(net, data["test"][0], batch_stars, device, amp)
    if shape == "regression":
        test_pred = test_pred * y_sd + y_mean
    y_test = data["test"][1]
    test_metric = score(shape, y_test, test_pred)

    # Selection-sanity check carried over from exp09: a run selecting at epoch 0 or at the cap is
    # reported with a flag, never shipped silently.
    epochs_ran = history[-1]["epoch"] + 1
    flags = []
    if best_epoch == 0:
        flags.append("selected_first_epoch")
    if best_epoch == max_epochs - 1:
        flags.append("selected_at_cap")
    if task.get("small_n"):
        flags.append("small_n")

    result = {"arm": arm_name, "roadmap_row": arm["roadmap_row"], "task": task_name,
              "block": task["block"], "population": task["population"], "shape": shape,
              "metric": task["metric"], "seed": seed,
              "score": test_metric, "val_metric": best_metric,
              "floor": metric_floor(shape, y_test),
              "n_train": int(len(data["train"][0])), "n_val": int(len(data["val"][0])),
              "n_test": int(len(y_test)),
              "n_test_pos": int(y_test.sum()) if shape != "regression" else -1,
              "n_params": int(n_params), "selected_epoch": int(best_epoch), "epochs_ran": int(epochs_ran),
              "flags": ";".join(flags), "minutes": round((time.time() - started) / 60.0, 2)}

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    with open(run_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    torch.save({"model": best_state, "result": result}, run_dir / "best.pt")
    wandb.log({"test/metric": test_metric, "selected_epoch": best_epoch})
    wandb.finish()
    log.info(f"[{arm_name}/{task_name}/s{seed}] TEST {task['metric']} {test_metric} "
             f"(floor {result['floor']}, selected epoch {best_epoch}, {result['minutes']} min)")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Train one C1/C2 supervised baseline cell.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-root", default=None, help="default: the manifest's paths.root")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="smoke only: cap kept stars per split")
    ap.add_argument("--wandb", default=None, choices=["online", "offline", "disabled"],
                    help="default: the manifest's wandb.mode")
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    out_root = Path(args.out_root) if args.out_root else repo_root / manifest["paths"]["root"]
    wandb_mode = args.wandb or manifest["wandb"]["mode"]
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available; pass --device cpu for a CPU smoke"
    result = train_one(manifest, args.arm, args.task, args.seed, out_root, args.device,
                       args.limit, wandb_mode)
    done = out_root / "runs" / args.arm / args.task / f"seed{args.seed}" / "DONE.txt"
    done.write_text(f"finished {pd.Timestamp.now().isoformat()} score {result['score']}\n",
                    encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
