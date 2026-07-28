"""
Tier-3 learned pooling for the MIL sweep (plan 2026-07-25): gated ABMIL and DSMIL on frozen mu.

These are the two methods a reviewer will ask about, so they are reported, but they are DIAGNOSTIC
ONLY: unlike every Tier-1/2 operator they train a small nonlinear head, which is the ADR-0008
exception that Prof. Theissen has not signed. The encoder itself is still frozen; only the pooling
head sees gradients.

Both are run with the three fixes the literature prescribes for our regime (200-900 positive bags,
where attention is known to overfit fast):
  small attention width L, so the head is ~2k parameters rather than ~16k;
  ACMIL stochastic top-k instance masking, which drops the currently most-attended windows during
  training so attention cannot collapse onto one or two of them;
  DTFD pseudo-bags, which split each star's windows into sub-bags inheriting the star label and so
  multiply the effective bag count, the standard remedy for few-bags-many-instances.

Bags are padded to a common length within a batch and masked, which is what makes this tractable:
a per-bag loop over 9,428 stars per epoch is dominated by kernel-launch overhead.

Run (swm env, from repo root, PYTHONPATH=src), after swm.eval.mil_cache:
    PYTHONUNBUFFERED=1 python -m swm.eval.mil_learned --scope first
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from swm.eval.mil_cache import cache_path, load_cache
from swm.eval.mil_sweep import star_labels
from swm.eval.skyline import _git_sha

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

repo_root = Path(__file__).resolve().parents[3]
tasks_default = ("pulsating", "eb", "rotation", "transit")


class GatedABMIL(nn.Module):
    """
    Ilse+2018 attention MIL with the gated attention branch, sized down for a few hundred positive bags.
    Attention weights a_k are a softmax over windows of w'(tanh(V h_k) * sigmoid(U h_k)); the bag
    embedding is their weighted sum, and one linear layer reads the bag label off it.
    """

    def __init__(self, dim: int, width: int = 16) -> None:
        super().__init__()
        self.v = nn.Linear(dim, width)
        self.u = nn.Linear(dim, width)
        self.w = nn.Linear(width, 1)
        self.head = nn.Linear(dim, 1)

    def attention(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gated = torch.tanh(self.v(x)) * torch.sigmoid(self.u(x)) # (B, K, width)
        logits = self.w(gated).squeeze(-1) # (B, K)
        logits = logits.masked_fill(~mask, float("-inf")) # padded windows must never receive attention
        return torch.softmax(logits, dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = self.attention(x, mask) # (B, K)
        bag = (a.unsqueeze(-1) * x).sum(dim=1) # (B, dim)
        return self.head(bag).squeeze(-1)


class DSMIL(nn.Module):
    """
    Li+2021 dual-stream MIL, the architecture built for frozen self-supervised embeddings whose
    positive instances are rare. Stream 1 takes the single highest-scoring window (the standard-MI
    view); stream 2 attends every window against that critical window (the collective view); the two
    scores are averaged, so the model degrades gracefully whichever assumption actually holds.
    """

    def __init__(self, dim: int, width: int = 32) -> None:
        super().__init__()
        self.instance = nn.Linear(dim, 1)
        self.q = nn.Linear(dim, width)
        self.v = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.instance(x).squeeze(-1).masked_fill(~mask, float("-inf")) # (B, K)
        critical = scores.argmax(dim=1)
        rows = torch.arange(x.shape[0], device=x.device)
        h_m = x[rows, critical] # (B, dim), the critical-instance embedding
        q = self.q(x) # (B, K, width)
        q_m = self.q(h_m).unsqueeze(1) # (B, 1, width)
        affinity = (q * q_m).sum(-1).masked_fill(~mask, float("-inf")) # (B, K)
        weights = torch.softmax(affinity, dim=1)
        bag = (weights.unsqueeze(-1) * self.v(x)).sum(dim=1)
        return 0.5 * (scores.gather(1, critical.unsqueeze(1)).squeeze(1) + self.head(bag).squeeze(-1))


def make_pseudo_bags(blocks: list[np.ndarray], y: np.ndarray, n_sub: int, rng: np.random.Generator,
                     ) -> tuple[list[np.ndarray], np.ndarray]:
    """
    DTFD pseudo-bags: shuffle each star's windows and deal them into n_sub sub-bags that inherit the
    star label. Multiplies the training bag count without new data, the standard remedy when there
    are only a few hundred positive bags. Bags too small to split are left whole.
    """
    out_blocks = []
    out_y = []
    for i, block in enumerate(blocks):
        if block.shape[0] < 2 * n_sub:
            out_blocks.append(block)
            out_y.append(y[i])
            continue
        order = rng.permutation(block.shape[0])
        for part in np.array_split(order, n_sub):
            out_blocks.append(block[part])
            out_y.append(y[i])
    return out_blocks, np.array(out_y, dtype=np.int64)


def pad_batch(blocks: list[np.ndarray], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length bags into (B, K_max, dim) plus a boolean mask marking the real windows."""
    lengths = []
    for block in blocks:
        lengths.append(block.shape[0])
    k_max = max(lengths)
    dim = blocks[0].shape[1]
    x = np.zeros((len(blocks), k_max, dim), dtype=np.float32)
    mask = np.zeros((len(blocks), k_max), dtype=bool)
    for i, block in enumerate(blocks):
        x[i, : block.shape[0]] = block
        mask[i, : block.shape[0]] = True
    return torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)


def apply_stkim(model: GatedABMIL, x: torch.Tensor, mask: torch.Tensor, top_k: int,
                rng: torch.Generator) -> torch.Tensor:
    """
    ACMIL stochastic top-k instance masking: hide the currently most-attended windows for this step,
    forcing the head to find redundant evidence instead of latching onto a single window.
    """
    with torch.no_grad():
        a = model.attention(x, mask)
    k = min(top_k, x.shape[1])
    drop = a.topk(k, dim=1).indices
    keep = mask.clone()
    coin = torch.rand(drop.shape, generator=rng, device=x.device) < 0.5
    keep.scatter_(1, drop, ~coin & keep.gather(1, drop))
    empty = ~keep.any(dim=1)
    keep[empty] = mask[empty] # never leave a bag with zero visible windows
    return keep


def train_head(kind: str, train_blocks: list[np.ndarray], y_train: np.ndarray,
               eval_blocks: dict[str, list[np.ndarray]], y_eval: dict[str, np.ndarray],
               device: str, epochs: int = 30, batch: int = 64, seed: int = 0) -> dict[str, float]:
    """
    Fit one learned pooling head on frozen mu and return val/test PR-AUC at the best val epoch.
    Selection on val mirrors the Tier-1/2 protocol, so the comparison is like-for-like.
    """
    torch.manual_seed(seed)
    dim = train_blocks[0].shape[1]
    if kind == "abmil":
        model = GatedABMIL(dim).to(device)
    elif kind == "dsmil":
        model = DSMIL(dim).to(device)
    else:
        raise ValueError(f"unknown learned pooling {kind!r}")
    positives = float(y_train.sum())
    pos_weight = torch.tensor([(len(y_train) - positives) / max(positives, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    best = {"pr_auc_val": -1.0, "pr_auc_test": float("nan"), "epoch": -1}
    for epoch in range(epochs):
        sub_blocks, sub_y = make_pseudo_bags(train_blocks, y_train, 4, rng)
        order = rng.permutation(len(sub_blocks))
        model.train()
        for start in range(0, len(order), batch):
            picks = order[start : start + batch]
            batch_blocks = []
            for pick in picks:
                batch_blocks.append(sub_blocks[pick])
            x, mask = pad_batch(batch_blocks, device)
            if kind == "abmil":
                mask_used = apply_stkim(model, x, mask, top_k=2, rng=gen)
            else:
                mask_used = mask
            target = torch.from_numpy(sub_y[picks].astype(np.float32)).to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, mask_used), target)
            loss.backward()
            optimizer.step()

        model.eval()
        scored = {}
        with torch.no_grad():
            for split, blocks in eval_blocks.items():
                values = []
                for start in range(0, len(blocks), batch):
                    x, mask = pad_batch(blocks[start : start + batch], device)
                    values.append(model(x, mask).cpu().numpy())
                scored[split] = np.concatenate(values)
        pr_val = float(average_precision_score(y_eval["val"], scored["val"]))
        if pr_val > best["pr_auc_val"]:
            best = {"pr_auc_val": pr_val,
                    "pr_auc_test": float(average_precision_score(y_eval["test"], scored["test"])),
                    "epoch": epoch}
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier-3 learned pooling reference for the MIL sweep")
    parser.add_argument("--cells", nargs="+", default=["exp05_comb_fbwd_c1p0"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--scope", default="first", choices=["first", "all"])
    parser.add_argument("--tasks", nargs="+", default=list(tasks_default))
    parser.add_argument("--kinds", nargs="+", default=["abmil", "dsmil"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out", default="experiments/mil_pooling/mil_learned.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    subset = pd.read_parquet(repo_root / "processed" / "subset" / "subset_tics.parquet")
    arms = []
    for cell in args.cells:
        for seed in args.seeds:
            arms.append((cell, seed, "trained"))
    arms.append(("untrained", 0, "untrained"))

    rows = []
    for cell, seed, kind_arm in tqdm(arms, desc="arms", total=len(arms)):
        path = cache_path(cell, seed, args.scope)
        if not path.exists():
            log.warning(f"{path.name}: no cache; skipped")
            continue
        cache = load_cache(path)
        blocks = {}
        tics = {}
        for split in ["train", "val", "test"]:
            tics[split], blocks[split], _ = cache[split]
        for task in args.tasks:
            y = {}
            for split in ["train", "val", "test"]:
                y[split] = star_labels(tics[split], subset, task)
            for kind in args.kinds:
                best = train_head(kind, blocks["train"], y["train"],
                                  {"val": blocks["val"], "test": blocks["test"]}, y, device, args.epochs, seed=seed)
                best.update({"exp_name": cell, "seed": seed, "arm_kind": kind_arm, "bag_scope": args.scope,
                             "family": "learned", "pooling": kind, "param": -1.0, "task": task,
                             "base_rate_test": float(y["test"].mean()), "n_test_pos": int(y["test"].sum())})
                rows.append(best)
                log.info(f"{cell} s{seed} {task} {kind}: val {best['pr_auc_val']} test {best['pr_auc_test']}")

    result = pd.DataFrame(rows)
    result["run_id"] = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    result["git_sha"] = _git_sha()
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        result = pd.concat([pd.read_csv(out_path), result], ignore_index=True)
    result.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(result)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
