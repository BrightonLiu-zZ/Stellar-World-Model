"""Phase 1 window-level transit eval: true window labels vs broadcast star labels on cached mu.

Plan: docs/plans/2026-07-16-phase1-window-level-transit-labels.md (D3/D4/D5). Consumes the
first-segment window-mu caches that readout_sweep already wrote (no encoder pass, no GPU) plus the
Phase-1 label table labels/qc/transit_window_labels_w256.parquet, and answers two questions per
(experiment x checkpoint x readout):

  (a) star-level: fit the MIL window readout on TRUE window labels instead of the broadcast star
      label, score test stars by max over their first-segment windows, and compare PR-AUC against
      the broadcast fit under the identical protocol --> "was label noise the bottleneck?"
  (b) window-level: PR-AUC of in-transit vs clean-negative windows on the test split --> "is an
      in-transit window separable in mu at all under perfect labels?" Reported for the full
      population (headline; every non-quarantined test window) and within-transit-star only
      (diagnostic; star identity held fixed).

Label semantics (three-state, from the label table): 1 = in-transit, 0 = clean negative,
-1 = quarantine. Quarantine windows are dropped from every fit and from the window-level metrics;
star-level max-pooling keeps ALL windows so the protocol stays exactly comparable to the
readout_sweep window_score rows. Windows of stars absent from the table are clean negatives.
Star-level PR-AUC is reported under both the v1 star labels (matches readout_sweep) and the
ADR-0009 coverage-filtered v2 labels.

Rows append to experiments/exp03_transit_window_eval.csv with run_id + git_sha (append-only).

Run (from repo root, swm env, PYTHONPATH=src; CPU-only, safe alongside anything):
    python -m swm.eval.transit_window_eval                                # all cached exp03 combos
    python -m swm.eval.transit_window_eval --exp-glob exp03_fb0p02_b0p1_lpsd --ckpts best_recon_aux
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from swm.eval.readout_sweep import cached_mu, fit_readout_scores
from swm.eval.skyline import _git_sha

log = logging.getLogger(__name__)

repo_root = Path(__file__).resolve().parents[3]
readouts_default = ("logistic", "gbm")
label_modes = ("broadcast", "true")


def first_segment_index(packed_dir: Path, split: str) -> pd.DataFrame:
    """One row per star: its first packed segment, in ascending tic order (mirrors load_first_segment_blocks)."""
    index = pd.read_parquet(packed_dir / f"{split}_index.parquet")
    first = index.sort_values(["tic_id", "sector", "seg_idx"]).drop_duplicates("tic_id").sort_values("tic_id")
    return first.reset_index(drop=True)


def window_label_blocks(first: pd.DataFrame, split_labels: pd.DataFrame) -> list[np.ndarray]:
    """
    Per-star window-label vectors aligned to the cached mu blocks.
    Window j of a star's first segment is packed row row_start+j; a row absent from the label table
    belongs to a non-transit star and is a clean negative (0).
    """
    lookup = dict(zip(split_labels["row"].tolist(), split_labels["label"].tolist()))
    blocks = []
    for row in first.itertuples(index=False):
        labels = np.zeros(int(row.n_win), dtype=np.int8)
        for j in range(int(row.n_win)):
            labels[j] = lookup.get(int(row.row_start) + j, 0)
        blocks.append(labels)
    return blocks


def score_arm(mu: dict, wlabels: dict, star_y: dict, readouts: tuple[str, ...], label: str) -> pd.DataFrame:
    """
    Score one encoder arm: every (readout x label_mode) cell on the shared first-segment protocol.
    mu / wlabels hold per-star window blocks per split; star_y holds the v1/v2 star label vectors.
    """
    train_tics, train_blocks = mu["train"]
    test_tics, test_blocks = mu["test"]
    x_train_all = np.concatenate(train_blocks, axis=0)  # (n_train_win, z)
    x_test_all = np.concatenate(test_blocks, axis=0)    # (n_test_win, z)
    wl_train = np.concatenate(wlabels["train"])         # (n_train_win,) in {1, 0, -1}
    wl_test = np.concatenate(wlabels["test"])           # (n_test_win,)
    test_star_of_window = np.repeat(np.arange(len(test_blocks)), [b.shape[0] for b in test_blocks])

    rows = []
    for readout in readouts:
        for mode in label_modes:
            if mode == "broadcast":  # the readout_sweep window_score protocol: star label on every window
                y_rows = np.concatenate(
                    [np.full(b.shape[0], star_y["train_v1"][i], dtype=np.int64) for i, b in enumerate(train_blocks)]
                )
                keep = np.ones(len(y_rows), dtype=bool)
            else:  # true window labels; quarantine (-1) excluded from the fit
                y_rows = (wl_train == 1).astype(np.int64)
                keep = wl_train != -1
            scores = fit_readout_scores(readout, x_train_all[keep], y_rows[keep], x_test_all)

            star_scores = np.zeros(len(test_blocks), dtype=np.float64)
            for i in range(len(test_blocks)):  # max over ALL of the star's windows (protocol-matched)
                star_scores[i] = scores[test_star_of_window == i].max()

            eval_mask = wl_test != -1  # quarantine excluded from window-level eval
            y_win = (wl_test == 1).astype(np.int64)
            within = eval_mask & np.isin(test_star_of_window, np.flatnonzero(star_y["test_v1"] == 1))
            rows.append({
                "readout": readout, "label_mode": mode,
                "star_pr_auc_v1": float(average_precision_score(star_y["test_v1"], star_scores)),
                "star_pr_auc_v2": float(average_precision_score(star_y["test_v2"], star_scores)),
                "win_pr_auc_full": float(average_precision_score(y_win[eval_mask], scores[eval_mask])),
                "win_pr_auc_within": float(average_precision_score(y_win[within], scores[within])),
                "n_train_win_kept": int(keep.sum()), "n_train_win_pos": int(y_rows[keep].sum()),
                "n_test_win": int(eval_mask.sum()), "n_test_win_pos": int(y_win[eval_mask].sum()),
                "n_test_stars_pos_v1": int(star_y["test_v1"].sum()), "n_test_stars_pos_v2": int(star_y["test_v2"].sum()),
            })
            log.info(f"{label} {readout}/{mode}: star_v1 {rows[-1]['star_pr_auc_v1']:.4f} "
                     f"win_full {rows[-1]['win_pr_auc_full']:.4f} win_within {rows[-1]['win_pr_auc_within']:.4f}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="window-level transit eval on cached first-segment mu")
    parser.add_argument("--exp-glob", default="exp03_*", help="glob under experiments/ selecting experiment folders")
    parser.add_argument("--ckpts", nargs="+", default=["best_recon_aux"], help="checkpoint stems to score")
    parser.add_argument("--readouts", nargs="+", default=list(readouts_default))
    parser.add_argument("--variant", default="B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels-parquet", default="labels/qc/transit_window_labels_w256.parquet")
    parser.add_argument("--untrained-cache", default="experiments/exp03_eval_cache")
    args = parser.parse_args()

    packed_dir = repo_root / "experiments" / "exp01_window256_seq16" / "packed"
    window_labels = pd.read_parquet(repo_root / args.labels_parquet)
    subset = pd.read_parquet(repo_root / "processed" / "subset" / "subset_tics.parquet")
    v2 = pd.read_csv(repo_root / "labels" / "variability_labels_star_v2.csv")
    v2_transit = dict(zip(v2["tic_id"].astype(int), pd.to_numeric(v2["transit"], errors="coerce").fillna(0).astype(int)))
    v1_transit = dict(zip(subset["tic_id"].astype(int), subset["transit"].astype(int)))

    # per-split alignment frames + label blocks are arm-independent: build once
    firsts, wlabels, star_y = {}, {}, {}
    for split in ["train", "test"]:
        first = first_segment_index(packed_dir, split)
        firsts[split] = first
        wlabels[split] = window_label_blocks(first, window_labels[window_labels["split"] == split])
        tics = first["tic_id"].astype(int).tolist()
        star_y[f"{split}_v1"] = np.array([v1_transit[t] for t in tics], dtype=np.int64)
        missing = [t for t in tics if t not in v2_transit]
        assert not missing, f"{len(missing)} packed tics missing from v2 labels (first: {missing[:5]})"
        star_y[f"{split}_v2"] = np.array([v2_transit[t] for t in tics], dtype=np.int64)

    arms = [("untrained_w256", "-", repo_root / args.untrained_cache / "untrained_mu_w256.npz")]
    for exp_dir in sorted((repo_root / "experiments").glob(args.exp_glob)):
        run_dir = exp_dir / "models" / f"{args.variant}_seed{args.seed}"
        for stem in args.ckpts:
            cache = run_dir / "extracted" / f"first_segment_window_mu_{stem}.npz"
            if cache.exists():
                arms.append((exp_dir.name, stem, cache))
            elif run_dir.exists():
                log.warning(f"{exp_dir.name}: no mu cache for {stem}; skipped (cell dropped, not silent)")
    log.info(f"{len(arms)} arms x {len(args.readouts)} readouts x {list(label_modes)} label modes")

    out_rows = []
    for exp_name, stem, cache in tqdm(arms, desc="arms", total=len(arms)):
        mu = cached_mu(cache, None, packed_dir, 256, "cpu", exp_name)  # cache exists: model/device unused
        for split in ["train", "test"]:
            cache_tics = [int(t) for t in mu[split][0]]
            index_tics = firsts[split]["tic_id"].astype(int).tolist()
            assert cache_tics == index_tics, f"{exp_name} [{split}]: cache tic order != index tic order"
            for block, wl in zip(mu[split][1], wlabels[split]):
                assert block.shape[0] == wl.shape[0], f"{exp_name} [{split}]: block/label length mismatch"
        cells = score_arm(mu, wlabels, star_y, tuple(args.readouts), exp_name)
        cells.insert(0, "exp_name", exp_name)
        cells.insert(1, "ckpt", stem)
        out_rows.append(cells)

    result = pd.concat(out_rows, ignore_index=True)
    result["run_id"] = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    result["git_sha"] = _git_sha()
    out_path = repo_root / "experiments" / "exp03_transit_window_eval.csv"
    if out_path.exists():
        result = pd.concat([pd.read_csv(out_path), result], ignore_index=True)  # append-only audit trail
    result.to_csv(out_path, index=False)
    log.info(f"wrote {out_path}")

    latest = result[result["run_id"] == result["run_id"].iloc[-1]]
    summary = latest.pivot_table(index=["exp_name", "ckpt"], columns=["readout", "label_mode"],
                                 values="win_pr_auc_full")
    log.info(f"win_pr_auc_full summary:\n{summary.round(4)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
