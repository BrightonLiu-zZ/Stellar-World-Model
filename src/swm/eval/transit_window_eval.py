"""Window-level transit eval: broadcast vs true (loose) vs true KP+CP-strict labels on cached mu.

Phase 1 (docs/plans/2026-07-16-phase1-window-level-transit-labels.md, D3/D4/D5) asked whether label
noise or dilution limits star-level transit; exp04 (grill 2026-07-20) adds the KP+CP-strict knob from
the label-sanity notebook (src/notebooks/transit_window_label_sanity.ipynb, cell dbc77fa3): positive =
a window that FULLY contains a transit of a CP/KP TOI (217 stars corpus-wide, 92% show a fold dip).
Consumes the first-segment window-mu caches readout_sweep already wrote (no encoder pass, no GPU) and
scores, per (experiment x seed x checkpoint x readout), three MIL fit modes on the shared protocol:

  broadcast  - the readout_sweep window_score protocol: v1 star label on every window (continuity)
  true       - loose window labels (any in-transit cadence, any {CP,KP,PC,APC} TOI), quarantine
               excluded from the fit
  true_kpcp  - strict window labels derived in-memory from the annot parquet (DISP {CP,KP},
               INCLUDE_PARTIAL off — the notebook's derive_label verbatim), quarantine excluded

Every fit mode is cross-scored on BOTH window-label definitions (loose and strict), isolating
fit-label effects from eval-label effects, and on three star-label sets:

  star_pr_auc_v1    - canonical v1 star labels (matches readout_sweep)
  star_pr_auc_v2    - ADR-0009 coverage-filtered star labels
  star_pr_auc_kpcp  - positive = star with >=1 strict-positive window; v1-transit stars OUTSIDE the
                      strict set are excluded from the metric (star-level quarantine, grill Q2
                      option 1: they do transit, so neither class may claim them)

Label semantics per definition: 1 = in-transit, 0 = clean negative, -1 = quarantine (dropped from
every fit and from the window-level metrics; under the strict definition a loose-positive that is not
KP+CP-full is quarantine, never a negative). Star-level max-pooling keeps ALL windows so the protocol
stays exactly comparable to the readout_sweep window_score rows.

Rows append to experiments/exp04_transit_window_eval.csv with run_id + git_sha (append-only). The
exp03 CSV is frozen — this schema adds columns (seed, strict metrics) the old file never had.

Run (from repo root, swm env, PYTHONPATH=src; CPU-only, safe alongside anything):
    python -m swm.eval.transit_window_eval --exp-globs "exp04_fb0p0*" exp04_winner_ep100 --seeds 0 1
    python -m swm.eval.transit_window_eval --exp-globs exp04_enc_z64 --seeds 0 1 2 \
        --untrained-cache experiments/exp04_eval_cache/enc_z64
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
label_modes = ("broadcast", "true", "true_kpcp")

STRICT_DISP = {"CP", "KP"}  # the KP+CP knob: confirmed/known planets only, full containment only


def derive_strict_labels(annot: pd.DataFrame) -> pd.Series:
    """Per-window 1/0/-1 label under the KP+CP-strict knob — the notebook's derive_label with
    DISP_SET={CP,KP}, INCLUDE_PARTIAL=False baked in. Any overlap that is not a strict positive
    (wrong disposition, partial clip, near buffer, unfoldable star) is quarantine, not negative."""
    pos = annot["full_best"].isin(STRICT_DISP)
    overlap = (
        (annot["full_best"] != "") | (annot["part_best"] != "") | (annot["near_best"] != "")
        | annot["unfoldable"]
    )
    return pd.Series(np.where(pos, 1, np.where(overlap, -1, 0)).astype(np.int8), index=annot.index)


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


def score_arm(mu: dict, wlabels: dict, wlabels_strict: dict, star_y: dict,
              readouts: tuple[str, ...], label: str) -> pd.DataFrame:
    """
    Score one encoder arm: every (readout x fit mode) cell on the shared first-segment protocol,
    cross-scored on both window-label definitions and all three star-label sets.
    mu / wlabels / wlabels_strict hold per-star window blocks per split; star_y holds the star-label
    vectors plus the kpcp keep-mask (v1-transit stars outside the strict set excluded).
    """
    train_tics, train_blocks = mu["train"]
    test_tics, test_blocks = mu["test"]
    x_train_all = np.concatenate(train_blocks, axis=0)  # (n_train_win, z)
    x_test_all = np.concatenate(test_blocks, axis=0)    # (n_test_win, z)
    wl_train = np.concatenate(wlabels["train"])         # (n_train_win,) loose labels in {1, 0, -1}
    wl_test = np.concatenate(wlabels["test"])           # (n_test_win,)
    ws_train = np.concatenate(wlabels_strict["train"])  # (n_train_win,) strict labels in {1, 0, -1}
    ws_test = np.concatenate(wlabels_strict["test"])    # (n_test_win,)
    test_star_of_window = np.repeat(np.arange(len(test_blocks)), [b.shape[0] for b in test_blocks])

    # window-level eval masks are fit-independent: build once per definition
    eval_loose = wl_test != -1
    eval_strict = ws_test != -1
    y_loose = (wl_test == 1).astype(np.int64)
    y_strict = (ws_test == 1).astype(np.int64)
    within_loose = eval_loose & np.isin(test_star_of_window, np.flatnonzero(star_y["test_v1"] == 1))
    within_strict = eval_strict & np.isin(test_star_of_window, np.flatnonzero(star_y["test_kpcp"] == 1))
    keep_star = star_y["test_kpcp_keep"]  # star-level quarantine: drop v1-transit stars outside the strict set

    rows = []
    for readout in readouts:
        for mode in label_modes:
            if mode == "broadcast":  # the readout_sweep window_score protocol: star label on every window
                y_rows = np.concatenate(
                    [np.full(b.shape[0], star_y["train_v1"][i], dtype=np.int64) for i, b in enumerate(train_blocks)]
                )
                keep = np.ones(len(y_rows), dtype=bool)
            elif mode == "true":  # loose window labels; quarantine (-1) excluded from the fit
                y_rows = (wl_train == 1).astype(np.int64)
                keep = wl_train != -1
            else:  # true_kpcp: strict window labels; quarantine excluded from the fit
                y_rows = (ws_train == 1).astype(np.int64)
                keep = ws_train != -1
            scores = fit_readout_scores(readout, x_train_all[keep], y_rows[keep], x_test_all)

            star_scores = np.zeros(len(test_blocks), dtype=np.float64)
            for i in range(len(test_blocks)):  # max over ALL of the star's windows (protocol-matched)
                star_scores[i] = scores[test_star_of_window == i].max()

            rows.append({
                "readout": readout, "label_mode": mode,
                "star_pr_auc_v1": float(average_precision_score(star_y["test_v1"], star_scores)),
                "star_pr_auc_v2": float(average_precision_score(star_y["test_v2"], star_scores)),
                "star_pr_auc_kpcp": float(
                    average_precision_score(star_y["test_kpcp"][keep_star], star_scores[keep_star])
                ),
                "win_pr_auc_full": float(average_precision_score(y_loose[eval_loose], scores[eval_loose])),
                "win_pr_auc_within": float(average_precision_score(y_loose[within_loose], scores[within_loose])),
                "win_pr_auc_full_kpcp": float(average_precision_score(y_strict[eval_strict], scores[eval_strict])),
                "win_pr_auc_within_kpcp": float(
                    average_precision_score(y_strict[within_strict], scores[within_strict])
                ),
                "n_train_win_kept": int(keep.sum()), "n_train_win_pos": int(y_rows[keep].sum()),
                "n_test_win": int(eval_loose.sum()), "n_test_win_pos": int(y_loose[eval_loose].sum()),
                "n_test_win_kpcp": int(eval_strict.sum()), "n_test_win_pos_kpcp": int(y_strict[eval_strict].sum()),
                "n_test_stars_pos_v1": int(star_y["test_v1"].sum()), "n_test_stars_pos_v2": int(star_y["test_v2"].sum()),
                "n_test_stars_pos_kpcp": int(star_y["test_kpcp"].sum()),
                "n_test_stars_excl_kpcp": int((~keep_star).sum()),
            })
            log.info(f"{label} {readout}/{mode}: star_v1 {rows[-1]['star_pr_auc_v1']:.4f} "
                     f"star_kpcp {rows[-1]['star_pr_auc_kpcp']:.4f} "
                     f"win_full {rows[-1]['win_pr_auc_full']:.4f} win_kpcp {rows[-1]['win_pr_auc_full_kpcp']:.4f}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="window-level transit eval on cached first-segment mu")
    parser.add_argument("--exp-globs", nargs="+", default=["exp04_*"],
                        help="globs under experiments/ selecting experiment folders")
    parser.add_argument("--ckpts", nargs="+", default=["best_recon_aux"], help="checkpoint stems to score")
    parser.add_argument("--readouts", nargs="+", default=list(readouts_default))
    parser.add_argument("--variant", default="B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="B_seed<N> run dirs to score")
    parser.add_argument("--labels-parquet", default="labels/qc/transit_window_labels_w256.parquet",
                        help="baked loose 1/0/-1 label table (any-cadence, full whitelist)")
    parser.add_argument("--annot-parquet", default="labels/qc/transit_window_labels_w256_annot.parquet",
                        help="rich per-window annotation the strict labels are derived from in-memory")
    parser.add_argument("--untrained-cache", default="experiments/exp03_eval_cache",
                        help="dir holding untrained_mu_w256.npz; MUST be capacity-matched to the "
                             "experiments scored (per-encoder-variant dirs for exp04 enc_* runs)")
    parser.add_argument("--skip-untrained", action="store_true",
                        help="score only the trained arms (untrained row already in the CSV)")
    parser.add_argument("--out", default="experiments/exp04_transit_window_eval.csv")
    args = parser.parse_args()

    packed_dir = repo_root / "experiments" / "exp01_window256_seq16" / "packed"
    window_labels = pd.read_parquet(repo_root / args.labels_parquet)
    annot = pd.read_parquet(repo_root / args.annot_parquet)
    strict = annot[["split", "row", "tic_id"]].copy()
    strict["label"] = derive_strict_labels(annot)
    strict_star_tics = set(strict.loc[strict["label"] == 1, "tic_id"].astype(int))
    log.info(f"strict KP+CP knob: {int((strict['label'] == 1).sum())} positive windows over "
             f"{len(strict_star_tics)} stars (notebook cell dbc77fa3 reference: 1751 / 217)")

    subset = pd.read_parquet(repo_root / "processed" / "subset" / "subset_tics.parquet")
    v2 = pd.read_csv(repo_root / "labels" / "variability_labels_star_v2.csv")
    v2_transit = dict(zip(v2["tic_id"].astype(int), pd.to_numeric(v2["transit"], errors="coerce").fillna(0).astype(int)))
    v1_transit = dict(zip(subset["tic_id"].astype(int), subset["transit"].astype(int)))

    # per-split alignment frames + label blocks are arm-independent: build once
    firsts, wlabels, wlabels_strict, star_y = {}, {}, {}, {}
    for split in ["train", "test"]:
        first = first_segment_index(packed_dir, split)
        firsts[split] = first
        wlabels[split] = window_label_blocks(first, window_labels[window_labels["split"] == split])
        wlabels_strict[split] = window_label_blocks(first, strict[strict["split"] == split])
        tics = first["tic_id"].astype(int).tolist()
        star_y[f"{split}_v1"] = np.array([v1_transit[t] for t in tics], dtype=np.int64)
        missing = [t for t in tics if t not in v2_transit]
        assert not missing, f"{len(missing)} packed tics missing from v2 labels (first: {missing[:5]})"
        star_y[f"{split}_v2"] = np.array([v2_transit[t] for t in tics], dtype=np.int64)
        star_y[f"{split}_kpcp"] = np.array([1 if t in strict_star_tics else 0 for t in tics], dtype=np.int64)
        # star-level quarantine: a v1-transit star outside the strict set is neither class (grill Q2)
        star_y[f"{split}_kpcp_keep"] = ~((star_y[f"{split}_v1"] == 1) & (star_y[f"{split}_kpcp"] == 0))
    log.info(f"test stars: v1_pos {int(star_y['test_v1'].sum())}, kpcp_pos {int(star_y['test_kpcp'].sum())}, "
             f"kpcp_excluded {int((~star_y['test_kpcp_keep']).sum())}")

    arms = []
    if not args.skip_untrained:
        # name the untrained arm after its cache dir: per-encoder-variant caches must not collide
        untrained_name = f"untrained_w256[{Path(args.untrained_cache).name}]"
        arms.append((untrained_name, "-", 0, repo_root / args.untrained_cache / "untrained_mu_w256.npz"))
    for glob in args.exp_globs:
        for exp_dir in sorted((repo_root / "experiments").glob(glob)):
            for seed in args.seeds:
                run_dir = exp_dir / "models" / f"{args.variant}_seed{seed}"
                if not run_dir.exists():
                    continue  # seed not trained for this experiment (ranked manifest, not a failure)
                for stem in args.ckpts:
                    cache = run_dir / "extracted" / f"first_segment_window_mu_{stem}.npz"
                    if cache.exists():
                        arms.append((exp_dir.name, stem, seed, cache))
                    else:
                        log.warning(f"{exp_dir.name} seed{seed}: no mu cache for {stem}; skipped (cell dropped, not silent)")
    log.info(f"{len(arms)} arms x {len(args.readouts)} readouts x {list(label_modes)} fit modes")

    out_rows = []
    for exp_name, stem, seed, cache in tqdm(arms, desc="arms", total=len(arms)):
        mu = cached_mu(cache, None, packed_dir, 256, "cpu", exp_name)  # cache exists: model/device unused
        for split in ["train", "test"]:
            cache_tics = [int(t) for t in mu[split][0]]
            index_tics = firsts[split]["tic_id"].astype(int).tolist()
            assert cache_tics == index_tics, f"{exp_name} [{split}]: cache tic order != index tic order"
            for block, wl in zip(mu[split][1], wlabels[split]):
                assert block.shape[0] == wl.shape[0], f"{exp_name} [{split}]: block/label length mismatch"
        cells = score_arm(mu, wlabels, wlabels_strict, star_y, tuple(args.readouts), f"{exp_name} seed{seed}")
        cells.insert(0, "exp_name", exp_name)
        cells.insert(1, "ckpt", stem)
        cells.insert(2, "seed", seed)
        out_rows.append(cells)

    result = pd.concat(out_rows, ignore_index=True)
    result["run_id"] = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    result["git_sha"] = _git_sha()
    out_path = repo_root / args.out
    if out_path.exists():
        result = pd.concat([pd.read_csv(out_path), result], ignore_index=True)  # append-only audit trail
    result.to_csv(out_path, index=False)
    log.info(f"wrote {out_path}")

    latest = result[result["run_id"] == result["run_id"].iloc[-1]]
    summary = latest.pivot_table(index=["exp_name", "seed"], columns=["readout", "label_mode"],
                                 values="star_pr_auc_kpcp")
    log.info(f"star_pr_auc_kpcp summary:\n{summary.round(4)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
