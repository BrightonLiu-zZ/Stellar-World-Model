"""Skyline ceiling suite (plan 2026-07-11): measure the headroom above the current 0.77 PR-AUC.

The frozen-encoder probe barely beats an untrained encoder, and exp01 (window 1024 -> 256) fixed the
reconstruction mechanism without closing that gap. Before spending training compute on an objective
change (exp02) we measure two distinct ceilings with a cheap, no-training-required diagnostic, all on
the identical first-segment protocol the exp01 gap table uses:

  Ceiling A (protocol-matched, could a better representation beat 0.77 under the same linear probe?)
    A1: logistic regression on T'DA-informed engineered features (swm.eval.features).
    A2: gradient-boosted trees on the same features, separating "features lack info" from "a linear
        probe cannot use it".
  Ceiling B (task ceiling, how much signal do data+labels contain at all?)
    B1: the same Conv1D encoder trunk (random init) + mean-pool + linear head, trained end-to-end
        supervised with class-weighted BCE, early-stopped on val PR-AUC, 3 seeds. Diagnostic only,
        never a v1 deliverable (the architecture-freeze rule governs the v1 model, not diagnostics).

  Disambiguating diagnostic (added 2026-07-11): a nonlinear GBM probe on the encoder mu (trained and
  untrained). If it beats the linear probe on the trained mu (info_in_mu=True) the signal is in mu but
  not linearly readable -> probe/pooling lever (Branch beta). If it does not (info_in_mu=False) the mu
  lacks the signal -> objective-change lever (Branch alpha). GBM on the untrained mu is the reference:
  when it exceeds GBM on the trained mu, SSL training is degrading mu below a random projection.

Every number carries a paired star-level bootstrap CI. The gate (decision 6) declares linear headroom
real iff A1 - untrained > 2 * SE_diff from a paired bootstrap of that difference. Results append to
experiments/<exp>/results/skyline_results.csv with a run_id + git_sha so re-runs are auditable.

Run (from repo src/ on PYTHONPATH, in the swm env):
    python -m swm.eval.skyline +experiment=exp01_window256_seq16
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from swm.eval.features import FEATURE_NAMES, extract_features
from swm.models import WorldModel
from swm.models.encoder import Encoder

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
TASKS_DEFAULT = ("pulsating", "eb", "rotation", "transit") # exp02 gates all four; transit is report-only
B1_SEEDS_DEFAULT = (0, 1, 2)


# ----------------------------------------------------------------------------------------------------
# First-segment loading (replicates the exp01 gap-table pipeline: one segment per star, train + test)
# ----------------------------------------------------------------------------------------------------
def load_first_segment_blocks(packed_dir: Path, split: str, window: int) -> tuple[list[int], list[np.ndarray]]:
    """
    Return each star's first-segment window block for one split, in ascending tic order.
    Mirrors the exp01 diagnostic: sort segments by (tic, sector, seg_idx), keep the first per star,
    and slice its contiguous window rows from the packed memmap. The stable tic ordering lets every
    downstream method (encoder mu, engineered features, B1) emit test scores in one aligned order.
    """
    index = pd.read_parquet(packed_dir / f"{split}_index.parquet")
    total = int(index["n_win"].sum())
    windows = np.memmap(packed_dir / f"{split}_windows.dat", dtype=np.float32, mode="r", shape=(total, window))
    first = index.sort_values(["tic_id", "sector", "seg_idx"]).drop_duplicates("tic_id").sort_values("tic_id")
    tics = []
    blocks = []
    for row in first.itertuples(index=False):
        block = np.array(windows[row.row_start : row.row_start + row.n_win], dtype=np.float32) # (n_win, window)
        tics.append(int(row.tic_id))
        blocks.append(block)
    return tics, blocks


def _attach_labels(tics: list[int], split: str, subset: pd.DataFrame, tasks: tuple[str, ...]) -> pd.DataFrame:
    """Build a (tic_id, split, task-label) frame in the given tic order, inner-joining the subset labels."""
    frame = pd.DataFrame({"tic_id": tics})
    frame["split"] = split
    labelled = frame.merge(subset[["tic_id", *tasks]], on="tic_id", how="inner") # every packed tic is in subset
    assert len(labelled) == len(frame), "a packed first-segment star is missing from the subset labels"
    return labelled


def feature_table(packed_dir: Path, window: int, subset: pd.DataFrame, tasks: tuple[str, ...]) -> pd.DataFrame:
    """
    Build the Ceiling-A input: one engineered feature vector per star (train + test first segments).
    Concatenates a star's first-segment windows into a contiguous flux series and reduces it with
    swm.eval.features.extract_features, then attaches the binary task labels. Rows follow the aligned
    ascending-tic order per split.
    """
    parts = []
    for split in ["train", "test"]:
        tics, blocks = load_first_segment_blocks(packed_dir, split, window)
        feats = np.zeros((len(tics), len(FEATURE_NAMES)), dtype=np.float64)
        for i in tqdm(range(len(blocks)), desc=f"features[{split}]", total=len(blocks)):
            feats[i] = extract_features(blocks[i].reshape(-1)) # first-segment concatenated flux
        labelled = _attach_labels(tics, split, subset, tasks)
        feat_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
        parts.append(pd.concat([labelled.reset_index(drop=True), feat_df], axis=1))
    return pd.concat(parts, ignore_index=True)


def encoder_mu_table(
    model: WorldModel, packed_dir: Path, window: int, subset: pd.DataFrame, tasks: tuple[str, ...],
    mu_cols: list[str], device: str,
) -> pd.DataFrame:
    """
    Build a per-star mean-posterior-mu table over first segments (train + test), matching the probe.
    Encodes each star's first-segment windows and mean-pools mu over them, exactly as the exp01 gap
    table does, so the trained/untrained numbers here reproduce that table.
    """
    parts = []
    for split in ["train", "test"]:
        tics, blocks = load_first_segment_blocks(packed_dir, split, window)
        mu_star = np.zeros((len(tics), len(mu_cols)), dtype=np.float32)
        for i in tqdm(range(len(blocks)), desc=f"encode-mu[{split}]", total=len(blocks)):
            x = torch.from_numpy(blocks[i]).unsqueeze(-1).to(device) # (n_win, window, 1)
            with torch.no_grad():
                mu, _ = model.encoder(x) # (n_win, z)
            mu_star[i] = mu.mean(0).cpu().numpy() # pool over the star's first-segment windows
        labelled = _attach_labels(tics, split, subset, tasks)
        mu_df = pd.DataFrame(mu_star, columns=mu_cols)
        parts.append(pd.concat([labelled.reset_index(drop=True), mu_df], axis=1))
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------------------------------
# Protocol-matched scorers: fit on train first segments, return aligned test-star (y, score) pairs
# ----------------------------------------------------------------------------------------------------
def logistic_scores(table: pd.DataFrame, cols: list[str], task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the protocol-matched logistic probe (standardize on train, balanced class weights) and score test.
    Returns (test tics, test labels, predicted P(positive)) aligned to the table's ascending-tic order.
    Used for the trained encoder, untrained encoder, and the A1 engineered-feature ceiling.
    """
    train = table[table["split"] == "train"]
    test = table[table["split"] == "test"]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[cols].to_numpy()) # learn mean/std on train only (no leakage)
    x_test = scaler.transform(test[cols].to_numpy())
    clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(x_train, train[task].to_numpy())
    scores = clf.predict_proba(x_test)[:, 1]
    return test["tic_id"].to_numpy(), test[task].to_numpy(), scores


def forest_scores(table: pd.DataFrame, cols: list[str], task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the A2 gradient-boosted-tree ceiling on the same engineered features and score test.
    A strong nonlinear model on identical inputs separates "the features lack the information" from
    "a linear probe cannot exploit it". No standardization (trees are scale-invariant).
    """
    train = table[table["split"] == "train"]
    test = table[table["split"] == "test"]
    clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=0) # gradient-boosted trees
    clf.fit(train[cols].to_numpy(), train[task].to_numpy())
    scores = clf.predict_proba(test[cols].to_numpy())[:, 1]
    return test["tic_id"].to_numpy(), test[task].to_numpy(), scores


# ----------------------------------------------------------------------------------------------------
# Ceiling B1: supervised conv-trunk skyline (same encoder architecture, random init, trained end-to-end)
# ----------------------------------------------------------------------------------------------------
class SupervisedTrunk(nn.Module):
    """
    Ceiling-B diagnostic: the locked Conv1D encoder trunk + mean-pool + linear head, trained supervised.
    Identical encoder layers/channels to the v1 world model but random-initialised and optimised
    end-to-end with labels, so its test PR-AUC upper-bounds what the architecture can extract from the
    first-segment data. This is a diagnostic model only, never a v1 deliverable.
    """

    def __init__(self, enc_channels: list[int], kernel_size: int, z_dim: int, window: int) -> None:
        super().__init__()
        self.encoder = Encoder(1, enc_channels, kernel_size, z_dim, window) # same trunk as the v1 world model
        self.head = nn.Linear(z_dim, 1)

    def forward(self, windows: torch.Tensor, seg_sizes: list[int]) -> torch.Tensor:
        # windows: (R, window, 1) = a minibatch of stars' first-segment windows concatenated
        mu, _ = self.encoder(windows) # (R, z)
        pooled = []
        start = 0
        for n in seg_sizes:
            pooled.append(mu[start : start + n].mean(0)) # mean-pool a star's windows
            start += n
        pooled = torch.stack(pooled, dim=0) # (n_stars, z)
        return self.head(pooled).squeeze(-1) # (n_stars,)


def _b1_batches(n_stars: int, batch_stars: int, rng: np.random.Generator, shuffle: bool) -> list[np.ndarray]:
    """Yield minibatches of star indices, optionally shuffled, for one B1 epoch."""
    order = np.arange(n_stars)
    if shuffle:
        rng.shuffle(order)
    batches = []
    for start in range(0, n_stars, batch_stars):
        batches.append(order[start : start + batch_stars])
    return batches


@torch.no_grad()
def _b1_predict(model: SupervisedTrunk, blocks: list[np.ndarray], batch_stars: int, device: str) -> np.ndarray:
    """Score every star's first segment with the B1 head, returning P(positive) in block order."""
    model.eval()
    scores = np.zeros(len(blocks), dtype=np.float64)
    for start in range(0, len(blocks), batch_stars):
        idx = list(range(start, min(start + batch_stars, len(blocks))))
        windows = np.concatenate([blocks[i] for i in idx], axis=0) # (sum n_win, window)
        seg_sizes = [blocks[i].shape[0] for i in idx]
        x = torch.from_numpy(windows).unsqueeze(-1).to(device)
        logits = model(x, seg_sizes)
        scores[start : start + len(idx)] = torch.sigmoid(logits).cpu().numpy()
    return scores


def train_b1_one_seed(
    train_blocks: list[np.ndarray], y_train: np.ndarray, val_blocks: list[np.ndarray], y_val: np.ndarray,
    enc_channels: list[int], kernel_size: int, z_dim: int, window: int, device: str, seed: int,
    max_epochs: int = 60, patience: int = 10, batch_stars: int = 64, lr: float = 3e-4,
) -> tuple[SupervisedTrunk, float]:
    """
    Train one B1 seed end-to-end with class-weighted BCE, early-stopping on val PR-AUC.
    Returns the best-val model and its val PR-AUC. Class imbalance is handled by pos_weight = neg/pos
    on the train split; only post-fit the caller scores the held-out test set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = SupervisedTrunk(enc_channels, kernel_size, z_dim, window).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device) # upweight rare positives
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)

    best_val = -1.0
    best_state = None
    patience_ctr = 0
    epoch_bar = tqdm(range(max_epochs), desc=f"B1 seed{seed}", total=max_epochs) # early-stops before max
    for epoch in epoch_bar:
        model.train()
        for batch in _b1_batches(len(train_blocks), batch_stars, rng, shuffle=True):
            windows = np.concatenate([train_blocks[i] for i in batch], axis=0)
            seg_sizes = [train_blocks[i].shape[0] for i in batch]
            x = torch.from_numpy(windows).unsqueeze(-1).to(device)
            logits = model(x, seg_sizes)
            loss = loss_fn(logits, y_train_t[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_scores = _b1_predict(model, val_blocks, batch_stars, device)
        val_ap = float(average_precision_score(y_val, val_scores))
        if val_ap > best_val:
            best_val = val_ap
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        epoch_bar.set_postfix(val_ap=round(val_ap, 4), best=round(best_val, 4), patience=patience_ctr)
        if patience_ctr >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_val


def b1_scores(
    packed_dir: Path, window: int, subset: pd.DataFrame, task: str, enc_channels: list[int],
    kernel_size: int, z_dim: int, device: str, seeds: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Train the B1 supervised trunk over `seeds` and return the seed-ensembled test scores.
    Averaging predicted probabilities across seeds gives one aligned score vector per test star for
    the paired bootstrap, while the per-seed test PR-AUCs are returned for transparency. Train/val/test
    are the same packed first-segment splits used everywhere else.
    """
    train_tics, train_blocks = load_first_segment_blocks(packed_dir, "train", window)
    val_tics, val_blocks = load_first_segment_blocks(packed_dir, "val", window)
    test_tics, test_blocks = load_first_segment_blocks(packed_dir, "test", window)
    label_of = dict(zip(subset["tic_id"].tolist(), subset[task].tolist()))
    y_train = np.array([label_of[t] for t in train_tics], dtype=np.int64)
    y_val = np.array([label_of[t] for t in val_tics], dtype=np.int64)
    y_test = np.array([label_of[t] for t in test_tics], dtype=np.int64)

    test_score_sum = np.zeros(len(test_blocks), dtype=np.float64)
    per_seed_ap = []
    for seed in seeds:
        model, val_ap = train_b1_one_seed(
            train_blocks, y_train, val_blocks, y_val, enc_channels, kernel_size, z_dim, window, device, seed
        )
        seed_scores = _b1_predict(model, test_blocks, 64, device)
        per_seed_ap.append(float(average_precision_score(y_test, seed_scores)))
        test_score_sum += seed_scores
        log.info(f"[B1 {task} seed{seed}] val PR-AUC {val_ap} test PR-AUC {per_seed_ap[-1]}")
    ensemble_scores = test_score_sum / len(seeds) # seed-ensemble P(positive)
    return np.array(test_tics), y_test, ensemble_scores, per_seed_ap


# ----------------------------------------------------------------------------------------------------
# Paired star-level bootstrap
# ----------------------------------------------------------------------------------------------------
def paired_bootstrap_ap(
    y_true: np.ndarray, scores_by_method: dict[str, np.ndarray], n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """
    Resample test stars with replacement and recompute PR-AUC for every method on each resample.
    All methods share the SAME resampled star indices per replicate, so any pairwise column
    difference (e.g. A1 minus untrained) is a paired difference and its std is the paired SE_diff the
    gate needs. Degenerate resamples (all one class) are skipped. Returns one row per valid replicate,
    one column per method.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    methods = list(scores_by_method.keys())
    records = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n) # star indices sampled with replacement
        y_boot = y_true[idx]
        if y_boot.sum() == 0 or y_boot.sum() == n: # PR-AUC undefined without both classes
            continue
        row = {}
        for method in methods:
            row[method] = average_precision_score(y_boot, scores_by_method[method][idx])
        records.append(row)
    return pd.DataFrame(records)


def _git_sha() -> str:
    """Return the short git SHA of the working tree, or 'nogit' if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "nogit"


def append_results(results_path: Path, rows: pd.DataFrame) -> None:
    """Append this run's skyline rows (stamped with run_id + git_sha) to the audit CSV, never overwriting."""
    stamped = rows.copy()
    stamped["run_id"] = datetime.now().strftime("%Y%m%dT%H%M%S")
    stamped["git_sha"] = _git_sha()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        previous = pd.read_csv(results_path)
        combined = pd.concat([previous, stamped], ignore_index=True)
    else:
        combined = stamped
    combined.to_csv(results_path, index=False)
    log.info(f"appended {len(stamped)} rows to {results_path}")


# ----------------------------------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------------------------------
def _make_untrained(enc_channels: list[int], kernel_size: int, z_dim: int, window: int,
                    gru_hidden: int, gru_layers: int, device: str) -> WorldModel:
    """Random-init world model with fixed seed 0, matching the exp01 gap-table untrained reference."""
    torch.manual_seed(0) # fixed init -> reproducible untrained baseline (reproduces gap_table untrained)
    model = WorldModel(
        in_ch=1, enc_channels=enc_channels, kernel_size=kernel_size, z_dim=z_dim,
        window=window, gru_hidden=gru_hidden, gru_layers=gru_layers,
    ).to(device)
    model.eval()
    return model


def run_suite(
    exp_name: str, tasks: tuple[str, ...] = TASKS_DEFAULT, b1_seeds: tuple[int, ...] = B1_SEEDS_DEFAULT,
    variant: str = "B", seed: int = 0, n_boot: int = 2000, device: str | None = None,
    root: Path = REPO_ROOT, write: bool = True, ckpt_stem: str = "best",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full skyline suite at one experiment config and return (summary, gate) frames.
    For each task it scores five methods (trained, untrained, A1 logistic, A2 GBM, B1 supervised) on
    the aligned test stars, attaches paired star-bootstrap CIs to every PR-AUC, applies the decision-6
    gate (A1 - untrained > 2 * SE_diff), and reports the supervised-gap fraction (trained - untrained)
    / (B1 - untrained). Also records an A1 label-shuffle control (should collapse to the base rate).
    Appends the per-method rows to the append-only results CSV when `write` is set.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = root / "experiments" / exp_name
    packed_dir = exp_dir / "packed"
    subset = pd.read_parquet(root / "processed" / "subset" / "subset_tics.parquet")

    ckpt = torch.load(
        exp_dir / "models" / f"{variant}_seed{seed}" / f"{ckpt_stem}.pt", map_location=device, weights_only=False
    )
    cfg = ckpt["cfg"]
    window = int(cfg["data"]["window"])
    mc = cfg["model"]
    enc_channels = list(mc["enc_channels"])
    kernel_size = int(mc["kernel_size"])
    z_dim = int(mc["z_dim"])
    mu_cols = [f"mu{j}" for j in range(z_dim)]

    trained = WorldModel(
        in_ch=1, enc_channels=enc_channels, kernel_size=kernel_size, z_dim=z_dim, window=window,
        gru_hidden=int(mc["gru_hidden"]), gru_layers=int(mc["gru_layers"]),
    ).to(device)
    trained.load_state_dict(ckpt["model"])
    trained.eval()
    untrained = _make_untrained(
        enc_channels, kernel_size, z_dim, window, int(mc["gru_hidden"]), int(mc["gru_layers"]), device
    )

    feats = feature_table(packed_dir, window, subset, tasks)
    mu_trained = encoder_mu_table(trained, packed_dir, window, subset, tasks, mu_cols, device)
    mu_untrained = encoder_mu_table(untrained, packed_dir, window, subset, tasks, mu_cols, device)

    summary_rows = []
    gate_rows = []
    for task in tasks:
        tics_tr, y, s_trained = logistic_scores(mu_trained, mu_cols, task)
        tics_un, _, s_untrained = logistic_scores(mu_untrained, mu_cols, task)
        tics_a1, _, s_a1 = logistic_scores(feats, FEATURE_NAMES, task)
        tics_a2, _, s_a2 = forest_scores(feats, FEATURE_NAMES, task)
        # Disambiguating diagnostic (grill 2026-07-11): a nonlinear probe on the encoder mu, to tell
        # "info is in mu but a linear probe cannot read it" (-> Branch beta) from "mu lacks the info"
        # (-> Branch alpha). GBM on the UNTRAINED mu is the reference: if it beats GBM on the trained
        # mu, the SSL objective is actively discarding discriminative variance a random projection kept.
        tics_tg, _, s_trained_gbm = forest_scores(mu_trained, mu_cols, task)
        tics_ug, _, s_untrained_gbm = forest_scores(mu_untrained, mu_cols, task)
        aligned = [tics_un, tics_a1, tics_a2, tics_tg, tics_ug]

        scores_by_method = {
            "trained": s_trained, "untrained": s_untrained,
            "A1_logistic": s_a1, "A2_gbm": s_a2,
            "trained_mu_gbm": s_trained_gbm, "untrained_mu_gbm": s_untrained_gbm,
        }
        # B1 is the objective-INDEPENDENT supervised-trunk ceiling (random init, trained with labels), so it
        # is computed once and reused across the sweep; b1_seeds=() skips it and leaves the B1 fields NaN.
        run_b1 = len(b1_seeds) > 0
        b1_per_seed = []
        if run_b1:
            tics_b1, y_b1, s_b1, b1_per_seed = b1_scores(
                packed_dir, window, subset, task, enc_channels, kernel_size, z_dim, device, b1_seeds
            )
            aligned.append(tics_b1)
            assert np.array_equal(y, y_b1), "B1 labels misaligned with the shared test order"
            scores_by_method["B1_supervised"] = s_b1
        for other in aligned:
            assert np.array_equal(tics_tr, other), "method test-star orders diverged; alignment broken"
        boot = paired_bootstrap_ap(y, scores_by_method, n_boot=n_boot, seed=0)
        base_rate = float(y.mean())

        point = {}
        for method, sc in scores_by_method.items():
            point[method] = float(average_precision_score(y, sc))
            lo, hi = np.percentile(boot[method].to_numpy(), [2.5, 97.5])
            summary_rows.append({
                "task": task, "method": method, "pr_auc": point[method],
                "ci_lo": float(lo), "ci_hi": float(hi), "se": float(boot[method].std()),
                "base_rate": base_rate, "n_test_pos": int(y.sum()), "n_test": int(len(y)),
            })

        # A1 label-shuffle control: permute labels, refit A1, expect collapse to the base rate.
        shuffled = feats.copy()
        shuffled[task] = np.random.default_rng(0).permutation(shuffled[task].to_numpy())
        _, y_sh, s_sh = logistic_scores(shuffled, FEATURE_NAMES, task)
        summary_rows.append({
            "task": task, "method": "A1_label_shuffle", "pr_auc": float(average_precision_score(y_sh, s_sh)),
            "ci_lo": np.nan, "ci_hi": np.nan, "se": np.nan, "base_rate": base_rate,
            "n_test_pos": int(y.sum()), "n_test": int(len(y)),
        })

        diff_a1 = boot["A1_logistic"] - boot["untrained"] # paired per-replicate difference
        se_diff = float(diff_a1.std())
        point_diff = point["A1_logistic"] - point["untrained"]
        headroom_real = bool(point_diff > 2 * se_diff)
        # trained - untrained is the primary exp02 bellwether; the B1 gap fraction is only defined when B1 ran.
        trained_gap = point["trained"] - point["untrained"]
        if run_b1:
            b1_point = point["B1_supervised"]
            gap_denom = b1_point - point["untrained"]
            supervised_gap_frac = trained_gap / gap_denom if abs(gap_denom) > 1e-9 else np.nan
            b1_per_seed_str = ";".join(str(round(v, 4)) for v in b1_per_seed)
        else:
            b1_point = np.nan
            supervised_gap_frac = np.nan
            b1_per_seed_str = ""
        # info_in_mu: does a nonlinear probe extract more from the trained mu than the linear probe?
        # False => mu lacks the signal (Branch alpha, objective change); True => probe/pooling (Branch beta).
        diff_mu = boot["trained_mu_gbm"] - boot["trained"]
        se_diff_mu = float(diff_mu.std())
        point_diff_mu = point["trained_mu_gbm"] - point["trained"]
        info_in_mu = bool(point_diff_mu > 2 * se_diff_mu)
        gate_rows.append({
            "task": task,
            "trained": point["trained"], "untrained": point["untrained"],
            "A1_logistic": point["A1_logistic"], "A2_gbm": point["A2_gbm"], "B1_supervised": b1_point,
            "b1_per_seed": b1_per_seed_str,
            "trained_minus_untrained": trained_gap,
            "A1_minus_untrained": point_diff, "se_diff": se_diff, "two_se_diff": 2 * se_diff,
            "headroom_real": headroom_real, "supervised_gap_frac": supervised_gap_frac,
            "trained_mu_gbm": point["trained_mu_gbm"], "untrained_mu_gbm": point["untrained_mu_gbm"],
            "mu_gbm_minus_lin": point_diff_mu, "two_se_diff_mu": 2 * se_diff_mu, "info_in_mu": info_in_mu,
            "base_rate": base_rate,
        })
        log.info(f"[{task}] gate A1-untrained {point_diff} vs 2*SE {2*se_diff} -> headroom_real={headroom_real}; "
                 f"mu_gbm-lin {point_diff_mu} vs 2*SE {2*se_diff_mu} -> info_in_mu={info_in_mu}")

    summary = pd.DataFrame(summary_rows)
    gate = pd.DataFrame(gate_rows)
    if write:
        results_path = exp_dir / "results" / "skyline_results.csv"
        stamped = summary.copy()
        stamped["exp_name"] = exp_name
        stamped["variant"] = variant
        stamped["seed"] = seed
        stamped["ckpt"] = ckpt_stem
        append_results(results_path, stamped)
        gate.to_csv(exp_dir / "results" / "skyline_gate.csv", index=False)
    return summary, gate


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Override tasks / B1 seeds from the CLI with a leading + (keys are not in the base config), e.g.
    # +skyline_b1_seeds=[] skips the objective-independent B1 during the exp02 sweep.
    tasks = tuple(cfg.get("skyline_tasks", TASKS_DEFAULT))
    b1_seeds = tuple(cfg.get("skyline_b1_seeds", B1_SEEDS_DEFAULT))
    ckpt_stem = str(cfg.get("skyline_ckpt", "best"))  # +skyline_ckpt=best_recon_aux for the exp04 gate
    summary, gate = run_suite(
        cfg.exp_name, tasks=tasks, b1_seeds=b1_seeds, variant=cfg.variant_name, seed=int(cfg.seed),
        ckpt_stem=ckpt_stem,
    )
    log.info(f"skyline summary:\n{summary}")
    log.info(f"skyline gate:\n{gate}")


if __name__ == "__main__":
    main()
