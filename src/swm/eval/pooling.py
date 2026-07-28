"""
Pooling operators that reduce a star's BAG of window-level mu rows to one star-level score
(plan 2026-07-25). Shared by swm.eval.readout_sweep (v1 subset) and swm.eval.new_task_scorecard
(new-task pool), so every current and future probe inherits the same operator menu.

Two families, both of which keep the readout LINEAR and therefore stay inside the v1 probe lock:

  feature-space  reduce the bag's (n_win, z) mu block to one vector, then fit the readout once.
                 mean / max / quantile3 / quantile5 / moments / pca32_quantile5 / rff_meanmap /
                 gmm_prototype. Some are fitted UNSUPERVISED on the train bags (pca, rff, gmm);
                 that fit consumes no labels, so it is a fixed feature map, not a learned head.

  score-space    fit the readout on window rows (star label broadcast to its windows), score every
                 window, then aggregate the per-window scores per star.
                 ws_mean / ws_max / ws_lse / ws_topk / ws_linsoftmax / ws_ppv_lspv / ws_smooth.

The governing quantity is the WITNESS RATE pi = fraction of a positive bag's windows that actually
contain the signal. pi ~ 1/K for transit (one 8.5 h window per epoch) selects max-like operators;
pi ~ 1 for pulsating/rotation selects mean-like operators. ws_lse spans both ends with one scalar
beta, so beta* fitted per task is an empirical estimate of that task's witness rate.

Bags are represented as a list of (n_win, z) float arrays in a fixed star order, optionally with
per-star segment offsets (the start row of each contiguous segment inside the block). Order-aware
operators (ws_lspv, ws_smooth) run WITHIN a segment and aggregate across segments, because
CLAUDE.md forbids crossing segment boundaries and smoothing across a sector gap is meaningless.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

log = logging.getLogger(__name__)

feature_poolings = ("mean", "max", "quantile3", "quantile5", "moments", "mean_std", "mean_skew",
                    "pca32_quantile5", "rff_meanmap", "gmm_prototype")
score_poolings = ("ws_mean", "ws_max", "ws_lse", "ws_topk", "ws_linsoftmax", "ws_ppv_lspv", "ws_smooth")

# One hyperparameter per operator, tuned on the val split (never on test). A single-element grid
# means the operator has no free knob and is evaluated once.
pooling_grids: dict[str, list] = {
    "mean": [None],
    "max": [None],
    "quantile3": [None],
    "quantile5": [None],
    "moments": [None],
    "mean_std": [None],                    # moments ablation: is the gain the spread term?
    "mean_skew": [None],                   # moments ablation: or the one-sidedness term?
    "pca32_quantile5": [16, 32, 64],       # PCA components kept before quantiling
    "rff_meanmap": [256, 512, 1024],       # number of random Fourier features
    "gmm_prototype": [4, 8, 16],           # mixture components in the unsupervised codebook
    "ws_mean": [None],
    "ws_max": [None],
    "ws_lse": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],  # temperature: 0 -> mean, inf -> max
    "ws_topk": [1, 2, 3, 5, 10, 20],       # witnesses averaged; k=1 is ws_max
    "ws_linsoftmax": [None],
    "ws_ppv_lspv": [0.5, 0.75, 0.9, 0.95, 0.99],  # train-score quantile used as the "positive" threshold
    "ws_smooth": [0.1, 0.3, 0.5],          # chain-graph smoothing strength alpha
}


def segment_offsets_from_counts(seg_counts: list[np.ndarray]) -> list[np.ndarray]:
    """
    Convert per-star segment window-counts into per-star row offsets inside that star's mu block.
    A star observed in three segments of 16 windows gives offsets [0, 16, 32], which the order-aware
    operators use to avoid smoothing or run-counting across a sector gap.
    """
    offsets = []
    for counts in seg_counts:
        starts = np.zeros(len(counts) + 1, dtype=np.int64)
        starts[1:] = np.cumsum(counts)
        offsets.append(starts)
    return offsets


def trivial_offsets(blocks: list[np.ndarray]) -> list[np.ndarray]:
    """Treat each bag as a single contiguous segment; the first-segment bag scope is exactly this case."""
    offsets = []
    for block in blocks:
        offsets.append(np.array([0, block.shape[0]], dtype=np.int64))
    return offsets


# ----------------------------------------------------------------------------------------------------
# feature-space pooling: bag mu block -> one vector, then the usual readout
# ----------------------------------------------------------------------------------------------------
class FeaturePooling:
    """
    A fixed feature map from a bag of window mu rows to one star-level vector.
    `fit` sees only the TRAIN bags and never any label, so operators with an unsupervised codebook
    (PCA, random Fourier features, GMM prototypes) remain feature maps rather than learned heads.
    """

    def __init__(self, kind: str, param=None) -> None:
        assert kind in feature_poolings, f"unknown feature pooling {kind!r}"
        self.kind = kind
        self.param = param
        self.pca: PCA | None = None
        self.rff_w: np.ndarray | None = None
        self.rff_b: np.ndarray | None = None
        self.gmm: GaussianMixture | None = None

    def fit(self, blocks: list[np.ndarray]) -> "FeaturePooling":
        """Learn any unsupervised state the operator needs from the pooled train windows."""
        if self.kind == "pca32_quantile5":
            rows = np.concatenate(blocks, axis=0)
            n_components = min(int(self.param), rows.shape[1])
            self.pca = PCA(n_components=n_components, random_state=0)
            self.pca.fit(rows) # unsupervised: learns the directions of largest window-mu variance
        elif self.kind == "rff_meanmap":
            rows = np.concatenate(blocks, axis=0)
            subsample = rows[np.random.default_rng(0).choice(len(rows), size=min(20000, len(rows)), replace=False)]
            pairwise = np.linalg.norm(subsample[:1000, None, :] - subsample[None, :1000, :], axis=-1)
            median_dist = float(np.median(pairwise[pairwise > 0]))
            gamma = 1.0 / (2.0 * median_dist ** 2) # median heuristic for the RBF bandwidth
            rng = np.random.default_rng(0)
            n_features = int(self.param)
            self.rff_w = rng.normal(scale=np.sqrt(2.0 * gamma), size=(rows.shape[1], n_features))
            self.rff_b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
        elif self.kind == "gmm_prototype":
            rows = np.concatenate(blocks, axis=0)
            rng = np.random.default_rng(0)
            subsample = rows[rng.choice(len(rows), size=min(50000, len(rows)), replace=False)]
            self.gmm = GaussianMixture(n_components=int(self.param), covariance_type="diag",
                                       random_state=0, max_iter=100)
            self.gmm.fit(subsample) # unsupervised codebook of recurring window morphologies
        return self

    def transform(self, blocks: list[np.ndarray]) -> np.ndarray:
        """Map every bag to its fixed-length star vector, in the given star order."""
        feats = []
        for block in blocks:
            feats.append(self._one(block))
        return np.stack(feats, axis=0)

    def _one(self, block: np.ndarray) -> np.ndarray:
        if self.kind == "mean":
            return block.mean(axis=0)
        if self.kind == "max":
            return block.max(axis=0)
        if self.kind == "quantile3":
            return np.quantile(block, [0.1, 0.5, 0.9], axis=0).reshape(-1)
        if self.kind == "quantile5":
            return np.quantile(block, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0).reshape(-1)
        if self.kind in ("moments", "mean_std", "mean_skew"):
            mean = block.mean(axis=0)
            std = block.std(axis=0)
            if self.kind == "mean_std":
                return np.concatenate([mean, std])
            centred = block - mean
            skew = (centred ** 3).mean(axis=0) / np.maximum(std ** 3, 1e-8) # one-sided dips give negative skew
            if self.kind == "mean_skew":
                return np.concatenate([mean, skew])
            return np.concatenate([mean, std, skew])
        if self.kind == "pca32_quantile5":
            assert self.pca is not None, "pca32_quantile5 used before fit()"
            projected = self.pca.transform(block)
            return np.quantile(projected, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0).reshape(-1)
        if self.kind == "rff_meanmap":
            assert self.rff_w is not None, "rff_meanmap used before fit()"
            phi = np.cos(block @ self.rff_w + self.rff_b) # approximates the RBF kernel mean embedding
            return phi.mean(axis=0) * np.sqrt(2.0 / self.rff_w.shape[1])
        if self.kind == "gmm_prototype":
            assert self.gmm is not None, "gmm_prototype used before fit()"
            responsibility = self.gmm.predict_proba(block) # (n_win, C) soft assignment to each prototype
            weight = responsibility.mean(axis=0)
            parts = [weight]
            for c in range(responsibility.shape[1]):
                mass = responsibility[:, c].sum()
                if mass < 1e-8:
                    parts.append(np.zeros(block.shape[1]))
                    parts.append(np.zeros(block.shape[1]))
                    continue
                weighted_mean = (responsibility[:, c : c + 1] * block).sum(axis=0) / mass
                centred = block - weighted_mean
                weighted_var = (responsibility[:, c : c + 1] * centred ** 2).sum(axis=0) / mass
                parts.append(weighted_mean)
                parts.append(np.sqrt(weighted_var))
            return np.concatenate(parts)
        raise ValueError(f"unhandled feature pooling {self.kind!r}")


# ----------------------------------------------------------------------------------------------------
# score-space pooling: per-window scores -> one score (or a few features) per star
# ----------------------------------------------------------------------------------------------------
def _bag_slices(counts: np.ndarray) -> list[slice]:
    """Row slice into the concatenated window-score vector for each star, in the given star order."""
    slices = []
    start = 0
    for count in counts:
        slices.append(slice(start, start + int(count)))
        start += int(count)
    return slices


def _smooth_within_segments(scores: np.ndarray, offsets: np.ndarray, alpha: float, n_iter: int = 10) -> np.ndarray:
    """
    Diffuse each window's score toward its immediate time neighbours, one segment at a time.
    Implements the Sm chain-graph operator (NeurIPS 2024) as n_iter steps of
    z <- (1-alpha)*z + alpha*(neighbour average); an eclipse spans adjacent windows and survives,
    an isolated noise spike is damped. Segments are smoothed independently because consecutive
    segments can be months apart.
    """
    out = scores.copy()
    for seg in range(len(offsets) - 1):
        lo, hi = int(offsets[seg]), int(offsets[seg + 1])
        if hi - lo < 2:
            continue
        z = scores[lo:hi].copy()
        for _ in range(n_iter):
            neighbour = np.empty_like(z)
            neighbour[0] = z[1]
            neighbour[-1] = z[-2]
            if len(z) > 2:
                neighbour[1:-1] = 0.5 * (z[:-2] + z[2:])
            z = (1.0 - alpha) * z + alpha * neighbour
        out[lo:hi] = z
    return out


def _longest_positive_run(mask: np.ndarray, offsets: np.ndarray) -> int:
    """Longest stretch of consecutive above-threshold windows, never counted across a segment gap."""
    best = 0
    for seg in range(len(offsets) - 1):
        lo, hi = int(offsets[seg]), int(offsets[seg + 1])
        run = 0
        for value in mask[lo:hi]:
            if value:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
    return best


def aggregate_scores(window_scores: np.ndarray, counts: np.ndarray, offsets: list[np.ndarray],
                     kind: str, param=None) -> np.ndarray:
    """
    Reduce the flat per-window score vector to one value (or a small feature row) per star.
    Returns (n_stars,) for single-score operators and (n_stars, n_feat) for ws_ppv_lspv, which
    needs a tiny second-stage logistic because its outputs are several complementary statistics.
    """
    assert kind in score_poolings, f"unknown score pooling {kind!r}"
    slices = _bag_slices(counts)
    assert len(slices) == len(offsets), "star count disagrees between counts and segment offsets"
    if kind == "ws_ppv_lspv":
        threshold = float(np.quantile(window_scores, float(param)))
        rows = []
        for i, sl in enumerate(slices):
            s = window_scores[sl]
            mask = s > threshold
            ppv = float(mask.mean()) # fraction of windows above threshold: the collective statistic
            mpv = float(s[mask].mean()) if mask.any() else float(s.max())
            lspv = _longest_positive_run(mask, offsets[i]) / max(1, len(s))
            rows.append([ppv, mpv, lspv, float(s.max()), float(s.mean())])
        return np.array(rows, dtype=np.float64)

    out = np.zeros(len(slices), dtype=np.float64)
    for i, sl in enumerate(slices):
        s = window_scores[sl]
        if kind == "ws_mean":
            out[i] = s.mean()
        elif kind == "ws_max":
            out[i] = s.max()
        elif kind == "ws_lse":
            beta = float(param)
            shifted = beta * s
            peak = shifted.max()
            out[i] = (peak + np.log(np.exp(shifted - peak).mean())) / beta # log-sum-exp, overflow-safe
        elif kind == "ws_topk":
            k = min(int(param), len(s))
            out[i] = np.sort(s)[-k:].mean()
        elif kind == "ws_linsoftmax":
            total = s.sum()
            if total <= 0:
                out[i] = s.max()
            else:
                out[i] = float((s ** 2).sum() / total) # zero-parameter self-weighted mean
        elif kind == "ws_smooth":
            smoothed = _smooth_within_segments(s, offsets[i], float(param))
            out[i] = smoothed.max()
        else:
            raise ValueError(f"unhandled score pooling {kind!r}")
    return out


def noisy_and(window_scores: np.ndarray, counts: np.ndarray, a: float = 10.0, b: float = 0.5) -> np.ndarray:
    """
    Kraus+2016 Noisy-AND, kept only as a documented rank-equivalence check, NOT as a sweep cell.
    It is a strictly increasing function of the bag mean, so under a frozen readout it produces the
    identical star ranking as ws_mean and therefore the identical PR-AUC. Noisy-AND only differs
    when it is trained jointly with the encoder, where it changes gradients rather than ranks.
    """
    slices = _bag_slices(counts)
    out = np.zeros(len(slices), dtype=np.float64)
    lo = 1.0 / (1.0 + np.exp(a * b))
    hi = 1.0 / (1.0 + np.exp(-a * (1.0 - b)))
    for i, sl in enumerate(slices):
        mean_p = window_scores[sl].mean()
        out[i] = (1.0 / (1.0 + np.exp(-a * (mean_p - b))) - lo) / (hi - lo)
    return out


def bag_size_features(blocks: list[np.ndarray]) -> np.ndarray:
    """
    The `bagsize_only` control: log window count per star and nothing else.
    max/LSE/top-k all rise with bag size on pure noise, and bag size tracks the number of observed
    sectors, which tracks ecliptic latitude; a pooling win must beat this baseline to mean anything.
    """
    sizes = []
    for block in blocks:
        sizes.append([np.log(block.shape[0])])
    return np.array(sizes, dtype=np.float64)


def subsample_bags(blocks: list[np.ndarray], offsets: list[np.ndarray], k0: int, seed: int,
                   ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Draw a fixed K0 windows per star, so an all-segment bag is compared against a first-segment-sized
    one at equal bag size. Separates a genuine COVERAGE gain (the event was finally observed) from a
    bag-size artifact. Stars with fewer than K0 windows are returned whole. Segment structure is
    rebuilt from which rows survived, so order-aware operators stay boundary-correct.
    """
    rng = np.random.default_rng(seed)
    out_blocks = []
    out_offsets = []
    for block, offset in zip(blocks, offsets):
        n = block.shape[0]
        if n <= k0:
            out_blocks.append(block)
            out_offsets.append(offset)
            continue
        keep = np.sort(rng.choice(n, size=k0, replace=False))
        out_blocks.append(block[keep])
        seg_of_row = np.searchsorted(offset[1:-1], keep, side="right")
        starts = [0]
        for seg in range(1, len(offset) - 1):
            starts.append(int((seg_of_row < seg).sum()))
        starts.append(k0)
        out_offsets.append(np.array(sorted(set(starts)), dtype=np.int64))
    return out_blocks, out_offsets
