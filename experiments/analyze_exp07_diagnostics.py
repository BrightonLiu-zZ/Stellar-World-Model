"""exp07 diagnostics: the three measurements the health-check notebook needs and no artifact holds.

exp07 froze `hann0p3` (combined aux @ 0.3, Hann-tapered log-PSD, fwd_bwd@60) as the ML4PS recipe on
the strength of an edge ratio that fell 31.4x -> 1.16x at probe parity. The probe/edge/amplitude CSVs
settle whether the fix worked. They cannot answer what it cost, what it changed, or where its one
significant probe win came from, because all three questions need quantities nothing has computed:

  spectral   What did the taper COST? A Hann window trades roughly 2x main-lobe width (frequency
             resolution) for far lower sidelobes (leakage). Scoring reconstruction spectra under
             EITHER training window rigs the comparison, so every model is scored under three
             referees - rectangular, Hann, and a DPSS multitaper neither model trained on - and the
             referees' disagreement is itself the leakage measurement. Resolution damage is then read
             off real oscillator peaks: reconstructed peak position, height, and width against truth.

  subspace   Is hann's latent the same latent as comb's, or only comb-without-the-artifact? mu
             coordinates are arbitrary across runs, so the comparison must be rotation-invariant:
             canonical correlations between the two runs' star-level mu, FITTED on train and scored
             on test, with same-recipe seed pairs as the null. If cross-recipe correlations sit inside
             the seed-to-seed band, the taper changed nothing structural. The sharp follow-up projects
             comb's shared subspace out of hann's mu and probes the residual.

  stars      Where does hann's pulsating win (+0.018 mean / +0.025 mean_std over 6 seeds) live?
             Per-star probe probabilities, so the notebook can bin the paired delta against the
             fractional-bin offset - the coordinate along which rectangular-window leakage is zero
             (integer cycles per window) or maximal (half-bin). PR-AUC is a set statistic and cannot
             be binned; per-star scores can.

Stage `mu` is a prerequisite for `subspace` and `stars`: it caches each run's star-level pooled mu
under experiments/exp07_forensics/mu_cache/ so the two downstream stages, and any later notebook
panel, cost no GPU. Every estimator here is copied from the scripts that produced the published
numbers (analyze_exp06_geometry_gap.py, analyze_exp07_c1_edge_sign.py) rather than re-derived, so the
`stars` stage's PR-AUC summary reproduces exp07_aux_gap_6seed.csv exactly; that reproduction is the
protocol check the notebook asserts before reading anything new.

Run (repo root, swm env, PYTHONPATH=src):
    PYTHONUNBUFFERED=1 python experiments/analyze_exp07_diagnostics.py
    python experiments/analyze_exp07_diagnostics.py --stages mu --seeds 0 --cells exp07_hann0p3_fbwd \
        --limit-stars 300 --n-windows 200
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal.windows import dpss
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from swm.eval.skyline import _make_untrained, load_first_segment_blocks, logistic_scores
from swm.models import WorldModel

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

# Every exp07 cell is w256 and its packed dir is a junction to this one; naming it explicitly makes
# the "identical windows for every run" contract visible rather than incidental (C1 convention).
PACKED = ROOT / "experiments" / "exp01_window256_seq16" / "packed"
MU_CACHE = ROOT / "experiments" / "exp07_forensics" / "mu_cache"

TASKS = ("pulsating", "eb", "rotation", "transit")
STAR_TASKS = ("pulsating", "eb") # the two exp07 gates: G-noregress lives on pulsating, G-C1 on eb
AMPLITUDE_COLS = ["p2p_scatter_ratio", "depth_5_95", "mad", "iqr"] # forensics 2.2 basis, periodicity-free

# 6-seed footing: winner + baseline, both arms. The subspace null needs seed pairs, so seed count is
# the binding constraint on these two stages and 4 seeds would leave only 6 null pairs per arm.
EXTENSION_CELLS = ["exp07_comb0p3_off", "exp07_comb0p3_fbwd", "exp07_hann0p3_off", "exp07_hann0p3_fbwd"]
EXTENSION_SEEDS = [0, 1, 2, 3, 4, 5]
# 4-seed footing: all five recipes on the dynamics-on arm, because form-vs-weight IS the taper-cost
# question and the extension never ran the lpsd / 0.1 cells.
SPECTRAL_CELLS = ["exp07_comb0p3_fbwd", "exp07_comb0p1_fbwd", "exp07_lpsd0p3_fbwd",
                  "exp07_lpsd0p1_fbwd", "exp07_hann0p3_fbwd"]
SPECTRAL_SEEDS = [0, 1, 2, 3]

WINDOW = 256
CADENCE_S = 120.0
BIN_UHZ = 1e6 / (WINDOW * CADENCE_S) # 32.55 uHz: the single-window frequency resolution at w256
PSD_EPS = 1e-8 # matches recon_aux.psd_eps in every exp07 cell
DPSS_NW = 4.0 # time-bandwidth product; 2*NW-1 = 7 usable tapers
DPSS_K = 7
PEAK_MIN_BIN = 2 # bins 0-1 are DC and the demeaning residue, never a physical peak
PEAK_PROMINENCE = 20.0 # peak power / median power; keeps the peak panel on windows that HAVE a peak
CCA_KS = (4, 8, 16) # 8 is primary: past the ~5-7 active units, short of fitting 128 noise directions


def load_model(ckpt_path: Path, device: str) -> tuple[WorldModel, dict]:
    """Rebuild one run's world model from its checkpoint; window and dyn_mode come from the stored cfg."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    mc = cfg["model"]
    model = WorldModel(
        in_ch=1, enc_channels=list(mc["enc_channels"]), kernel_size=int(mc["kernel_size"]),
        z_dim=int(mc["z_dim"]), window=int(cfg["data"]["window"]),
        gru_hidden=int(mc["gru_hidden"]), gru_layers=int(mc["gru_layers"]),
        dyn_mode=str(mc.get("dyn_mode", "fwd")),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval() # switch off any train-mode layer behaviour before measuring
    return model, cfg


def ckpt_path_of(cell: str, seed: int, variant: str, ckpt: str) -> Path:
    """Locate one run's evaluation checkpoint under the per-cell experiment folder."""
    return ROOT / "experiments" / cell / "models" / f"{variant}_seed{seed}" / f"{ckpt}.pt"


def untrained_ref_ckpt(cells: list[str], variant: str, ckpt: str) -> Path:
    """
    Find a checkpoint to read the ARCHITECTURE spec from when building the untrained reference.

    The untrained arm never loads weights -- it only needs cfg["model"] (enc_channels, kernel_size,
    z_dim, gru_hidden, gru_layers), which is identical across every cell in these experiments. The
    original code read it from a hardcoded EXTENSION_CELLS[0] using the CALLER'S ckpt name, so any
    experiment introducing a new checkpoint name (exp09's best_recon_only.pt) crashed on a cell that
    has never heard of it. Prefer a cell actually being scored, then fall back through the checkpoint
    names that every historical run carries. Fail loud rather than silently scoring a different arm.
    """
    candidates = [*cells, *EXTENSION_CELLS]
    for cell in candidates:
        if cell.startswith("untrained"):
            continue
        for name in (ckpt, "best_recon_aux", "best"):
            for seed in (0, 1, 2, 3, 4, 5):
                path = ckpt_path_of(cell, seed, variant, name)
                if path.exists():
                    return path
    raise FileNotFoundError(f"no checkpoint found to source the untrained architecture from: {candidates}")


# ----------------------------------------------------------------------------------------------------
# stage: mu - star-level pooled mu for every run, cached so the downstream stages need no GPU
# ----------------------------------------------------------------------------------------------------
@torch.no_grad() # no gradients needed at eval time; halves the memory of a full-split encode
def encode_blocks_batched(model: WorldModel, blocks: list[np.ndarray], device: str,
                          batch: int = 4096) -> list[np.ndarray]:
    """Encode every star's window block in concatenated batches, then split back per star (gap-script estimator)."""
    counts = []
    for block in blocks:
        counts.append(block.shape[0])
    flat = np.concatenate(blocks, axis=0) # (total_windows, window)
    chunks = []
    for start in range(0, flat.shape[0], batch):
        x = torch.from_numpy(flat[start : start + batch]).unsqueeze(-1).to(device) # (b, window, 1)
        mu, _ = model.encoder(x) # (b, z)
        chunks.append(mu.float().cpu().numpy())
    mu_all = np.concatenate(chunks, axis=0)
    out = []
    start = 0
    for n in counts:
        out.append(mu_all[start : start + n])
        start += n
    return out


def pool_blocks(mu_blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Reduce each star's bag of window mu to the two pooled vectors the v1 protocol scores (mean, std)."""
    means = []
    stds = []
    for block in mu_blocks:
        means.append(block.mean(axis=0))
        stds.append(block.std(axis=0))
    return np.stack(means).astype(np.float32), np.stack(stds).astype(np.float32)


def cache_mu(cell: str, seed: int, variant: str, ckpt: str, device: str,
             blocks: dict, tics: dict, cache_dir: Path, arch_cells: list[str] | None = None) -> Path:
    """
    Encode one run's first-segment bags and store the star-level pooled mu.
    Cached because `subspace` and `stars` both need every run's mu and re-encoding 24 runs for each
    stage would triple the GPU cost of the notebook for no new information.
    """
    out_path = cache_dir / f"{cell}_seed{seed}.npz"
    if out_path.exists():
        return out_path
    if cell.startswith("untrained"):
        ref_ckpt = torch.load(untrained_ref_ckpt(arch_cells or [], variant, ckpt),
                              map_location="cpu", weights_only=False)
        mc = ref_ckpt["cfg"]["model"]
        model = _make_untrained(list(mc["enc_channels"]), int(mc["kernel_size"]), int(mc["z_dim"]),
                                WINDOW, int(mc["gru_hidden"]), int(mc["gru_layers"]), device)
    else:
        model, _ = load_model(ckpt_path_of(cell, seed, variant, ckpt), device)
    payload = {}
    for split in ["train", "test"]:
        mu_blocks = encode_blocks_batched(model, blocks[split], device)
        mean, std = pool_blocks(mu_blocks)
        payload[f"tics_{split}"] = np.asarray(tics[split], dtype=np.int64)
        payload[f"mean_{split}"] = mean
        payload[f"std_{split}"] = std
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    return out_path


# ----------------------------------------------------------------------------------------------------
# stage: spectral - what the taper cost, scored under three referees
# ----------------------------------------------------------------------------------------------------
def taper_power(x: torch.Tensor, taper: torch.Tensor | None, weighted_demean: bool) -> torch.Tensor:
    """
    Periodogram of a batch of windows under one taper.
    `weighted_demean` reproduces the training estimator for the hann cells: weighting the mean by the
    taper gives the endpoint samples zero weight in BOTH the mean and the signal, so an edge impulse
    cannot leak back in through the mean. It is only defined for a strictly non-negative taper. DPSS
    tapers alternate in sign and the odd-order ones sum to ~0, so dividing by that sum would explode;
    those get plain demeaning, which is standard multitaper practice anyway.
    """
    if taper is None:
        centred = x - x.mean(dim=-1, keepdim=True) # (n, window)
    elif weighted_demean:
        weighted_mean = (x * taper).sum(dim=-1, keepdim=True) / taper.sum()
        centred = (x - weighted_mean) * taper
    else:
        centred = (x - x.mean(dim=-1, keepdim=True)) * taper
    return torch.fft.rfft(centred, dim=-1).abs() ** 2 # (n, window//2 + 1)


def multitaper_power(x: torch.Tensor, tapers: torch.Tensor) -> torch.Tensor:
    """
    Average the tapered periodograms over a DPSS taper family: the neutral referee.
    Slepian tapers are the standard low-variance spectral estimator and neither exp07 recipe trained
    against them, so neither model can be favoured by the choice; averaging K of them also removes the
    single-taper variance that would otherwise dominate a per-bin error curve.
    """
    total = torch.zeros(x.shape[0], x.shape[-1] // 2 + 1, device=x.device, dtype=torch.float32)
    for k in range(tapers.shape[0]):
        total = total + taper_power(x, tapers[k], weighted_demean=False)
    return total / float(tapers.shape[0])


def referee_windows(device: str) -> dict[str, tuple[torch.Tensor | None, bool]]:
    """The three scoring windows: the two the models trained under, plus the arbiter neither saw."""
    hann = torch.hann_window(WINDOW, periodic=False, device=device, dtype=torch.float32)
    slepian = torch.from_numpy(np.asarray(dpss(WINDOW, DPSS_NW, DPSS_K), dtype=np.float32)).to(device)
    return {"rect": (None, False), "hann": (hann, True), "dpss": (slepian, False)}


@torch.no_grad()
def reconstruct(model: WorldModel, x: torch.Tensor, batch: int) -> torch.Tensor:
    """Encode-then-decode a stack of windows, returning the reconstruction as (n, window)."""
    out = []
    for start in range(0, x.shape[0], batch):
        chunk = x[start : start + batch] # (b, window, 1)
        mu, _ = model.encoder(chunk) # (b, z)
        out.append(model.decoder(mu)[:, :, 0]) # (b, window)
    return torch.cat(out, dim=0)


def spectral_error_rows(recon: torch.Tensor, truth: torch.Tensor, referees: dict,
                        meta: dict) -> list[dict]:
    """
    Per-frequency-bin agreement between reconstructed and true spectra, once per referee.
    `err_sq` is the log-power squared error, the same functional form the aux term optimises, so a
    hann-trained model's advantage under its own referee is directly comparable to what it was
    trained to minimise. `bias` is the signed mean, which separates the two ways a spectrum can be
    wrong: positive means the reconstruction carries MORE power than the truth in that band, the
    signature of the edge impulse the untapered term rewards.
    """
    rows = []
    for name, (taper, weighted) in referees.items():
        if name == "dpss":
            pr = multitaper_power(recon, taper)
            pt = multitaper_power(truth, taper)
        else:
            pr = taper_power(recon, taper, weighted)
            pt = taper_power(truth, taper, weighted)
        diff = torch.log(pr + PSD_EPS) - torch.log(pt + PSD_EPS) # (n, n_bins)
        err_sq = (diff ** 2).mean(dim=0).cpu().numpy()
        bias = diff.mean(dim=0).cpu().numpy()
        for b in range(len(err_sq)):
            rows.append({**meta, "referee": name, "bin": b, "freq_uhz": b * BIN_UHZ,
                         "err_sq": float(err_sq[b]), "bias": float(bias[b])})
    return rows


def peak_fidelity(recon: torch.Tensor, truth: torch.Tensor, tapers: torch.Tensor,
                  meta: dict) -> pd.DataFrame:
    """
    Resolution damage read off real oscillator peaks rather than asserted from window theory.
    For every window whose true spectrum carries a dominant peak, three quantities are compared under
    the SAME neutral referee, so the referee's own main-lobe width cancels: where the reconstruction
    puts its peak, how much of the peak's power it keeps, and how sharply it concentrates that power
    against the surrounding bins. `contrast` is the sharpness measure and is defined even when the
    reconstruction has no local maximum at all, which a half-power width is not: these decoders
    routinely return a peak at 1-7% of true amplitude with no resolvable shoulder.
    """
    pr = multitaper_power(recon, tapers).cpu().numpy()
    pt = multitaper_power(truth, tapers).cpu().numpy()
    peak_bins = pt[:, PEAK_MIN_BIN:].argmax(axis=1) + PEAK_MIN_BIN
    rows = np.arange(pt.shape[0])
    prominent = pt[rows, peak_bins] > PEAK_PROMINENCE * np.median(pt[:, PEAK_MIN_BIN:], axis=1)
    keep = rows[prominent]
    truth_peak = peak_bins[prominent]
    recon_peak = pr[keep, PEAK_MIN_BIN:].argmax(axis=1) + PEAK_MIN_BIN
    contrast_truth, contrast_recon = [], []
    for i, row in enumerate(keep):
        shoulder = neighbour_bins(truth_peak[i], pt.shape[1])
        contrast_truth.append(pt[row, truth_peak[i]] / np.median(pt[row, shoulder]))
        contrast_recon.append(pr[row, truth_peak[i]] / np.median(pr[row, shoulder]))
    frame = pd.DataFrame({
        **meta,
        "truth_peak_bin": truth_peak,
        "truth_peak_uhz": truth_peak * BIN_UHZ,
        "d_peak_bins": recon_peak - truth_peak,
        "amp_ratio": pr[keep, truth_peak] / pt[keep, truth_peak],
        "contrast_truth": contrast_truth,
        "contrast_recon": contrast_recon,
    })
    frame["contrast_ratio"] = frame["contrast_recon"] / frame["contrast_truth"]
    return frame


def neighbour_bins(peak: int, n_bins: int) -> np.ndarray:
    """The shoulder bins a peak is measured against: 3-6 bins either side, clipped to the spectrum."""
    candidates = []
    for offset in [-6, -5, -4, -3, 3, 4, 5, 6]:
        b = peak + offset
        if PEAK_MIN_BIN <= b < n_bins:
            candidates.append(b)
    return np.asarray(candidates, dtype=int)


def sample_windows(split: str, n: int, seed: int) -> np.ndarray:
    """
    Draw the fixed random window sample every run is measured on (same estimator and seed as C1).
    The seed is fixed on purpose: this is a measurement, so a difference between two runs must not be
    able to come from the sample moving.
    """
    index = pd.read_parquet(PACKED / f"{split}_index.parquet")
    total = int(index["n_win"].sum())
    memmap = np.memmap(PACKED / f"{split}_windows.dat", dtype=np.float32, mode="r", shape=(total, WINDOW))
    rows = np.sort(np.random.default_rng(seed).choice(total, min(n, total), replace=False))
    return np.array(memmap[rows], dtype=np.float32) # (n, window)


def pulsating_windows(subset: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """
    Windows drawn from pulsating-positive test stars, the population the peak-fidelity panel is about.
    The random test sample is ~90% quiet stars, whose spectra have no peak to resolve, so measuring
    resolution damage on it would be measuring the noise floor.
    """
    tics, blocks = load_first_segment_blocks(PACKED, "test", WINDOW)
    positive = set(subset[subset["pulsating"] == 1]["tic_id"].tolist())
    keep = []
    for tic, block in zip(tics, blocks):
        if tic in positive:
            keep.append(block)
    stacked = np.concatenate(keep, axis=0)
    rng = np.random.default_rng(seed)
    if stacked.shape[0] > n:
        rows = np.sort(rng.choice(stacked.shape[0], n, replace=False))
        stacked = stacked[rows]
    return np.asarray(stacked, dtype=np.float32)


# ----------------------------------------------------------------------------------------------------
# stage: subspace - is hann's latent the same latent as comb's?
# ----------------------------------------------------------------------------------------------------
def canonical_correlations(a_train: np.ndarray, b_train: np.ndarray, a_test: np.ndarray,
                           b_test: np.ndarray, k: int) -> np.ndarray:
    """
    Held-out canonical correlations between two runs' star-level mu.
    Two encoders put their information in arbitrary and unrelated coordinates, so a dimension-wise
    comparison is meaningless; what IS comparable is the subspace each spans over the SAME stars.
    Canonical correlations are the cosines of the principal angles between those two star-space
    subspaces. Each side is PCA-whitened on train to k directions first (128 raw dims are mostly
    inactive under this objective, and an unregularised fit would align noise directions perfectly),
    the rotation is solved on train, and the correlation is then reported on held-out test stars, so
    an overfitted alignment shows up as a drop rather than as a perfect score.
    """
    pca_a = PCA(n_components=k, whiten=True).fit(a_train) # whiten: unit-variance scores on train
    pca_b = PCA(n_components=k, whiten=True).fit(b_train)
    za, zb = pca_a.transform(a_train), pca_b.transform(b_train)
    cross = za.T @ zb / (za.shape[0] - 1) # (k, k) cross-covariance of whitened train scores
    u, _, vt = np.linalg.svd(cross)
    ca = pca_a.transform(a_test) @ u
    cb = pca_b.transform(b_test) @ vt.T
    out = []
    for j in range(k):
        out.append(float(np.corrcoef(ca[:, j], cb[:, j])[0, 1]))
    return np.asarray(out)


def residual_probe(a: dict, b: dict, labels: pd.DataFrame, meta: dict) -> list[dict]:
    """
    Probe what run A's mu carries that run B's mu cannot linearly predict.
    Fitting the A-from-B map on train and probing A's residual on test asks the sharp version of "same
    latent or not": if hann's pulsating advantage survives having comb's shared content projected out,
    the taper added information rather than rearranging it. Scored against the same probe on A's full
    mu, and against the same operation between two seeds of one recipe, which is the null.
    """
    fitter = LinearRegression().fit(b["mean_train"], a["mean_train"])
    resid_train = a["mean_train"] - fitter.predict(b["mean_train"])
    resid_test = a["mean_test"] - fitter.predict(b["mean_test"])
    rows = []
    full_scores = probe_table_scores(a["mean_train"], a["mean_test"], a["tics_train"], a["tics_test"], labels)
    resid_scores = probe_table_scores(resid_train, resid_test, a["tics_train"], a["tics_test"], labels)
    for task in TASKS:
        rows.append({**meta, "task": task, "pr_auc_full": full_scores[task],
                     "pr_auc_residual": resid_scores[task],
                     "resid_var_frac": float(resid_train.var(axis=0).sum() / a["mean_train"].var(axis=0).sum())})
    return rows


def probe_table_scores(train_x: np.ndarray, test_x: np.ndarray, train_tics: np.ndarray,
                       test_tics: np.ndarray, labels: pd.DataFrame) -> dict[str, float]:
    """PR-AUC of the protocol-matched logistic probe on one feature matrix, every v1 task."""
    table = build_table(train_x, test_x, train_tics, test_tics, labels)
    cols = []
    for c in table.columns:
        if c.startswith("f"):
            cols.append(c)
    out = {}
    for task in TASKS:
        _, y, scores = logistic_scores(table, cols, task)
        out[task] = float(average_precision_score(y, scores))
    return out


def build_table(train_x: np.ndarray, test_x: np.ndarray, train_tics: np.ndarray,
                test_tics: np.ndarray, labels: pd.DataFrame) -> pd.DataFrame:
    """Assemble the star-level feature table the v1 probe expects (train+test rows, labels attached)."""
    parts = []
    for split, x, tics in [("train", train_x, train_tics), ("test", test_x, test_tics)]:
        frame = pd.DataFrame(x, columns=[f"f{j}" for j in range(x.shape[1])])
        frame.insert(0, "tic_id", tics)
        frame.insert(1, "split", split)
        parts.append(frame)
    table = pd.concat(parts, ignore_index=True)
    merged = table.merge(labels[["tic_id", *TASKS, *AMPLITUDE_COLS]], on="tic_id", how="inner")
    assert len(merged) == len(table), "a packed star is missing from the feature cache"
    return merged


def residualize(table: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove what the amplitude basis linearly predicts from each feature dim (fit on train only)."""
    out = table.copy()
    train = table[table["split"] == "train"]
    fitter = LinearRegression().fit(train[AMPLITUDE_COLS].to_numpy(), train[cols].to_numpy())
    out[cols] = table[cols].to_numpy() - fitter.predict(table[AMPLITUDE_COLS].to_numpy())
    return out


# ----------------------------------------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------------------------------------
def run_mu_stage(cells: list[str], seeds: list[int], variant: str, ckpt: str, device: str,
                 limit_stars: int, cache_dir: Path) -> None:
    """Cache star-level pooled mu for every (cell, seed) the downstream stages read, plus the untrained arm."""
    tics, blocks = {}, {}
    for split in ["train", "test"]:
        t, b = load_first_segment_blocks(PACKED, split, WINDOW)
        if limit_stars:
            t, b = t[:limit_stars], b[:limit_stars]
        tics[split], blocks[split] = t, b
    log.info(f"first-segment bags: {len(tics['train'])} train / {len(tics['test'])} test stars")
    jobs = [("untrained_w256", -1)]
    for cell in cells:
        for seed in seeds:
            jobs.append((cell, seed))
    for cell, seed in tqdm(jobs, desc="mu cache", total=len(jobs)):
        cache_mu(cell, seed, variant, ckpt, device, blocks, tics, cache_dir, arch_cells=cells)


def run_spectral_stage(cells: list[str], seeds: list[int], variant: str, ckpt: str, device: str,
                       n_windows: int, batch: int, out_bins: Path, out_peaks: Path) -> None:
    """Score every recipe's reconstruction spectrum under three referees, plus peak fidelity on pulsators."""
    subset = pd.read_parquet(ROOT / "processed" / "subset" / "subset_tics.parquet")
    referees = referee_windows(device)
    random_np = sample_windows("test", n_windows, seed=0)
    puls_np = pulsating_windows(subset, n_windows, seed=0)
    random_x = torch.from_numpy(random_np).to(device)
    puls_x = torch.from_numpy(puls_np).to(device)
    log.info(f"spectral sample: {random_np.shape[0]} random test windows, {puls_np.shape[0]} pulsator windows")

    jobs = [("untrained_w256", -1)]
    for cell in cells:
        for seed in seeds:
            jobs.append((cell, seed))
    bin_rows, peak_rows = [], []
    for cell, seed in tqdm(jobs, desc="spectral", total=len(jobs)):
        if cell.startswith("untrained"):
            ref = torch.load(ckpt_path_of(cells[0], seeds[0], variant, ckpt), map_location="cpu",
                             weights_only=False)
            mc = ref["cfg"]["model"]
            model = _make_untrained(list(mc["enc_channels"]), int(mc["kernel_size"]), int(mc["z_dim"]),
                                    WINDOW, int(mc["gru_hidden"]), int(mc["gru_layers"]), device)
            aux_type, aux_weight, psd_window = "none", 0.0, "none"
        else:
            model, cfg = load_model(ckpt_path_of(cell, seed, variant, ckpt), device)
            aux = cfg["train"]["recon_aux"]
            aux_type = str(aux["type"])
            aux_weight = float(aux["weight"])
            psd_window = str(aux.get("psd_window", "none"))
        meta = {"cell": cell, "seed": seed, "aux_type": aux_type, "aux_weight": aux_weight,
                "psd_window": psd_window}
        recon_random = reconstruct(model, random_x.unsqueeze(-1), batch)
        bin_rows += spectral_error_rows(recon_random, random_x, referees, {**meta, "pool": "random"})
        recon_puls = reconstruct(model, puls_x.unsqueeze(-1), batch)
        bin_rows += spectral_error_rows(recon_puls, puls_x, referees, {**meta, "pool": "pulsating"})
        peak_rows.append(peak_fidelity(recon_puls, puls_x, referees["dpss"][0], meta))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        pd.DataFrame(bin_rows).to_csv(out_bins, index=False) # rewrite per run: a late crash keeps the work
        pd.concat(peak_rows, ignore_index=True).to_csv(out_peaks, index=False)
    log.info(f"wrote {out_bins} ({len(bin_rows)} rows) and {out_peaks}")


def load_cached(cell: str, seed: int, cache_dir: Path) -> dict:
    """Read one run's cached star-level mu, keyed the way the subspace and stars stages consume it."""
    data = np.load(cache_dir / f"{cell}_seed{seed}.npz")
    out = {}
    for key in data.files:
        out[key] = data[key]
    return out


def run_subspace_stage(seeds: list[int], labels: pd.DataFrame, cache_dir: Path, out_cca: Path,
                       out_resid: Path) -> None:
    """Canonical correlations and the hann-unique residual probe, cross-recipe against the seed-pair null."""
    cca_rows, resid_rows = [], []
    pairs = []
    for arm in ["off", "fbwd"]:
        comb, hann = f"exp07_comb0p3_{arm}", f"exp07_hann0p3_{arm}"
        for seed in seeds:
            pairs.append((arm, "cross", hann, seed, comb, seed))
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i + 1:]:
                pairs.append((arm, "null_comb", comb, seed_a, comb, seed_b))
                pairs.append((arm, "null_hann", hann, seed_a, hann, seed_b))
    for arm, kind, cell_a, seed_a, cell_b, seed_b in tqdm(pairs, desc="subspace", total=len(pairs)):
        a, b = load_cached(cell_a, seed_a, cache_dir), load_cached(cell_b, seed_b, cache_dir)
        assert np.array_equal(a["tics_test"], b["tics_test"]), "the two runs were scored on different stars"
        meta = {"arm": arm, "pair": kind, "cell_a": cell_a, "seed_a": seed_a,
                "cell_b": cell_b, "seed_b": seed_b}
        for k in CCA_KS:
            corrs = canonical_correlations(a["mean_train"], b["mean_train"], a["mean_test"], b["mean_test"], k)
            for j, corr in enumerate(corrs):
                angle = float(np.degrees(np.arccos(np.clip(abs(corr), 0.0, 1.0))))
                cca_rows.append({**meta, "k": k, "component": j + 1, "cancorr_test": float(corr),
                                 "angle_deg": angle})
        if kind != "null_hann": # the residual probe asks what A adds to B; hann-on-comb and its comb null
            resid_rows += residual_probe(a, b, labels, meta)
        pd.DataFrame(cca_rows).to_csv(out_cca, index=False)
        pd.DataFrame(resid_rows).to_csv(out_resid, index=False)
    log.info(f"wrote {out_cca} ({len(cca_rows)} rows) and {out_resid} ({len(resid_rows)} rows)")


def run_stars_stage(cells: list[str], seeds: list[int], labels: pd.DataFrame, cache_dir: Path,
                    out_stars: Path, out_summary: Path) -> None:
    """
    Per-star probe probabilities for every run and pooling, plus the PR-AUC summary that must
    reproduce exp07_aux_gap_6seed.csv. The reproduction is the footing check: if it drifts, the
    per-star scores are not measuring the same probe the published gates were decided on.
    """
    star_rows, summary_rows = [], []
    jobs = [("untrained_w256", -1)]
    for cell in cells:
        for seed in seeds:
            jobs.append((cell, seed))
    for cell, seed in tqdm(jobs, desc="star scores", total=len(jobs)):
        cached = load_cached(cell, seed, cache_dir)
        mean_table = build_table(cached["mean_train"], cached["mean_test"],
                                 cached["tics_train"], cached["tics_test"], labels)
        cols = []
        for c in mean_table.columns:
            if c.startswith("f"):
                cols.append(c)
        std_table = build_table(np.concatenate([cached["mean_train"], cached["std_train"]], axis=1),
                                np.concatenate([cached["mean_test"], cached["std_test"]], axis=1),
                                cached["tics_train"], cached["tics_test"], labels)
        std_cols = []
        for c in std_table.columns:
            if c.startswith("f"):
                std_cols.append(c)
        arms = {"mean": (mean_table, cols), "mean_resid": (residualize(mean_table, cols), cols),
                "mean_std": (std_table, std_cols)}
        for pooling, (table, use_cols) in arms.items():
            for task in TASKS:
                tics, y, scores = logistic_scores(table, use_cols, task)
                summary_rows.append({"cell": cell, "seed": seed, "pooling": pooling, "task": task,
                                     "pr_auc": float(average_precision_score(y, scores))})
                if task not in STAR_TASKS:
                    continue
                star_rows.append(pd.DataFrame({"cell": cell, "seed": seed, "pooling": pooling,
                                               "task": task, "tic_id": tics, "y": y, "score": scores}))
        pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    pd.concat(star_rows, ignore_index=True).to_parquet(out_stars, index=False)
    log.info(f"wrote {out_stars} and {out_summary} ({len(summary_rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description="exp07 diagnostics: taper cost, latent identity, per-star scores")
    ap.add_argument("--stages", nargs="+", default=["mu", "spectral", "subspace", "stars"],
                    choices=["mu", "spectral", "subspace", "stars"])
    ap.add_argument("--cells", nargs="+", default=EXTENSION_CELLS, help="cells for the mu/stars stages")
    ap.add_argument("--seeds", nargs="+", type=int, default=EXTENSION_SEEDS)
    ap.add_argument("--spectral-cells", nargs="+", default=SPECTRAL_CELLS)
    ap.add_argument("--spectral-seeds", nargs="+", type=int, default=SPECTRAL_SEEDS)
    ap.add_argument("--variant", default="B")
    ap.add_argument("--ckpt", default="best_recon_aux")
    ap.add_argument("--feature-cache", default="experiments/exp06_features_cache.parquet")
    ap.add_argument("--n-windows", type=int, default=2000, help="matches the exp05-07 edge samples")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit-stars", type=int, default=0, help="smoke test: truncate each split")
    ap.add_argument("--cache-dir", default="", help="smoke test: write mu elsewhere than the real cache")
    ap.add_argument("--out-prefix", default="experiments/exp07_diag")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prefix = ROOT / args.out_prefix
    cache_dir = Path(args.cache_dir) if args.cache_dir else MU_CACHE
    log.info(f"device {device} | stages {args.stages} | mu cache {cache_dir}")

    labels = pd.read_parquet(ROOT / args.feature_cache)[["tic_id", *TASKS, *AMPLITUDE_COLS]]
    labels = labels.drop_duplicates("tic_id")

    if "mu" in args.stages:
        run_mu_stage(args.cells, args.seeds, args.variant, args.ckpt, device, args.limit_stars, cache_dir)
    if "spectral" in args.stages:
        run_spectral_stage(args.spectral_cells, args.spectral_seeds, args.variant, args.ckpt, device,
                           args.n_windows, args.batch,
                           Path(f"{prefix}_spectral_bins.csv"), Path(f"{prefix}_spectral_peaks.csv"))
    if "subspace" in args.stages:
        run_subspace_stage(args.seeds, labels, cache_dir, Path(f"{prefix}_cca.csv"),
                           Path(f"{prefix}_residual_probe.csv"))
    if "stars" in args.stages:
        run_stars_stage(args.cells, args.seeds, labels, cache_dir,
                        Path(f"{prefix}_star_scores.parquet"), Path(f"{prefix}_probe_summary.csv"))


if __name__ == "__main__":
    main()
