"""R11 (roadmap 2026-08-11) = Q16 test 1 -- does the learned linear latent map transport phase?

The exp08 `linear_fbwd` arm replaced the GRU with a literal affine map per direction,
`mu_{t+1} ~ A_fwd mu_t + b_fwd` and `mu_{t-1} ~ A_bwd mu_t + b_bwd` (`dynamics.py:48-62`,
`world_model.py:107-118`). Eigen-decomposing A is the most direct test the project can run of the
strong framing "the latent tracks the state of an oscillator": phase transport means complex
eigenpairs with |lambda| near 1, whose argument theta implies a period 2*pi*dt/theta that tracks the
corpus period distribution. A real, sub-unit spectrum means the map is a smoother, full stop, which
Q16's decision rule treats as equally publishable.

Pre-registered before any number was looked at (Q16 flags confirmation pressure as the main hazard):
    O1 existence      >=1 complex pair with mu-mass weight in the top decile and |lambda| >= 0.9
    O2 persistence    mu-mass-weighted median |lambda| of complex modes >= 0.9 AND above the
                      random-init null's 95th percentile
    O3 population     implied P inside the resolvable band [0.711 d, 5.69 d] AND >=4 of 6 seeds carry
                      a mode within +-20% of a common P AND theta non-uniform vs the null
PASS needs all three; O1 alone is PARTIAL; anything less is FAIL ("the map is a smoother").

Band edges: consecutive window centres are dt = 256 cadences x 2 min = 8.533 h apart, so rotation
aliases below 2*dt = 0.711 d, and the training sequence spans 16 windows = 5.69 d.

Three controls, because a spectrum on its own cannot be read:
    random null   6 fresh LinearDynamics(128) inits. A random real matrix has almost ALL eigenvalues
                  in complex pairs with theta uniform, so "found complex pairs" is not evidence; only
                  |lambda| near 1 and clustered theta are.
    ols           the least-squares optimal one-step map fitted on the cached mu. Separates "training
                  chose a contraction" from "the latent affords nothing but a contraction".
    reciprocity   phase transport is invertible, so A_bwd should invert A_fwd (|lambda| reciprocal).
                  A contraction in BOTH time directions is impossible for transport and is the
                  cleanest smoother signature available.

Three views of every spectrum, since the linear arm holds only a handful of active mu dimensions and
eigenvalues of the full 128x128 map include directions mu never uses:
    full          eigen(A), every mode weighted by the mu-mass its eigenvector carries. The verdict.
    active_sub    eigen(A[S, S]) on the mu-variance-active dims S. Only legitimate if S is close to
                  A-invariant, so the leakage ||(I-P_S) A P_S|| / ||A P_S|| is reported alongside.
    pca           eigen(U^T A U) on the top-|S| principal components of mu. The basis the probe sees.

A is non-normal, so near-unit eigenvalues are necessary and not sufficient (transient amplification
lives in the pseudospectrum). Eigenvalues answer the asymptotic question the framing asks; the
non-normality of each map is reported rather than assumed away.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, seconds):
    PYTHONUNBUFFERED=1 python experiments/analyze_exp08_linear_eigen.py
    python experiments/analyze_exp08_linear_eigen.py --seeds 0 1 --no-plot
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("exp08_linear_eigen")

repo_root = Path(__file__).resolve().parents[1]
CKPT_DIR = repo_root / "experiments" / "exp08_linear" / "models"
MU_CACHE = repo_root / "experiments" / "exp08_prechecks" / "mu_cache"
LABELS = repo_root / "labels" / "variability_labels_star.csv"

DT_HOURS = 256 * 2 / 60  # 8.533 h between consecutive 256-cadence window centres
SEQ_LEN = 16
P_MIN_DAYS = 2 * DT_HOURS / 24  # 0.711 d, the Nyquist floor of the window cadence
P_MAX_DAYS = SEQ_LEN * DT_HOURS / 24  # 5.69 d, one sequence span
ACTIVE_VAR_FRAC = 0.01  # a mu dim is active if its variance is >=1% of the largest dim's
LAMBDA_GATE = 0.9  # O1/O2 modulus floor: tau = 3.4 d, ~10 prediction steps
MASS_DECILE = 0.9  # O1 mu-mass quantile a mode must clear
PERIOD_TOL = 0.2  # O3 seed-agreement tolerance on implied P
MIN_SEEDS_AGREE = 4


def load_maps(seed: int, ckpt_name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pull the two affine latent maps out of one linear_fbwd checkpoint, no model construction.

    nn.Linear stores y = x W^T + b, so with mu as a column vector the transition matrix IS `weight`.
    """
    ckpt = torch.load(CKPT_DIR / f"B_seed{seed}" / ckpt_name, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    maps = {}
    for direction, prefix in [("fwd", "dynamics"), ("bwd", "dynamics_bwd")]:
        weight = state[f"{prefix}.head.weight"].double().numpy()
        bias = state[f"{prefix}.head.bias"].double().numpy()
        maps[direction] = (weight, bias)
    return maps


def random_null_maps(draw: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Untrained LinearDynamics heads at PyTorch's default init, the null spectrum every gate is read against."""
    from swm.models.dynamics import LinearDynamics

    maps = {}
    for i, direction in enumerate(["fwd", "bwd"]):
        torch.manual_seed(1000 * (draw + 1) + i)
        head = LinearDynamics(128).head
        maps[direction] = (head.weight.detach().double().numpy(), head.bias.detach().double().numpy())
    return maps


def star_blocks(seed: int, split: str) -> list[np.ndarray]:
    """Per-star (n_windows, z) mu blocks for the linear arm; rows within a block are consecutive in time."""
    payload = np.load(MU_CACHE / f"linear_s{seed}.npz")
    flat = payload[f"{split}_mu"].astype(np.float64)
    counts = payload[f"{split}_counts"]
    blocks = []
    start = 0
    for count in counts:
        blocks.append(flat[start : start + int(count)])
        start += int(count)
    return blocks


def ols_maps(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Least-squares optimal one-step affine map on the cached mu, the "what the data affords" control."""
    blocks = star_blocks(seed, "train")
    current = []
    nxt = []
    for block in blocks:
        current.append(block[:-1])
        nxt.append(block[1:])
    current = np.concatenate(current, axis=0)
    nxt = np.concatenate(nxt, axis=0)
    maps = {}
    for direction, (x, y) in [("fwd", (current, nxt)), ("bwd", (nxt, current))]:
        design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        maps[direction] = (coef[:-1].T, coef[-1])
    return maps


def mu_covariance(seed: int) -> np.ndarray:
    """Centred window-level mu covariance on the train split, the mass measure every mode is weighted by."""
    flat = np.concatenate(star_blocks(seed, "train"), axis=0)
    return np.cov(flat, rowvar=False)


def active_dims(cov: np.ndarray) -> np.ndarray:
    """Latent dimensions mu actually uses, by variance relative to the largest dim (exp08 counts 4-29 here)."""
    var = np.diag(cov)
    return np.flatnonzero(var >= ACTIVE_VAR_FRAC * var.max())


def spectrum_rows(matrix: np.ndarray, cov: np.ndarray | None) -> pd.DataFrame:
    """Eigen-decompose one map and turn every mode into a physical row: modulus, angle, period, timescale.

    `cov` supplies the mu-mass weight w_k = |v_k^* Sigma v_k| normalised over modes; None leaves it NaN
    (the random null has no mu of its own to weight against).
    """
    values, vectors = np.linalg.eig(matrix)
    theta = np.abs(np.angle(values))
    modulus = np.abs(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        implied_p = np.where(theta > 0, 2 * np.pi * DT_HOURS / np.where(theta > 0, theta, np.nan) / 24, np.nan)
        tau = np.where(modulus > 0, -DT_HOURS / np.log(np.where(modulus > 0, modulus, np.nan)) / 24, np.nan)
    if cov is None:
        weight = np.full(len(values), np.nan)
    else:
        raw = np.empty(len(values))
        for k in range(len(values)):
            v = vectors[:, k]
            raw[k] = np.abs(np.conj(v) @ cov @ v)
        weight = raw / raw.sum()
    frame = pd.DataFrame({
        "mode": np.arange(len(values)),
        "re": values.real,
        "im": values.imag,
        "abs_lambda": modulus,
        "theta": theta,
        "implied_p_days": implied_p,
        "tau_days": tau,
        "mu_mass_weight": weight,
        "is_complex": np.abs(values.imag) > 1e-9,
    })
    frame["in_band"] = frame["implied_p_days"].between(P_MIN_DAYS, P_MAX_DAYS)
    return frame


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of `values` under mu-mass weights, used for O2 so dead directions cannot vote."""
    if len(values) == 0 or weights.sum() <= 0:
        return np.nan
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) / weights[order].sum()
    return float(values[order][np.searchsorted(cum, 0.5)])


def summarise(matrix: np.ndarray, bias: np.ndarray, cov: np.ndarray | None, rows: pd.DataFrame,
              dims: np.ndarray) -> dict:
    """Collapse one map to the row the verdict is read off: radius, mass-weighted persistence, geometry checks."""
    projector = np.zeros_like(matrix)
    projector[dims, dims] = 1.0
    projected = matrix @ projector
    leakage = np.linalg.norm(projected - projector @ projected, "fro") / max(np.linalg.norm(projected, "fro"), 1e-12)
    commutator = matrix.T @ matrix - matrix @ matrix.T
    complex_rows = rows[rows["is_complex"]]
    if cov is None:
        weights = np.ones(len(complex_rows))
    else:
        weights = complex_rows["mu_mass_weight"].to_numpy()
    sub = matrix[np.ix_(dims, dims)]
    identity = np.eye(len(dims))
    fixed_point = np.linalg.lstsq(identity - sub, bias[dims], rcond=None)[0]
    return {
        "n_active_dims": len(dims),
        "spectral_radius": float(rows["abs_lambda"].max()),
        "median_abs_lambda_complex_weighted": weighted_median(complex_rows["abs_lambda"].to_numpy(), weights),
        "frac_complex": float(rows["is_complex"].mean()),
        "mass_in_band": float(rows.loc[rows["in_band"] & rows["is_complex"], "mu_mass_weight"].sum()),
        "leakage_active": float(leakage),
        "non_normality": float(np.linalg.norm(commutator, "fro") / max(np.linalg.norm(matrix, "fro") ** 2, 1e-12)),
        "bias_norm": float(np.linalg.norm(bias)),
        "fixed_point_norm": float(np.linalg.norm(fixed_point)),
        "cond_i_minus_a": float(np.linalg.cond(identity - sub)),
    }


def analyse_arm(source: str, seed: int, maps: dict, cov: np.ndarray | None,
                dims: np.ndarray) -> tuple[pd.DataFrame, list[dict]]:
    """Run all three views for one (source, seed) pair and return its mode rows plus per-direction summaries."""
    mode_frames = []
    summaries = []
    for direction in ["fwd", "bwd"]:
        matrix, bias = maps[direction]
        views = {"full": matrix}
        views["active_sub"] = matrix[np.ix_(dims, dims)]
        if cov is None:
            views["pca"] = None
        else:
            eigvals, eigvecs = np.linalg.eigh(cov)
            basis = eigvecs[:, np.argsort(eigvals)[::-1][: len(dims)]]
            views["pca"] = basis.T @ matrix @ basis
        for view, view_matrix in views.items():
            if view_matrix is None:
                continue
            view_cov = cov
            if view == "active_sub" and cov is not None:
                view_cov = cov[np.ix_(dims, dims)]
            if view == "pca" and cov is not None:
                view_cov = basis.T @ cov @ basis
            rows = spectrum_rows(view_matrix, view_cov)
            rows.insert(0, "view", view)
            rows.insert(0, "direction", direction)
            rows.insert(0, "seed", seed)
            rows.insert(0, "source", source)
            mode_frames.append(rows)
            if view == "full":
                summary = summarise(matrix, bias, cov, rows, dims)
                summary.update({"source": source, "seed": seed, "direction": direction})
                summaries.append(summary)
    reciprocity = maps["bwd"][0] @ maps["fwd"][0]
    residual = reciprocity[np.ix_(dims, dims)] - np.eye(len(dims))
    for summary in summaries:
        summary["reciprocity_residual"] = float(np.linalg.norm(residual, "fro") / np.sqrt(len(dims)))
    return pd.concat(mode_frames, ignore_index=True), summaries


def theta_uniformity(rows: pd.DataFrame) -> float:
    """KS statistic of |theta| against uniform on (0, pi], the test that separates clustered phase from a random matrix."""
    theta = rows.loc[rows["is_complex"], "theta"].to_numpy()
    if len(theta) < 5:
        return np.nan
    return float(stats.kstest(theta / np.pi, "uniform").statistic)


def seed_period_agreement(band: pd.DataFrame, seeds: list[int]) -> tuple[int, float]:
    """Largest number of seeds carrying an in-band mode within +-20% of a common implied period.

    Run on the random null as well as the trained maps: with ~100 in-band complex modes per seed the
    criterion is close to trivially satisfiable, and the null's count is what shows that.
    """
    best = 0
    best_period = np.nan
    for _, candidate in band.iterrows():
        hit_seeds = set()
        for seed in seeds:
            same = band[(band["seed"] == seed)
                        & (np.abs(band["implied_p_days"] - candidate["implied_p_days"])
                           <= PERIOD_TOL * candidate["implied_p_days"])]
            if len(same) > 0:
                hit_seeds.add(seed)
        if len(hit_seeds) > best:
            best = len(hit_seeds)
            best_period = float(candidate["implied_p_days"])
    return best, best_period


def evaluate_gates(modes: pd.DataFrame, summary: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """Score the pre-registered O1/O2/O3 gates per seed and direction against the random-init null."""
    null_median = summary.loc[summary["source"] == "random_null", "median_abs_lambda_complex_weighted"]
    null_radius = summary.loc[summary["source"] == "random_null", "spectral_radius"]
    null_cut = float(np.nanpercentile(null_radius, 95))
    null_ks = theta_uniformity(modes[(modes["source"] == "random_null") & (modes["view"] == "full")])
    log.info(f"null: median radius {float(np.nanmedian(null_radius))}, 95th pct {null_cut}, "
             f"weighted-median |lambda| {float(np.nanmedian(null_median))}, theta KS {null_ks}")

    trained = modes[(modes["source"] == "trained") & (modes["view"] == "full")]
    null_modes = modes[(modes["source"] == "random_null") & (modes["view"] == "full")]
    rows = []
    for direction in ["fwd", "bwd"]:
        band = trained[(trained["direction"] == direction) & trained["is_complex"] & trained["in_band"]]
        agreeing, common_p = seed_period_agreement(band, seeds)
        null_band = null_modes[(null_modes["direction"] == direction) & null_modes["is_complex"]
                               & null_modes["in_band"]]
        null_agreeing, _ = seed_period_agreement(null_band, seeds)
        for seed in seeds:
            cell = trained[(trained["seed"] == seed) & (trained["direction"] == direction)]
            mass_cut = cell["mu_mass_weight"].quantile(MASS_DECILE)
            o1 = bool(((cell["is_complex"]) & (cell["mu_mass_weight"] >= mass_cut)
                       & (cell["abs_lambda"] >= LAMBDA_GATE)).any())
            cell_summary = summary[(summary["source"] == "trained") & (summary["seed"] == seed)
                                   & (summary["direction"] == direction)]
            weighted = float(cell_summary["median_abs_lambda_complex_weighted"].iloc[0])
            o2 = bool(weighted >= LAMBDA_GATE and cell_summary["spectral_radius"].iloc[0] > null_cut)
            ks = theta_uniformity(cell)
            o3 = bool(agreeing >= MIN_SEEDS_AGREE and np.isfinite(ks) and ks > null_ks)
            rows.append({"direction": direction, "seed": seed, "O1_existence": o1, "O2_persistence": o2,
                         "O3_population": o3, "weighted_median_abs_lambda": weighted, "theta_ks": ks,
                         "null_theta_ks": null_ks, "null_radius_p95": null_cut,
                         "seeds_agreeing_on_period": agreeing, "common_implied_p_days": common_p,
                         "null_seeds_agreeing_on_period": null_agreeing})
    return pd.DataFrame(rows)


def corpus_periods(seed: int) -> pd.DataFrame:
    """Catalogued periods for the stars the linear arm's mu was measured on, the population implied P is judged against."""
    payload = np.load(MU_CACHE / f"linear_s{seed}.npz")
    tics = set(payload["train_tics"].tolist()) | set(payload["test_tics"].tolist())
    labels = pd.read_csv(LABELS)
    labels["tic_id"] = labels["tic_id"].astype(int)
    return labels[labels["tic_id"].isin(tics)]


def make_figure(modes: pd.DataFrame, periods: pd.DataFrame, out_path: Path) -> None:
    """Two panels: where the trained modes sit relative to the unit circle and the null, and implied P vs the corpus."""
    trained = modes[(modes["source"] == "trained") & (modes["view"] == "full")]
    null = modes[(modes["source"] == "random_null") & (modes["view"] == "full")]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    circle = np.linspace(0, 2 * np.pi, 400)
    plt.plot(np.cos(circle), np.sin(circle), color="black", linewidth=1)
    plt.scatter(null["re"], null["im"], s=4, alpha=0.15, color="grey", label="random init")
    sizes = 4 + 400 * trained["mu_mass_weight"].fillna(0)
    plt.scatter(trained["re"], trained["im"], s=sizes, alpha=0.5, color="tab:red", label="trained (size = mu mass)")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.gca().set_aspect("equal")
    plt.xlabel("Re(lambda)")
    plt.ylabel("Im(lambda)")
    plt.title("linear_fbwd spectrum, 6 seeds x {fwd, bwd}")
    plt.legend(loc="upper left", fontsize=8)

    plt.subplot(1, 2, 2)
    bins = np.logspace(-1.7, 1.7, 50)
    centres = np.sqrt(bins[:-1] * bins[1:])
    for column, label in [("rotation_period", "rotation"), ("eb_period", "eb"), ("pulsating_period", "pulsating")]:
        values = pd.to_numeric(periods[column], errors="coerce").dropna()
        counts, _ = np.histogram(values, bins=bins)
        plt.step(centres, counts / max(counts.max(), 1), where="mid", label=f"{label} (n={len(values)})")
    complex_modes = trained[trained["is_complex"]].dropna(subset=["implied_p_days"])
    counts, _ = np.histogram(complex_modes["implied_p_days"], bins=bins,
                             weights=complex_modes["mu_mass_weight"]) # mass-weighted: dead directions cannot vote
    plt.fill_between(centres, counts / max(counts.max(), 1e-12), step="mid", alpha=0.35, color="tab:red",
                     label="implied P (mu-mass weighted)")
    null_complex = null[null["is_complex"]].dropna(subset=["implied_p_days"])
    counts, _ = np.histogram(null_complex["implied_p_days"], bins=bins)
    plt.step(centres, counts / max(counts.max(), 1), where="mid", color="grey", label="implied P, random init")
    plt.axvspan(P_MIN_DAYS, P_MAX_DAYS, color="tab:green", alpha=0.1)
    plt.xscale("log")
    plt.xlabel("period [d]")
    plt.ylabel("counts, each series scaled to its own max")
    plt.title(f"implied vs catalogued periods, band {round(P_MIN_DAYS, 3)}-{round(P_MAX_DAYS, 2)} d")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    log.info(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Q16 test 1: eigen-decompose the exp08 linear latent map.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--ckpt-name", default="best_recon_aux.pt", help="selection checkpoint (Q8)")
    ap.add_argument("--out-dir", default=str(repo_root / "experiments"))
    ap.add_argument("--fig-path", default=str(repo_root / "experiments" / "figs" / "exp08_linear_eigen.png"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    mode_frames = []
    summaries = []
    for seed in args.seeds:
        cov = mu_covariance(seed)
        dims = active_dims(cov)
        log.info(f"seed {seed}: {len(dims)} active mu dims")
        frame, rows = analyse_arm("trained", seed, load_maps(seed, args.ckpt_name), cov, dims)
        mode_frames.append(frame)
        summaries.extend(rows)

        frame, rows = analyse_arm("ols", seed, ols_maps(seed), cov, dims)
        mode_frames.append(frame)
        summaries.extend(rows)

        frame, rows = analyse_arm("random_null", seed, random_null_maps(seed), None, dims)
        mode_frames.append(frame)
        summaries.extend(rows)

        last = load_maps(seed, "last.pt")
        for direction in ["fwd", "bwd"]:
            radius = float(np.abs(np.linalg.eigvals(last[direction][0])).max())
            for summary in summaries:
                if summary["source"] == "trained" and summary["seed"] == seed and summary["direction"] == direction:
                    summary["spectral_radius_last_ckpt"] = radius

    modes = pd.concat(mode_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    gates = evaluate_gates(modes, summary, args.seeds)

    out_dir = Path(args.out_dir)
    modes.to_csv(out_dir / "exp08_linear_eigen.csv", index=False)
    summary.to_csv(out_dir / "exp08_linear_eigen_summary.csv", index=False)
    gates.to_csv(out_dir / "exp08_linear_eigen_gates.csv", index=False)
    log.info(f"wrote {out_dir / 'exp08_linear_eigen.csv'} ({len(modes)} modes)")

    verdict = "PASS"
    if not gates[["O1_existence", "O2_persistence", "O3_population"]].all(axis=None):
        if gates["O1_existence"].any():
            verdict = "PARTIAL"
        else:
            verdict = "FAIL (smoother)"
    log.info(f"pre-registered verdict: {verdict}")
    print(summary.groupby(["source", "direction"])[
        ["spectral_radius", "median_abs_lambda_complex_weighted", "mass_in_band", "leakage_active",
         "non_normality", "bias_norm", "reciprocity_residual"]].median())
    print(gates.groupby("direction")[["O1_existence", "O2_persistence", "O3_population"]].sum())

    if not args.no_plot:
        make_figure(modes, corpus_periods(args.seeds[0]), Path(args.fig_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
