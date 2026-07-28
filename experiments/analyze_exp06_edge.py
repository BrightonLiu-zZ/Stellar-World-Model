"""exp06 pre-design forensics, part 1: what causes the position-0 reconstruction spike?

The exp05 notebook (section C1b) established the symptom and ruled out three explanations, but could
not identify the cause: `comb` and `lpsd` differ in TWO knobs at once (`free_bits` 0.0 vs 0.02 and
`recon_aux` comb vs log-PSD), so the 48 existing runs cannot say which one confers immunity.

This script adds the four measurements that can discriminate, all on checkpoints already on disk:

  1. profile        per-position MSE for every run, so the notebook can band it by group instead of
                    quoting a single seed.
  2. bias split     is the spike a constant the decoder emits regardless of input, or star-dependent?
                    mean(err)^2 / mean(err^2) at p=0. A near-constant spike is a trivial fix (trim or
                    bias correction); a star-dependent one is not.
  3. jacobian       ||d recon[p] / d z|| by position, estimated with central differences along random
                    latent directions, for the trained model AND for a random-init one. The decoder's
                    ConvTranspose1d(k=4, s=2, p=1) stack has an unpaired tap at the boundary, so the
                    architecture may supply a structural asymmetry that training then amplifies. The
                    open issue calls the defect "not architectural" on the strength of untrained MSE
                    being flat, which is a weaker instrument than the Jacobian.
  4. aux cost       the counterfactual log-PSD penalty. A time-domain spike is broadband in frequency,
                    so the log-PSD auxiliary term should punish it far out of proportion to its
                    time-domain cost. Measured by replacing the first k reconstructed samples with the
                    input's own values and re-scoring both terms. If the aux term drops much more than
                    the time term, `recon_aux` is what makes `lpsd` immune, and the proposed 2x2 corner
                    collapses to a single confirmation cell.

Also measured: whether the defect reaches the encoder at all (perturb x[0] vs an interior sample and
compare ||d mu||), since criterion 1's "the probe reads the encoder, the defect is in the decoder"
argument is currently assumed rather than tested.

Not measured: what checkpoint selection would have picked with the edge excluded. That needs per-epoch
checkpoints, and only the selected ones were kept. The edge's share of the recon loss is reported
instead, as a bound on how much selection could have been distorted.

Writes experiments/exp06_edge.csv (one row per run) and experiments/exp06_edge_profile.parquet
(one row per run x position). Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp06_edge.py
    python experiments/analyze_exp06_edge.py --cells exp05_comb_off exp05_comb_fbwd_c1p0 --seeds 0
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from swm.train.losses import spectral_recon_loss
from swm.models import WorldModel

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

EDGE_K = 4 # samples at each end treated as "the edge"; C1b showed the profile settles by p ~ 4
INTERIOR_LO = 8 # interior band used as the reference level, matching C1b
INTERIOR_HI = 8


def discover_runs(cells: list[str] | None, seeds: list[int], variant: str, ckpt: str) -> list[tuple[str, int, Path]]:
    """
    Find every (cell, seed) whose checkpoint exists on disk.
    When --cells is omitted, every exp05_* experiment directory is taken, so the table covers all 48
    runs without the caller having to name them.
    """
    if cells is None:
        cells = sorted(p.name for p in (ROOT / "experiments").glob("exp05_*") if p.is_dir())
    runs = []
    for cell in cells:
        for seed in seeds:
            path = ROOT / "experiments" / cell / "models" / f"{variant}_seed{seed}" / f"{ckpt}.pt"
            if path.exists():
                runs.append((cell, seed, path))
            else:
                log.warning(f"{cell} seed {seed}: no {ckpt}.pt; skipped")
    return runs


def load_model(ckpt_path: Path, device: str) -> tuple[WorldModel, dict]:
    """
    Rebuild one run's world model from its checkpoint.
    dyn_mode comes from the stored config so fwd_bwd runs strict-load their backward head instead of
    silently dropping it, which is what strict=False does in the exp05 notebook.
    """
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
    model.eval()
    return model, cfg


def sample_windows(packed_dir: Path, split: str, window: int, n: int, seed: int) -> np.ndarray:
    """
    Draw a fixed random set of packed windows to measure on.
    The RNG is fixed on purpose: this is a measurement, not an illustration, so every run must see the
    identical windows or the between-run comparison moves with the sample.
    """
    index = pd.read_parquet(packed_dir / f"{split}_index.parquet")
    total = int(index["n_win"].sum())
    memmap = np.memmap(packed_dir / f"{split}_windows.dat", dtype=np.float32, mode="r", shape=(total, window))
    rows = np.sort(np.random.default_rng(seed).choice(total, min(n, total), replace=False))
    return np.array(memmap[rows], dtype=np.float32) # (n, window)


@torch.no_grad()
def reconstruct(model: WorldModel, x: torch.Tensor, batch: int) -> torch.Tensor:
    """Encode-then-decode a stack of windows in batches, returning the reconstruction as (n, window)."""
    out = []
    for start in range(0, x.shape[0], batch):
        chunk = x[start : start + batch]
        mu, _ = model.encoder(chunk) # (b, z)
        out.append(model.decoder(mu)[:, :, 0]) # (b, window)
    return torch.cat(out, dim=0)


@torch.no_grad()
def jacobian_by_position(model: WorldModel, z: torch.Tensor, n_dirs: int, eps: float) -> np.ndarray:
    """
    Estimate ||d recon[p] / d z|| at every position with central differences along random directions.
    A full Jacobian is window x z_dim per star, which is wasteful when only the per-position norm is
    wanted. Averaging squared directional derivatives over isotropic unit directions recovers that norm
    up to a constant shared by all positions, which is all the position-to-position comparison needs.
    """
    window = model.decoder(z[:1]).shape[1]
    total = torch.zeros(window, device=z.device) # (window,)
    for _ in range(n_dirs):
        direction = torch.randn_like(z)
        direction = direction / direction.norm(dim=1, keepdim=True)
        plus = model.decoder(z + eps * direction)[:, :, 0] # (b, window)
        minus = model.decoder(z - eps * direction)[:, :, 0]
        total += (((plus - minus) / (2 * eps)) ** 2).mean(dim=0)
    return torch.sqrt(total / n_dirs).cpu().numpy() # (window,)


@torch.no_grad()
def encoder_edge_sensitivity(model: WorldModel, x: torch.Tensor, delta: float) -> tuple[float, float]:
    """
    How much does mu move when the FIRST sample of the input is perturbed, versus an interior sample?
    If the encoder is roughly translation-invariant these are comparable and the edge defect is purely
    a decoder phenomenon, which is what the criterion-1 impact argument assumes.
    """
    mu0, _ = model.encoder(x)
    edge = x.clone()
    edge[:, 0, 0] += delta
    mu_edge, _ = model.encoder(edge)
    mid = x.clone()
    mid[:, x.shape[1] // 2, 0] += delta
    mu_mid, _ = model.encoder(mid)
    d_edge = float((mu_edge - mu0).norm(dim=1).mean())
    d_mid = float((mu_mid - mu0).norm(dim=1).mean())
    return d_edge, d_mid


@torch.no_grad()
def spike_sensitivity(recon: torch.Tensor, x: torch.Tensor, amp: float) -> dict[str, float]:
    """
    Which loss term punishes an edge spike harder, plain MSE or the log-PSD auxiliary?
    Both obvious counterfactuals are confounded: splicing the input's true values over the edge injects
    a fresh discontinuity at the splice, and trimming the ends changes the FFT length, which moves the
    spectral loss for reasons unrelated to the edge. This instead INJECTS a controlled impulse of the
    given amplitude at position 0 of the model's own reconstruction, which is length-preserving and adds
    nothing but the spike, then reports each term's relative increase.
    An impulse is broadband, so a spectral term is expected to react far more strongly than time-domain
    MSE. The ratio of the two responses is what decides whether `recon_aux` rather than `free_bits` is
    the knob that makes the lpsd recipe immune.
    """
    shape = (recon.shape[0], 1, recon.shape[1], 1)
    spiked = recon.clone()
    spiked[:, 0] += amp
    time_base = float(torch.nn.functional.mse_loss(recon.reshape(shape), x.reshape(shape)))
    time_spiked = float(torch.nn.functional.mse_loss(spiked.reshape(shape), x.reshape(shape)))
    aux_base = float(spectral_recon_loss(recon.reshape(shape), x.reshape(shape)))
    aux_spiked = float(spectral_recon_loss(spiked.reshape(shape), x.reshape(shape)))

    resid = (recon - x).float()
    power = torch.fft.rfft(resid - resid.mean(dim=-1, keepdim=True), dim=-1).abs() ** 2 # (n, window//2+1)
    hf_frac = float(power[:, power.shape[-1] // 2:].sum() / power.sum())
    time_response = (time_spiked - time_base) / time_base
    aux_response = (aux_spiked - aux_base) / aux_base
    return {
        "time_base": time_base, "aux_base": aux_base,
        "time_response": time_response, "aux_response": aux_response,
        "aux_over_time_response": aux_response / time_response,
        "hf_resid_frac": hf_frac,
    }


def measure_run(model: WorldModel, cfg: dict, x_np: np.ndarray, device: str,
                n_dirs: int, jac_stars: int, batch: int,
                spike_amp: float) -> tuple[dict[str, float], np.ndarray]:
    """
    Run every edge measurement for one checkpoint and return (scalar row, per-position MSE profile).
    Everything shares the same fixed window sample so the scalars and the profile describe one object.
    """
    x = torch.from_numpy(x_np).unsqueeze(-1).to(device) # (n, window, 1)
    recon = reconstruct(model, x, batch) # (n, window)
    flat = x[:, :, 0] # (n, window)
    profile = ((recon - flat) ** 2).mean(dim=0).cpu().numpy() # (window,)

    err0 = (recon[:, 0] - flat[:, 0]).cpu().numpy()
    mse0 = float((err0 ** 2).mean())
    interior = float(profile[INTERIOR_LO:-INTERIOR_HI].mean())

    with torch.no_grad():
        mu, _ = model.encoder(x[:jac_stars])
    jac = jacobian_by_position(model, mu, n_dirs, eps=1e-3)
    d_edge, d_mid = encoder_edge_sensitivity(model, x[:batch], delta=1.0)
    aux = spike_sensitivity(recon, flat, amp=spike_amp)

    edge_share = float(profile[:EDGE_K].sum() + profile[-EDGE_K:].sum()) / float(profile.sum())
    row = {
        "lam": float(cfg["train"]["lambda_dyn"]),
        "dyn_mode": str(cfg["model"].get("dyn_mode", "fwd")),
        "free_bits": float(cfg["train"]["free_bits"]),
        "p0": float(profile[0]), "p_last": float(profile[-1]), "interior": interior,
        "edge_ratio": float(profile[0]) / interior,
        "edge_share_of_recon": edge_share,
        "bias_frac_p0": float(err0.mean() ** 2 / mse0), # 1.0 = the spike is a pure constant offset
        "recon_sd_p0": float(recon[:, 0].std()), "input_sd_p0": float(flat[:, 0].std()),
        "corr_p0": float(np.corrcoef(recon[:, 0].cpu().numpy(), flat[:, 0].cpu().numpy())[0, 1]),
        "corr_mid": float(np.corrcoef(recon[:, recon.shape[1] // 2].cpu().numpy(),
                                     flat[:, flat.shape[1] // 2].cpu().numpy())[0, 1]),
        "jac_p0": float(jac[0]), "jac_interior": float(jac[INTERIOR_LO:-INTERIOR_HI].mean()),
        "jac_ratio": float(jac[0]) / float(jac[INTERIOR_LO:-INTERIOR_HI].mean()),
        "dmu_edge": d_edge, "dmu_mid": d_mid, "dmu_ratio": d_edge / d_mid,
        **aux,
    }
    return row, profile


def main() -> None:
    ap = argparse.ArgumentParser(description="exp06 pre-design forensics: window-edge spike")
    ap.add_argument("--cells", nargs="+", default=None, help="default: every exp05_* experiment dir")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--variant", default="B")
    ap.add_argument("--ckpt", default="best_recon_aux")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n-windows", type=int, default=2000, help="matches the exp05 notebook's C1b sample")
    ap.add_argument("--n-dirs", type=int, default=32, help="random directions for the Jacobian estimate")
    ap.add_argument("--jac-stars", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--spike-amp", type=float, default=3.0,
                    help="injected impulse amplitude, in MAD-normalised flux units")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: keep only this many runs")
    ap.add_argument("--out", default="experiments/exp06_edge.csv")
    ap.add_argument("--out-profile", default="experiments/exp06_edge_profile.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = discover_runs(args.cells, args.seeds, args.variant, args.ckpt)
    if args.limit:
        runs = runs[: args.limit]
    if not runs:
        raise SystemExit("no checkpoints found - check --cells/--seeds")
    log.info(f"device {device} | {len(runs)} runs | {args.n_windows} windows")

    rows = []
    profiles = []
    x_np = None
    for cell, seed, path in tqdm(runs, desc="edge forensics", total=len(runs)):
        model, cfg = load_model(path, device)
        window = int(cfg["data"]["window"])
        if x_np is None:
            x_np = sample_windows(ROOT / "experiments" / cell / "packed", args.split, window,
                                  args.n_windows, seed=0)
            log.info(f"sampled {x_np.shape[0]} windows of {window} from the {args.split} split")
        row, profile = measure_run(model, cfg, x_np, device, args.n_dirs, args.jac_stars, args.batch,
                                   args.spike_amp)
        recipe = "lpsd" if "lpsd" in cell else "comb"
        rows.append({"cell": cell.replace("exp05_", ""), "recipe": recipe, "seed": seed,
                     "dyn_on": row["lam"] > 0, **row})
        profiles.append(pd.DataFrame({"cell": cell.replace("exp05_", ""), "recipe": recipe, "seed": seed,
                                      "dyn_on": row["lam"] > 0, "pos": np.arange(len(profile)),
                                      "mse": profile}))

    # The untrained reference: same architecture, random init, so the Jacobian shows what the
    # ConvTranspose stack does before any training shapes it.
    torch.manual_seed(0) # fixed init, so the untrained reference is reproducible across invocations
    untrained = WorldModel(
        in_ch=1, enc_channels=list(cfg["model"]["enc_channels"]), kernel_size=int(cfg["model"]["kernel_size"]),
        z_dim=int(cfg["model"]["z_dim"]), window=int(cfg["data"]["window"]),
        gru_hidden=int(cfg["model"]["gru_hidden"]), gru_layers=int(cfg["model"]["gru_layers"]),
    ).to(device).eval()
    row, profile = measure_run(untrained, cfg, x_np, device, args.n_dirs, args.jac_stars, args.batch,
                               args.spike_amp)
    row["lam"] = 0.0 # the row is measured on a random-init model, so the last cfg's knobs do not apply
    row["free_bits"] = float("nan")
    row["dyn_mode"] = "none"
    rows.append({"cell": "untrained", "recipe": "untrained", "seed": 0, "dyn_on": False, **row})
    profiles.append(pd.DataFrame({"cell": "untrained", "recipe": "untrained", "seed": 0, "dyn_on": False,
                                  "pos": np.arange(len(profile)), "mse": profile}))

    edge = pd.DataFrame(rows)
    edge.to_csv(ROOT / args.out, index=False)
    pd.concat(profiles, ignore_index=True).to_parquet(ROOT / args.out_profile, index=False)
    log.info(f"\nwrote {ROOT / args.out} ({len(edge)} runs) and {ROOT / args.out_profile}")

    pd.set_option("display.width", 220)
    print()
    print(edge.groupby(["recipe", "dyn_on"]).agg(
        n=("edge_ratio", "size"), edge_ratio=("edge_ratio", "mean"), bias_frac=("bias_frac_p0", "mean"),
        jac_ratio=("jac_ratio", "mean"), dmu_ratio=("dmu_ratio", "mean"),
        time_resp=("time_response", "mean"), aux_resp=("aux_response", "mean"),
        aux_over_time=("aux_over_time_response", "mean")).round(4).to_string())
    print(f"\nresponse to an injected impulse of amplitude {args.spike_amp} at position 0.")
    print("aux_over_time >> 1 means the spectral term punishes a spike far harder than plain MSE,")
    print("which would make recon_aux (not free_bits) the knob that confers lpsd's immunity.")


if __name__ == "__main__":
    main()
