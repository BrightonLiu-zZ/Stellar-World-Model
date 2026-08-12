"""exp07 pre-check C1: is the window-edge spike buying high-frequency PSD power?

`analyze_exp06_edge.py` established that the combined aux term punishes an INJECTED impulse far harder
than time-MSE does, and concluded that `recon_aux` is the knob conferring lpsd's immunity. That is a
statement about the loss functions, measured on a counterfactual spike. It does not say which way the
model's OWN edge error points, nor whether that error is something the aux term rewards. If the decoder
is under-powered at high frequency (the low-pass shortcut) it could be paying the log-PSD term down with
a cheap broadband edge impulse, in which case a Hann taper on the log-PSD term is a mechanism-targeted
fix and belongs in exp07; if not, the fix has no mechanism behind it and the immunity attribution rests
on the aux 2x2 alone.

Three measurements, all on checkpoints already on disk:

  1. sign        the mean SIGNED error mean(recon - input) at p0, p1, p254, p255 against the interior
                 mean. The existing profile parquet stores MSE only, which is sign-blind.

  2. ablate      the decisive counterfactual, and the mirror image of the exp06 injection test: REMOVE
                 the model's own edge samples and re-score. Three removals bracket the ambiguity in
                 "replace with the neighbours", since p0 and p255 each have interior neighbours on one
                 side only:
                   linear  extrapolate from the two nearest interior samples (2*r[1] - r[2])
                   copy    hold the nearest interior sample (r[1])
                   truth   splice the input's own value, so the edge residual becomes exactly zero
                 `truth` is the strongest form of the question: if aux is HIGHER with a perfect edge
                 than with the model's spike, the spike is buying aux down and nothing about the
                 comparison depends on how well an extrapolator guesses.

  3. decompose   the combined aux is log_psd + hf_weight * hf_time, and the two sub-terms have opposite
                 stakes in an edge impulse (broadband power helps a power-starved log-PSD; a step
                 hurts the first-difference term). The verdict needs them separated, so each is
                 reported alongside the combined value the training config actually optimises.

Loss config is read from each checkpoint's own stored cfg, so the recomputed aux is the exact term that
run trained against rather than an assumed one.

Writes experiments/exp07_c1_edge_sign.csv (one row per run x ablation) and
experiments/exp07_c1_signed_profile.parquet (one row per run x position). Run (repo root, swm env,
PYTHONPATH=src):
    python experiments/analyze_exp07_c1_edge_sign.py
    python experiments/analyze_exp07_c1_edge_sign.py --runs exp06_edge_comb_fb0p02:0 --n-windows 500
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from swm.models import WorldModel
from swm.train.losses import hf_time_loss, spectral_recon_loss

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

# Shared packed windows: every cell below is w256 and its packed dir is a junction to this one, so
# sampling here makes the "identical windows for every run" contract explicit rather than incidental.
PACKED = ROOT / "experiments" / "exp01_window256_seq16" / "packed"

INTERIOR_LO = 8 # interior band used as the reference level, matching analyze_exp06_edge.py
INTERIOR_HI = 8

# cell:seed. The three spiky arms the handoff names, plus two immune controls: the ablation itself
# perturbs the reconstruction, so a rise in aux only implicates the SPIKE if it does not also appear
# on models that have no spike to remove.
DEFAULT_RUNS = [
    "exp06_edge_comb_fb0p02:0",
    "exp05_comb_fbwd_c1p0:0", "exp05_comb_fbwd_c1p0:1", "exp05_comb_fbwd_c1p0:2",
    "exp06_w256_fbwd:0", "exp06_w256_fbwd:1", "exp06_w256_fbwd:2",
    "exp05_comb_off:0", # comb recipe, dynamics off: mild edge structure only
    "exp05_lpsd_multi_c1p0:0", # lpsd recipe, dynamics on: the immune arm
]
# truth_mid is the location control: the SAME operation (two residuals forced to zero) applied far from
# the boundary. Removing any two samples perturbs a 256-bin spectrum, so without it a rise in log-PSD
# cannot be attributed to the edge rather than to the ablation itself.
ABLATIONS = ("none", "linear", "copy", "truth", "truth_mid")
MID_POSITIONS = (96, 160) # interior control positions, both outside the p<8 / p>247 edge bands


def load_model(ckpt_path: Path, device: str) -> tuple[WorldModel, dict]:
    """Rebuild one run's world model from its checkpoint, strict-loading the backward head when present."""
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


def sample_windows(split: str, window: int, n: int, seed: int) -> np.ndarray:
    """
    Draw the fixed random window sample every run is measured on.
    The RNG seed is fixed on purpose: this is a measurement, so a between-run difference must not be
    able to come from the sample moving. Same estimator as analyze_exp06_edge.sample_windows.
    """
    index = pd.read_parquet(PACKED / f"{split}_index.parquet")
    total = int(index["n_win"].sum())
    memmap = np.memmap(PACKED / f"{split}_windows.dat", dtype=np.float32, mode="r", shape=(total, window))
    rows = np.sort(np.random.default_rng(seed).choice(total, min(n, total), replace=False))
    return np.array(memmap[rows], dtype=np.float32) # (n, window)


@torch.no_grad()
def reconstruct(model: WorldModel, x: torch.Tensor, batch: int) -> torch.Tensor:
    """Encode-then-decode a stack of windows in batches, returning the reconstruction as (n, window)."""
    out = []
    for start in range(0, x.shape[0], batch):
        chunk = x[start : start + batch] # (b, window, 1)
        mu, _ = model.encoder(chunk) # (b, z)
        out.append(model.decoder(mu)[:, :, 0]) # (b, window)
    return torch.cat(out, dim=0)


def ablate_edges(recon: torch.Tensor, x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Return a copy of the reconstruction with positions {0, -1} replaced, leaving the interior untouched.
    Each mode answers the same question under a different assumption about what "no spike" would mean:
    linear extrapolates the interior trend, copy holds the nearest interior sample, and truth substitutes
    the input's own value (a perfect edge, so the comparison does not depend on an extrapolator).
    """
    out = recon.clone()
    if mode == "none":
        return out
    if mode == "linear":
        out[:, 0] = 2.0 * recon[:, 1] - recon[:, 2]
        out[:, -1] = 2.0 * recon[:, -2] - recon[:, -3]
    elif mode == "copy":
        out[:, 0] = recon[:, 1]
        out[:, -1] = recon[:, -2]
    elif mode == "truth":
        out[:, 0] = x[:, 0]
        out[:, -1] = x[:, -1]
    elif mode == "truth_mid":
        for pos in MID_POSITIONS:
            out[:, pos] = x[:, pos]
    else:
        raise ValueError(f"unknown ablation {mode!r}")
    return out


def hann_log_psd(recon: torch.Tensor, x: torch.Tensor, eps: float) -> float:
    """
    The proposed exp07 fix, scored as a diagnostic: log-PSD MSE with a Hann taper applied before the FFT.
    A taper multiplies the boundary samples by ~0, so an edge impulse can no longer inject broadband
    power into the recon's spectrum. Comparing this term's response to the ablation against the untapered
    term's response says directly whether the taper removes the incentive the untapered term creates.
    Local to this diagnostic on purpose: changing swm.train.losses is an exp07 implementation decision.
    """
    r = recon.float()
    t = x.float()
    r = r - r.mean(dim=-1, keepdim=True) # drop DC, matching spectral_recon_loss
    t = t - t.mean(dim=-1, keepdim=True)
    taper = torch.hann_window(r.shape[-1], periodic=False, device=r.device) # (window,)
    pr = torch.fft.rfft(r * taper, dim=-1).abs() ** 2 # (n, window//2 + 1)
    pt = torch.fft.rfft(t * taper, dim=-1).abs() ** 2
    return float(F.mse_loss(torch.log(pr + eps), torch.log(pt + eps)))


def score_losses(recon: torch.Tensor, x: torch.Tensor, aux_cfg: dict) -> dict[str, float]:
    """
    Recompute the training loss terms on one (possibly edge-ablated) reconstruction.
    Shapes are lifted to the (B, S, window, 1) the loss functions expect, with S = 1: the aux terms are
    per-window operations, so folding the sample into the batch axis reproduces the training value.
    The combined term is reported both raw and weighted, since `val/aux` logs the raw value while the
    optimised objective carries the weight.
    """
    shape = (recon.shape[0], 1, recon.shape[1], 1)
    r = recon.reshape(shape)
    t = x.reshape(shape)
    log_psd = float(spectral_recon_loss(r, t, normalize=bool(aux_cfg["psd_normalize"]),
                                        eps=float(aux_cfg["psd_eps"]),
                                        window_fn=str(aux_cfg.get("psd_window", "none")))) # exp07 hann cells
    hf_time = float(hf_time_loss(r, t))
    if aux_cfg["type"] == "combined":
        aux = log_psd + float(aux_cfg["hf_weight"]) * hf_time
    elif aux_cfg["type"] == "log_psd":
        aux = log_psd
    else:
        raise ValueError(f"unexpected recon_aux type {aux_cfg['type']!r} for an edge-aux measurement")
    return {"time_mse": float(F.mse_loss(r, t)), "aux": aux,
            "aux_weighted": float(aux_cfg["weight"]) * aux,
            "log_psd": log_psd, "hf_time": hf_time,
            "log_psd_hann": hann_log_psd(recon, x, float(aux_cfg["psd_eps"]))}


def amplitude_sweep(recon: torch.Tensor, x: torch.Tensor, aux_cfg: dict,
                    scales: np.ndarray) -> list[dict]:
    """
    Score the aux terms with the model's own edge DEVIATION rescaled by a range of factors.
    The ablation says removing the edge raises log-PSD, and the exp06 injection test says adding a
    further impulse also raises it. Both can hold only if the learned edge value sits at a log-PSD
    optimum, which is a far stronger claim than a monotone response: it means the amplitude is tuned
    rather than incidental. Scale 0 is the interior-extrapolated edge, scale 1 is the model as trained.
    """
    base = ablate_edges(recon, x, "linear") # the no-spike reference the deviation is measured from
    rows = []
    for scale in scales:
        probe = base.clone()
        probe[:, 0] = base[:, 0] + float(scale) * (recon[:, 0] - base[:, 0])
        probe[:, -1] = base[:, -1] + float(scale) * (recon[:, -1] - base[:, -1])
        scored = score_losses(probe, x, aux_cfg)
        rows.append({"scale": float(scale), **scored})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="exp07 C1: signed edge profile + ablate-the-spike aux response")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="cell:seed pairs")
    ap.add_argument("--variant", default="B")
    ap.add_argument("--ckpt", default="best_recon_aux")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n-windows", type=int, default=2000, help="matches the exp05/exp06 edge samples")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="experiments/exp07_c1_edge_sign.csv")
    ap.add_argument("--out-profile", default="experiments/exp07_c1_signed_profile.parquet")
    ap.add_argument("--out-sweep", default="experiments/exp07_c1_amplitude_sweep.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_np = sample_windows(args.split, 256, args.n_windows, seed=0)
    x = torch.from_numpy(x_np).unsqueeze(-1).to(device) # (n, window, 1)
    flat = x[:, :, 0] # (n, window)
    log.info(f"device {device} | {len(args.runs)} runs | {x_np.shape[0]} windows of 256 ({args.split})")

    rows = []
    profiles = []
    sweeps = []
    sweep_scales = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
    for spec in tqdm(args.runs, desc="edge sign", total=len(args.runs)):
        cell, seed = spec.split(":")
        ckpt_path = ROOT / "experiments" / cell / "models" / f"{args.variant}_seed{seed}" / f"{args.ckpt}.pt"
        assert ckpt_path.exists(), f"missing {ckpt_path}"
        model, cfg = load_model(ckpt_path, device)
        aux_cfg = cfg["train"]["recon_aux"]
        recon = reconstruct(model, x, args.batch) # (n, window)

        err = (recon - flat).cpu().numpy() # (n, window) signed reconstruction error
        signed = err.mean(axis=0) # (window,)
        mse = (err ** 2).mean(axis=0) # (window,)
        interior_signed = float(signed[INTERIOR_LO:-INTERIOR_HI].mean())
        interior_mse = float(mse[INTERIOR_LO:-INTERIOR_HI].mean())
        profiles.append(pd.DataFrame({"cell": cell, "seed": int(seed), "pos": np.arange(len(signed)),
                                      "signed_err": signed, "mse": mse}))

        base = score_losses(recon, flat, aux_cfg)
        for mode in ABLATIONS:
            scored = score_losses(ablate_edges(recon, flat, mode), flat, aux_cfg)
            rows.append({
                "cell": cell, "seed": int(seed), "ablation": mode,
                "aux_type": aux_cfg["type"], "aux_weight": float(aux_cfg["weight"]),
                "lam": float(cfg["train"]["lambda_dyn"]), "free_bits": float(cfg["train"]["free_bits"]),
                "signed_p0": float(signed[0]), "signed_p1": float(signed[1]),
                "signed_p254": float(signed[-2]), "signed_p255": float(signed[-1]),
                "signed_interior": interior_signed,
                "edge_ratio": float(mse[0]) / interior_mse,
                "edge_ratio_last": float(mse[-1]) / interior_mse,
                **scored,
                "d_aux": scored["aux"] - base["aux"],
                "d_aux_rel": (scored["aux"] - base["aux"]) / base["aux"],
                "d_log_psd_rel": (scored["log_psd"] - base["log_psd"]) / base["log_psd"],
                "d_log_psd_hann_rel": (scored["log_psd_hann"] - base["log_psd_hann"]) / base["log_psd_hann"],
                "d_hf_time_rel": (scored["hf_time"] - base["hf_time"]) / base["hf_time"],
                "d_time_mse_rel": (scored["time_mse"] - base["time_mse"]) / base["time_mse"],
            })
        for sweep_row in amplitude_sweep(recon, flat, aux_cfg, sweep_scales):
            sweeps.append({"cell": cell, "seed": int(seed), "aux_type": aux_cfg["type"], **sweep_row})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / args.out, index=False)
    pd.concat(profiles, ignore_index=True).to_parquet(ROOT / args.out_profile, index=False)
    sweep = pd.DataFrame(sweeps)
    sweep.to_csv(ROOT / args.out_sweep, index=False)

    pd.set_option("display.width", 260)
    print("\nsigned edge profile (mean(recon - input); one row per run)")
    print(table[table["ablation"] == "none"][
        ["cell", "seed", "aux_type", "free_bits", "edge_ratio", "edge_ratio_last",
         "signed_p0", "signed_p1", "signed_p254", "signed_p255", "signed_interior"]
    ].round(4).to_string(index=False))
    print("\nablate-the-spike: relative change in each loss term when the named positions are replaced")
    print(table[table["ablation"] != "none"][
        ["cell", "seed", "ablation", "d_time_mse_rel", "d_aux_rel", "d_log_psd_rel",
         "d_log_psd_hann_rel", "d_hf_time_rel"]
    ].round(4).to_string(index=False))
    print("\npositive d_aux_rel = removing the edge RAISES the aux term, i.e. the spike was paying aux")
    print("down (HF-power-buying CONFIRMED). Negative = the edge cost aux, mechanism REFUTED.")
    print("truth vs truth_mid isolates location; d_log_psd_hann_rel is the same test under the taper.")
    print("\nedge-amplitude sweep: log-PSD against the model's own edge deviation rescaled (1.0 = as trained)")
    print(sweep.pivot_table(index="scale", columns="cell", values="log_psd").round(3).to_string())
    print("a minimum at scale ~1 means the edge amplitude is TUNED to the log-PSD term, not incidental.")
    log.info(f"\nwrote {ROOT / args.out} and {ROOT / args.out_profile}")


if __name__ == "__main__":
    main()
