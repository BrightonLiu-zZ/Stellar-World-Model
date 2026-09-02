"""Wave-7 cells: the free_bits contrast at aux weight 0.03, composed and trained the way the runner does.

Wave 7 exists because wave 6 lost a seed of `w0p025` to total posterior collapse (KL 0.000, 0 active
units). The three new cells complete a 2x2 whose fourth corner is that already-run wave-6 cell:

                   free_bits 0.00                      free_bits 0.01
    w = 0.025      exp09_dpss_impulse_w0p025 (run)     exp09_dpss_impulse_w0p025_fb0p01
    w = 0.030      exp09_dpss_impulse_w0p03            exp09_dpss_impulse_w0p03_fb0p01

Both one-factor contrasts (free_bits at fixed weight, weight at fixed floor) are unreadable if anything
else differs along the way -- which is what these tests pin, edge by edge.

The rationale for testing this way rather than smoking the GPU queue is in test_wave6_cells.py and is
unchanged: composing `+experiment=exp09/<cell>` and driving `run_epoch` exercises Hydra resolution, the
manifest -> config contract and the loss path, but NOT the packed reader, the 100-epoch schedule,
checkpoint resume, or W&B. That gap is disclosed in the handoff message rather than papered over.

One wave-7-specific trap this pins. `free_bits` is a PER-DIMENSION KL floor, so the total floor is
free_bits * z_dim. The value 0.01 was chosen against exp09's measured KL regime (1.588 nats at w=0.025
and 1.638 at w=0.03; z_dim 128 -> floor 1.28 nats = 0.81x and 0.78x of achieved), reproducing the
geometry that made exp03's fb=0.02 the winner there ("with the floor below true KL, beta finally exerts
live gradient"). If z_dim ever changes, 0.01 silently stops meaning what the manifest says it means --
hence the z_dim assertion.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

import swm
from swm.models import WorldModel
from swm.train.loop import run_epoch

CONFIG_DIR = Path(swm.__file__).resolve().parent / "configs"

# (cell, aux weight, free_bits) -- the manifest's intent, written out independently here so a manifest
# typo cannot silently agree with itself.
WAVE7 = [
    ("exp09_dpss_impulse_w0p03", 0.03, 0.0),
    ("exp09_dpss_impulse_w0p03_fb0p01", 0.03, 0.01),
    ("exp09_dpss_impulse_w0p025_fb0p01", 0.025, 0.01),
]
Z_DIM_AT_DESIGN_TIME = 128  # the floor is free_bits * z_dim; 0.01 was derived against this value
# achieved (unfloored) val/kl_total measured on the wave-6 curves, per aux weight
ACHIEVED_KL = {0.025: 1.588, 0.03: 1.638}


def compose_cell(cell: str):
    """Resolve one cell the way `python -m swm.train +experiment=exp09/<cell>` does."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=[f"+experiment=exp09/{cell}", "variant=B", "seed=0"])


@pytest.mark.parametrize("cell,weight,free_bits", WAVE7)
def test_cell_resolves_to_its_intended_knobs(cell, weight, free_bits):
    cfg = compose_cell(cell)
    aux = cfg.train.recon_aux
    assert float(cfg.train.free_bits) == free_bits
    assert float(aux.weight) == weight
    assert aux.psd_window == "dpss"
    assert float(cfg.train.impulse_penalty_weight) == 0.1
    assert cfg.model.dyn_mode == "fwd_bwd" and float(cfg.train.lambda_dyn) == 60
    # wave-5 floor machinery and wave-6 augmentation must both stay off, or this re-runs an older wave
    assert aux.spectral_floor is None
    assert float(cfg.train.augment.hf_noise_sigma) == 0.0
    assert cfg.data.window == 256 and cfg.data.seq_len == 16


@pytest.mark.parametrize("cell,weight,free_bits", WAVE7)
def test_cell_trains_one_epoch(cell, weight, free_bits):
    """One real `run_epoch` per cell -- the loss path must run, not merely resolve."""
    cfg = compose_cell(cell)
    cfg.train.amp = False              # autocast('cuda') is a no-op on CPU
    cfg.data.window = 32
    cfg.model.z_dim = 8                # run_epoch sizes its KL accumulator from cfg.model.z_dim
    torch.manual_seed(0)
    model = WorldModel(in_ch=1, enc_channels=[4, 8], kernel_size=5, z_dim=8, window=32,
                       gru_hidden=16, gru_layers=1, dyn_mode=cfg.model.dyn_mode)
    loader = [torch.randn(2, 4, 32, 1) for _ in range(2)]   # (B, S, window, 1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    out = run_epoch(model, loader, None, scaler, cfg, beta=0.1, device="cpu", train=False)
    assert torch.isfinite(torch.tensor(out["total"])), f"{cell}: non-finite loss"
    assert out["aux"] > 0.0, f"{cell}: the aux term is silent"


def assert_only_difference(cell_a: str, cell_b: str, allowed: str) -> None:
    """Both cells resolve identically except on `allowed` -- the definition of a one-factor contrast.

    Checked key by key rather than by comparing whole configs, because exp09 has twice been bitten by a
    knob that moved silently while nothing failed loudly (impulse_penalty_weight nested under
    recon_aux; spectral_floor_mode likewise). A named-key sweep names the offender in the failure.
    """
    a, b = compose_cell(cell_a), compose_cell(cell_b)
    aux_keys = ("weight", "psd_window", "psd_normalize", "hf_weight", "spectral_floor",
                "spectral_floor_mode", "psd_taper_nw", "psd_taper_k", "type", "mask_frac")
    train_keys = ("lambda_dyn", "impulse_penalty_weight", "beta_target", "beta_warmup_epochs",
                  "free_bits", "lr", "max_epochs")
    for key in aux_keys:
        if f"recon_aux.{key}" == allowed:
            assert a.train.recon_aux[key] != b.train.recon_aux[key], f"{allowed} did NOT differ"
            continue
        assert a.train.recon_aux[key] == b.train.recon_aux[key], f"unexpected drift on aux.{key}"
    for key in train_keys:
        if key == allowed:
            assert a.train[key] != b.train[key], f"{allowed} did NOT differ"
            continue
        assert a.train[key] == b.train[key], f"unexpected drift on train.{key}"
    assert a.model.dyn_mode == b.model.dyn_mode
    assert a.data.window == b.data.window and a.data.seq_len == b.data.seq_len


def test_free_bits_edge_at_w0p03():
    """2x2 edge: free_bits at fixed weight 0.03."""
    assert_only_difference("exp09_dpss_impulse_w0p03", "exp09_dpss_impulse_w0p03_fb0p01", "free_bits")


def test_free_bits_edge_at_w0p025():
    """2x2 edge: free_bits at fixed weight 0.025, against the already-run wave-6 cell that collapsed."""
    assert_only_difference("exp09_dpss_impulse_w0p025", "exp09_dpss_impulse_w0p025_fb0p01", "free_bits")


def test_weight_edge_at_fb0():
    """2x2 edge: weight at fixed floor 0.00. w=0.03 is the low end of the untested 0.025 < w < 0.05."""
    assert_only_difference("exp09_dpss_impulse_w0p025", "exp09_dpss_impulse_w0p03", "recon_aux.weight")


def test_weight_edge_at_fb0p01():
    """2x2 edge: weight at fixed floor 0.01."""
    assert_only_difference("exp09_dpss_impulse_w0p025_fb0p01", "exp09_dpss_impulse_w0p03_fb0p01",
                           "recon_aux.weight")


@pytest.mark.parametrize("cell,weight,free_bits", [c for c in WAVE7 if c[2] > 0])
def test_the_free_bits_floor_means_what_the_manifest_says_it_means(cell, weight, free_bits):
    """free_bits is PER DIMENSION; the total floor is free_bits * z_dim. 0.01 was derived to sit BELOW
    exp09's measured KL at each weight, so beta keeps a live gradient -- the property that made exp03's
    floor work ("with the floor below true KL, beta finally exerts live gradient"). A z_dim change, or a
    weight whose achieved KL drops under the floor, silently breaks that derivation."""
    cfg = compose_cell(cell)
    assert int(cfg.model.z_dim) == Z_DIM_AT_DESIGN_TIME
    total_floor = float(cfg.train.free_bits) * int(cfg.model.z_dim)
    assert total_floor == pytest.approx(1.28)
    achieved = ACHIEVED_KL[weight]
    assert total_floor < achieved, (
        f"{cell}: floor {total_floor} must sit BELOW achieved KL {achieved}, or it binds on every seed")
    # exp03's winner sat at floor/KL = 0.85; stay in the same neighbourhood rather than merely under it
    assert 0.6 < total_floor / achieved < 0.95, f"{cell}: floor/KL {total_floor / achieved:.3f} off-design"
