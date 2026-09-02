"""Wave-6 cells: compose each generated config through Hydra exactly as the runner does, then train on it.

WHY THIS EXISTS AS A TEST RATHER THAN AN AD-HOC SCRIPT. The long-run guard forbids training inside
Claude Code, which collides with the smoke-test-before-handoff rule -- we cannot verify a GPU queue by
running it. Wave 5's accepted resolution, reused here: compose every new cell the way the runner does
(`+experiment=exp09/<cell>`, the SAME override string the .ps1 emits) and drive it through `run_epoch`
on synthetic data. That exercises Hydra resolution, the manifest -> config contract, and the loss path.

WHAT IT DOES NOT EXERCISE, disclosed rather than papered over: the real packed data reader, the
optimizer/scheduler over 100 epochs, checkpoint save/resume, and W&B. A cell can pass everything here
and still die on the user's machine for a resume/RNG reason -- which has happened before on this
project, and is why the handoff message says so out loud.

The knob assertions are the point. exp09 has already been bitten twice by a mis-nested key that
trained a cell as its own control while nothing failed loudly (`impulse_penalty_weight` under
recon_aux; `spectral_floor_mode` likewise). Wave 6 adds a cell whose entire identity is two knobs
being ZERO (`w0p025_off`), which is exactly the shape that fails silently.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

import swm
from swm.models import WorldModel
from swm.train.loop import run_epoch

# swm/configs is a config DIRECTORY, not an importable module (no __init__.py) -- swm.train.__main__
# reaches it as config_path="../configs", so the test has to resolve the same directory by path.
CONFIG_DIR = Path(swm.__file__).resolve().parent / "configs"

# (cell, dyn_mode, lambda_dyn, aux weight, psd_window, kurtosis weight) -- the manifest's intent,
# written out independently here so a manifest typo cannot silently agree with itself.
WAVE6 = [
    ("exp09_dpss_impulse_w0p025", "fwd_bwd", 60, 0.025, "dpss", 0.1),
    ("exp07_hann0p3_fbwd", "fwd_bwd", 60, 0.3, "hann", 0.0),
    ("exp09_dpss_impulse_w0p025_off", "fwd", 0, 0.025, "dpss", 0.1),
    ("exp09_hann_w0p025", "fwd_bwd", 60, 0.025, "hann", 0.0),
    ("exp09_hann_w0p10", "fwd_bwd", 60, 0.10, "hann", 0.0),
]


def compose_cell(cell: str):
    """Resolve one cell the way `python -m swm.train +experiment=exp09/<cell>` does."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=[f"+experiment=exp09/{cell}", "variant=B", "seed=0"])


@pytest.mark.parametrize("cell,dyn,lam,weight,window,kurt", WAVE6)
def test_cell_resolves_to_its_intended_knobs(cell, dyn, lam, weight, window, kurt):
    cfg = compose_cell(cell)
    aux = cfg.train.recon_aux
    assert cfg.model.dyn_mode == dyn
    assert float(cfg.train.lambda_dyn) == lam
    assert float(aux.weight) == weight
    assert aux.psd_window == window
    assert float(cfg.train.impulse_penalty_weight) == kurt
    # every wave-6 cell holds the wave-5 floor machinery OFF; a stray floor would silently re-run wave 5
    assert aux.spectral_floor is None
    assert float(cfg.train.augment.hf_noise_sigma) == 0.0
    assert cfg.data.window == 256 and cfg.data.seq_len == 16


@pytest.mark.parametrize("cell,dyn,lam,weight,window,kurt", WAVE6)
def test_cell_trains_one_epoch(cell, dyn, lam, weight, window, kurt):
    """One real `run_epoch` per cell on synthetic windows -- the loss path must run, not just resolve."""
    cfg = compose_cell(cell)
    cfg.train.amp = False  # autocast('cuda') is a no-op on CPU
    # Shrink the geometry so the smoke is cheap. window/z_dim must be shrunk in the CONFIG as well as in
    # the model: run_epoch sizes its KL accumulator from cfg.model.z_dim, so a mismatch is a shape error,
    # not a silently wrong number. The real geometry is asserted in the resolution test above.
    cfg.data.window = 32
    cfg.model.z_dim = 8
    torch.manual_seed(0)
    model = WorldModel(in_ch=1, enc_channels=[4, 8], kernel_size=5, z_dim=8, window=32,
                       gru_hidden=16, gru_layers=1, dyn_mode=cfg.model.dyn_mode)
    loader = [torch.randn(2, 4, 32, 1) for _ in range(2)]  # (B, S, window, 1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    out = run_epoch(model, loader, None, scaler, cfg, beta=0.1, device="cpu", train=False)
    assert torch.isfinite(torch.tensor(out["total"])), f"{cell}: non-finite loss"
    assert out["aux"] > 0.0, f"{cell}: the aux term is silent"


def test_the_off_arm_actually_switches_the_dynamics_term_off():
    """`off` in this repo is dyn_mode=fwd + lambda_dyn=0, not a dyn_mode value (world_model.py asserts
    the allowed set). Both halves must land, or the 'off' arm is a fwd arm at full dose wearing a label."""
    off = compose_cell("exp09_dpss_impulse_w0p025_off")
    on = compose_cell("exp09_dpss_impulse_w0p025")
    assert float(off.train.lambda_dyn) == 0.0 and off.model.dyn_mode == "fwd"
    assert float(on.train.lambda_dyn) == 60.0 and on.model.dyn_mode == "fwd_bwd"
    # ...and the ONLY difference between the pair is the dynamics axis, so the contrast is one-factor
    differing = {k for k in ("weight", "psd_window", "psd_normalize", "hf_weight", "spectral_floor")
                 if off.train.recon_aux[k] != on.train.recon_aux[k]}
    assert not differing, f"off arm differs from its parent on the aux axis too: {differing}"
    assert float(off.train.impulse_penalty_weight) == float(on.train.impulse_penalty_weight)


def test_the_hann_ladder_differs_from_its_weight_matched_twin_only_in_mechanism():
    """exp09_hann_w0p025 vs exp09_dpss_impulse_w0p025 is the wave-6 mechanism contrast: same weight
    (= same spectral pressure), taper and kurtosis term removed. If the weights ever drift apart the
    contrast stops being one-factor and P11's multiplier is unreadable."""
    hann = compose_cell("exp09_hann_w0p025")
    twin = compose_cell("exp09_dpss_impulse_w0p025")
    assert float(hann.train.recon_aux.weight) == float(twin.train.recon_aux.weight)
    assert float(hann.train.lambda_dyn) == float(twin.train.lambda_dyn)
    assert hann.train.recon_aux.psd_window == "hann" and twin.train.recon_aux.psd_window == "dpss"
    assert float(hann.train.impulse_penalty_weight) == 0.0
    assert float(twin.train.impulse_penalty_weight) == 0.1

    hi = compose_cell("exp09_hann_w0p10")
    hi_twin = compose_cell("exp09_dpss_impulse_w0p10")
    assert float(hi.train.recon_aux.weight) == float(hi_twin.train.recon_aux.weight) == 0.10
