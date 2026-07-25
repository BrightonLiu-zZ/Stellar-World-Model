"""Unit tests for the exp05 forward+backward latent-dynamics toggle (model.dyn_mode).

Guards the two risks the reverse-time term introduces: (T1) silently breaking exp04 checkpoint
reuse, and (T2/T3) the backward term leaking gradient through its stop-grad target or into the
wrong module. See src/swm/models/world_model.py and docs handoff 2026-07-21 task 4.
"""
from __future__ import annotations

import torch

from swm.models import WorldModel
from swm.train.losses import dynamics_loss


def make_small_model(dyn_mode: str = "fwd") -> WorldModel:
    return WorldModel(
        in_ch=1, enc_channels=[8, 16, 32, 64], kernel_size=5, z_dim=8, window=64,
        gru_hidden=16, gru_layers=1, dyn_mode=dyn_mode,
    )


def test_fwd_statedict_is_exp04_identical_and_strict_loads():
    """T1 -- reuse guard: fwd tree == fwd_bwd tree minus dynamics_bwd.*, and fwd strict-loads fwd."""
    fwd_keys = set(make_small_model("fwd").state_dict().keys())
    bwd_keys = set(make_small_model("fwd_bwd").state_dict().keys())

    # fwd_bwd adds EXACTLY the backward module's parameters, nothing else.
    extra = bwd_keys - fwd_keys
    assert extra, "fwd_bwd must add backward-module keys"
    assert all(k.startswith("dynamics_bwd.") for k in extra)
    assert fwd_keys == bwd_keys - extra # fwd is the exp00-04 tree, untouched

    # A fwd checkpoint loads into a fresh fwd model with strict=True (the exp04 mu-reuse path).
    src = make_small_model("fwd")
    dst = make_small_model("fwd")
    dst.load_state_dict(src.state_dict()) # strict=True default; raises on any key mismatch
    # And an exp04-style checkpoint (no dynamics_bwd) would fail strict-load into a fwd_bwd model,
    # which is exactly why fwd cells must build with dyn_mode="fwd".
    try:
        make_small_model("fwd_bwd").load_state_dict(src.state_dict())
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "fwd_bwd model must reject a fwd (dynamics_bwd-less) checkpoint under strict load"


def test_prev_keys_presence_and_shape_and_stopgrad():
    """T2 -- fwd emits no prev keys; fwd_bwd emits (B, S-1, z) pred/target with a stop-grad target."""
    x = torch.randn(3, 4, 64, 1) # (B, S, window, 1)

    fwd_out = make_small_model("fwd")(x)
    assert "pred_prev" not in fwd_out and "target_prev" not in fwd_out

    bwd_out = make_small_model("fwd_bwd")(x)
    assert bwd_out["pred_prev"].shape == (3, 3, 8) # (B, S-1, z)
    assert bwd_out["target_prev"].shape == (3, 3, 8)
    assert bwd_out["target_prev"].requires_grad is False # stop-grad holds
    # target_prev is the flipped mu_seq shifted by one, detached (mirror of target_next).
    rev = torch.flip(bwd_out["mu_seq"], dims=[1])
    assert torch.equal(bwd_out["target_prev"], rev[:, 1:, :].detach())


def test_backward_loss_trains_only_the_intended_paths():
    """T3 -- backward MSE alone gives grad to dynamics_bwd + encoder, but NOT to the forward GRU."""
    model = make_small_model("fwd_bwd")
    x = torch.randn(3, 4, 64, 1)
    out = model(x)
    dynamics_loss(out["pred_prev"], out["target_prev"]).backward() # backward term in isolation

    def has_grad(module) -> bool:
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    assert has_grad(model.dynamics_bwd) # the backward predictor learns
    assert has_grad(model.encoder) # gradient reaches the encoder via the (non-detached) input latents
    assert not has_grad(model.dynamics) # the forward GRU is untouched by the backward term
