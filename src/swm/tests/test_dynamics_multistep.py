"""Unit tests for the exp05 multistep (free-running rollout) latent-dynamics mode (model.dyn_mode).

Guards the risks the rollout term introduces: (T1) it must NOT add params or break exp04 checkpoint
reuse (multistep reuses the SAME forward GRU), (T2) it must emit correctly-shaped stop-grad rollout
targets, and (T3) the rollout loss must route gradient into the forward dynamics + encoder. Plus a
compounding sanity check (free-running rollout is harder than one-step). See
src/swm/models/{world_model,dynamics}.py and docs/plans/2026-07-22-exp05-dynamics-axis.md.
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


def test_multistep_statedict_is_fwd_identical_and_strict_loads():
    """T1 -- reuse guard: multistep adds NO params (reuses self.dynamics), so its tree == the fwd tree
    and a fwd checkpoint strict-loads into a multistep model (and vice versa)."""
    fwd_keys = set(make_small_model("fwd").state_dict().keys())
    multi_keys = set(make_small_model("multistep").state_dict().keys())
    assert fwd_keys == multi_keys, "multistep must not add or drop any state_dict keys vs fwd"

    # A fwd checkpoint loads into a fresh multistep model with strict=True (and the reverse), because
    # multistep reuses the forward GRU rather than instantiating a new module.
    src = make_small_model("fwd")
    make_small_model("multistep").load_state_dict(src.state_dict()) # strict=True default; raises on mismatch
    make_small_model("fwd").load_state_dict(make_small_model("multistep").state_dict())


def test_roll_keys_presence_and_shape_and_stopgrad():
    """T2 -- fwd emits no roll keys; multistep emits (B, S-1, z) pred/target with a stop-grad target
    equal to the shifted latent sequence."""
    x = torch.randn(3, 4, 64, 1) # (B, S, window, 1)

    fwd_out = make_small_model("fwd")(x)
    assert "pred_roll" not in fwd_out and "target_roll" not in fwd_out

    multi_out = make_small_model("multistep")(x)
    assert multi_out["pred_roll"].shape == (3, 3, 8) # (B, S-1, z)
    assert multi_out["target_roll"].shape == (3, 3, 8)
    assert multi_out["target_roll"].requires_grad is False # stop-grad holds
    # target_roll is mu_seq shifted by one, detached (z_2..z_S).
    assert torch.equal(multi_out["target_roll"], multi_out["mu_seq"][:, 1:, :].detach())
    # pred_next is still emitted (one-step diagnostic), so both are available.
    assert "pred_next" in multi_out


def test_rollout_loss_trains_dynamics_and_encoder():
    """T3 -- the rollout MSE alone gives gradient to the forward GRU (self.dynamics) AND the encoder
    (via the non-detached z_1 / fed-back predictions)."""
    model = make_small_model("multistep")
    x = torch.randn(3, 4, 64, 1)
    out = model(x)
    dynamics_loss(out["pred_roll"], out["target_roll"]).backward() # rollout term in isolation

    def has_grad(module) -> bool:
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    assert has_grad(model.dynamics) # the forward GRU learns the rollout
    assert has_grad(model.encoder) # gradient reaches the encoder via z_1 and the fed-back predictions


def test_rollout_is_harder_than_one_step():
    """Sanity -- free-running rollout compounds error, so its MSE should exceed the one-step MSE for
    the SAME model/inputs (the point of the difficulty axis)."""
    torch.manual_seed(0)
    model = make_small_model("multistep")
    x = torch.randn(8, 6, 64, 1) # a longer sequence makes the compounding gap clearer
    out = model(x)
    one_step = dynamics_loss(out["pred_next"], out["target_next"]).item()
    rollout = dynamics_loss(out["pred_roll"], out["target_roll"]).item()
    assert rollout > one_step, f"rollout MSE {rollout} should exceed one-step MSE {one_step}"
