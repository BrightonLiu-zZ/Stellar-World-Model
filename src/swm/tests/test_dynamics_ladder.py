"""Unit tests for the exp08 dynamics-ladder modes (model.dyn_mode: smooth / linear_fbwd / frozen_fbwd).

Guards the three risks the Q4 ablation arms introduce: (T1) a smooth-mode model silently carrying
predictor params (the arm's whole point is that none exist), (T2) the frozen arm's GRUs learning
anyway, and (T3) a ladder term failing to reach the encoder -- the only path through which any of
these objectives can shape the representation. Mirrors test_dynamics_bwd.py.
"""
from __future__ import annotations

import torch

from swm.models import WorldModel
from swm.train.losses import dynamics_loss, smoothness_loss


def make_small_model(dyn_mode: str = "fwd") -> WorldModel:
    return WorldModel(
        in_ch=1, enc_channels=[8, 16, 32, 64], kernel_size=5, z_dim=8, window=64,
        gru_hidden=16, gru_layers=1, dyn_mode=dyn_mode,
    )


def has_grad(module) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_smooth_has_no_predictor_params_and_emits_no_pred_keys():
    """T1 -- smooth = encoder+decoder only: no dynamics keys in the tree, no pred keys in forward."""
    model = make_small_model("smooth")
    assert model.dynamics is None and model.dynamics_bwd is None
    keys = set(model.state_dict().keys())
    assert not any(k.startswith("dynamics") for k in keys)
    # state_dict is exactly the fwd tree minus every dynamics.* key (encoder/decoder untouched)
    fwd_keys = set(make_small_model("fwd").state_dict().keys())
    assert keys == {k for k in fwd_keys if not k.startswith("dynamics")}

    out = model(torch.randn(3, 4, 64, 1)) # (B, S, window, 1)
    assert "pred_next" not in out and "pred_prev" not in out and "pred_roll" not in out
    assert out["mu_seq"].shape == (3, 4, 8) # (B, S, z)


def test_smoothness_loss_reaches_encoder_and_matches_first_difference():
    """T3 (smooth) -- the penalty is the consecutive-mu MSE and its gradient reaches the encoder."""
    model = make_small_model("smooth")
    out = model(torch.randn(3, 4, 64, 1))
    sl = smoothness_loss(out["mu_seq"])
    expected = (out["mu_seq"][:, 1:, :] - out["mu_seq"][:, :-1, :]).pow(2).mean()
    assert torch.allclose(sl, expected)
    sl.backward()
    assert has_grad(model.encoder)
    # A constant latent sequence is the penalty's trivial minimum (recon is what blocks it in training).
    flat = torch.ones(2, 4, 8)
    assert smoothness_loss(flat).item() == 0.0


def test_linear_fbwd_mirrors_fwd_bwd_contract_with_linear_heads():
    """T3 (linear) -- same pred/target keys, shapes and stop-grad as fwd_bwd; both maps are linear."""
    model = make_small_model("linear_fbwd")
    # single linear map per direction: no GRU parameters anywhere in the predictors
    for module in (model.dynamics, model.dynamics_bwd):
        names = [n for n, _ in module.named_parameters()]
        assert set(names) == {"head.weight", "head.bias"}
    x = torch.randn(3, 4, 64, 1)
    out = model(x)
    assert out["pred_next"].shape == (3, 3, 8) # (B, S-1, z)
    assert out["pred_prev"].shape == (3, 3, 8)
    assert out["target_next"].requires_grad is False and out["target_prev"].requires_grad is False

    # the backward term in isolation trains dynamics_bwd + encoder, never the forward map
    dynamics_loss(out["pred_prev"], out["target_prev"]).backward()
    assert has_grad(model.dynamics_bwd)
    assert has_grad(model.encoder)
    assert not has_grad(model.dynamics)


def test_frozen_fbwd_gru_params_stay_frozen_but_encoder_still_learns():
    """T2 -- frozen arm: full fwd_bwd loss gives the GRUs no grad while the encoder gets one."""
    model = make_small_model("frozen_fbwd")
    # tree is byte-identical in KEYS to fwd_bwd (checkpoints round-trip strict)
    assert set(model.state_dict().keys()) == set(make_small_model("fwd_bwd").state_dict().keys())
    assert all(not p.requires_grad for p in model.dynamics.parameters())
    assert all(not p.requires_grad for p in model.dynamics_bwd.parameters())

    before = [p.clone() for p in model.dynamics.parameters()]
    out = model(torch.randn(3, 4, 64, 1))
    loss = dynamics_loss(out["pred_next"], out["target_next"]) + dynamics_loss(out["pred_prev"], out["target_prev"])
    loss.backward()
    assert has_grad(model.encoder) # pressure reaches the representation through the input latents
    assert all(p.grad is None for p in model.dynamics.parameters())
    assert all(p.grad is None for p in model.dynamics_bwd.parameters())
    # one optimizer step over all params must leave the frozen predictors bit-identical
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.step()
    for p_before, p_after in zip(before, model.dynamics.parameters()):
        assert torch.equal(p_before, p_after)
