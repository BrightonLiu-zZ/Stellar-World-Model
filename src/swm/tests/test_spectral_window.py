"""exp07: the Hann-tapered log-PSD must be blind to window-edge samples and nothing else.

Pins the mechanism-targeted edge fix (pre-check C1): a symmetric Hann window is exactly zero at both
endpoints, so a decoder impulse at p0/p-1 buys zero spectral power under the tapered term while the
untapered (rectangular) term still responds. Also pins backward compatibility: psd_window absent or
"none" reproduces the exp00-06 loss exactly.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from swm.train.loop import additive_aux_loss
from swm.train.losses import hf_time_loss, spectral_recon_loss

B, S, W = 2, 3, 64


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    target = torch.randn(B, S, W, 1, generator=g) # (B, S, window, 1)
    recon = target + 0.1 * torch.randn(B, S, W, 1, generator=g)
    return recon, target


def test_hann_blind_to_edge_impulse() -> None:
    recon, target = _pair()
    spiked = recon.clone()
    spiked[:, :, 0, 0] += 5.0
    spiked[:, :, -1, 0] += 5.0
    base = spectral_recon_loss(recon, target, window_fn="hann")
    with_spike = spectral_recon_loss(spiked, target, window_fn="hann")
    # taper-weighted demeaning gives the endpoint samples zero weight in the mean and the signal,
    # so the edge impulse is EXACTLY invisible, not merely attenuated
    assert torch.isclose(base, with_spike, rtol=0, atol=1e-6)
    rect_base = spectral_recon_loss(recon, target, window_fn="none")
    rect_spiked = spectral_recon_loss(spiked, target, window_fn="none")
    assert (rect_spiked - rect_base).abs() > 100 * (with_spike - base).abs()


def test_hann_still_sees_interior() -> None:
    recon, target = _pair()
    spiked = recon.clone()
    spiked[:, :, W // 2, 0] += 5.0 # taper ~= 1 at mid-window
    base = spectral_recon_loss(recon, target, window_fn="hann")
    with_spike = spectral_recon_loss(spiked, target, window_fn="hann")
    assert (with_spike - base).abs() > 0.1


def test_none_reproduces_legacy() -> None:
    recon, target = _pair()
    legacy = spectral_recon_loss(recon, target)
    explicit = spectral_recon_loss(recon, target, window_fn="none")
    assert torch.equal(legacy, explicit)


def test_unknown_window_fn_raises() -> None:
    recon, target = _pair()
    with pytest.raises(ValueError, match="psd window_fn"):
        spectral_recon_loss(recon, target, window_fn="hamming")


def test_additive_aux_routes_psd_window() -> None:
    recon, target = _pair()
    cfg_hann = OmegaConf.create({"type": "combined", "weight": 0.3, "psd_normalize": False,
                                 "psd_eps": 1e-8, "hf_weight": 1.0, "psd_window": "hann",
                                 "mask_frac": 0.0, "mask_span": 8})
    expected = spectral_recon_loss(recon, target, window_fn="hann") + hf_time_loss(recon, target)
    assert torch.isclose(additive_aux_loss(recon, target, cfg_hann), expected)

    # pre-exp07 configs carry no psd_window key --> must resolve to the rectangular legacy term
    cfg_old = OmegaConf.create({"type": "combined", "weight": 0.3, "psd_normalize": False,
                                "psd_eps": 1e-8, "hf_weight": 1.0, "mask_frac": 0.0, "mask_span": 8})
    legacy = spectral_recon_loss(recon, target, window_fn="none") + hf_time_loss(recon, target)
    assert torch.isclose(additive_aux_loss(recon, target, cfg_old), legacy)
