"""Unit tests for the exp02 reconstruction-objective terms (swm.train.losses).

The spectral and high-pass terms must vanish on a perfect reconstruction, the log-PSD term must penalize a
low-passed reconstruction of a fast oscillation (the shortcut it exists to break), and the denoising mask
must corrupt roughly the requested fraction of each window.
"""
from __future__ import annotations

import torch

from swm.train.losses import hf_time_loss, make_keep_mask, spectral_recon_loss


def _highfreq_windows(b: int = 2, s: int = 3, window: int = 64) -> torch.Tensor:
    """A batch of identical fast sinusoid windows shaped (B, S, window, 1) for the reconstruction terms."""
    t = torch.arange(window, dtype=torch.float32)
    wave = torch.sin(2 * torch.pi * (window / 4) * t / window) # a fast oscillation across the window
    return wave.view(1, 1, window, 1).expand(b, s, window, 1).contiguous()


def test_spectral_loss_zero_on_perfect_recon():
    x = _highfreq_windows()
    assert spectral_recon_loss(x, x).item() < 1e-6 # identical spectra --> zero
    assert spectral_recon_loss(x, x, normalize=True).item() < 1e-6 # shape-only variant too


def test_hf_time_loss_zero_on_perfect_recon():
    x = _highfreq_windows()
    assert hf_time_loss(x, x).item() < 1e-6


def test_spectral_loss_penalizes_lowpass_recon():
    target = _highfreq_windows()
    lowpass = target.mean(dim=2, keepdim=True).expand_as(target) # flat recon = the low-pass shortcut
    penalty = spectral_recon_loss(lowpass, target).item()
    assert penalty > spectral_recon_loss(target, target).item() # low-passing the fast tone costs spectral loss
    assert penalty > 1.0 # log-PSD makes the missing high-frequency power a large, non-trivial penalty


def test_keep_mask_shape_and_fraction():
    n_seq, window = 200, 256
    keep = make_keep_mask(n_seq, window, mask_frac=0.25, mask_span=8, device="cpu")
    assert keep.shape == (n_seq, window)
    assert torch.all((keep == 0.0) | (keep == 1.0)) # a binary keep-mask
    corrupted_frac = float((keep == 0.0).float().mean())
    assert 0.1 < corrupted_frac < 0.35 # near 0.25; span overlaps only ever reduce the corrupted fraction
