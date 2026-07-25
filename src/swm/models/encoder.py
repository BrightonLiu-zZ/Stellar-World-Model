from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Conv1D-VAE encoder.
    Maps one flux window to a Gaussian posterior (mu, logvar) over the latent z.
    Four Conv-BN-ReLU-MaxPool stages each halve the time axis (1024 --> 64), then a
    linear head produces mu and logvar. Architecture is locked for the A/B/C ablation.
    """

    def __init__(self, in_ch: int, enc_channels: list[int], kernel_size: int, z_dim: int, window: int) -> None:
        super().__init__()
        pad = kernel_size // 2 # 'same' length before pooling
        stages: list[nn.Module] = []
        prev = in_ch
        for ch in enc_channels:
            stages.append(nn.Conv1d(prev, ch, kernel_size, padding=pad))
            stages.append(nn.BatchNorm1d(ch))
            stages.append(nn.ReLU())
            stages.append(nn.MaxPool1d(2)) # halve length
            prev = ch
        self.conv = nn.Sequential(*stages)
        self.bottleneck_len = window // (2 ** len(enc_channels)) # 1024 / 16 = 64
        self.flat_dim = enc_channels[-1] * self.bottleneck_len # 256 * 64 = 16384
        self.fc_mu = nn.Linear(self.flat_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, window, 1)
        h = x.transpose(1, 2) # (B, 1, window) -- Conv1d wants channels-first
        h = self.conv(h) # (B, C_last, bottleneck_len)
        h = h.flatten(1) # (B, flat_dim)
        mu = self.fc_mu(h) # (B, z)
        logvar = self.fc_logvar(h) # (B, z)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0) # keep exp(logvar) finite under fp16 AMP (no inf -> no nan)
        return mu, logvar

