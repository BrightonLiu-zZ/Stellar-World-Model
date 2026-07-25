from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Conv1D-VAE decoder, a mirror of the encoder.
    Maps a latent z back to a reconstructed flux window. A linear layer lifts z to the
    bottleneck feature map, then four ConvTranspose stages (kernel 4, stride 2) each
    double the time axis (64 --> 1024). The final stage emits the single flux channel raw.
    """

    def __init__(self, in_ch: int, enc_channels: list[int], z_dim: int, bottleneck_len: int) -> None:
        super().__init__()
        self.last_ch = enc_channels[-1]
        self.bottleneck_len = bottleneck_len
        self.fc = nn.Linear(z_dim, self.last_ch * bottleneck_len) # 128 --> 256*64
        rev = list(reversed(enc_channels)) # [256, 128, 64, 32]
        out_channels = rev[1:] + [in_ch] # [128, 64, 32, 1]
        stages: list[nn.Module] = []
        prev = rev[0]
        for i, ch in enumerate(out_channels):
            stages.append(nn.ConvTranspose1d(prev, ch, kernel_size=4, stride=2, padding=1)) # doubles length
            is_last = i == len(out_channels) - 1
            if not is_last:
                stages.append(nn.BatchNorm1d(ch))
                stages.append(nn.ReLU())
            prev = ch
        self.deconv = nn.Sequential(*stages)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, z)
        h = self.fc(z) # (B, last_ch*bottleneck_len)
        h = h.view(h.shape[0], self.last_ch, self.bottleneck_len) # (B, C_last, bottleneck_len)
        h = self.deconv(h) # (B, 1, window)
        return h.transpose(1, 2) # (B, window, 1)
