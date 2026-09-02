from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Conv1D-VAE decoder, a mirror of the encoder.
    Maps a latent z back to a reconstructed flux window. A linear layer lifts z to the
    bottleneck feature map, then four ConvTranspose stages (kernel 4, stride 2) each
    double the time axis (64 --> 1024). The final stage emits the single flux channel raw.

    cond_dim (exp10 E1) widens ONLY that first linear layer, so the decoder can read the star's
    standardized engineered features alongside z: content the features already carry no longer has to
    be paid for out of the latent, and the encoder's capacity is free to move elsewhere. The conv
    stack is untouched (the CLAUDE.md architecture lock; D-E10.7 rules the fc width a hyperparameter),
    and cond_dim=0 leaves nn.Linear(z_dim, ...) exactly as it was for exp00-09.
    """

    def __init__(self, in_ch: int, enc_channels: list[int], z_dim: int, bottleneck_len: int,
                 cond_dim: int = 0) -> None:
        super().__init__()
        self.last_ch = enc_channels[-1]
        self.bottleneck_len = bottleneck_len
        self.cond_dim = cond_dim
        self.fc = nn.Linear(z_dim + cond_dim, self.last_ch * bottleneck_len) # 128 (+25) --> 256*64
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

    def forward(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # z: (B, z); cond: (B, cond_dim) standardized per-star features, or None when cond_dim == 0
        if self.cond_dim > 0:
            assert cond is not None, "decoder cond_dim > 0 but no conditioning vector was passed"
            assert cond.shape[-1] == self.cond_dim, f"cond width {cond.shape[-1]} != cond_dim {self.cond_dim}"
            z = torch.cat([z, cond.to(z.dtype)], dim=-1) # (B, z + cond_dim)
        h = self.fc(z) # (B, last_ch*bottleneck_len)
        h = h.view(h.shape[0], self.last_ch, self.bottleneck_len) # (B, C_last, bottleneck_len)
        h = self.deconv(h) # (B, 1, window)
        return h.transpose(1, 2) # (B, window, 1)
