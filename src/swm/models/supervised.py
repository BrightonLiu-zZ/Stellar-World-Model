"""The two supervised-baseline architectures for roadmap rows C1 and C2.

Both are the SAME network except for the trunk, which is the point: C1 minus C2 then reads as "the
convolutional inductive bias" rather than as an uncontrolled architecture-and-input change.

    trunk  -->  Linear(flat_dim, z_dim)  -->  Dropout  -->  Linear(z_dim, 1)
                ^ `fc_mu`, taken verbatim from the shipped Encoder in the conv arm

C1's trunk IS swm.models.encoder.Encoder's conv stack -- the module objects, not a copy -- so the
architecture cannot drift from the encoder the SSL arm uses. Its `fc_logvar` is dropped rather than
carried dead: this network has no posterior, no sampling and no KL, only a deterministic 128-d
bottleneck.

Why a 128-d bottleneck costs no ceiling: for a scalar output, trunk -> Linear(4096,128) ->
Linear(128,1) spans the same functions as trunk -> Linear(4096,1), because any 1x4096 linear
functional factors through 128 dimensions. The bottleneck buys a state_dict that mirrors the SSL
encoder without buying a handicap.

These are EXTERNAL BASELINES under ADR-0012 decision 3, never the probe.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from swm.models.encoder import Encoder


class SupervisedNet(nn.Module):
    """Window encoder + task head, scored per star by the mean of its windows' outputs.

    The bag mean lives in `pool_bags` rather than in the training loop because it is part of the
    model's definition of a star score (roadmap D20: star score = mean over window logits), and the
    training loss and the reported metric must both go through it.
    """

    def __init__(self, trunk: str, window: int, z_dim: int, dropout: float,
                 enc_channels: list[int] | None = None, kernel_size: int = 5,
                 hidden: list[int] | None = None) -> None:
        super().__init__()
        self.trunk_kind = trunk
        if trunk == "conv":
            assert enc_channels is not None, "conv trunk needs enc_channels"
            encoder = Encoder(1, list(enc_channels), kernel_size, z_dim, window)
            self.trunk = encoder.conv # the shipped Encoder's Conv-BN-ReLU-MaxPool stack, verbatim
            self.to_z = encoder.fc_mu # (flat_dim -> z_dim); fc_logvar is dropped, there is no posterior
            flat_dim = encoder.flat_dim
        elif trunk == "dense":
            assert hidden is not None, "dense trunk needs hidden sizes"
            layers: list[nn.Module] = []
            prev = window
            for width in hidden:
                layers.append(nn.Linear(prev, width))
                layers.append(nn.ReLU())
                prev = width
            self.trunk = nn.Sequential(*layers)
            self.to_z = nn.Linear(prev, z_dim)
            flat_dim = prev
        else:
            raise ValueError(f"unknown trunk {trunk!r}; expected 'conv' or 'dense'")
        self.flat_dim = flat_dim
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(z_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_windows, window, 1)
        if self.trunk_kind == "conv":
            h = x.transpose(1, 2) # (n_windows, 1, window) -- Conv1d wants channels-first
            h = self.trunk(h) # (n_windows, C_last, bottleneck_len)
            h = h.flatten(1) # (n_windows, flat_dim)
        else:
            h = self.trunk(x.squeeze(-1)) # (n_windows, hidden[-1])
        z = self.to_z(h) # (n_windows, z_dim)
        return self.head(self.dropout(z)).squeeze(-1) # (n_windows,)


def pool_bags(window_out: torch.Tensor, star_index: torch.Tensor, n_stars: int) -> torch.Tensor:
    """Mean of a star's window outputs -- the star score of roadmap D20, computed inside the graph.

    Kept differentiable on purpose. Broadcasting the star label onto every window instead would force
    the model to call every window of a positive star positive, which on a localized task is a
    manufactured label noise; pooling here lets a featureless window score near zero while the one
    informative window carries the bag.
    """
    totals = torch.zeros(n_stars, dtype=window_out.dtype, device=window_out.device)
    counts = torch.zeros(n_stars, dtype=window_out.dtype, device=window_out.device)
    totals = totals.index_add(0, star_index, window_out) # sum each bag's window outputs
    counts = counts.index_add(0, star_index, torch.ones_like(window_out))
    return totals / counts
