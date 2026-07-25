from __future__ import annotations

import torch
import torch.nn as nn

from swm.models.decoder import Decoder
from swm.models.dynamics import Dynamics
from swm.models.encoder import Encoder


class WorldModel(nn.Module):
    """
    Full Conv1D-VAE + GRU world model used for every variant in the A/B/C ablation.
    Variants differ only in the dynamics-loss weight lambda applied to this module's
    `pred_next` vs `target_next` outputs; the module itself is identical across variants.
    forward() consumes a batch of sequences and returns everything the loss needs.
    encode_mu() is the deterministic eval path used by the linear probe.

    dyn_mode selects the latent-dynamics objective (exp05):
      "fwd"       -- one-step forward prediction only (the exp00-04 model, unchanged).
      "fwd_bwd"   -- also predict the time-reversed sequence with a SEPARATE backward GRU
                     (`dynamics_bwd`), so the latent must be predictable in both time directions.
      "multistep" -- free-running rollout: from z_1 only, roll `self.dynamics` forward S-1 steps
                     feeding predictions back (`pred_roll`/`target_roll`). Reuses the SAME forward
                     GRU (no new params), so its state_dict equals the fwd tree.
    The backward module is instantiated ONLY in fwd_bwd mode; fwd AND multistep leave the state_dict
    byte-for-byte the exp00-04 tree, so old checkpoints load with strict=True (the free mu-reuse for
    exp05's fwd cells depends on this).
    """

    def __init__(
        self,
        in_ch: int,
        enc_channels: list[int],
        kernel_size: int,
        z_dim: int,
        window: int,
        gru_hidden: int,
        gru_layers: int,
        dyn_mode: str = "fwd",
    ) -> None:
        super().__init__()
        assert dyn_mode in ("fwd", "fwd_bwd", "multistep"), f"unknown dyn_mode {dyn_mode!r}"
        self.dyn_mode = dyn_mode
        self.encoder = Encoder(in_ch, enc_channels, kernel_size, z_dim, window)
        self.decoder = Decoder(in_ch, enc_channels, z_dim, self.encoder.bottleneck_len)
        self.dynamics = Dynamics(z_dim, gru_hidden, gru_layers)
        # Instantiate the backward predictor only when needed; see class docstring on strict-load reuse.
        self.dynamics_bwd = Dynamics(z_dim, gru_hidden, gru_layers) if dyn_mode == "fwd_bwd" else None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Draw z ~ N(mu, sigma^2) with the reparameterization trick so gradients flow through mu/logvar.
        At eval time the probe uses mu directly instead of a sampled z.
        """
        std = torch.exp(0.5 * logvar) # (.., z)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        # x_seq: (B, S, window, 1)
        bsz, seq_len = x_seq.shape[0], x_seq.shape[1]
        flat = x_seq.reshape(bsz * seq_len, x_seq.shape[2], x_seq.shape[3]) # (B*S, window, 1)
        mu, logvar = self.encoder(flat) # (B*S, z), (B*S, z)
        z = self.reparameterize(mu, logvar) # (B*S, z)
        recon = self.decoder(z) # (B*S, window, 1)

        recon_seq = recon.reshape(bsz, seq_len, recon.shape[1], recon.shape[2]) # (B, S, window, 1)
        z_dim = mu.shape[1]
        mu_seq = mu.reshape(bsz, seq_len, z_dim) # (B, S, z)
        logvar_seq = logvar.reshape(bsz, seq_len, z_dim) # (B, S, z)

        pred_next = self.dynamics(mu_seq[:, :-1, :]) # (B, S-1, z) -- predicts steps 2..S
        target_next = mu_seq[:, 1:, :].detach() # (B, S-1, z) -- STOP-GRADIENT on the target

        out = {
            "recon": recon_seq,
            "mu_seq": mu_seq,
            "logvar_seq": logvar_seq,
            "pred_next": pred_next,
            "target_next": target_next,
        }

        if self.dynamics_bwd is not None:
            # Reverse-time one-step prediction (exp05): time-mirror of the forward path. Feed the
            # flipped latent sequence z_S..z_2 to a SEPARATE GRU and predict z_{S-1}..z_1. The input
            # side is NOT detached (backward loss flows into the encoder via the input latents, as
            # forward does); the target IS stop-gradient'd, so only the prediction path learns.
            rev = torch.flip(mu_seq, dims=[1]) # (B, S, z) -- z_S, z_{S-1}, ..., z_1
            out["pred_prev"] = self.dynamics_bwd(rev[:, :-1, :]) # (B, S-1, z) -- predicts z_{S-1}..z_1
            out["target_prev"] = rev[:, 1:, :].detach() # (B, S-1, z) -- STOP-GRADIENT, same reversed order

        if self.dyn_mode == "multistep":
            # Free-running rollout (exp05): from the real z_1 only, roll the SAME forward GRU S-1 steps
            # feeding predictions back, and match z_2..z_S. Errors compound, so this is the hard
            # "learned physics" objective. pred_next (teacher-forced one-step) is still emitted above so
            # the one-step `dyn` stays diagnosable; the training loss uses pred_roll (see loop.py).
            out["pred_roll"] = self.dynamics.rollout(mu_seq[:, 0, :], seq_len - 1) # (B, S-1, z) -- z_2..z_S
            out["target_roll"] = mu_seq[:, 1:, :].detach() # (B, S-1, z) -- STOP-GRADIENT

        return out

    @torch.no_grad()
    def encode_mu(self, windows: torch.Tensor) -> torch.Tensor:
        """
        Deterministic eval encoding: return posterior means only (no sampling) for a batch of windows.
        The probe mean-pools these over all windows of a segment, then over a star's segments.
        """
        mu, _ = self.encoder(windows) # (B, window, 1) --> (B, z)
        return mu
