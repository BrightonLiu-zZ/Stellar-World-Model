from __future__ import annotations

import torch
import torch.nn.functional as F


def recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction term: mean squared error per element between decoded and input windows.
    Mean (not sum) reduction keeps the scale stable across batch and sequence length so beta
    and lambda stay interpretable.
    """
    # recon, target: (B, S, window, 1)
    return F.mse_loss(recon, target)


def kl_free_bits(
    mu: torch.Tensor, logvar: torch.Tensor, free_bits: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    KL(q(z|x) || N(0, I)) with a per-dimension free-bits floor.
    Free bits stop the optimizer from driving any latent dimension's average KL below `free_bits`
    nats, the main guard against posterior collapse. Returns the floored KL used in the loss, the
    true (unfloored) total KL for monitoring, and the per-dim mean KL used to count active units.
    """
    # mu, logvar: (B, S, z)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0) # (B, S, z)
    kl_per_dim_mean = kl_per_dim.mean(dim=(0, 1)) # (z,) average over batch and sequence
    kl_floored = torch.clamp(kl_per_dim_mean, min=free_bits) # apply the per-dim floor
    kl_loss = kl_floored.sum() # scalar, summed over latent dims
    kl_total = kl_per_dim_mean.sum() # scalar, true KL without the floor (monitoring)
    return kl_loss, kl_total, kl_per_dim_mean


def dynamics_loss(pred_next: torch.Tensor, target_next: torch.Tensor) -> torch.Tensor:
    """
    Latent-dynamics term: MSE between the GRU's predicted next latent and the encoder's actual
    next latent. The target is already stop-gradient'd in the model, so only the prediction side
    learns; this is the guard against the trivial all-latents-equal collapse.
    """
    # pred_next, target_next: (B, S-1, z)
    return F.mse_loss(pred_next, target_next)
