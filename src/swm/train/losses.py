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


def spectral_recon_loss(
    recon: torch.Tensor, target: torch.Tensor, normalize: bool = False, eps: float = 1e-8
) -> torch.Tensor:
    """
    Log-power-spectrum reconstruction term (exp02): MSE between the log-PSD of the decoded and the input
    window along the time axis.
    The rFFT runs in float32 because half-precision rFFT is unsupported and its |.|**2 power overflows.
    Each window is mean-subtracted first, dropping the DC bin so the term describes oscillatory content only,
    matching swm.eval.features._periodogram.
    The log compresses the multi-decade power range so low-amplitude high-frequency bins receive gradient
    comparable to the dominant low-frequency bins, which is what breaks the time-MSE low-pass shortcut.
    normalize=False keeps raw power (pulsation amplitude vs the noise floor is retained; flux is already
    per-segment MAD-normalized so amplitudes are comparable across stars); normalize=True rescales each
    window's power to sum one, so the term describes spectral shape only.
    """
    # recon, target: (B, S, window, 1)
    r = recon.squeeze(-1).float() # (B, S, window); fp32 for a stable, non-overflowing FFT
    t = target.squeeze(-1).float() # (B, S, window)
    r = r - r.mean(dim=-1, keepdim=True) # subtract per-window mean --> drops the DC bin
    t = t - t.mean(dim=-1, keepdim=True)
    pr = torch.fft.rfft(r, dim=-1).abs() ** 2 # (B, S, window//2 + 1) power spectrum
    pt = torch.fft.rfft(t, dim=-1).abs() ** 2
    if normalize:
        pr = pr / (pr.sum(dim=-1, keepdim=True) + eps) # spectral shape, amplitude-invariant
        pt = pt / (pt.sum(dim=-1, keepdim=True) + eps)
    return F.mse_loss(torch.log(pr + eps), torch.log(pt + eps))


def hf_time_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    High-pass time-domain reconstruction term (exp02): MSE on the first difference of the flux along time.
    Differencing is a cheap high-pass filter, so this penalizes errors in fast structure while leaving the
    slow envelope to the plain time-MSE.
    It is expected to help pulsating and eclipse edges but, unlike the full-spectrum log-PSD term, it
    de-emphasizes the slow modulation the rotation task depends on.
    """
    # recon, target: (B, S, window, 1)
    dr = recon[:, :, 1:, :] - recon[:, :, :-1, :] # (B, S, window-1, 1) discrete high-pass
    dt = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.mse_loss(dr, dt)


def make_keep_mask(
    n_seq: int, window: int, mask_frac: float, mask_span: int, device: str
) -> torch.Tensor:
    """
    Build a per-window keep-mask for the masked denoising variant (exp02).
    Zeroes contiguous spans covering about mask_frac of each window's timesteps, in blocks of mask_span
    cadences at random positions, so the encoder must infer the corrupted structure from surrounding
    context. Returns a (n_seq, window) float mask (1 = keep, 0 = corrupt); the caller multiplies the input
    by it and reconstructs the clean target.
    """
    keep = torch.ones(n_seq, window, device=device) # (n_seq, window)
    n_span = max(1, int(round(mask_frac * window / max(mask_span, 1)))) # spans needed to hit the target fraction
    starts = torch.randint(0, max(1, window - mask_span + 1), (n_seq, n_span), device=device) # (n_seq, n_span)
    for j in range(mask_span):
        idx = (starts + j).clamp(max=window - 1) # (n_seq, n_span) columns to zero for this offset
        keep.scatter_(1, idx, 0.0) # set the masked timesteps to 0
    return keep


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
