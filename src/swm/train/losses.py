from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=8)
def _dpss_tapers(window: int, nw: float, k: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Cached DPSS (Slepian) taper family for the multitaper log-PSD term (exp09).
    Returns (k, window). scipy computes them once per (shape, NW, K, device, dtype); the cache matters
    because this sits inside the training inner loop.
    NW is the time-bandwidth product and K the number of tapers, with the standard K <= 2*NW - 1 so
    every retained taper is well concentrated in the NW/(window) band.
    """
    from scipy.signal.windows import dpss as _scipy_dpss

    tapers = _scipy_dpss(window, NW=nw, Kmax=k, sym=True) # (k, window) float64
    # .copy(): scipy returns a reversed view with a negative stride, which torch.tensor rejects
    return torch.tensor(tapers.copy(), device=device, dtype=dtype) # (k, window)


def recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction term: mean squared error per element between decoded and input windows.
    Mean (not sum) reduction keeps the scale stable across batch and sequence length so beta
    and lambda stay interpretable.
    """
    # recon, target: (B, S, window, 1)
    return F.mse_loss(recon, target)


def spectral_recon_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-8,
    window_fn: str = "none",
    taper_nw: float = 4.0,
    taper_k: int = 7,
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
    window_fn="hann" applies a symmetric Hann taper after demeaning (Welch-style periodogram). The
    symmetric window is exactly zero at both endpoints, so the edge samples contribute nothing to the
    spectrum: a decoder cannot buy spectral power with a window-edge impulse (exp07 edge fix; the
    untapered rectangular FFT rewards exactly that, see exp07 pre-check C1).
    window_fn="dpss" averages the log-PSD MSE over a DPSS (Slepian) taper family of `taper_k` tapers at
    time-bandwidth product `taper_nw` (exp09). Motivation: Hann did NOT remove the impulse, it RELOCATED
    it (F27) -- a single taper always has one position of maximal weight, and that position is the
    cheapest place for the decoder to plant broadband power. A taper family has no single such position,
    so the exploit has no fixed address. Note this ATTENUATES rather than abolishes edge sensitivity
    (taper 6's endpoint weight is 0.065, against Hann's exact 0) -- spreading the cost across positions
    is the point, not zeroing one of them.
    """
    # recon, target: (B, S, window, 1)
    r = recon.squeeze(-1).float() # (B, S, window); fp32 for a stable, non-overflowing FFT
    t = target.squeeze(-1).float() # (B, S, window)
    if window_fn == "hann":
        taper = torch.hann_window(r.shape[-1], periodic=False, device=r.device, dtype=r.dtype) # (window,), 0 at both ends
        # taper-WEIGHTED demeaning: plain demeaning would let an edge impulse shift the mean and leak
        # back in as a taper-shaped broadband component; weighting by the taper gives edge samples zero
        # weight in the mean AND in the signal, and still zeroes the DC bin exactly
        # (sum(taper*(x - m)) = 0 by construction).
        r = (r - (r * taper).sum(dim=-1, keepdim=True) / taper.sum()) * taper
        t = (t - (t * taper).sum(dim=-1, keepdim=True) / taper.sum()) * taper
    elif window_fn == "dpss":
        tapers = _dpss_tapers(r.shape[-1], float(taper_nw), int(taper_k), str(r.device), r.dtype) # (k, window)
        # Demean with TAPER 0 only. Odd-order DPSS tapers are antisymmetric and sum to 0, so the
        # per-taper weighted demeaning used for Hann would divide by zero. Taper 0 is strictly positive
        # with near-zero endpoints (7e-6), so it carries the same anti-leak property: an edge impulse
        # cannot shift the mean and re-enter as a taper-shaped broadband component, and the DC bin of
        # the taper-0 channel is exactly zeroed.
        t0 = tapers[0] # (window,)
        r = r - (r * t0).sum(dim=-1, keepdim=True) / t0.sum()
        t = t - (t * t0).sum(dim=-1, keepdim=True) / t0.sum()
        r = r.unsqueeze(-2) * tapers # (B, S, k, window) one tapered copy per taper
        t = t.unsqueeze(-2) * tapers # (B, S, k, window)
    elif window_fn == "none":
        r = r - r.mean(dim=-1, keepdim=True) # subtract per-window mean --> drops the DC bin
        t = t - t.mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown psd window_fn {window_fn!r}; expected 'none', 'hann' or 'dpss'")
    pr = torch.fft.rfft(r, dim=-1).abs() ** 2 # (B, S, [k,] window//2 + 1) power spectrum
    pt = torch.fft.rfft(t, dim=-1).abs() ** 2
    if normalize:
        pr = pr / (pr.sum(dim=-1, keepdim=True) + eps) # spectral shape, amplitude-invariant
        pt = pt / (pt.sum(dim=-1, keepdim=True) + eps)
    return F.mse_loss(torch.log(pr + eps), torch.log(pt + eps))


def impulse_penalty_loss(recon: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    exp09 impulse penalty: excess kurtosis of the reconstruction residual, per window.

    Targets the exploit that is actually IN the objective. Every loss term is computed inside one
    256-cadence window (no term spans a window boundary), so what the decoder buys is a single
    within-window IMPULSE with a broadband spectrum -- the visible "comb" is only an artifact of laying
    decoded windows end to end for a plot. Penalising spectral periodicity instead would attack real
    eclipsing-binary and overtone-pulsator harmonic combs, i.e. our strongest task.

    Kurtosis is the natural impulse statistic: a residual with one large excursion has heavy tails and
    high fourth moment, while a residual of ordinary broadband error is near-Gaussian (kurtosis 3). The
    relu keeps the term one-sided -- there is no reward for making the residual sub-Gaussian -- and the
    statistic is frequency-agnostic by construction, so nothing about the exploit's rate is hard-coded.
    Kurtosis is also scale-free, so the penalty cannot be triggered or evaded by residual magnitude
    alone; eps is 1e-12 rather than the usual 1e-8 because at 1e-8 it perturbs the ratio for
    well-reconstructed windows (m2^2 approaching eps) and breaks that scale invariance.
    """
    # recon, target: (B, S, window, 1)
    d = (recon - target).squeeze(-1).float() # (B, S, window) residual; fp32 for a stable 4th moment
    d = d - d.mean(dim=-1, keepdim=True) # per-window centring; kurtosis is defined about the mean
    m2 = (d**2).mean(dim=-1) # (B, S)
    m4 = (d**4).mean(dim=-1) # (B, S)
    kurtosis = m4 / (m2**2 + eps) # (B, S); 3.0 for a Gaussian residual
    return torch.relu(kurtosis - 3.0).mean() # one-sided: only heavy tails are penalised


def hf_noise_min_bin(window: int, min_uhz: float, cadence_s: float = 120.0) -> int:
    """
    First rFFT bin at or above `min_uhz`, for the exp09 band-limited noise augmentation.
    TESS 2-min cadence gives df = 1e6 / (120 * window) uHz -- 32.55 uHz at window 256, so a 1 mHz floor
    starts at bin 31. Everything below that bin stays clean, which is what keeps the 65-260 uHz pulsator
    band provably untouched by the augmentation.
    """
    df_uhz = 1.0e6 / (cadence_s * window)
    return int(math.ceil(min_uhz / df_uhz))


def hf_noise_augment(x: torch.Tensor, sigma: float, min_bin: int) -> torch.Tensor:
    """
    exp09 augmentation: add band-limited high-frequency noise to a batch of windows.

    Synthesised in the frequency domain -- white noise, rFFT, zero every bin below `min_bin`, back to
    time -- so the perturbation is confined to frequencies above the cut by construction rather than by
    a filter's rolloff. Rescaled to `sigma` in MAD units (flux is already per-segment MAD-normalised).

    Applied to the SAMPLE, i.e. to input and reconstruction target together, so the model still
    reconstructs what it sees; adding noise to the input alone would make this a denoising autoencoder,
    a different intervention. TRAIN SPLIT ONLY -- noising validation would make val/recon incomparable
    across cells, and val/recon is both the checkpoint selector and the x-axis of the G9-select gate.
    """
    # x: (B, S, window, 1)
    window = x.shape[-2]
    noise = torch.randn(x.shape[:-1], device=x.device, dtype=torch.float32) # (B, S, window)
    spectrum = torch.fft.rfft(noise, dim=-1) # (B, S, window//2 + 1)
    spectrum[..., :min_bin] = 0.0 # kill everything below the cut, including DC
    hp = torch.fft.irfft(spectrum, n=window, dim=-1) # (B, S, window) band-limited noise
    hp = hp / (hp.std(dim=-1, keepdim=True) + 1e-8) * sigma # per-window rescale to sigma MAD units
    return x + hp.unsqueeze(-1).to(x.dtype) # (B, S, window, 1)


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


def decorr_loss(mu_seq: torch.Tensor, feats: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    exp10 E2: mean squared Pearson correlation between every latent dim and every engineered feature.

    The forensics measured the features --> mu map as nearly LINEAR (F-B: gbm minus ridge = 0.030
    against a 0.15 bar), so a linear instrument is the indicated one and HSIC/adversarial machinery is
    not. This is that instrument: pressure to move mu's content OFF the feature manifold, which is what
    the fusion spine needs if the GBM is to gain from mu rather than merely use it (F-F).

    CORRELATION, never covariance (D-E10.11, the dim-51 hazard): mu's variance is 84-86% concentrated in
    one dimension, so a covariance penalty would spend nearly all its pressure there and act as a
    disguised variance penalty on a single dim. Per-dim standardization makes every dim cost the same.

    The statistic is computed on MU, not on the sampled z. Everything the penalty is meant to move --
    the probe features, the F1 fusion columns, the G10-mech predictability read -- is computed from mu,
    and sampling noise would only attenuate the estimate. Features arrive already standardized on the
    train split (their columns are re-centred here anyway, since a batch mean is not the global mean).
    """
    # mu_seq: (B, S, z); feats: (B, n_feat) -- one vector per star, shared by that star's S windows
    bsz, seq_len, z_dim = mu_seq.shape
    z = mu_seq.reshape(bsz * seq_len, z_dim).float() # (B*S, z); fp32 for a stable correlation under AMP
    f = feats.unsqueeze(1).expand(bsz, seq_len, feats.shape[-1]) # (B, S, n_feat) -- broadcast, no copy
    f = f.reshape(bsz * seq_len, feats.shape[-1]).float() # (B*S, n_feat)
    n = z.shape[0]
    z = z - z.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, unbiased=False, keepdim=True) + eps) # per-dim batch standardization
    f = f - f.mean(dim=0, keepdim=True)
    f = f / (f.std(dim=0, unbiased=False, keepdim=True) + eps)
    corr = (z.t() @ f) / n # (z, n_feat) Pearson correlations, both sides already unit-variance
    return (corr**2).mean()


def smoothness_loss(mu_seq: torch.Tensor) -> torch.Tensor:
    """
    exp08 dynamics-free temporal prior: MSE between consecutive latents (an SFA-style slowness
    penalty). No stop-gradient on either side -- the symmetric pull has the same encoder gradients
    as the fwd+bwd stop-grad pair, and unlike the GRU term there is no prediction path to protect.
    Same MSE reduction over (B, S-1, z) as dynamics_loss, so lambda-dose bookkeeping is comparable.
    """
    # mu_seq: (B, S, z)
    return F.mse_loss(mu_seq[:, 1:, :], mu_seq[:, :-1, :])
