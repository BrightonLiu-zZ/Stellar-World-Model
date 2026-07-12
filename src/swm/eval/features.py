"""Ceiling-A feature extractor: T'DA-informed per-star features from first-segment flux.

The skyline suite (plan 2026-07-11) needs a hand-engineered representation to bound the headroom
above the current 0.77 PR-AUC under the protocol-matched linear probe. We reduce each star's
first-segment flux to a fixed ~25-feature vector modelled on the T'DA classification pipeline
(Audenaert+2021), whose ensemble of engineered periodogram/entropy/robust-stat features reached
94.9% on a Kepler class set close to ours (delta Sct, RR Lyrae, EB, rotation, aperiodic): the top
Lomb-Scargle peak structure, log-spaced power-spectral band shape, spectral entropy, and robust
time-domain scatter statistics.

A packed first segment is contiguous, gap-guarded, and uniformly sampled at the TESS 2-min cadence,
so the Lomb-Scargle periodogram of an evenly-sampled series reduces exactly to the FFT periodogram.
We compute it with numpy directly (astropy is absent from the swm CUDA env and buys nothing on a
uniform grid). The extractor is deterministic and NaN-free by contract.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

# TESS 2-min cadence expressed in days; the packed grid is uniform at this spacing.
CADENCE_DAYS = 2.0 / 1440.0

# Physical band edges (cycles/day), log-spaced from ~3-day periods to just under the 2-min Nyquist
# (360 c/d). Fixed edges keep the band-power features comparable across stars of differing length.
_N_BANDS = 8
_BAND_EDGES = np.logspace(np.log10(0.3), np.log10(360.0), _N_BANDS + 1)

# Fixed feature order. Every downstream table (A1/A2/B1) indexes columns by this list.
FEATURE_NAMES: list[str] = [
    "ls_f1_log10", # log10 dominant periodogram peak frequency (c/d)
    "ls_f2_log10",
    "ls_f3_log10",
    "ls_p1", # normalized power of peak 1 (fraction of total periodogram power)
    "ls_p2",
    "ls_p3",
    "ls_p2_p1", # peak-2 / peak-1 power ratio
    "ls_p3_p1",
    "ls_harmonic_ratio", # power near 2*f1 relative to peak-1 power
    "psd_band0", # log10 fractional power in each of 8 log-spaced frequency bands
    "psd_band1",
    "psd_band2",
    "psd_band3",
    "psd_band4",
    "psd_band5",
    "psd_band6",
    "psd_band7",
    "spectral_entropy", # Shannon entropy of the normalized PSD, in [0, 1]
    "spectral_centroid_log10", # log10 power-weighted mean frequency (c/d)
    "skew",
    "kurtosis",
    "p2p_scatter_ratio", # std(diff(flux)) / std(flux); high-frequency roughness
    "depth_5_95", # 95th minus 5th flux percentile (robust peak-to-peak)
    "mad", # median absolute deviation
    "iqr", # inter-quartile range
]

_EPS = 1e-12


def _periodogram(flux: np.ndarray, cadence_days: float) -> tuple[np.ndarray, np.ndarray]:
    """
    FFT periodogram of the mean-subtracted flux on the uniform cadence grid.
    Returns the non-DC frequencies (cycles/day) and their power, normalized to sum to one so every
    spectral feature describes spectral SHAPE and is invariant to the per-segment MAD flux scale.
    On a degenerate (constant or zero-power) segment the returned power is a flat distribution.
    """
    y = flux - flux.mean()
    freqs = np.fft.rfftfreq(len(flux), d=cadence_days) # (n//2 + 1,) in c/d
    power = np.abs(np.fft.rfft(y)) ** 2
    freqs = freqs[1:] # drop the DC bin
    power = power[1:]
    total = power.sum()
    if total <= _EPS:
        power = np.ones_like(power) / max(len(power), 1)
    else:
        power = power / total
    return freqs, power


def _top_peaks(freqs: np.ndarray, power: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the frequencies and (normalized) powers of the k strongest periodogram peaks.
    Uses local maxima (find_peaks) so a single broad feature is not counted k times; if fewer than k
    true peaks exist (e.g. white noise), pads with the globally strongest remaining bins. Frequencies
    and powers are returned in descending-power order.
    """
    peak_idx, _ = find_peaks(power) # indices of local maxima only
    if len(peak_idx) < k:
        order = np.argsort(power)[::-1] # fall back to strongest bins overall
        peak_idx = order[: max(k, len(peak_idx))]
    peak_idx = peak_idx[np.argsort(power[peak_idx])[::-1]] # strongest first
    peak_idx = peak_idx[:k]
    return freqs[peak_idx], power[peak_idx]


def extract_features(flux: np.ndarray, cadence_days: float = CADENCE_DAYS) -> np.ndarray:
    """
    Reduce one star's first-segment flux to the fixed FEATURE_NAMES vector.
    Computes the FFT periodogram (Lomb-Scargle on a uniform grid), its top-3 peak structure and
    first-harmonic ratio, 8 log-spaced band powers, spectral entropy/centroid, and robust
    time-domain scatter statistics. Input must be a 1-D NaN-free flux array (fail-loud contract);
    the output is deterministic and ordered to match FEATURE_NAMES.
    """
    flux = np.asarray(flux, dtype=np.float64).reshape(-1)
    assert flux.size >= 8, f"first-segment flux too short for spectral features: {flux.size}"
    assert np.isfinite(flux).all(), "flux must be NaN-free (discard-NaN contract upstream)"

    freqs, power = _periodogram(flux, cadence_days)
    peak_f, peak_p = _top_peaks(freqs, power, 3)

    # Peak structure. Log10 frequencies linearise the multi-decade pulsation range for the linear probe.
    f1, f2, f3 = peak_f
    p1, p2, p3 = peak_p
    p1 = max(p1, _EPS)
    harmonic_freq = 2.0 * f1
    harmonic_bin = int(np.argmin(np.abs(freqs - harmonic_freq)))
    harmonic_ratio = power[harmonic_bin] / p1 if harmonic_freq < freqs[-1] else 0.0

    # Log-spaced band powers: fractional power summed in each fixed frequency band.
    band_powers = np.zeros(_N_BANDS)
    for b in range(_N_BANDS):
        in_band = (freqs >= _BAND_EDGES[b]) & (freqs < _BAND_EDGES[b + 1])
        band_powers[b] = np.log10(power[in_band].sum() + _EPS)

    # Spectral shape.
    entropy = float(-np.sum(power * np.log(power + _EPS)) / np.log(len(power)))
    centroid = float(np.sum(freqs * power))
    centroid_log10 = float(np.log10(centroid + _EPS))

    # Robust time-domain scatter.
    std = float(np.std(flux))
    skew = float(stats.skew(flux)) if std > _EPS else 0.0
    kurtosis = float(stats.kurtosis(flux)) if std > _EPS else 0.0
    p2p_ratio = float(np.std(np.diff(flux)) / std) if std > _EPS else 0.0
    lo, hi = np.percentile(flux, [5, 95])
    depth = float(hi - lo)
    mad = float(np.median(np.abs(flux - np.median(flux))))
    q25, q75 = np.percentile(flux, [25, 75])
    iqr = float(q75 - q25)

    feats = [
        float(np.log10(max(f1, _EPS))),
        float(np.log10(max(f2, _EPS))),
        float(np.log10(max(f3, _EPS))),
        float(p1),
        float(p2),
        float(p3),
        float(p2 / p1),
        float(p3 / p1),
        float(harmonic_ratio),
    ]
    for b in range(_N_BANDS):
        feats.append(float(band_powers[b]))
    feats.extend([entropy, centroid_log10, skew, kurtosis, p2p_ratio, depth, mad, iqr])
    return np.asarray(feats, dtype=np.float64)
