"""Unit tests for the Ceiling-A feature extractor (swm.eval.features).

A synthetic sinusoid must be recovered as the dominant periodogram peak, white noise must produce a
flat band spectrum with a weak dominant peak, the NaN-free contract must fail loud, and the vector
must be the fixed length and deterministic.
"""
from __future__ import annotations

import numpy as np
import pytest

from swm.eval.features import CADENCE_DAYS, FEATURE_NAMES, extract_features


def _feat(name: str, vec: np.ndarray) -> float:
    return float(vec[FEATURE_NAMES.index(name)])


def test_sinusoid_recovers_peak_frequency():
    f0 = 10.0 # cycles/day, a delta-Sct-like frequency
    n = 4096 # a typical first-segment length
    t = np.arange(n) * CADENCE_DAYS
    flux = np.sin(2 * np.pi * f0 * t)
    vec = extract_features(flux)
    recovered = 10 ** _feat("ls_f1_log10", vec)
    assert recovered == pytest.approx(f0, rel=0.05) # dominant peak within 5% of injected freq
    assert _feat("ls_p1", vec) > 0.5 # a clean tone concentrates most spectral power in peak 1


def test_white_noise_flat_bands_weak_peak():
    rng = np.random.default_rng(0)
    flux = rng.standard_normal(4096)
    vec = extract_features(flux)
    bands = np.array([_feat(f"psd_band{b}", vec) for b in range(8)])
    assert bands.std() < 1.0 # log10 band powers stay within ~1 decade of each other (no dominant band)
    assert _feat("ls_p1", vec) < 0.1 # no single bin holds much of the flat-spectrum power
    assert _feat("spectral_entropy", vec) > 0.8 # near-uniform PSD -> high entropy


def test_nan_free_contract_fails_loud():
    flux = np.ones(4096)
    flux[10] = np.nan
    with pytest.raises(AssertionError):
        extract_features(flux)


def test_vector_length_and_determinism():
    rng = np.random.default_rng(1)
    flux = rng.standard_normal(4096)
    a = extract_features(flux)
    b = extract_features(flux)
    assert len(a) == len(FEATURE_NAMES)
    assert np.array_equal(a, b) # deterministic: identical input -> identical output


def test_constant_flux_is_finite():
    vec = extract_features(np.full(4096, 3.14))
    assert np.isfinite(vec).all() # degenerate zero-variance segment must not produce NaN/inf
