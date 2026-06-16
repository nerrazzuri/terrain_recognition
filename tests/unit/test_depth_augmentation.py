"""Unit tests for depth augmentation (P6-M3-T1). Pure numpy, no torch.

Augmentations simulate real depth-sensor degradation (roadmap §10.6) so the CReF policy is
robust to it. Tests check shape preservation, clipping bounds, dropout fraction, and seeded
reproducibility.
"""
import numpy as np
import pytest

from x2_locomotion.cref import depth_augmentation as da


def _depth(h=48, w=64, val=2.0):
    return np.full((h, w), val, dtype=float)


def test_dropout_zeros_some_pixels():
    rng = np.random.default_rng(0)
    out = da.pixel_dropout(_depth(), prob=0.3, rng=rng)
    assert out.shape == (48, 64)
    frac = np.mean(out == 0.0)
    assert 0.2 < frac < 0.4  # ~30%


def test_dropout_zero_prob_is_identity():
    d = _depth()
    out = da.pixel_dropout(d, prob=0.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(out, d)


def test_near_far_clipping():
    d = np.linspace(0.0, 10.0, 100).reshape(10, 10)
    out = da.clip_range(d, near=0.5, far=5.0)
    assert out.min() >= 0.5 and out.max() <= 5.0


def test_additive_noise_preserves_shape_and_is_bounded():
    rng = np.random.default_rng(1)
    d = _depth()
    out = da.additive_noise(d, sigma=0.05, rng=rng)
    assert out.shape == d.shape
    assert np.abs(out - d).max() < 1.0  # noise is small


def test_missing_pixels_become_nan():
    rng = np.random.default_rng(2)
    out = da.missing_pixels(_depth(), prob=0.2, rng=rng)
    assert np.isnan(out).any()


def test_pipeline_is_seeded_reproducible():
    cfg = {"dropout_prob": 0.2, "noise_sigma": 0.03, "near": 0.3, "far": 6.0}
    a = da.augment(_depth(), cfg, rng=np.random.default_rng(42))
    b = da.augment(_depth(), cfg, rng=np.random.default_rng(42))
    np.testing.assert_allclose(np.nan_to_num(a), np.nan_to_num(b))


def test_pipeline_preserves_shape():
    out = da.augment(_depth(), {"dropout_prob": 0.1, "near": 0.3, "far": 6.0},
                     rng=np.random.default_rng(0))
    assert out.shape == (48, 64)
