"""Depth augmentation (P6-M3-T1). Pure numpy, no torch.

Simulates real depth-sensor degradation (roadmap §10.6) so the raw-depth CReF policy
transfers: pixel dropout, missing pixels, additive/reflective noise, near/far clipping. Pure
+ seedable so it is reproducible and unit-testable; the dataset/training pipeline calls these.
"""
from __future__ import annotations

import numpy as np


def pixel_dropout(depth: np.ndarray, prob: float, rng) -> np.ndarray:
    """Randomly zero a fraction ``prob`` of pixels (dropout)."""
    d = np.asarray(depth, dtype=float).copy()
    if prob <= 0.0:
        return d
    mask = rng.random(d.shape) < prob
    d[mask] = 0.0
    return d


def missing_pixels(depth: np.ndarray, prob: float, rng) -> np.ndarray:
    """Randomly mark a fraction ``prob`` of pixels as missing (NaN)."""
    d = np.asarray(depth, dtype=float).copy()
    if prob <= 0.0:
        return d
    mask = rng.random(d.shape) < prob
    d[mask] = np.nan
    return d


def additive_noise(depth: np.ndarray, sigma: float, rng) -> np.ndarray:
    """Add zero-mean Gaussian noise of std ``sigma`` (metres)."""
    d = np.asarray(depth, dtype=float)
    return d + rng.normal(0.0, sigma, size=d.shape)


def clip_range(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Clip depths to the sensor's valid [near, far] range."""
    return np.clip(np.asarray(depth, dtype=float), near, far)


def augment(depth: np.ndarray, cfg: dict, rng) -> np.ndarray:
    """Apply the configured augmentation pipeline in a fixed order (reproducible per rng)."""
    d = np.asarray(depth, dtype=float)
    if "noise_sigma" in cfg:
        d = additive_noise(d, float(cfg["noise_sigma"]), rng)
    if "dropout_prob" in cfg:
        d = pixel_dropout(d, float(cfg["dropout_prob"]), rng)
    if "missing_prob" in cfg:
        d = missing_pixels(d, float(cfg["missing_prob"]), rng)
    if "near" in cfg and "far" in cfg:
        # clip only the finite (non-missing) pixels
        finite = np.isfinite(d)
        d[finite] = np.clip(d[finite], float(cfg["near"]), float(cfg["far"]))
    return d
