"""Helpers to turn a flat TerrainGrid payload into 2D arrays + a forward x axis.

Pure logic (operates on plain numbers/arrays), so it is unit-testable without ROS2.
"""
from __future__ import annotations

import numpy as np

from .heightmap import HeightMapConfig


def grid_from_flat(width, height, resolution_m, origin_x_m, origin_y_m,
                   height_m, confidence):
    """Return ``(cfg, height_2d, conf_2d, x_positions)``.

    Cells with confidence <= 0 are treated as unknown and set to NaN in ``height_2d`` so that
    downstream logic never mistakes an unobserved cell for flat ground.
    """
    cfg = HeightMapConfig(float(resolution_m), int(width), int(height),
                          float(origin_x_m), float(origin_y_m))
    h = np.asarray(height_m, dtype=float).reshape(cfg.height, cfg.width)
    c = np.asarray(confidence, dtype=float).reshape(cfg.height, cfg.width)
    h = np.where(c > 0.0, h, np.nan)
    x_positions = cfg.origin_x_m + (np.arange(cfg.width) + 0.5) * cfg.resolution_m
    return cfg, h, c, x_positions
