"""Per-cell traversability scoring — pure logic, no ROS2.

Maps a height grid + confidence grid to a uint8 traversability grid feeding the
``traversability[]`` field of TerrainGrid. Encoding (matches TerrainCell):
0 = unknown, 1..254 score (higher = easier), 255 = blocked.

Score is driven by the local height step to neighbours: small steps are easy, steps above
``max_step_m`` are blocked. Unknown cells (zero confidence) score 0.
"""
from __future__ import annotations

import numpy as np


def estimate(height_grid_2d, confidence_grid_2d, resolution_m: float,
             max_step_m: float = 0.15) -> np.ndarray:
    heights = np.asarray(height_grid_2d, dtype=float)
    conf = np.asarray(confidence_grid_2d, dtype=float)
    out = np.zeros(heights.shape, dtype=np.uint8)

    known = (conf > 0.0) & np.isfinite(heights)
    if not known.any():
        return out

    # Local max absolute height step to 4-neighbours (treat unknown neighbours as no info).
    filled = np.where(known, heights, np.nan)
    max_step = np.zeros(heights.shape, dtype=float)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.full(heights.shape, np.nan)
        ys = slice(max(0, dy), heights.shape[0] + min(0, dy))
        xs = slice(max(0, dx), heights.shape[1] + min(0, dx))
        ys2 = slice(max(0, -dy), heights.shape[0] + min(0, -dy))
        xs2 = slice(max(0, -dx), heights.shape[1] + min(0, -dx))
        shifted[ys2, xs2] = filled[ys, xs]
        diff = np.abs(filled - shifted)
        max_step = np.fmax(max_step, np.nan_to_num(diff, nan=0.0))

    # Map step -> score. step 0 -> ~254, step==max_step_m -> ~1, above -> blocked(255).
    score = 254.0 * (1.0 - np.clip(max_step / max_step_m, 0.0, 1.0))
    score = np.clip(score, 1.0, 254.0).astype(np.uint8)
    out[known] = score[known]
    out[known & (max_step > max_step_m)] = 255
    return out
