"""Roughness and single-step estimation from a height grid — pure logic, no ROS2."""
from __future__ import annotations

import numpy as np


def roughness_m(height_grid_2d) -> float:
    """Local height standard deviation over known cells (a roughness proxy)."""
    h = np.asarray(height_grid_2d, dtype=float)
    vals = h[np.isfinite(h)]
    if vals.size < 2:
        return 0.0
    # Detrend by subtracting the per-column (forward) mean so a uniform slope is not "rough".
    return float(np.nanstd(vals - np.nanmean(vals)))


def max_single_step_m(profile_z) -> float:
    """Largest single forward height jump in the profile (a curb/step proxy)."""
    z = np.asarray(profile_z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(z))))
