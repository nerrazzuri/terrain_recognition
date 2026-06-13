"""Height-sample extraction around the robot (P3-M3-T3). Pure logic, no Isaac Lab.

Samples a fixed grid of terrain heights (default 11x11 = 121) in the robot's base frame and
returns them relative to the base — exactly the ``height_samples`` observation component used
in training (docs/training_method.md). The terrain is provided as a height function
``z = f(x, y)`` over world coordinates, so the same code works for analytic terrain in tests
and for a sampled heightfield in sim.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SampleGrid:
    nx: int
    ny: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def points_base(self) -> np.ndarray:
        """Return the (nx*ny, 2) sample points in the base frame (row-major over x then y)."""
        xs = np.linspace(self.x_min, self.x_max, self.nx)
        ys = np.linspace(self.y_min, self.y_max, self.ny)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        return np.column_stack([gx.reshape(-1), gy.reshape(-1)])


def sample_heights(grid: SampleGrid, height_fn, base_xy, base_yaw, base_height) -> np.ndarray:
    """Sample ``height_fn`` at the grid points transformed into the world, base-relative.

    - ``height_fn(x, y)`` accepts numpy arrays and returns world terrain heights.
    - Points are rotated by ``base_yaw`` and translated by ``base_xy`` into the world.
    - Output = terrain_height_world - base_height (negative when terrain is below the base).
    """
    pts = grid.points_base()
    c, s = np.cos(base_yaw), np.sin(base_yaw)
    wx = base_xy[0] + c * pts[:, 0] - s * pts[:, 1]
    wy = base_xy[1] + s * pts[:, 0] + c * pts[:, 1]
    z = np.asarray(height_fn(wx, wy), dtype=float)
    return z - base_height
