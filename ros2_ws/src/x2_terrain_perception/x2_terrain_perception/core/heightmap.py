"""Robot-centred elevation map — pure logic, no ROS2.

Grid convention (matches x2_terrain_msgs/TerrainGrid and docs/data_contracts.md):
- ``ix`` indexes x (forward), ``iy`` indexes y (lateral); ``width`` = x cells,
  ``height`` = y cells.
- cell (0,0) lower corner sits at ``(origin_x_m, origin_y_m)`` in the base frame.
- row-major flatten: ``idx = iy * width + ix``.

Fail-safe rule: a never-observed cell is **unknown** (confidence 0, height NaN), never
flat. Downstream classification must treat unknown as unsafe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeightMapConfig:
    resolution_m: float
    width: int          # cells in x (forward)
    height: int         # cells in y (lateral)
    origin_x_m: float
    origin_y_m: float

    @classmethod
    def from_dict(cls, d: dict) -> "HeightMapConfig":
        """Build from the ``heightmap`` block of configs/terrain_perception.yaml."""
        return cls(
            resolution_m=float(d["resolution_m"]),
            width=int(d["width_cells"]),
            height=int(d["height_cells"]),
            origin_x_m=float(d.get("origin_x_m", 0.0)),
            origin_y_m=float(d.get("origin_y_m", -float(d["map_size_y_m"]) / 2.0)),
        )


def world_to_cell(cfg: HeightMapConfig, x: float, y: float) -> tuple[int, int] | None:
    """Map a base-frame (x, y) to integer cell indices, or ``None`` if outside the grid."""
    ix = int(math.floor((x - cfg.origin_x_m) / cfg.resolution_m))
    iy = int(math.floor((y - cfg.origin_y_m) / cfg.resolution_m))
    if 0 <= ix < cfg.width and 0 <= iy < cfg.height:
        return ix, iy
    return None


def cell_to_world(cfg: HeightMapConfig, ix: int, iy: int) -> tuple[float, float]:
    """Return the base-frame coordinates of the centre of cell ``(ix, iy)``."""
    x = cfg.origin_x_m + (ix + 0.5) * cfg.resolution_m
    y = cfg.origin_y_m + (iy + 0.5) * cfg.resolution_m
    return x, y


def cell_index(cfg: HeightMapConfig, ix: int, iy: int) -> int:
    """Row-major flat index for cell ``(ix, iy)``."""
    return iy * cfg.width + ix


class HeightMap:
    """Mutable elevation map with per-cell confidence and exponential time decay.

    ``decay`` is the blend weight given to a *new* confident measurement when fusing into an
    existing cell (0..1). The first observation of a cell sets it outright.
    """

    def __init__(self, cfg: HeightMapConfig, decay: float = 0.5):
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.cfg = cfg
        self.decay = decay
        n = cfg.width * cfg.height
        self._height = np.full(n, np.nan, dtype=float)
        self._conf = np.zeros(n, dtype=float)

    # --- queries ---

    def height_at(self, ix: int, iy: int) -> float:
        return float(self._height[cell_index(self.cfg, ix, iy)])

    def confidence_at(self, ix: int, iy: int) -> float:
        return float(self._conf[cell_index(self.cfg, ix, iy)])

    # --- update ---

    def update(self, points: np.ndarray, measurement_confidence: float = 1.0) -> None:
        """Fuse an ``(N, 3)`` array of base-frame points into the map.

        Per cell the measured height is the **max** z of points falling in it (the terrain
        top). Empty cells are left untouched (stay at their previous/unknown value).
        """
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be (N, 3)")

        res = self.cfg.resolution_m
        ix = np.floor((points[:, 0] - self.cfg.origin_x_m) / res).astype(int)
        iy = np.floor((points[:, 1] - self.cfg.origin_y_m) / res).astype(int)
        z = points[:, 2]
        in_bounds = (ix >= 0) & (ix < self.cfg.width) & (iy >= 0) & (iy < self.cfg.height)
        ix, iy, z = ix[in_bounds], iy[in_bounds], z[in_bounds]
        if ix.size == 0:
            return
        flat = iy * self.cfg.width + ix

        # Per-cell max z via np.maximum.at over the touched cells.
        measured = np.full(self._height.shape, -np.inf)
        np.maximum.at(measured, flat, z)
        touched = np.isfinite(measured) & (measured > -np.inf)
        touched_idx = np.nonzero(touched)[0]

        for i in touched_idx:
            m = measured[i]
            if self._conf[i] <= 0.0 or math.isnan(self._height[i]):
                self._height[i] = m
            else:
                a = self.decay
                self._height[i] = a * m + (1.0 - a) * self._height[i]
            self._conf[i] = min(
                1.0, (1.0 - self.decay) * self._conf[i] + self.decay * measurement_confidence
            )

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(height_m, confidence, traversability)`` flat arrays for TerrainGrid.

        Traversability here is a placeholder (0 = unknown); the traversability estimator
        fills real scores. Unknown/NaN heights are reported as 0 height with 0 confidence so
        the message stays numeric, but confidence==0 is the signal that the cell is unknown.
        """
        heights = np.where(np.isnan(self._height), 0.0, self._height).astype(np.float32)
        conf = self._conf.astype(np.float32)
        trav = np.zeros(self._height.shape, dtype=np.uint8)
        return heights, conf, trav

    def height_grid_2d(self) -> np.ndarray:
        """Return heights as a ``(height, width)`` 2D array (NaN where unknown)."""
        return self._height.reshape(self.cfg.height, self.cfg.width)

    def confidence_grid_2d(self) -> np.ndarray:
        return self._conf.reshape(self.cfg.height, self.cfg.width)
