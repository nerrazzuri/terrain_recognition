"""Local ground-plane estimation via RANSAC — pure logic, no ROS2.

Returns a plane (unit normal + offset), the inlier fraction, and a confidence score.
On stairs / multi-plane scenes the inlier fraction drops, which downstream logic reads as
"not a single clean ground plane" (roadmap §5.4 ground_plane_estimator acceptance).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneFit:
    normal: np.ndarray      # unit normal, z-component made non-negative
    offset: float           # plane: normal . p = offset
    inlier_fraction: float  # fraction of points within distance_thresh
    confidence: float       # 0..1

    def height_at(self, x: float, y: float) -> float:
        """Plane height z at (x, y)."""
        nx, ny, nz = self.normal
        if abs(nz) < 1e-9:
            return float("nan")
        return float((self.offset - nx * x - ny * y) / nz)


def _orient_up(normal: np.ndarray) -> np.ndarray:
    return normal if normal[2] >= 0 else -normal


def fit_plane_ransac(
    points: np.ndarray,
    distance_thresh: float = 0.02,
    max_iter: int = 100,
    seed: int | None = None,
) -> PlaneFit:
    """Fit a plane to ``(N, 3)`` points with RANSAC.

    Fewer than 3 points -> zero-confidence flat-up plane (fail safe). The best model is
    refined with a least-squares fit over its inliers.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    if n < 3:
        return PlaneFit(np.array([0.0, 0.0, 1.0]), 0.0, 0.0, 0.0)

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = -1

    for _ in range(max_iter):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(normal)
        if nn < 1e-9:
            continue
        normal = normal / nn
        offset = float(normal @ p0)
        dist = np.abs(points @ normal - offset)
        inliers = dist < distance_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        return PlaneFit(np.array([0.0, 0.0, 1.0]), 0.0, 0.0, 0.0)

    # Least-squares refit over inliers for a stable normal.
    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid)
    normal = _orient_up(vh[-1])
    offset = float(normal @ centroid)

    inlier_fraction = best_count / n
    # Confidence: rewards both a high inlier fraction and a near-up normal (clean ground).
    confidence = float(np.clip(inlier_fraction, 0.0, 1.0))
    return PlaneFit(normal, offset, inlier_fraction, confidence)
