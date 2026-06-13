"""Small numpy-based geometry helpers (frame transforms, gravity projection).

Pure logic, no ROS2/tf2 dependency so it is unit-testable. For live TF lookups in nodes,
wrap a ``tf2_ros.Buffer`` and feed the resulting 4x4 matrices into these helpers.
"""
from __future__ import annotations

import numpy as np


def quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a (x, y, z, w) quaternion to a 3x3 rotation matrix.

    The quaternion is normalised first; a zero quaternion raises ``ValueError``.
    """
    n = math_sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def math_sqrt(x: float) -> float:
    return float(np.sqrt(x))


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and a length-3 translation."""
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float).reshape(3)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an ``(N, 3)`` array of points, returning ``(N, 3)``."""
    transform = np.asarray(transform, dtype=float)
    points = np.asarray(points, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    out = homog @ transform.T
    return out[:, :3]


def projected_gravity(rotation_world_to_base: np.ndarray) -> np.ndarray:
    """Gravity unit vector expressed in the base frame (an observation component).

    ``rotation_world_to_base`` rotates world vectors into the base frame. On flat ground
    with an upright base this returns ``[0, 0, -1]``.
    """
    g_world = np.array([0.0, 0.0, -1.0])
    return np.asarray(rotation_world_to_base, dtype=float) @ g_world


def drop_nonfinite(points: np.ndarray) -> np.ndarray:
    """Remove rows containing NaN/inf from an ``(N, 3)`` point array."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points.reshape(0, 3)
    mask = np.isfinite(points).all(axis=1)
    return points[mask]


def crop_roi(
    points: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> np.ndarray:
    """Keep only points inside the axis-aligned ROI box (base frame)."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points.reshape(0, 3)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = (
        (x >= x_range[0]) & (x <= x_range[1])
        & (y >= y_range[0]) & (y <= y_range[1])
        & (z >= z_range[0]) & (z <= z_range[1])
    )
    return points[mask]
