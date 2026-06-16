"""Unit tests for ground-plane estimation and slope derivation (pure logic)."""
import math

import numpy as np
import pytest

from x2_terrain_perception.core import ground_plane as gp
from x2_terrain_perception.core import slope as sl


def _plane_points(slope_deg=0.0, n=400, noise=0.0, seed=0):
    """Sample points on z = tan(slope) * x (slope about the y-axis -> rises in +x)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    z = math.tan(math.radians(slope_deg)) * xy[:, 0]
    if noise:
        z = z + rng.normal(0.0, noise, size=n)
    return np.column_stack([xy, z])


def test_flat_plane_normal_is_up():
    res = gp.fit_plane_ransac(_plane_points(0.0), distance_thresh=0.01, max_iter=100, seed=1)
    assert res.normal[2] == pytest.approx(1.0, abs=1e-3)
    assert res.confidence > 0.95


def test_flat_plane_slope_within_tolerance():
    res = gp.fit_plane_ransac(_plane_points(0.0, noise=0.005), distance_thresh=0.02, seed=2)
    assert sl.slope_angle_deg(res.normal) < 2.0  # acceptance: flat within +/-2 deg


def test_ramp_slope_is_directionally_correct():
    res = gp.fit_plane_ransac(_plane_points(10.0), distance_thresh=0.01, seed=3)
    assert sl.slope_angle_deg(res.normal) == pytest.approx(10.0, abs=1.5)
    assert sl.slope_direction(res.normal, threshold_deg=3.0) == "up"


def test_downward_ramp_direction():
    res = gp.fit_plane_ransac(_plane_points(-10.0), distance_thresh=0.01, seed=4)
    assert sl.slope_direction(res.normal, threshold_deg=3.0) == "down"


def test_flat_direction_is_none():
    res = gp.fit_plane_ransac(_plane_points(0.0), distance_thresh=0.01, seed=5)
    assert sl.slope_direction(res.normal, threshold_deg=3.0) == "none"


def test_outliers_are_rejected():
    pts = _plane_points(0.0, n=300, seed=6)
    # add 60 gross outliers well off the plane
    rng = np.random.default_rng(7)
    outliers = np.column_stack([rng.uniform(-1, 1, 60), rng.uniform(-1, 1, 60),
                                rng.uniform(0.3, 0.6, 60)])
    pts = np.vstack([pts, outliers])
    res = gp.fit_plane_ransac(pts, distance_thresh=0.02, max_iter=200, seed=8)
    assert sl.slope_angle_deg(res.normal) < 3.0
    assert res.inlier_fraction > 0.7  # the 300 plane points dominate


def test_multiplane_stairs_lower_confidence():
    # two stacked planes (like stair treads) -> no single plane explains most points
    lower = _plane_points(0.0, n=200, seed=9)
    upper = _plane_points(0.0, n=200, seed=10)
    upper[:, 2] += 0.15
    res = gp.fit_plane_ransac(np.vstack([lower, upper]), distance_thresh=0.01, seed=11)
    assert res.inlier_fraction < 0.75  # confidence/structure ambiguous


def test_too_few_points_low_confidence():
    res = gp.fit_plane_ransac(np.zeros((2, 3)), distance_thresh=0.01)
    assert res.confidence == 0.0


def test_slope_angle_clamps_unit_normal():
    assert sl.slope_angle_deg(np.array([0.0, 0.0, 2.0])) == pytest.approx(0.0, abs=1e-6)
