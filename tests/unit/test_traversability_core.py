"""Unit tests for per-cell traversability scoring (pure logic).

Encoding (matches TerrainCell): 0 = unknown, 1..254 score (higher = easier), 255 = blocked.
"""
import numpy as np

from x2_terrain_perception.core import traversability as tv


def test_unknown_cells_score_zero():
    heights = np.full((5, 5), np.nan)
    conf = np.zeros((5, 5))
    out = tv.estimate(heights, conf, resolution_m=0.04, max_step_m=0.15)
    assert np.all(out == 0)


def test_flat_known_ground_is_highly_traversable():
    heights = np.zeros((5, 5))
    conf = np.ones((5, 5))
    out = tv.estimate(heights, conf, resolution_m=0.04, max_step_m=0.15)
    assert out[2, 2] > 200
    assert np.all(out[conf > 0] >= 1)


def test_large_step_is_blocked():
    heights = np.zeros((5, 5))
    heights[:, 3:] = 0.30  # a 0.30 m wall, above max_step
    conf = np.ones((5, 5))
    out = tv.estimate(heights, conf, resolution_m=0.04, max_step_m=0.15)
    # cells along the discontinuity are blocked
    assert out[2, 2] > out[2, 3] or out[2, 3] == 255
    assert (out == 255).any()


def test_output_shape_and_dtype():
    heights = np.zeros((4, 6))
    conf = np.ones((4, 6))
    out = tv.estimate(heights, conf, resolution_m=0.04, max_step_m=0.15)
    assert out.shape == (4, 6)
    assert out.dtype == np.uint8
