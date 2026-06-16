"""Unit tests for grid_msg.grid_from_flat (flat TerrainGrid payload -> 2D arrays)."""
import numpy as np

from x2_terrain_perception.core import grid_msg


def test_unknown_cells_become_nan():
    width, height = 3, 2
    heights = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    conf = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]  # cells 1 and 5 unknown
    cfg, h2d, c2d, xpos = grid_msg.grid_from_flat(
        width, height, 0.04, 0.0, -0.04, heights, conf)
    assert h2d.shape == (2, 3)
    assert np.isnan(h2d[0, 1])      # idx 1 unknown
    assert np.isnan(h2d[1, 2])      # idx 5 unknown
    assert h2d[0, 0] == 0.0
    assert h2d[1, 0] == 3.0


def test_x_positions_are_cell_centers():
    cfg, h2d, c2d, xpos = grid_msg.grid_from_flat(
        3, 2, 0.04, 0.0, -0.04, [0] * 6, [1] * 6)
    np.testing.assert_allclose(xpos, [0.02, 0.06, 0.10])


def test_row_major_reshape_matches_index_convention():
    # idx = iy*width + ix ; value encodes (iy, ix)
    width, height = 4, 3
    heights = [iy * 10 + ix for iy in range(height) for ix in range(width)]
    _, h2d, _, _ = grid_msg.grid_from_flat(
        width, height, 0.04, 0.0, 0.0, heights, [1] * (width * height))
    assert h2d[2, 3] == 23
    assert h2d[1, 0] == 10
