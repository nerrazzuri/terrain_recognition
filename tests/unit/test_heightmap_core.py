"""Unit tests for the height-map core (coordinate conversion, indexing, build, decay).

Pure logic, no ROS2. Acceptance (roadmap §5.4 heightmap_node): coordinate conversion and
grid indexing covered; a 10-15 cm step appears at ~correct height; unknown cells are marked
unknown, not flat.
"""
import numpy as np
import pytest

from x2_terrain_perception.core import heightmap as hm


def cfg():
    # 50 x 40 cells, 0.04 m, origin at (0, -0.8): x in [0,2), y in [-0.8,0.8)
    return hm.HeightMapConfig(
        resolution_m=0.04, width=50, height=40, origin_x_m=0.0, origin_y_m=-0.8
    )


# --- coordinate conversion / indexing ---

def test_world_to_cell_origin_corner():
    assert hm.world_to_cell(cfg(), 0.0, -0.8) == (0, 0)


def test_world_to_cell_interior():
    # x=1.0 -> ix=25 ; y=0.0 -> iy=20
    assert hm.world_to_cell(cfg(), 1.0, 0.0) == (25, 20)


def test_world_to_cell_out_of_bounds_returns_none():
    assert hm.world_to_cell(cfg(), -0.1, 0.0) is None
    assert hm.world_to_cell(cfg(), 2.5, 0.0) is None
    assert hm.world_to_cell(cfg(), 1.0, 1.0) is None


def test_cell_to_world_is_cell_center():
    x, y = hm.cell_to_world(cfg(), 0, 0)
    assert x == pytest.approx(0.02)
    assert y == pytest.approx(-0.78)


def test_round_trip_world_cell_world():
    c = cfg()
    for (wx, wy) in [(0.10, -0.30), (1.97, 0.50), (0.50, 0.0)]:
        ix, iy = hm.world_to_cell(c, wx, wy)
        cx, cy = hm.cell_to_world(c, ix, iy)
        # recovered center is within half a cell of the original point
        assert abs(cx - wx) <= c.resolution_m / 2 + 1e-9
        assert abs(cy - wy) <= c.resolution_m / 2 + 1e-9


def test_cell_index_row_major():
    c = cfg()
    assert hm.cell_index(c, 0, 0) == 0
    assert hm.cell_index(c, 1, 0) == 1
    assert hm.cell_index(c, 0, 1) == c.width
    assert hm.cell_index(c, 49, 39) == c.width * c.height - 1


# --- map building ---

def test_unknown_cells_are_unknown_not_flat():
    c = cfg()
    grid = hm.HeightMap(c)
    # no points observed yet -> everything unknown
    assert grid.confidence_at(25, 20) == 0.0
    assert np.isnan(grid.height_at(25, 20))


def test_step_appears_at_correct_height():
    c = cfg()
    grid = hm.HeightMap(c)
    # flat ground at z=0 for x<1.0, a 0.12 m step for x>=1.0
    # sample finer than the 0.04 m cell size so every queried cell is populated
    pts = []
    for x in np.arange(0.0, 2.0, 0.02):
        z = 0.12 if x >= 1.0 else 0.0
        for y in np.arange(-0.4, 0.4, 0.02):
            pts.append((x, y, z))
    grid.update(np.array(pts), measurement_confidence=1.0)
    # cell at x=0.5 ~ 0.0, cell at x=1.5 ~ 0.12
    ix_low, iy = hm.world_to_cell(c, 0.5, 0.0)
    ix_high, _ = hm.world_to_cell(c, 1.5, 0.0)
    assert grid.height_at(ix_low, iy) == pytest.approx(0.0, abs=0.01)
    assert grid.height_at(ix_high, iy) == pytest.approx(0.12, abs=0.01)


def test_decay_blends_then_forgets():
    c = cfg()
    grid = hm.HeightMap(c, decay=0.5)
    ix, iy = 25, 20
    cx, cy = hm.cell_to_world(c, ix, iy)
    grid.update(np.array([[cx, cy, 0.10]]), measurement_confidence=1.0)
    assert grid.height_at(ix, iy) == pytest.approx(0.10, abs=1e-6)
    # a second, different observation blends toward the new value
    grid.update(np.array([[cx, cy, 0.20]]), measurement_confidence=1.0)
    assert 0.10 < grid.height_at(ix, iy) < 0.20


def test_to_arrays_lengths_match_grid():
    c = cfg()
    grid = hm.HeightMap(c)
    heights, conf, trav = grid.to_arrays()
    n = c.width * c.height
    assert heights.shape == (n,)
    assert conf.shape == (n,)
    assert trav.shape == (n,)
