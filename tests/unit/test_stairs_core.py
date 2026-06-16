"""Unit tests for stair detection from a forward height profile (pure logic).

Acceptance (roadmap §5.4 stair_detector): detect clear stairs; reject clutter lacking
repeated structure; first-step distance within tolerance; never confident under ambiguity.
"""
import numpy as np
import pytest

from x2_terrain_perception.core import stairs as st


def params(**kw):
    base = dict(
        min_rise_m=0.05, max_rise_m=0.20, min_tread_m=0.20, max_tread_m=0.40,
        min_repeated_steps=2, riser_edge_min_dh_m=0.04, stop_distance_margin_m=0.3,
    )
    base.update(kw)
    return st.StairParams(**base)


def _staircase(first_step_x=0.6, rise=0.15, tread=0.30, n_steps=4, direction=1, dx=0.02):
    """Build a (x, z) forward profile of a flat run then a staircase."""
    xs = np.arange(0.0, 2.0, dx)
    zs = np.zeros_like(xs)
    for i, x in enumerate(xs):
        k = int(np.floor((x - first_step_x) / tread)) + 1
        if x < first_step_x:
            zs[i] = 0.0
        else:
            zs[i] = direction * rise * min(k, n_steps)
    return xs, zs


def test_detects_clear_upstairs():
    xs, zs = _staircase()
    res = st.detect_stairs(xs, zs, params())
    assert res.stairs_detected
    assert res.direction == "up"
    assert res.rise_m == pytest.approx(0.15, abs=0.03)
    assert res.tread_m == pytest.approx(0.30, abs=0.05)
    assert res.visible_step_count >= 2
    assert res.confidence > 0.6


def test_first_step_distance_within_tolerance():
    xs, zs = _staircase(first_step_x=0.6)
    res = st.detect_stairs(xs, zs, params())
    assert res.first_step_distance_m == pytest.approx(0.6, abs=0.05)
    # recommended stop is strictly before the first step
    assert res.recommended_stop_distance_m < res.first_step_distance_m


def test_detects_downstairs_direction():
    xs, zs = _staircase(direction=-1)
    res = st.detect_stairs(xs, zs, params())
    assert res.stairs_detected
    assert res.direction == "down"


def test_flat_ground_no_stairs():
    xs = np.arange(0.0, 2.0, 0.02)
    zs = np.zeros_like(xs)
    res = st.detect_stairs(xs, zs, params())
    assert not res.stairs_detected
    assert res.confidence < 0.5


def test_clutter_rejected():
    rng = np.random.default_rng(0)
    xs = np.arange(0.0, 2.0, 0.02)
    zs = rng.normal(0.0, 0.05, size=xs.shape)  # noise, no repeated structure
    res = st.detect_stairs(xs, zs, params())
    assert not res.stairs_detected


def test_single_step_is_not_stairs():
    # one edge only -> a curb, not a staircase (needs >= min_repeated_steps)
    xs, zs = _staircase(first_step_x=0.6, n_steps=1)
    res = st.detect_stairs(xs, zs, params())
    assert not res.stairs_detected


def test_never_confident_without_repeated_structure():
    # two edges with wildly inconsistent spacing -> not regular stairs
    xs = np.arange(0.0, 2.0, 0.02)
    zs = np.zeros_like(xs)
    zs[xs >= 0.4] = 0.15
    zs[xs >= 1.5] = 0.30
    res = st.detect_stairs(xs, zs, params())
    assert res.confidence < 0.6
