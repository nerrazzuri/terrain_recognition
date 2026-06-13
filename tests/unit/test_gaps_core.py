"""Unit tests for gap/drop-off detection (pure logic).

Acceptance (roadmap §5.4 gap_detector): detect an open gap; treat unknown regions as unsafe
unless confidence is high; publish a reason string.
"""
import numpy as np
import pytest

from x2_terrain_perception.core import gaps as gp


def params(**kw):
    base = dict(min_drop_m=0.10, min_gap_width_m=0.08, unknown_is_unsafe=True)
    base.update(kw)
    return gp.GapParams(**base)


def test_flat_profile_no_gap():
    xs = np.arange(0.0, 2.0, 0.02)
    zs = np.zeros_like(xs)
    conf = np.ones_like(xs)
    res = gp.detect_gap(xs, zs, conf, params())
    assert not res.gap_detected
    assert not res.unknown_ahead


def test_drop_off_detected():
    xs = np.arange(0.0, 2.0, 0.02)
    zs = np.zeros_like(xs)
    zs[(xs >= 1.0) & (xs < 1.3)] = -0.30  # a hole 0.3 m wide, 0.3 m deep
    conf = np.ones_like(xs)
    res = gp.detect_gap(xs, zs, conf, params())
    assert res.gap_detected
    assert res.distance_m == pytest.approx(1.0, abs=0.05)
    assert res.gap_width_m == pytest.approx(0.3, abs=0.06)
    assert res.reason


def test_small_dip_below_threshold_ignored():
    xs = np.arange(0.0, 2.0, 0.02)
    zs = np.zeros_like(xs)
    zs[(xs >= 1.0) & (xs < 1.1)] = -0.05  # shallow, below min_drop
    conf = np.ones_like(xs)
    res = gp.detect_gap(xs, zs, conf, params())
    assert not res.gap_detected


def test_unknown_region_is_unsafe():
    xs = np.arange(0.0, 2.0, 0.02)
    zs = np.zeros_like(xs)
    conf = np.ones_like(xs)
    conf[(xs >= 0.8) & (xs < 1.4)] = 0.0   # unknown band ahead
    res = gp.detect_gap(xs, zs, conf, params())
    assert res.unknown_ahead
    assert "unknown" in res.reason.lower()


def test_unknown_ignored_when_disabled():
    xs = np.arange(0.0, 2.0, 0.02)
    zs = np.zeros_like(xs)
    conf = np.zeros_like(xs)
    res = gp.detect_gap(xs, zs, conf, params(unknown_is_unsafe=False))
    assert not res.unknown_ahead
