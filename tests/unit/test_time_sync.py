"""Unit tests for x2_common.time_sync (pure logic)."""
import pytest

from x2_common import time_sync as ts


def test_stamp_to_sec():
    assert ts.stamp_to_sec(2, 500_000_000) == pytest.approx(2.5)


def test_is_stale_fail_closed_on_none():
    assert ts.is_stale(None, now_sec=100.0, timeout_s=0.5) is True


def test_is_stale():
    assert ts.is_stale(99.8, now_sec=100.0, timeout_s=0.5) is False
    assert ts.is_stale(99.0, now_sec=100.0, timeout_s=0.5) is True


def test_is_stale_future_stamp_not_stale():
    assert ts.is_stale(101.0, now_sec=100.0, timeout_s=0.5) is False


def test_nearest_index():
    stamps = [1.0, 2.0, 3.0]
    assert ts.nearest_index(2.6, stamps) == 2
    assert ts.nearest_index(1.1, stamps) == 0


def test_nearest_index_empty_raises():
    with pytest.raises(ValueError):
        ts.nearest_index(1.0, [])


def test_within_tolerance():
    assert ts.within_tolerance(1.0, 1.02, 0.05)
    assert not ts.within_tolerance(1.0, 1.2, 0.05)
