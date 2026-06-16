"""Unit tests for x2_common.safety_limits (pure logic, no ROS2)."""
import math

import pytest

from x2_common import safety_limits as sl


def test_clamp_basic():
    assert sl.clamp(5, 0, 10) == 5
    assert sl.clamp(-1, 0, 10) == 0
    assert sl.clamp(11, 0, 10) == 10


def test_clamp_inverted_bounds_raises():
    with pytest.raises(ValueError):
        sl.clamp(1, 10, 0)


def test_clamp_symmetric():
    assert sl.clamp_symmetric(5, 3) == 3
    assert sl.clamp_symmetric(-5, 3) == -3
    assert sl.clamp_symmetric(2, -3) == 2  # magnitude is abs()


def test_is_finite():
    assert sl.is_finite(1.0, -2.0, 0.0)
    assert not sl.is_finite(1.0, float("nan"))
    assert not sl.is_finite(float("inf"))


def test_rate_limit():
    assert sl.rate_limit(0.0, 1.0, 0.1) == pytest.approx(0.1)
    assert sl.rate_limit(0.0, -1.0, 0.1) == pytest.approx(-0.1)
    assert sl.rate_limit(0.0, 0.05, 0.1) == pytest.approx(0.05)


def test_rate_limit_negative_delta_raises():
    with pytest.raises(ValueError):
        sl.rate_limit(0.0, 1.0, -0.1)


def test_freshness_watchdog_fail_closed_on_none():
    wd = sl.FreshnessWatchdog(timeout_s=0.5)
    assert wd.is_fresh(None, now=100.0) is False


def test_freshness_watchdog():
    wd = sl.FreshnessWatchdog(timeout_s=0.5)
    assert wd.is_fresh(last_stamp=99.8, now=100.0) is True
    assert wd.is_fresh(last_stamp=99.0, now=100.0) is False  # too old


def test_tilt_exceeded():
    limit = math.radians(15)
    assert not sl.tilt_exceeded(math.radians(10), math.radians(-10), limit)
    assert sl.tilt_exceeded(math.radians(20), 0.0, limit)
    assert sl.tilt_exceeded(0.0, math.radians(-20), limit)


def test_velocity_limits_apply():
    lim = sl.VelocityLimits(0.12, 0.0, 0.3)
    fwd, lat, yaw = lim.apply(1.0, 1.0, 1.0)
    assert fwd == pytest.approx(0.12)
    assert lat == pytest.approx(0.0)
    assert yaw == pytest.approx(0.3)
