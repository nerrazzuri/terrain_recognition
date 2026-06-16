"""Safety-limit helpers: clamps, freshness watchdogs, fall detection.

Pure logic, no ROS2 dependency — so it is unit-testable with extreme inputs (AGENTS.md §4).
The cross-cutting safety rule is **fail closed**: when in doubt, the safe value is *stop*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to ``[low, high]``. Raises if the bounds are inverted."""
    if low > high:
        raise ValueError(f"inverted clamp bounds: low={low} > high={high}")
    return max(low, min(high, value))


def clamp_symmetric(value: float, magnitude: float) -> float:
    """Clamp ``value`` to ``[-|magnitude|, +|magnitude|]``."""
    m = abs(magnitude)
    return clamp(value, -m, m)


def is_finite(*values: float) -> bool:
    """True only if every value is finite (no NaN / inf). Bad policy output ⇒ stop."""
    return all(math.isfinite(v) for v in values)


def rate_limit(current: float, target: float, max_delta: float) -> float:
    """Step ``current`` toward ``target`` by at most ``max_delta`` (>= 0)."""
    if max_delta < 0:
        raise ValueError("max_delta must be >= 0")
    delta = clamp(target - current, -max_delta, max_delta)
    return current + delta


@dataclass(frozen=True)
class FreshnessWatchdog:
    """Stop if an input has not been updated within ``timeout_s``.

    Stamps are monotonic seconds (e.g. ``time.monotonic()`` or ROS clock seconds).
    Fail closed: a never-seen input (``last_stamp is None``) is treated as stale.
    """

    timeout_s: float

    def is_fresh(self, last_stamp: float | None, now: float) -> bool:
        if last_stamp is None:
            return False
        age = now - last_stamp
        return 0.0 <= age <= self.timeout_s


def tilt_exceeded(roll_rad: float, pitch_rad: float, limit_rad: float) -> bool:
    """True if |roll| or |pitch| exceeds ``limit_rad`` — IMU fall/tip detector."""
    return abs(roll_rad) > limit_rad or abs(pitch_rad) > limit_rad


@dataclass(frozen=True)
class VelocityLimits:
    """Per-axis velocity envelope for the safe locomotion adapter."""

    max_forward_mps: float
    max_lateral_mps: float
    max_yaw_radps: float

    def apply(self, forward: float, lateral: float, yaw: float) -> tuple[float, float, float]:
        return (
            clamp_symmetric(forward, self.max_forward_mps),
            clamp_symmetric(lateral, self.max_lateral_mps),
            clamp_symmetric(yaw, self.max_yaw_radps),
        )
