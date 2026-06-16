"""Time / message synchronisation helpers.

Pure logic (operates on float seconds) so it is unit-testable without ROS2. Convert ROS
``builtin_interfaces/Time`` to seconds with :func:`stamp_to_sec` before using these.
"""
from __future__ import annotations

from typing import Sequence


def stamp_to_sec(sec: int, nanosec: int) -> float:
    """Convert a ROS time (sec, nanosec) to float seconds."""
    return float(sec) + float(nanosec) * 1e-9


def age_sec(stamp_sec: float, now_sec: float) -> float:
    """Age of a stamp relative to ``now`` (negative if the stamp is in the future)."""
    return now_sec - stamp_sec


def is_stale(stamp_sec: float | None, now_sec: float, timeout_s: float) -> bool:
    """Fail-closed staleness check: ``None`` (never seen) is stale; future stamps are not."""
    if stamp_sec is None:
        return True
    age = now_sec - stamp_sec
    if age < 0:
        return False
    return age > timeout_s


def nearest_index(query_sec: float, stamps_sec: Sequence[float]) -> int:
    """Index of the stamp closest to ``query_sec``. Raises on empty input."""
    if len(stamps_sec) == 0:
        raise ValueError("empty stamp sequence")
    best_i, best_d = 0, abs(stamps_sec[0] - query_sec)
    for i, s in enumerate(stamps_sec[1:], start=1):
        d = abs(s - query_sec)
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def within_tolerance(a_sec: float, b_sec: float, tol_s: float) -> bool:
    """True if two stamps are within ``tol_s`` seconds — a simple approximate-sync test."""
    return abs(a_sec - b_sec) <= tol_s
