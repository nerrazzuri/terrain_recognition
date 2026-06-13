"""Termination conditions (P4-M2-T2). Pure logic, no Isaac Lab.

Fall / collision detection per roadmap §8.9. Returns ``(done, reason)`` so episodes end on a
fall and the reason can be logged.
"""
from __future__ import annotations


def should_terminate(base_height: float, min_height: float, roll: float, pitch: float,
                     max_tilt: float, bad_contact: bool) -> tuple[bool, str]:
    if base_height < min_height:
        return True, f"base height {base_height:.2f} below {min_height:.2f}"
    if abs(roll) > max_tilt:
        return True, f"roll {roll:.2f} over tilt limit {max_tilt:.2f}"
    if abs(pitch) > max_tilt:
        return True, f"pitch {pitch:.2f} over tilt limit {max_tilt:.2f}"
    if bad_contact:
        return True, "invalid contact (head/torso/knee or undesired body)"
    return False, "ok"
