"""Gap / drop-off detection from a forward profile — pure logic, no ROS2.

A gap is a contiguous run of cells dropping >= ``min_drop_m`` below the near-ground
reference and at least ``min_gap_width_m`` wide. Unknown (zero-confidence) regions ahead are
reported separately as unsafe (roadmap §5.4 gap_detector). Fail safe: unknown ⇒ unsafe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GapParams:
    min_drop_m: float
    min_gap_width_m: float
    unknown_is_unsafe: bool

    @classmethod
    def from_dict(cls, d: dict) -> "GapParams":
        return cls(
            min_drop_m=float(d["min_drop_m"]),
            min_gap_width_m=float(d["min_gap_width_m"]),
            unknown_is_unsafe=bool(d["unknown_is_unsafe"]),
        )


@dataclass(frozen=True)
class GapResult:
    gap_detected: bool
    gap_width_m: float
    distance_m: float
    unknown_ahead: bool
    reason: str


def _contiguous_runs(mask: np.ndarray):
    """Yield (start_idx, end_idx_exclusive) for each run of True in a 1D bool mask."""
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def detect_gap(profile_x, profile_z, profile_conf, p: GapParams) -> GapResult:
    xs = np.asarray(profile_x, dtype=float)
    zs = np.asarray(profile_z, dtype=float)
    conf = np.asarray(profile_conf, dtype=float)
    if xs.size < 2:
        return GapResult(False, 0.0, 0.0, False, "insufficient profile")

    # Near-ground reference: the closest known cells (robust median of first quarter).
    known = conf > 0.0
    ref_region = zs[known][: max(1, np.count_nonzero(known) // 4)] if known.any() else zs[:1]
    reference = float(np.median(ref_region)) if ref_region.size else 0.0

    # Drop mask: cells well below the reference (only where we have data).
    drop_mask = known & (zs < reference - p.min_drop_m)
    gap_detected = False
    gap_width = 0.0
    distance = 0.0
    reason = ""
    for (s, e) in _contiguous_runs(drop_mask):
        width = float(xs[e - 1] - xs[s])
        if width >= p.min_gap_width_m:
            gap_detected = True
            gap_width = width
            distance = float(xs[s])
            reason = f"drop-off {gap_width:.2f} m wide at {distance:.2f} m ahead"
            break

    # Unknown-ahead check.
    unknown_ahead = False
    if p.unknown_is_unsafe:
        unknown_mask = conf <= 0.0
        for (s, e) in _contiguous_runs(unknown_mask):
            width = float(xs[e - 1] - xs[s]) if e > s else 0.0
            if width >= p.min_gap_width_m:
                unknown_ahead = True
                if not reason:
                    reason = f"unknown region {width:.2f} m wide at {float(xs[s]):.2f} m ahead"
                break

    if not reason:
        reason = "no gap detected"
    return GapResult(gap_detected, gap_width, distance, unknown_ahead, reason)
