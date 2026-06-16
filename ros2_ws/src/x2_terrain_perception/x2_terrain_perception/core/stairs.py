"""Stair detection from a forward height profile — pure logic, no ROS2.

Method (roadmap §5.4 stair_detector, first version):
1. take a forward height profile (height vs x), already averaged over a central y band;
2. find step-like discontinuities (risers) above a threshold;
3. require several edges with consistent rise and tread (repeated structure);
4. estimate rise/tread, first-step distance, recommended stop distance, and confidence.

Fail safe: ambiguous / irregular structure yields low confidence and ``stairs_detected``
False — the classifier then keeps the terrain unsafe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StairParams:
    min_rise_m: float
    max_rise_m: float
    min_tread_m: float
    max_tread_m: float
    min_repeated_steps: int
    riser_edge_min_dh_m: float
    stop_distance_margin_m: float

    @classmethod
    def from_dict(cls, d: dict) -> "StairParams":
        return cls(
            min_rise_m=float(d["min_rise_m"]),
            max_rise_m=float(d["max_rise_m"]),
            min_tread_m=float(d["min_tread_m"]),
            max_tread_m=float(d["max_tread_m"]),
            min_repeated_steps=int(d["min_repeated_steps"]),
            riser_edge_min_dh_m=float(d["riser_edge_min_dh_m"]),
            stop_distance_margin_m=float(d["stop_distance_margin_m"]),
        )


@dataclass(frozen=True)
class StairResult:
    stairs_detected: bool
    direction: str            # "up" | "down" | "none"
    confidence: float
    rise_m: float
    tread_m: float
    visible_step_count: int
    first_step_distance_m: float
    recommended_stop_distance_m: float


_NONE = StairResult(False, "none", 0.0, 0.0, 0.0, 0, 0.0, 0.0)


def _find_edges(xs: np.ndarray, zs: np.ndarray, min_dh: float):
    """Find riser edges: cumulative monotonic height changes >= min_dh.

    Returns a list of (x_position, signed_height_change) for each detected step.
    """
    edges = []
    anchor_z = zs[0]
    anchor_x = xs[0]
    for i in range(1, len(zs)):
        dz = zs[i] - anchor_z
        if abs(dz) >= min_dh:
            edges.append((float(xs[i]), float(dz)))
            anchor_z = zs[i]
            anchor_x = xs[i]
        elif np.sign(zs[i] - zs[i - 1]) != np.sign(dz) and dz != 0:
            # height reversed before reaching a full riser -> reset anchor (flat tread)
            anchor_z = zs[i]
            anchor_x = xs[i]
    return edges


def detect_stairs(profile_x: np.ndarray, profile_z: np.ndarray, p: StairParams) -> StairResult:
    xs = np.asarray(profile_x, dtype=float)
    zs = np.asarray(profile_z, dtype=float)
    if xs.size < 3 or xs.size != zs.size:
        return _NONE

    edges = _find_edges(xs, zs, p.riser_edge_min_dh_m)
    if len(edges) < p.min_repeated_steps:
        return _NONE

    edge_x = np.array([e[0] for e in edges])
    edge_dz = np.array([e[1] for e in edges])

    # Direction: consistent sign of risers.
    up_frac = float(np.mean(edge_dz > 0))
    direction = "up" if up_frac >= 0.5 else "down"
    sign = 1.0 if direction == "up" else -1.0
    consistent = edge_dz[np.sign(edge_dz) == sign]
    if consistent.size < p.min_repeated_steps:
        return _NONE

    rise = float(np.median(np.abs(consistent)))
    treads = np.diff(edge_x)
    tread = float(np.median(treads)) if treads.size else 0.0

    # Validate against expected stair geometry.
    rise_ok = p.min_rise_m <= rise <= p.max_rise_m
    tread_ok = p.min_tread_m <= tread <= p.max_tread_m

    # Regularity: low coefficient of variation in rise and tread -> real staircase.
    def cov(a):
        a = np.asarray(a, dtype=float)
        m = np.mean(np.abs(a))
        return float(np.std(a) / m) if m > 1e-6 else 1.0

    rise_cov = cov(consistent)
    tread_cov = cov(treads) if treads.size else 1.0
    regularity = max(0.0, 1.0 - 0.5 * (rise_cov + tread_cov))

    step_count = int(consistent.size)
    enough_steps = step_count >= p.min_repeated_steps

    detected = bool(rise_ok and tread_ok and enough_steps and regularity > 0.5)
    confidence = float(np.clip(regularity * min(1.0, step_count / 3.0), 0.0, 1.0))
    if not (rise_ok and tread_ok):
        confidence = min(confidence, 0.4)
    if not detected:
        confidence = min(confidence, 0.5)

    first_step = float(edge_x.min())
    stop = max(0.0, first_step - p.stop_distance_margin_m)

    return StairResult(
        stairs_detected=detected,
        direction=direction if detected else "none",
        confidence=confidence,
        rise_m=rise,
        tread_m=tread,
        visible_step_count=step_count,
        first_step_distance_m=first_step,
        recommended_stop_distance_m=stop,
    )


def forward_height_profile(height_grid_2d: np.ndarray, x_positions: np.ndarray,
                           y_center_band: int = 6):
    """Collapse a (height, width) grid into a forward (x, z) profile.

    Averages over the central ``y_center_band`` rows, ignoring NaN (unknown) cells. Columns
    that are entirely unknown are dropped so they do not masquerade as flat ground.
    """
    grid = np.asarray(height_grid_2d, dtype=float)
    n_y, n_x = grid.shape
    c = n_y // 2
    half = max(1, y_center_band // 2)
    band = grid[max(0, c - half): min(n_y, c + half), :]
    # Columns that are entirely unknown (all NaN) are expected; suppress the all-NaN-slice
    # warning and drop them via the validity mask below.
    with np.errstate(invalid="ignore"):
        all_nan = np.all(np.isnan(band), axis=0)
        prof = np.full(band.shape[1], np.nan)
        prof[~all_nan] = np.nanmean(band[:, ~all_nan], axis=0)
    valid = ~np.isnan(prof)
    return np.asarray(x_positions)[valid], prof[valid]
