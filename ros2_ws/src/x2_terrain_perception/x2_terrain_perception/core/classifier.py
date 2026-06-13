"""Terrain classification decision logic — pure logic, no ROS2.

Fuses ground-plane slope, height-map roughness, stair detection and gap detection into a
single terrain type + confidence + reason, following the decision policy in roadmap §5.4.

Safety: ``safe_to_continue`` is never true under low confidence or for
curb/stairs/gap/unknown/platform classes. Reason is always populated.
"""
from __future__ import annotations

from dataclasses import dataclass

from .stairs import StairResult
from .gaps import GapResult

# Classes the robot may move over (slowly); everything else means stop.
_TRAVERSABLE = {"flat_ground", "rough_ground", "slope_up", "slope_down"}


@dataclass(frozen=True)
class ClassifierParams:
    low_confidence_threshold: float
    single_step_min_height_m: float
    roughness_threshold_m: float
    slope_threshold_deg: float
    stair_min_rise_m: float

    @classmethod
    def from_dict(cls, perception: dict) -> "ClassifierParams":
        c = perception["terrain_classifier"]
        return cls(
            low_confidence_threshold=float(c["low_confidence_threshold"]),
            single_step_min_height_m=float(c["single_step_min_height_m"]),
            roughness_threshold_m=float(c["roughness_threshold_m"]),
            slope_threshold_deg=float(perception["slope_detector"]["slope_threshold_deg"]),
            stair_min_rise_m=float(perception["stair_detector"]["min_rise_m"]),
        )


@dataclass(frozen=True)
class ClassifierInputs:
    overall_confidence: float
    slope_angle_deg: float
    slope_direction: str          # "up" | "down" | "none"
    roughness_m: float
    single_step_height_m: float
    max_obstacle_height_m: float
    stairs: StairResult
    gap: GapResult


@dataclass(frozen=True)
class Classification:
    terrain_type: str
    confidence: float
    safe_to_continue: bool
    reason: str
    slope_angle_deg: float
    max_obstacle_height_m: float
    estimated_step_height_m: float
    estimated_step_depth_m: float
    gap_width_m: float


def classify(i: ClassifierInputs, p: ClassifierParams) -> Classification:
    conf = float(i.overall_confidence)

    def out(ttype: str, reason: str, *, step_h=0.0, step_d=0.0):
        safe = ttype in _TRAVERSABLE and conf >= p.low_confidence_threshold
        return Classification(
            terrain_type=ttype,
            confidence=conf,
            safe_to_continue=safe,
            reason=reason,
            slope_angle_deg=i.slope_angle_deg,
            max_obstacle_height_m=i.max_obstacle_height_m,
            estimated_step_height_m=step_h,
            estimated_step_depth_m=step_d,
            gap_width_m=i.gap.gap_width_m,
        )

    # 1. Low confidence or unknown region ahead -> unsafe.
    if conf < p.low_confidence_threshold:
        return out("unknown_unsafe", f"overall confidence {conf:.2f} below threshold")
    if i.gap.unknown_ahead:
        return out("unknown_unsafe", i.gap.reason)

    # 2. Gap / drop-off.
    if i.gap.gap_detected:
        return out("gap_or_hole", i.gap.reason)

    # 3. Stairs.
    if i.stairs.stairs_detected and i.stairs.rise_m >= p.stair_min_rise_m:
        ttype = "stairs_up" if i.stairs.direction == "up" else "stairs_down"
        return out(
            ttype,
            f"{ttype} rise={i.stairs.rise_m:.2f} tread={i.stairs.tread_m:.2f} "
            f"conf={i.stairs.confidence:.2f}",
            step_h=i.stairs.rise_m, step_d=i.stairs.tread_m,
        )

    # 4. Single step / curb.
    if i.single_step_height_m >= p.single_step_min_height_m:
        return out("curb_or_single_step",
                   f"single step {i.single_step_height_m:.2f} m ahead",
                   step_h=i.single_step_height_m)

    # 5. Slope.
    if abs(i.slope_angle_deg) > p.slope_threshold_deg and i.slope_direction in ("up", "down"):
        ttype = "slope_up" if i.slope_direction == "up" else "slope_down"
        return out(ttype, f"{ttype} {i.slope_angle_deg:.1f} deg")

    # 6. Rough ground.
    if i.roughness_m > p.roughness_threshold_m:
        return out("rough_ground", f"roughness {i.roughness_m:.3f} m")

    # 7. Flat.
    return out("flat_ground", "flat ground, clear")
