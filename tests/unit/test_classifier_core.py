"""Unit tests for the terrain-classification decision logic (roadmap §5.4).

This is the safety-critical fusion step. Precedence:
low-confidence -> gap -> stairs -> single step -> slope -> rough -> flat.
``safe_to_continue`` is never true under uncertainty or for curb/stairs/gap/unknown.
"""
import pytest

from x2_terrain_perception.core import classifier as cl
from x2_terrain_perception.core.stairs import StairResult
from x2_terrain_perception.core.gaps import GapResult


def no_stairs():
    return StairResult(False, "none", 0.0, 0.0, 0.0, 0, 0.0, 0.0)


def no_gap():
    return GapResult(False, 0.0, 0.0, False, "")


def inp(**kw):
    base = dict(
        overall_confidence=0.9, slope_angle_deg=0.0, slope_direction="none",
        roughness_m=0.0, single_step_height_m=0.0, max_obstacle_height_m=0.0,
        stairs=no_stairs(), gap=no_gap(),
    )
    base.update(kw)
    return cl.ClassifierInputs(**base)


def prm(**kw):
    base = dict(
        low_confidence_threshold=0.4, single_step_min_height_m=0.05,
        roughness_threshold_m=0.03, slope_threshold_deg=6.0, stair_min_rise_m=0.05,
    )
    base.update(kw)
    return cl.ClassifierParams(**base)


def test_low_confidence_is_unknown_unsafe():
    out = cl.classify(inp(overall_confidence=0.2), prm())
    assert out.terrain_type == "unknown_unsafe"
    assert not out.safe_to_continue


def test_gap_takes_precedence_over_stairs():
    out = cl.classify(
        inp(gap=GapResult(True, 0.3, 1.0, False, "drop"),
            stairs=StairResult(True, "up", 0.9, 0.15, 0.30, 3, 0.6, 0.3)),
        prm(),
    )
    assert out.terrain_type == "gap_or_hole"
    assert not out.safe_to_continue


def test_stairs_up_classified():
    out = cl.classify(
        inp(stairs=StairResult(True, "up", 0.9, 0.15, 0.30, 3, 0.6, 0.3)), prm())
    assert out.terrain_type == "stairs_up"
    assert not out.safe_to_continue


def test_stairs_down_classified():
    out = cl.classify(
        inp(stairs=StairResult(True, "down", 0.9, 0.15, 0.30, 3, 0.6, 0.3)), prm())
    assert out.terrain_type == "stairs_down"


def test_single_step_is_curb():
    out = cl.classify(inp(single_step_height_m=0.10), prm())
    assert out.terrain_type == "curb_or_single_step"
    assert not out.safe_to_continue


def test_slope_up():
    out = cl.classify(inp(slope_angle_deg=9.0, slope_direction="up"), prm())
    assert out.terrain_type == "slope_up"
    assert out.safe_to_continue          # mild slope is traversable (slowly)


def test_rough_ground():
    out = cl.classify(inp(roughness_m=0.05), prm())
    assert out.terrain_type == "rough_ground"
    assert out.safe_to_continue


def test_flat_ground_is_safe():
    out = cl.classify(inp(), prm())
    assert out.terrain_type == "flat_ground"
    assert out.safe_to_continue
    assert out.confidence == pytest.approx(0.9)


def test_reason_always_populated():
    for out in [cl.classify(inp(overall_confidence=0.1), prm()), cl.classify(inp(), prm())]:
        assert isinstance(out.reason, str) and out.reason
