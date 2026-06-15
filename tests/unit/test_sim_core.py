"""Unit tests for the testable Phase 3 sim cores: joint map, height sampling, terrain spec.

These are pure logic (no Isaac Lab) and cover the safety-relevant pieces: joint ordering
(sim<->AimDK), height-sample extraction, and progressive terrain parameters.
"""
import numpy as np
import pytest

from x2_locomotion.robots import x2_joint_map as jm
from x2_locomotion.tasks.common import height_samples as hs
from x2_locomotion.tasks.common import terrain_spec as ts


# --- joint map (P3-M2-T1) ---

def test_aimdk_leg_order_is_12_left_then_right():
    order = jm.aimdk_leg_order()
    assert len(order) == 12
    assert order[0] == "left_hip_pitch_joint"   # X2 URDF v1.3.0 names
    assert order[6] == "right_hip_pitch_joint"
    assert order[5] == "left_ankle_roll_joint"


def test_round_trip_sim_to_aimdk_and_back():
    sim_names = list(reversed(jm.aimdk_leg_order()))  # some other ordering
    m = jm.JointMap(sim_names)
    sim_vec = np.arange(12, dtype=float)
    aimdk_vec = m.to_aimdk(sim_vec)
    back = m.to_sim(aimdk_vec)
    np.testing.assert_allclose(back, sim_vec)


def test_to_aimdk_reorders_correctly():
    sim_names = list(reversed(jm.aimdk_leg_order()))
    m = jm.JointMap(sim_names)
    # value i sits at sim position i; after mapping it must sit at aimdk index of that name
    sim_vec = np.arange(12, dtype=float)
    aimdk_vec = m.to_aimdk(sim_vec)
    for sim_idx, name in enumerate(sim_names):
        aimdk_idx = jm.aimdk_leg_order().index(name)
        assert aimdk_vec[aimdk_idx] == sim_vec[sim_idx]


def test_missing_joint_raises():
    with pytest.raises(jm.JointMapError):
        jm.JointMap(jm.aimdk_leg_order()[:-1])  # missing one joint


def test_left_right_counts_balanced():
    order = jm.aimdk_leg_order()
    assert sum(n.startswith("left_") for n in order) == 6
    assert sum(n.startswith("right_") for n in order) == 6


def test_leg_order_is_prefix_of_verified_body_order():
    # legs must be the first 12 of the full AimDK body order (robot MC robot_model.yaml)
    assert jm.aimdk_leg_order() == jm.AIMDK_BODY_ORDER[:12]
    assert len(jm.AIMDK_BODY_ORDER) == 31   # 12 leg + 3 waist + 2 head + 14 arm (x2_31dof)


# --- height samples (P3-M3-T3) ---

def test_height_grid_shape():
    spec = hs.SampleGrid(nx=11, ny=11, x_min=-0.4, x_max=1.2, y_min=-0.5, y_max=0.5)
    assert spec.points_base().shape == (121, 2)


def test_samples_flat_ground_relative_to_base():
    spec = hs.SampleGrid(nx=11, ny=11, x_min=-0.4, x_max=1.2, y_min=-0.5, y_max=0.5)

    def flat(x, y):  # ground at z=0 everywhere
        return np.zeros_like(x)

    out = hs.sample_heights(spec, flat, base_xy=(0.0, 0.0), base_yaw=0.0, base_height=0.55)
    # relative to a base 0.55 m up, flat ground is -0.55 everywhere
    np.testing.assert_allclose(out, -0.55, atol=1e-6)
    assert out.shape == (121,)


def test_samples_respect_yaw_rotation():
    spec = hs.SampleGrid(nx=3, ny=1, x_min=1.0, x_max=1.0, y_min=0.0, y_max=0.0)

    def ramp(x, y):  # height rises with world-x
        return 0.1 * x

    straight = hs.sample_heights(spec, ramp, (0.0, 0.0), 0.0, 0.0)
    rotated = hs.sample_heights(spec, ramp, (0.0, 0.0), np.pi / 2, 0.0)
    # yaw 90deg turns forward(+x) into world +y, where the ramp is flat -> different samples
    assert not np.allclose(straight, rotated)


# --- terrain spec (P3-M3-T2) ---

def test_terrain_levels_0_to_6_exist():
    for level in range(7):
        spec = ts.level_params(level)
        assert spec.name


def test_flat_level_has_no_obstacles():
    spec = ts.level_params(0)
    assert spec.max_step_height_m == 0.0
    assert spec.slope_deg == 0.0


def test_stairs_up_level_has_valid_rise_tread():
    spec = ts.level_params(4)
    assert spec.stair_rise_m[0] >= 0.05 and spec.stair_rise_m[1] <= 0.18
    assert spec.stair_tread_m[0] >= 0.24


def test_difficulty_is_monotonic():
    diffs = [ts.level_params(i).difficulty for i in range(7)]
    assert diffs == sorted(diffs)


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        ts.level_params(99)
