"""Unit tests for the safe-locomotion core policy + safety supervisor (pure logic).

These encode the Phase 2 safety contract (roadmap §6.4): slow on flat, stop before
stairs/gaps/unknown, smooth commands, and fail closed on any missing/critical input.
"""
import pytest

from x2_safe_locomotion.core import velocity_policy as vp
from x2_safe_locomotion.core import supervisor as sup
from x2_safe_locomotion.core import smoother as sm


VEL_BY_TERRAIN = {
    "flat_ground": 0.12, "rough_ground": 0.06, "slope_up": 0.04, "slope_down": 0.04,
    "curb_or_single_step": 0.0, "stairs_up": 0.0, "stairs_down": 0.0,
    "gap_or_hole": 0.0, "platform": 0.0, "unknown_unsafe": 0.0,
}


# --- velocity policy ---

def policy():
    return vp.VelocityPolicy(VEL_BY_TERRAIN, max_forward_mps=0.12)


def test_flat_allows_slow_forward():
    assert policy().safe_velocity(0.5, "flat_ground", safe_to_continue=True) == pytest.approx(0.12)


def test_desired_below_cap_passes_through():
    assert policy().safe_velocity(0.03, "flat_ground", safe_to_continue=True) == pytest.approx(0.03)


def test_rough_is_slower():
    assert policy().safe_velocity(0.5, "rough_ground", safe_to_continue=True) == pytest.approx(0.06)


def test_stairs_force_stop():
    assert policy().safe_velocity(0.5, "stairs_up", safe_to_continue=True) == 0.0


def test_gap_forces_stop():
    assert policy().safe_velocity(0.5, "gap_or_hole", safe_to_continue=True) == 0.0


def test_unknown_terrain_forces_stop():
    assert policy().safe_velocity(0.5, "unknown_unsafe", safe_to_continue=True) == 0.0


def test_unlisted_terrain_fails_closed():
    assert policy().safe_velocity(0.5, "something_new", safe_to_continue=True) == 0.0


def test_not_safe_to_continue_forces_stop_even_on_flat():
    assert policy().safe_velocity(0.5, "flat_ground", safe_to_continue=False) == 0.0


def test_negative_desired_never_drives_backward_past_cap():
    # backward requests are clamped to >= 0 in this forward-only safe adapter
    assert policy().safe_velocity(-0.5, "flat_ground", safe_to_continue=True) == 0.0


# --- safety supervisor ---

def state(**kw):
    base = dict(
        terrain_fresh=True, imu_fresh=True, command_fresh=True,
        roll_deg=0.0, pitch_deg=0.0, max_roll_deg=15.0, max_pitch_deg=15.0,
        terrain_type="flat_ground", safe_to_continue=True, operator_estop=False,
        robot_mode_ok=True,
    )
    base.update(kw)
    return sup.SupervisorState(**base)


def test_supervisor_allows_motion_when_all_ok():
    stop, reason = sup.evaluate_stop(state())
    assert stop is False


def test_stale_terrain_stops():
    stop, reason = sup.evaluate_stop(state(terrain_fresh=False))
    assert stop and "terrain" in reason.lower()


def test_stale_imu_stops():
    stop, reason = sup.evaluate_stop(state(imu_fresh=False))
    assert stop and "imu" in reason.lower()


def test_command_timeout_stops():
    stop, reason = sup.evaluate_stop(state(command_fresh=False))
    assert stop and "command" in reason.lower()


def test_excess_roll_stops():
    stop, reason = sup.evaluate_stop(state(roll_deg=20.0))
    assert stop and "roll" in reason.lower()


def test_excess_pitch_stops():
    stop, reason = sup.evaluate_stop(state(pitch_deg=-20.0))
    assert stop and "pitch" in reason.lower()


def test_unsafe_terrain_ahead_stops():
    stop, reason = sup.evaluate_stop(state(terrain_type="stairs_up", safe_to_continue=False))
    assert stop


def test_operator_estop_overrides_everything():
    stop, reason = sup.evaluate_stop(state(operator_estop=True))
    assert stop and "operator" in reason.lower()


def test_unexpected_mode_stops():
    stop, reason = sup.evaluate_stop(state(robot_mode_ok=False))
    assert stop and "mode" in reason.lower()


# --- command smoother ---

def test_smoother_ramps_up():
    s = sm.CommandSmoother(max_forward_accel=0.05, max_yaw_accel=0.10)
    out = s.step(target_forward=0.12, target_yaw=0.0, dt=0.1)
    assert out.forward == pytest.approx(0.005)  # 0.05 * 0.1


def test_smoother_reaches_target_over_time():
    s = sm.CommandSmoother(max_forward_accel=0.05, max_yaw_accel=0.10)
    f = 0.0
    for _ in range(100):
        f = s.step(0.12, 0.0, dt=0.1).forward
    assert f == pytest.approx(0.12, abs=1e-6)


def test_emergency_stop_is_immediate_zero():
    s = sm.CommandSmoother(max_forward_accel=0.05, max_yaw_accel=0.10)
    s.step(0.12, 0.0, dt=1.0)
    out = s.emergency_stop()
    assert out.forward == 0.0 and out.yaw == 0.0
