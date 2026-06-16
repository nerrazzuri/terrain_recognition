"""Unit tests for the pure-logic Phase 4 training cores (no torch / Isaac Lab).

Covers the deployment-critical pieces: observation ordering + normalization (must match the
real-robot observation_builder exactly), reward component math, termination conditions,
curriculum progression, and domain-randomization sampling ranges.
"""
import numpy as np
import pytest

from x2_locomotion.tasks.common import observations as obs
from x2_locomotion.tasks.common import rewards as rw
from x2_locomotion.tasks.common import terminations as tm
from x2_locomotion.tasks.common import curriculum as cu
from x2_locomotion.tasks.common import domain_randomization as dr


# --- observations (P4-M1-T2) ---

def _parts():
    return {
        "command_velocity": np.zeros(3),
        "base_angular_velocity": np.zeros(3),
        "projected_gravity": np.array([0, 0, -1.0]),
        "joint_position_error": np.zeros(12),
        "joint_velocity": np.zeros(12),
        "previous_action": np.zeros(12),
        "gait_phase": np.array([0.0, 1.0]),
        "height_samples": np.zeros(121),
    }


def test_observation_dim_is_168():
    v = obs.assemble(_parts())
    assert v.shape == (168,)


def test_observation_order_is_stable():
    parts = _parts()
    parts["command_velocity"] = np.array([1.0, 2.0, 3.0])
    v = obs.assemble(parts)
    np.testing.assert_allclose(v[:3], [1.0, 2.0, 3.0])  # command velocity is first


def test_observation_missing_part_raises():
    parts = _parts()
    del parts["gait_phase"]
    with pytest.raises(obs.ObservationError):
        obs.assemble(parts)


def test_observation_wrong_dim_raises():
    parts = _parts()
    parts["height_samples"] = np.zeros(100)
    with pytest.raises(obs.ObservationError):
        obs.assemble(parts)


def test_normalizer_zero_mean_unit_std():
    n = obs.Normalizer(mean=np.full(168, 2.0), std=np.full(168, 2.0))
    v = obs.assemble(_parts())
    out = n.normalize(v)
    assert out.shape == (168,)
    # a value equal to the mean normalizes to 0
    assert out[0] == pytest.approx((0.0 - 2.0) / 2.0)


def test_normalizer_guards_zero_std():
    n = obs.Normalizer(mean=np.zeros(168), std=np.zeros(168))
    out = n.normalize(obs.assemble(_parts()))
    assert np.all(np.isfinite(out))


# --- rewards (P4-M2-T1) ---

def test_velocity_tracking_is_max_at_zero_error():
    perfect = rw.velocity_tracking(np.array([0.3, 0, 0]), np.array([0.3, 0, 0]), sigma=0.25)
    worse = rw.velocity_tracking(np.array([0.3, 0, 0]), np.array([0.0, 0, 0]), sigma=0.25)
    assert perfect == pytest.approx(1.0)
    assert worse < perfect


def test_action_rate_penalty_zero_when_unchanged():
    a = np.ones(12)
    assert rw.action_rate_penalty(a, a) == 0.0
    assert rw.action_rate_penalty(np.ones(12), np.zeros(12)) > 0.0


def test_torso_stability_penalises_tilt():
    upright = rw.torso_stability(roll=0.0, pitch=0.0, ang_vel=np.zeros(3))
    tilted = rw.torso_stability(roll=0.3, pitch=0.2, ang_vel=np.zeros(3))
    assert upright > tilted


def test_joint_limit_penalty_zero_inside_limits():
    q = np.zeros(12)
    lo = np.full(12, -1.0)
    hi = np.full(12, 1.0)
    assert rw.joint_limit_penalty(q, lo, hi) == 0.0
    q[0] = 0.99
    assert rw.joint_limit_penalty(q, lo, hi) > 0.0


# --- terminations (P4-M2-T2) ---

def test_terminate_on_low_base():
    done, reason = tm.should_terminate(base_height=0.2, min_height=0.35,
                                       roll=0.0, pitch=0.0, max_tilt=0.7, bad_contact=False)
    assert done and "height" in reason.lower()


def test_terminate_on_excess_tilt():
    done, reason = tm.should_terminate(base_height=0.55, min_height=0.35,
                                       roll=1.0, pitch=0.0, max_tilt=0.7, bad_contact=False)
    assert done and ("roll" in reason.lower() or "tilt" in reason.lower())


def test_terminate_on_bad_contact():
    done, reason = tm.should_terminate(base_height=0.55, min_height=0.35,
                                       roll=0.0, pitch=0.0, max_tilt=0.7, bad_contact=True)
    assert done


def test_no_termination_when_healthy():
    done, _ = tm.should_terminate(base_height=0.55, min_height=0.35,
                                  roll=0.0, pitch=0.0, max_tilt=0.7, bad_contact=False)
    assert not done


# --- curriculum (P4-M2-T3) ---

def test_curriculum_starts_at_first_stage():
    c = cu.Curriculum(["standing", "flat_walk", "rough"], advance_threshold=0.8)
    assert c.current_stage == "standing"


def test_curriculum_advances_on_success():
    c = cu.Curriculum(["standing", "flat_walk"], advance_threshold=0.8)
    c.update(success_rate=0.9)
    assert c.current_stage == "flat_walk"


def test_curriculum_holds_below_threshold():
    c = cu.Curriculum(["standing", "flat_walk"], advance_threshold=0.8)
    c.update(success_rate=0.5)
    assert c.current_stage == "standing"


def test_curriculum_does_not_advance_past_last():
    c = cu.Curriculum(["standing"], advance_threshold=0.8)
    c.update(success_rate=1.0)
    assert c.current_stage == "standing"
    assert c.is_complete


# --- domain randomization (P4-M2-T4) ---

def test_domain_randomization_within_ranges():
    cfg = {"body_mass_pct": 0.10, "motor_strength_pct": 0.20,
           "terrain_friction": [0.4, 1.2], "action_delay_steps": [1, 3]}
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = dr.sample(cfg, rng)
        assert 0.9 <= s["body_mass_scale"] <= 1.1
        assert 0.8 <= s["motor_strength_scale"] <= 1.2
        assert 0.4 <= s["terrain_friction"] <= 1.2
        assert 1 <= s["action_delay_steps"] <= 3


def test_domain_randomization_is_seeded_reproducible():
    cfg = {"body_mass_pct": 0.10, "terrain_friction": [0.4, 1.2]}
    a = dr.sample(cfg, np.random.default_rng(42))
    b = dr.sample(cfg, np.random.default_rng(42))
    assert a == b
