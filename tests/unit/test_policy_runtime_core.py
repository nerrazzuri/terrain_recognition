"""Unit tests for the Phase 5 policy-runtime cores (pure logic, no torch / ROS2).

These are safety-critical: the action filter and policy safety supervisor are the last line
of defence before a real leg joint command. Fail closed everywhere; reject NaN/Inf; never let
an output exceed the joint soft limits.
"""
import numpy as np
import pytest

from x2_policy_runtime.core import action_filter as af
from x2_policy_runtime.core import policy_supervisor as ps
from x2_policy_runtime.core import observation_builder as ob


def limits():
    lo = np.full(12, -1.0)
    hi = np.full(12, 1.0)
    return lo, hi


# --- action filter (P5-M1-T3) ---

def filt():
    lo, hi = limits()
    return af.ActionFilter(q_min=lo, q_max=hi, max_rate=0.5, max_joint_vel=2.0,
                           low_pass_alpha=1.0, soft_margin=0.05)


def test_clamps_to_soft_limits():
    f = filt()
    out, ok = f.filter(np.full(12, 5.0), prev=np.zeros(12), dt=1.0)
    assert ok
    assert np.all(out <= 1.0 - 0.05 + 1e-9)


def test_rate_limit_blocks_spikes():
    f = filt()
    out, ok = f.filter(np.full(12, 1.0), prev=np.zeros(12), dt=0.1)
    # max_rate 0.5/s * 0.1 s = 0.05 step max
    assert np.all(np.abs(out) <= 0.05 + 1e-9)


def test_nonfinite_action_is_rejected():
    f = filt()
    bad = np.zeros(12)
    bad[3] = np.nan
    out, ok = f.filter(bad, prev=np.zeros(12), dt=0.1)
    assert not ok  # caller must safe-stop


def test_inf_action_is_rejected():
    f = filt()
    bad = np.zeros(12)
    bad[0] = np.inf
    _, ok = f.filter(bad, prev=np.zeros(12), dt=0.1)
    assert not ok


def test_low_pass_smooths():
    lo, hi = limits()
    f = af.ActionFilter(lo, hi, max_rate=10.0, max_joint_vel=10.0,
                        low_pass_alpha=0.5, soft_margin=0.05)
    out, ok = f.filter(np.full(12, 0.4), prev=np.zeros(12), dt=1.0)
    assert ok and np.allclose(out, 0.2)  # 0.5*0.4 + 0.5*0.0


def test_output_never_exceeds_limits_under_extremes():
    f = filt()
    prev = np.zeros(12)
    for _ in range(50):
        raw = np.random.default_rng().uniform(-100, 100, 12)
        out, ok = f.filter(raw, prev, dt=0.02)
        if ok:
            assert np.all(out <= 1.0 - 0.05 + 1e-9)
            assert np.all(out >= -1.0 + 0.05 - 1e-9)
            prev = out


# --- policy safety supervisor (P5-M2-T1) ---

def pstate(**kw):
    base = dict(
        roll=0.0, pitch=0.0, max_tilt=0.7, joint_fresh=True, imu_fresh=True,
        inference_ok=True, action_finite=True, target_in_limits=True,
        operator_stop=False, base_stable=True,
    )
    base.update(kw)
    return ps.PolicyState(**base)


def test_supervisor_allows_when_ok():
    cut, reason = ps.evaluate(pstate())
    assert not cut


def test_supervisor_cuts_on_tilt():
    cut, reason = ps.evaluate(pstate(roll=1.0))
    assert cut and ("roll" in reason.lower() or "tilt" in reason.lower())


def test_supervisor_cuts_on_missing_joint_state():
    cut, _ = ps.evaluate(pstate(joint_fresh=False))
    assert cut


def test_supervisor_cuts_on_inference_timeout():
    cut, reason = ps.evaluate(pstate(inference_ok=False))
    assert cut and "infer" in reason.lower()


def test_supervisor_cuts_on_nonfinite_action():
    cut, _ = ps.evaluate(pstate(action_finite=False))
    assert cut


def test_supervisor_cuts_on_target_outside_limit():
    cut, _ = ps.evaluate(pstate(target_in_limits=False))
    assert cut


def test_supervisor_operator_stop_overrides():
    cut, reason = ps.evaluate(pstate(operator_stop=True))
    assert cut and "operator" in reason.lower()


# --- observation builder (P5-M1-T1) ---

def test_observation_builder_matches_training_dim():
    parts = {name: np.zeros(dim) for name, dim in ob.OBSERVATION_LAYOUT}
    v, ok = ob.build(parts, ob.Normalizer.identity())
    assert ok and v.shape == (168,)


def test_observation_builder_missing_input_safe_stops():
    parts = {name: np.zeros(dim) for name, dim in ob.OBSERVATION_LAYOUT}
    del parts["joint_velocity"]
    v, ok = ob.build(parts, ob.Normalizer.identity())
    assert not ok  # missing sensor -> safe stop


def test_observation_builder_nonfinite_safe_stops():
    parts = {name: np.zeros(dim) for name, dim in ob.OBSERVATION_LAYOUT}
    parts["base_angular_velocity"] = np.array([np.nan, 0, 0])
    v, ok = ob.build(parts, ob.Normalizer.identity())
    assert not ok


def test_runtime_layout_matches_training_layout():
    # the deployment observation layout MUST equal the training one exactly
    from x2_locomotion.tasks.common.observations import OBSERVATION_LAYOUT as TRAIN
    assert ob.OBSERVATION_LAYOUT == TRAIN
