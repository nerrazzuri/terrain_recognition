"""Policy safety supervisor (P5-M2-T1) — pure logic, no ROS2.

Decides whether to cut the policy output and switch to damping/zero (roadmap §9.2). Fail
closed: any missing/critical condition cuts. Operator stop overrides everything. Returns
``(cut, reason)`` for the deployment log.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyState:
    roll: float
    pitch: float
    max_tilt: float
    joint_fresh: bool
    imu_fresh: bool
    inference_ok: bool          # inference completed within the period budget
    action_finite: bool         # raw action has no NaN/Inf
    target_in_limits: bool      # filtered targets inside the soft envelope
    operator_stop: bool
    base_stable: bool


def evaluate(s: PolicyState) -> tuple[bool, str]:
    if s.operator_stop:
        return True, "operator stop requested"
    if not s.joint_fresh:
        return True, "joint state missing/stale"
    if not s.imu_fresh:
        return True, "IMU missing/stale"
    if not s.inference_ok:
        return True, "policy inference timeout"
    if not s.action_finite:
        return True, "action NaN/Inf"
    if not s.target_in_limits:
        return True, "joint target outside soft limit"
    if abs(s.roll) > s.max_tilt:
        return True, f"roll {s.roll:.2f} over tilt limit {s.max_tilt:.2f}"
    if abs(s.pitch) > s.max_tilt:
        return True, f"pitch {s.pitch:.2f} over tilt limit {s.max_tilt:.2f}"
    if not s.base_stable:
        return True, "base instability detected"
    return False, "ok"
