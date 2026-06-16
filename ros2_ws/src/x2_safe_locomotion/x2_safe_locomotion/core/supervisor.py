"""Safety supervisor decision logic — pure logic, no ROS2.

Evaluates whether motion must be stopped (roadmap §6.4). Fail closed: any missing/critical
input → stop. Operator e-stop overrides everything. Always returns a human-readable reason
for the logs. This is the safety-critical heart of Phase 2 and is exhaustively unit-tested.
"""
from __future__ import annotations

from dataclasses import dataclass

# Terrain classes the adapter is allowed to move over (others ⇒ stop).
_TRAVERSABLE = {"flat_ground", "rough_ground", "slope_up", "slope_down"}


@dataclass(frozen=True)
class SupervisorState:
    terrain_fresh: bool
    imu_fresh: bool
    command_fresh: bool
    roll_deg: float
    pitch_deg: float
    max_roll_deg: float
    max_pitch_deg: float
    terrain_type: str
    safe_to_continue: bool
    operator_estop: bool
    robot_mode_ok: bool


def evaluate_stop(s: SupervisorState) -> tuple[bool, str]:
    """Return ``(stop, reason)``. ``stop`` True means motion must be zeroed.

    Order matters: the operator e-stop is checked first so it overrides all else.
    """
    if s.operator_estop:
        return True, "operator emergency stop requested"
    if not s.robot_mode_ok:
        return True, "unexpected robot mode"
    if not s.imu_fresh:
        return True, "IMU data missing/stale"
    if not s.terrain_fresh:
        return True, "terrain status missing/stale"
    if not s.command_fresh:
        return True, "command timeout (no fresh command)"
    if abs(s.roll_deg) > s.max_roll_deg:
        return True, f"roll {s.roll_deg:.1f} deg over limit {s.max_roll_deg:.1f}"
    if abs(s.pitch_deg) > s.max_pitch_deg:
        return True, f"pitch {s.pitch_deg:.1f} deg over limit {s.max_pitch_deg:.1f}"
    if not s.safe_to_continue or s.terrain_type not in _TRAVERSABLE:
        return True, f"unsafe terrain ahead: {s.terrain_type}"
    return False, "ok"
