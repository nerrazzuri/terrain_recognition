"""Command smoother — pure logic, no ROS2.

Applies acceleration ramp limits to forward/yaw velocity so commands are smooth, not jerky
(roadmap §6.4). Emergency stop bypasses the ramp and outputs zero immediately.
"""
from __future__ import annotations

from dataclasses import dataclass

from x2_common.safety_limits import rate_limit


@dataclass(frozen=True)
class Command:
    forward: float
    yaw: float


class CommandSmoother:
    def __init__(self, max_forward_accel: float, max_yaw_accel: float):
        self.max_forward_accel = float(max_forward_accel)
        self.max_yaw_accel = float(max_yaw_accel)
        self._forward = 0.0
        self._yaw = 0.0

    def step(self, target_forward: float, target_yaw: float, dt: float) -> Command:
        """Advance one tick, ramping toward the targets within accel limits."""
        if dt <= 0:
            raise ValueError("dt must be > 0")
        self._forward = rate_limit(self._forward, target_forward, self.max_forward_accel * dt)
        self._yaw = rate_limit(self._yaw, target_yaw, self.max_yaw_accel * dt)
        return Command(self._forward, self._yaw)

    def emergency_stop(self) -> Command:
        """Immediate zero, bypassing ramps."""
        self._forward = 0.0
        self._yaw = 0.0
        return Command(0.0, 0.0)

    @property
    def current(self) -> Command:
        return Command(self._forward, self._yaw)
