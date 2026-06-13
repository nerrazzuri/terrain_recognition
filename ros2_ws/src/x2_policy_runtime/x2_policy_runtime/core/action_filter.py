"""Action filter (P5-M1-T3) — pure logic, no ROS2 / torch.

The last numeric guard before a real leg-joint target. Applies, in order: NaN/Inf rejection,
soft-limit clamp, per-joint rate limit, joint-velocity limit, low-pass smoothing. Returns
``(filtered, ok)``; ``ok`` False means the caller must safe-stop (roadmap §9.2 action_filter).

No output may ever exceed the configured soft limits — exhaustively unit-tested with extreme
inputs.
"""
from __future__ import annotations

import numpy as np


class ActionFilter:
    def __init__(self, q_min, q_max, max_rate: float, max_joint_vel: float,
                 low_pass_alpha: float, soft_margin: float):
        self.q_min = np.asarray(q_min, dtype=float)
        self.q_max = np.asarray(q_max, dtype=float)
        self.max_rate = float(max_rate)
        self.max_joint_vel = float(max_joint_vel)
        self.alpha = float(low_pass_alpha)
        self.margin = float(soft_margin)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("low_pass_alpha must be in (0, 1]")

    @classmethod
    def from_configs(cls, joint_limits: dict, safety_limits: dict, order: list[str]):
        af = safety_limits["action_filter"]
        joints = joint_limits["joints"]
        q_min = np.array([joints[n]["min"] for n in order], dtype=float)
        q_max = np.array([joints[n]["max"] for n in order], dtype=float)
        # max_rate is a per-step magnitude derived from joint target rate (rad/s).
        return cls(q_min, q_max,
                   max_rate=float(af["max_joint_target_rate_radps"]),
                   max_joint_vel=float(af["max_joint_velocity_radps"]),
                   low_pass_alpha=float(af["low_pass_alpha"]),
                   soft_margin=float(joint_limits["soft_limit_margin_rad"]))

    def filter(self, raw, prev, dt: float):
        """Return ``(filtered_action, ok)``. ``ok`` False ⇒ caller must safe-stop."""
        raw = np.asarray(raw, dtype=float)
        prev = np.asarray(prev, dtype=float)
        if dt <= 0:
            raise ValueError("dt must be > 0")
        if not np.all(np.isfinite(raw)):
            return prev, False

        # 1. low-pass toward the raw target
        target = self.alpha * raw + (1.0 - self.alpha) * prev
        # 2. rate limit (per joint, magnitude max_rate * dt) and joint-velocity limit
        step_cap = min(self.max_rate, self.max_joint_vel) * dt
        delta = np.clip(target - prev, -step_cap, step_cap)
        out = prev + delta
        # 3. hard soft-limit clamp (never exceed)
        lo = self.q_min + self.margin
        hi = self.q_max - self.margin
        out = np.clip(out, lo, hi)
        return out, True
