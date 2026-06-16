"""x2_joint_map (P3-M2-T1) — sim <-> AimDK joint ordering. Pure logic, no Isaac Lab.

Joint order is never assumed (AGENTS.md §3): the canonical AimDK leg order is left leg then
right leg, each ``hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll``. A JointMap
reorders vectors between an arbitrary simulator joint order and this canonical order, and
verifies the joint set matches. Limits are loaded from config, never hardcoded here.

Names match the official X2 URDF v1.3.0 (``*_joint`` suffix). The index order of the AimDK
``aimdk_msgs/msg/JointCommandArray`` for ``/aima/hal/joint/leg/command`` must still be
confirmed against ``/aima/hal/joint/leg/state`` on the real robot — that runtime check is the
whole point of this map (use tools/check_joint_order.py).
"""
from __future__ import annotations

import numpy as np

from x2_common.joint_map import aimdk_leg_order  # noqa: F401 — single source of truth

# Full AimDK body joint order — VERIFIED against the robot MC param
# (software/mc_param/robot/lx2501_3_t2d5/robot_model.yaml): legs(12) -> waist(3) -> head(2)
# -> left arm(7) -> right arm(7) = 31 (the "x2_31dof" model). The leg slice (indices 0-11) is
# what the 12-DoF policy drives; this documents the rest for waist/arm expansion (roadmap §8.4).
AIMDK_BODY_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
]


class JointMapError(RuntimeError):
    """Raised when the simulator joint set does not match the expected AimDK leg joints."""


class JointMap:
    """Bidirectional index map between a simulator joint order and the AimDK order."""

    def __init__(self, sim_joint_names: list[str]):
        self.sim_joint_names = list(sim_joint_names)
        self.aimdk_names = aimdk_leg_order()
        expected = set(self.aimdk_names)
        got = set(self.sim_joint_names)
        if got != expected:
            missing = expected - got
            extra = got - expected
            raise JointMapError(
                f"sim joints do not match AimDK leg set; missing={sorted(missing)} "
                f"extra={sorted(extra)}")
        # sim_to_aimdk[a] = index in sim vector of the joint at AimDK index a
        self._sim_index = {name: i for i, name in enumerate(self.sim_joint_names)}
        self._sim_to_aimdk = [self._sim_index[name] for name in self.aimdk_names]
        self._aimdk_index = {name: i for i, name in enumerate(self.aimdk_names)}
        self._aimdk_to_sim = [self._aimdk_index[name] for name in self.sim_joint_names]

    def to_aimdk(self, sim_vector) -> np.ndarray:
        """Reorder a length-12 simulator-ordered vector into AimDK order."""
        v = np.asarray(sim_vector, dtype=float)
        if v.shape[-1] != 12:
            raise JointMapError("expected length-12 vector")
        return v[..., self._sim_to_aimdk]

    def to_sim(self, aimdk_vector) -> np.ndarray:
        """Reorder a length-12 AimDK-ordered vector into simulator order."""
        v = np.asarray(aimdk_vector, dtype=float)
        if v.shape[-1] != 12:
            raise JointMapError("expected length-12 vector")
        return v[..., self._aimdk_to_sim]

    def describe(self) -> str:
        """Human-readable side-by-side ordering for verification logs / check tool."""
        lines = ["idx  aimdk_name           sim_index  sim_name"]
        for a, name in enumerate(self.aimdk_names):
            si = self._sim_index[name]
            lines.append(f"{a:>3}  {name:<20} {si:>9}  {self.sim_joint_names[si]}")
        return "\n".join(lines)
