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

_PER_LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
_JOINT_SUFFIX = "_joint"


class JointMapError(RuntimeError):
    """Raised when the simulator joint set does not match the expected AimDK leg joints."""


def aimdk_leg_order() -> list[str]:
    """Canonical 12-DoF leg joint order (X2 URDF v1.3.0 names): left leg then right leg."""
    return [f"{side}_{j}{_JOINT_SUFFIX}" for side in ("left", "right") for j in _PER_LEG]


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
