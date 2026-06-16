"""Canonical AimDK leg joint order — shared between ROS2 runtime and Isaac Lab training.

This is the single source of truth for the 12-DoF leg joint names and order.
The training x2_joint_map.py imports from here so both stacks stay in sync.
"""
from __future__ import annotations

_PER_LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
_JOINT_SUFFIX = "_joint"


def aimdk_leg_order() -> list[str]:
    """Canonical 12-DoF leg joint order (X2 URDF v1.3.0): left leg then right leg."""
    return [f"{side}_{j}{_JOINT_SUFFIX}" for side in ("left", "right") for j in _PER_LEG]
