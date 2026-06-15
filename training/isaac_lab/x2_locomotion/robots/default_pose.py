"""Default X2 standing pose — pure data, no Isaac Lab dependency.

Shared by the Isaac Lab robot cfg and the MuJoCo standing controller so both use one source.
Leg joint names are the real X2 URDF v1.3.0 names (``*_joint``). Slightly bent knees; validate
that it stands under PD (training/mujoco/scripts/stand.py).
"""
from __future__ import annotations

BASE_HEIGHT_M = 0.67  # validated stand height in MuJoCo (soles on floor, straight legs)

# Validated standing pose: straight legs (zeros) stands stably under PD in MuJoCo
# (training/mujoco/scripts/stand.py -> base_z ~0.67, upright ~1.0 for 3 s). A slightly
# bent-knee nominal pose is preferable for locomotion but needs gravity compensation / higher
# gains to hold statically — revisit when the RL standing task (Stage A) is tuned.
DEFAULT_LEG_POSE = {
    "left_hip_pitch_joint": 0.0, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0, "left_ankle_pitch_joint": 0.0, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0, "right_ankle_pitch_joint": 0.0, "right_ankle_roll_joint": 0.0,
}
