"""x2_actuator_cfg (P3-M2-T4) — actuator / PD parameters for ALL X2 joints.

Builds Isaac Lab ImplicitActuatorCfg groups covering all 31 joints (legs + waist + head +
arms), keyed by joint-name regex. Every joint must be actuated or the unconstrained upper
body folds the robot over (legs alone cannot hold it). Stiffness/damping are PD gains for the
implicit position drive; first estimates, validated by the spawn-and-stand check
(scripts/spawn_x2.py). Effort/velocity limits come from the USD (imported from the URDF), so
we don't set the deprecated effort_limit/velocity_limit here.

BLOCKED: requires Isaac Lab installed. Importing this module needs ``isaaclab``.
"""
from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402  (Isaac Lab)

# PD gains by joint group (implicit position drive). Tuned so the default pose holds upright.
_GROUPS = {
    "hips_knees": {
        "expr": [".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"],
        "stiffness": 300.0, "damping": 8.0,
    },
    "ankles": {
        "expr": [".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        "stiffness": 120.0, "damping": 4.0,
    },
    "waist": {
        "expr": [".*waist.*"],
        "stiffness": 300.0, "damping": 8.0,
    },
    "arms": {
        "expr": [".*shoulder.*", ".*elbow.*", ".*wrist.*"],
        "stiffness": 80.0, "damping": 3.0,
    },
    "head": {
        "expr": [".*head.*"],
        "stiffness": 20.0, "damping": 1.0,
    },
}


def build_actuator_cfgs() -> dict[str, ImplicitActuatorCfg]:
    """Return actuator groups covering all 31 joints (legs + waist + head + arms)."""
    return {
        name: ImplicitActuatorCfg(
            joint_names_expr=list(g["expr"]),
            stiffness=g["stiffness"],
            damping=g["damping"],
        )
        for name, g in _GROUPS.items()
    }
