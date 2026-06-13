"""x2_robot_cfg (P3-M2-T5) — Isaac Lab articulation config for the X2 legs.

Defines the asset path, default standing pose, base height, joint limits (from config), PD
gains (from x2_actuator_cfg), and contact/termination bodies and feet names.

BLOCKED on two things:
  1. Isaac Lab installed (``isaaclab``);
  2. the X2 USD asset (P3-M1-T1/T2) — set ``X2_USD_PATH`` or drop it under
     training/isaac_lab/assets/x2.usd. Until then ``X2_ROBOT_CFG`` cannot spawn.

Acceptance (roadmap §7.3): spawns without exploding, stands under gravity with stable PD,
correct foot contact, no major mesh/collision mismatch — verify once the asset exists.
"""
from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils  # noqa: E402  (Isaac Lab)
from isaaclab.assets import ArticulationCfg  # noqa: E402

from x2_common import config_loader
from .x2_actuator_cfg import build_actuator_cfgs

_DEFAULT_USD = Path(__file__).resolve().parents[2] / "assets" / "x2.usd"
X2_USD_PATH = os.environ.get("X2_USD_PATH", str(_DEFAULT_USD))

# Default standing pose (rad), canonical AimDK leg order. PLACEHOLDER — validate in sim.
DEFAULT_LEG_POSE = {
    "left_hip_pitch": -0.20, "left_hip_roll": 0.0, "left_hip_yaw": 0.0,
    "left_knee": 0.45, "left_ankle_pitch": -0.25, "left_ankle_roll": 0.0,
    "right_hip_pitch": -0.20, "right_hip_roll": 0.0, "right_hip_yaw": 0.0,
    "right_knee": 0.45, "right_ankle_pitch": -0.25, "right_ankle_roll": 0.0,
}
BASE_HEIGHT_M = 0.55  # initial spawn height — validate against the real model

FEET_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]      # verify in URDF
TERMINATION_BODY_NAMES = ["torso_link", "head_link", "left_knee_link", "right_knee_link"]


def build_robot_cfg() -> ArticulationCfg:
    limits = config_loader.load_config("joint_limits_x2_ultra")
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(usd_path=X2_USD_PATH),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, BASE_HEIGHT_M),
            joint_pos=dict(DEFAULT_LEG_POSE),
        ),
        actuators=build_actuator_cfgs(),
        soft_joint_pos_limit_factor=1.0 - float(limits["soft_limit_margin_rad"]),
    )


def assets_available() -> bool:
    """True if the X2 USD asset exists (P3-M1 gate)."""
    return Path(X2_USD_PATH).is_file()
