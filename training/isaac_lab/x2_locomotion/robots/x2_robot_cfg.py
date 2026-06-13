"""x2_robot_cfg (P3-M2-T5) — Isaac Lab articulation config for the X2 legs.

Defines the asset path, default standing pose, base height, joint limits (from config), PD
gains (from x2_actuator_cfg), and contact/termination bodies and feet names.

Assets are present (X2_URDF-v1.3.0): ``assets/x2_ultra_simple_collision.urdf`` + meshes. Isaac
Lab can import the URDF directly, or convert it to USD once (see assets/SOURCES.md); set
``X2_USD_PATH`` to use a converted USD. Still BLOCKED to spawn until Isaac Lab is installed;
the spawn-and-stand AC (roadmap §7.3) must be validated there.
"""
from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils  # noqa: E402  (Isaac Lab)
from isaaclab.assets import ArticulationCfg  # noqa: E402

from x2_common import config_loader
from .x2_actuator_cfg import build_actuator_cfgs

_ASSETS = Path(__file__).resolve().parents[2] / "assets"
X2_URDF_PATH = os.environ.get("X2_URDF_PATH", str(_ASSETS / "x2_ultra_simple_collision.urdf"))
X2_USD_PATH = os.environ.get("X2_USD_PATH", str(_ASSETS / "x2.usd"))

# Default standing pose (rad), real X2 URDF v1.3.0 leg joint names. Slightly bent knees;
# validate in sim (stable PD stand). Knee min is 0 so the bend must stay positive.
DEFAULT_LEG_POSE = {
    "left_hip_pitch_joint": -0.20, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.45, "left_ankle_pitch_joint": -0.25, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.20, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.45, "right_ankle_pitch_joint": -0.25, "right_ankle_roll_joint": 0.0,
}
BASE_HEIGHT_M = 0.55  # initial spawn height — validate against the real model

# Link names from X2_URDF-v1.3.0 meshes.
FEET_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
TERMINATION_BODY_NAMES = ["torso_link", "head_pitch_link", "head_yaw_link",
                          "left_knee_link", "right_knee_link"]


def _spawn_cfg():
    """Spawn from USD if a converted asset exists, else from the URDF importer."""
    if Path(X2_USD_PATH).is_file():
        return sim_utils.UsdFileCfg(usd_path=X2_USD_PATH)
    return sim_utils.UrdfFileCfg(asset_path=X2_URDF_PATH, fix_base=False)


def build_robot_cfg() -> ArticulationCfg:
    limits = config_loader.load_config("joint_limits_x2_ultra")
    return ArticulationCfg(
        spawn=_spawn_cfg(),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, BASE_HEIGHT_M),
            joint_pos=dict(DEFAULT_LEG_POSE),
        ),
        actuators=build_actuator_cfgs(),
        soft_joint_pos_limit_factor=1.0 - float(limits["soft_limit_margin_rad"]),
    )


def assets_available() -> bool:
    """True if a usable X2 asset exists (URDF present, or a converted USD)."""
    return Path(X2_URDF_PATH).is_file() or Path(X2_USD_PATH).is_file()
