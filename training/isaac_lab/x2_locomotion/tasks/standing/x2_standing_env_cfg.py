"""x2_standing_env_cfg (P3-M3-T1 / P4-M3-T1) — Stage A standing environment.

Flat ground, standing-pose target, low disturbance, terminate on fall. First sim environment
and first curriculum stage. Acceptance (roadmap §7.3): X2 stands 30 s with fixed PD,
reasonable contact forces, stable base height/orientation.

BLOCKED: requires Isaac Lab + the X2 USD asset (see x2_robot_cfg).
"""
from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402  (Isaac Lab)
from isaaclab.utils import configclass  # noqa: E402

from x2_common import config_loader
from ...robots.x2_robot_cfg import build_robot_cfg, BASE_HEIGHT_M, TERMINATION_BODY_NAMES


@configclass
class X2StandingEnvCfg(ManagerBasedRLEnvCfg):
    """Stage A: stand still on flat ground.

    Timing from configs/training_default.yaml (200 Hz physics, 50 Hz policy, decimation 4).
    Reward/observation/termination managers are wired in tasks/common (rewards.py,
    observations.py, terminations.py) once Isaac Lab is available.
    """

    def __post_init__(self):
        train = config_loader.load_config("training_default")
        sim = train["sim"]
        self.decimation = int(sim["control_decimation"])
        self.episode_length_s = 30.0
        self.sim.dt = float(sim["physics_dt_s"])
        self.scene.num_envs = int(sim["num_envs"])
        self.scene.robot = build_robot_cfg()
        # Standing target = default base height; terminate if the base falls below it.
        self.base_height_target = BASE_HEIGHT_M
        self.termination_bodies = TERMINATION_BODY_NAMES
