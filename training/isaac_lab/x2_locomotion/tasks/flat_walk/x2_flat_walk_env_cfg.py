"""x2_flat_walk_env_cfg (P4-M3-T2) — Stage B flat walking.

Same env as standing, but the velocity command is non-zero (fwd 0–0.3 m/s, yaw ±0.3 rad/s,
roadmap §8.8) and ~20% of envs still get a zero command, so one policy learns to stand
planted AND walk. The locomotion rewards (velocity tracking, feet air-time, hip-deviation)
live in the shared standing env; this only widens the command.
"""
from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from ..standing.x2_standing_env_cfg import X2StandingEnvCfg


@configclass
class X2FlatWalkEnvCfg(X2StandingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        cmd = self.commands.base_velocity
        cmd.rel_standing_envs = 0.2          # ~20% of envs command "stand still"
        cmd.resampling_time_range = (5.0, 10.0)
        cmd.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.3), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.3, 0.3))
