"""x2_flat_walk_env_cfg (P4-M3-T2) — Stage B flat walking.

Same env as standing, but the velocity command is a clear WALK command so one policy learns
to stand planted AND walk.

History / fix: the first run commanded only 0–0.3 m/s with ~20% standing envs. At those tiny
speeds (and reward std 0.5) a robot that just *stands still* scored ~90% of the velocity
reward with zero fall risk, so the policy learned to stand, not walk (confirmed on video).
Fix: command a clear forward speed (0.3–0.8 m/s), fewer standing envs (10%), and tighten the
tracking reward std (in the shared standing RewardsCfg) so velocity error actually bites.
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
        cmd.rel_standing_envs = 0.1          # ~10% stand; the rest get a full 0–1.0 m/s spread
        cmd.resampling_time_range = (5.0, 10.0)
        # Standard biped velocity range: include low speeds (from 0) so easy commands exist to
        # bootstrap stepping, up to 1.0 m/s. Paired with the feet_air_time gait reward, this is
        # the Unitree H1/G1 walking recipe (vs the v2 run's 0.3–0.8 with no gait reward).
        cmd.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0))
