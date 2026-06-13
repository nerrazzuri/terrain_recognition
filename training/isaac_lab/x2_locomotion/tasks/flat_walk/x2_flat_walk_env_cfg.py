"""x2_flat_walk_env_cfg (P4-M3-T2) — Stage B flat walking.

Flat ground; commands fwd 0-0.3 m/s, yaw ±0.3 rad/s; goals: stable walking, low foot slip,
reasonable energy. BLOCKED to run: requires Isaac Lab + X2 asset.
"""
from __future__ import annotations

from isaaclab.utils import configclass  # noqa: E402

from ..standing.x2_standing_env_cfg import X2StandingEnvCfg


@configclass
class X2FlatWalkEnvCfg(X2StandingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.command_ranges = {
            "forward_velocity_mps": (0.0, 0.3),
            "lateral_velocity_mps": (0.0, 0.0),
            "yaw_velocity_radps": (-0.3, 0.3),
        }
        self.episode_length_s = 20.0
