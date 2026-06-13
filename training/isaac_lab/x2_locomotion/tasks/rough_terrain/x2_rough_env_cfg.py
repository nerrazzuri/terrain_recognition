"""x2_rough_env_cfg (P4-M3-T3) — Stage C rough terrain.

1-5 cm height noise + mild slopes (terrain levels 1-2); goals: stable torso, foot clearance,
no tripping. BLOCKED to run: requires Isaac Lab + X2 asset.
"""
from __future__ import annotations

from isaaclab.utils import configclass  # noqa: E402

from ..flat_walk.x2_flat_walk_env_cfg import X2FlatWalkEnvCfg
from ..common.terrain_generator import build_terrain_cfg


@configclass
class X2RoughEnvCfg(X2FlatWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # progressive terrain; curriculum starts at the rough/slope levels
        self.scene.terrain = build_terrain_cfg()
        self.terrain_start_level = 1
