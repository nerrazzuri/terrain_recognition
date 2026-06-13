"""x2_stairs_env_cfg (P4-M3-T4..T7) — Stages D-G: single step, stairs up/down, mixed.

Single step 2-15 cm; stairs rise 5-18 cm tread 24-35 cm; then mixed generalization (terrain
levels 3-6). Goals: clear step edge, continuous ascent/descent, safe footholds, no unsafe
stepping. BLOCKED to run: requires Isaac Lab + X2 asset.
"""
from __future__ import annotations

from isaaclab.utils import configclass  # noqa: E402

from ..rough_terrain.x2_rough_env_cfg import X2RoughEnvCfg


@configclass
class X2StairsEnvCfg(X2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # advance the curriculum into the step/stairs/mixed levels
        self.terrain_start_level = 3
        self.episode_length_s = 25.0
