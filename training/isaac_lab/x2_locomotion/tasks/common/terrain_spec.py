"""Progressive terrain level specs (P3-M3-T2). Pure logic, no Isaac Lab.

Defines parameters for terrain levels 0-6 (flat -> rough -> slope -> single step ->
stairs up -> stairs down -> mixed) per roadmap §7.4. The Isaac Lab terrain_generator adapter
consumes these specs to build the actual sub-terrains; keeping the numbers here makes them
testable and the single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TerrainLevel:
    level: int
    name: str
    difficulty: float
    roughness_m: tuple[float, float] = (0.0, 0.0)
    slope_deg: float = 0.0
    max_step_height_m: float = 0.0
    stair_rise_m: tuple[float, float] = (0.0, 0.0)
    stair_tread_m: tuple[float, float] = (0.0, 0.0)
    friction: tuple[float, float] = (0.4, 1.2)
    components: tuple[str, ...] = field(default_factory=tuple)


_LEVELS = {
    0: TerrainLevel(0, "flat", 0.0),
    1: TerrainLevel(1, "rough_low", 0.15, roughness_m=(0.01, 0.05)),
    2: TerrainLevel(2, "slope", 0.30, slope_deg=12.0, roughness_m=(0.0, 0.02)),
    3: TerrainLevel(3, "single_step", 0.45, max_step_height_m=0.15),
    4: TerrainLevel(4, "stairs_up", 0.65, stair_rise_m=(0.05, 0.18), stair_tread_m=(0.24, 0.35)),
    5: TerrainLevel(5, "stairs_down", 0.80, stair_rise_m=(0.05, 0.15), stair_tread_m=(0.24, 0.35)),
    6: TerrainLevel(6, "mixed", 1.0, roughness_m=(0.01, 0.05), slope_deg=8.0,
                    max_step_height_m=0.12, stair_rise_m=(0.05, 0.15), stair_tread_m=(0.24, 0.35),
                    components=("flat", "rough", "curb", "stairs", "platform", "gap_edge")),
}


def level_params(level: int) -> TerrainLevel:
    """Return the spec for a terrain level (0-6). Raises ValueError for unknown levels."""
    if level not in _LEVELS:
        raise ValueError(f"unknown terrain level {level}; valid: {sorted(_LEVELS)}")
    return _LEVELS[level]


def num_levels() -> int:
    return len(_LEVELS)
