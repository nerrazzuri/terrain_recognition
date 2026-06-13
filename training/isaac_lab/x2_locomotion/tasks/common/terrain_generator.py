"""terrain_generator (P3-M3-T2) — Isaac Lab terrain from the tested terrain specs.

Builds a progressive multi-level terrain (flat -> rough -> slope -> step -> stairs up ->
stairs down -> mixed) by translating the pure-logic specs in ``terrain_spec`` into Isaac Lab
sub-terrain configs. The numbers live in terrain_spec (unit-tested); this module is the thin
Isaac Lab adapter.

BLOCKED: requires Isaac Lab (``isaaclab.terrains``).
"""
from __future__ import annotations

import isaaclab.terrains as terrain_gen  # noqa: E402  (Isaac Lab)

from . import terrain_spec


def build_terrain_cfg(rows: int = 10, cols: int = 20, size=(8.0, 8.0)):
    """Return a TerrainGeneratorCfg with one sub-terrain per level (0-6).

    Difficulty comes from ``terrain_spec.level_params(i).difficulty`` so the curriculum and
    the sim terrain share a single source of truth.
    """
    sub_terrains = {}
    for i in range(terrain_spec.num_levels()):
        spec = terrain_spec.level_params(i)
        sub_terrains[spec.name] = _sub_terrain_for(spec)
    return terrain_gen.TerrainGeneratorCfg(
        size=size, num_rows=rows, num_cols=cols, sub_terrains=sub_terrains,
        curriculum=True, difficulty_range=(0.0, 1.0))


def _sub_terrain_for(spec: "terrain_spec.TerrainLevel"):
    """Map one TerrainLevel to an Isaac Lab sub-terrain cfg."""
    if spec.name in ("stairs_up", "stairs_down"):
        return terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=spec.stair_rise_m,
            step_width=sum(spec.stair_tread_m) / 2.0,
            platform_width=2.0,
        )
    if spec.name == "single_step":
        return terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45,
            grid_height_range=(0.02, spec.max_step_height_m), platform_width=2.0)
    if spec.name == "slope":
        return terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.0, spec.slope_deg / 60.0), platform_width=2.0)
    if spec.name in ("rough_low", "mixed"):
        return terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=spec.roughness_m, noise_step=0.01, border_width=0.25)
    # flat (level 0)
    return terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
