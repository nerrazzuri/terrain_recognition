"""Domain randomization sampling (P4-M2-T4). Pure numpy, no Isaac Lab.

Samples per-episode physics/sensor randomizations within the configured ranges (roadmap
§8.10), mandatory for sim-to-real transfer. Pure + seedable so it is reproducible and
unit-testable; the Isaac Lab env applies the sampled values to the scene.
"""
from __future__ import annotations

import numpy as np


def _scale_from_pct(pct: float, rng) -> float:
    """Sample a multiplicative scale in [1-pct, 1+pct]."""
    return float(rng.uniform(1.0 - pct, 1.0 + pct))


def sample(cfg: dict, rng) -> dict:
    """Return a dict of sampled randomization values within the configured ranges."""
    out: dict = {}
    if "body_mass_pct" in cfg:
        out["body_mass_scale"] = _scale_from_pct(float(cfg["body_mass_pct"]), rng)
    if "link_inertia_pct" in cfg:
        out["link_inertia_scale"] = _scale_from_pct(float(cfg["link_inertia_pct"]), rng)
    if "motor_strength_pct" in cfg:
        out["motor_strength_scale"] = _scale_from_pct(float(cfg["motor_strength_pct"]), rng)
    if "pd_gain_pct" in cfg:
        out["pd_gain_scale"] = _scale_from_pct(float(cfg["pd_gain_pct"]), rng)
    if "com_offset_m" in cfg:
        m = float(cfg["com_offset_m"])
        out["com_offset"] = rng.uniform(-m, m, size=3).tolist()
    if "terrain_friction" in cfg:
        lo, hi = cfg["terrain_friction"]
        out["terrain_friction"] = float(rng.uniform(lo, hi))
    if "action_delay_steps" in cfg:
        lo, hi = cfg["action_delay_steps"]
        out["action_delay_steps"] = int(rng.integers(lo, hi + 1))
    if "sensor_latency_ms" in cfg:
        lo, hi = cfg["sensor_latency_ms"]
        out["sensor_latency_ms"] = float(rng.uniform(lo, hi))
    if "heightmap_noise_m" in cfg:
        out["heightmap_noise_m"] = float(cfg["heightmap_noise_m"])
    return out
