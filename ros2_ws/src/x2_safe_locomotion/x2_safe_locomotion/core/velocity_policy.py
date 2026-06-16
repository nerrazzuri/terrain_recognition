"""Terrain-gated velocity policy — pure logic, no ROS2.

Maps a desired forward velocity to a safe one given the terrain type and the perception
``safe_to_continue`` flag (roadmap §6.4). Fail closed: any terrain not explicitly allowed,
or ``safe_to_continue`` false, yields zero. Forward-only safe adapter (no backward driving).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VelocityPolicy:
    velocity_by_terrain: dict[str, float]
    max_forward_mps: float

    @classmethod
    def from_dict(cls, safe_cfg: dict) -> "VelocityPolicy":
        return cls(
            velocity_by_terrain=dict(safe_cfg["velocity_by_terrain"]),
            max_forward_mps=float(safe_cfg["velocity_limits"]["max_forward_mps"]),
        )

    def max_forward_for(self, terrain_type: str) -> float:
        """Configured cap for a terrain type; unknown/unlisted types fail closed to 0."""
        return float(self.velocity_by_terrain.get(terrain_type, 0.0))

    def safe_velocity(self, desired_forward: float, terrain_type: str,
                      safe_to_continue: bool) -> float:
        if not safe_to_continue:
            return 0.0
        cap = min(self.max_forward_for(terrain_type), self.max_forward_mps)
        if cap <= 0.0:
            return 0.0
        # forward-only: clamp to [0, cap]
        return float(max(0.0, min(desired_forward, cap)))
