"""Observation builder (P5-M1-T1) — pure logic, no ROS2.

Builds the policy observation from real-robot inputs, matching the training layout and
normalization **exactly** (AGENTS.md §4). Missing or non-finite inputs return ``ok=False`` so
the caller safe-stops (roadmap §9.2). The layout below MUST equal the training layout — a
unit test enforces equality with x2_locomotion.tasks.common.observations.OBSERVATION_LAYOUT.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Canonical observation layout — identical to training (enforced by test).
OBSERVATION_LAYOUT: list[tuple[str, int]] = [
    ("command_velocity", 3),
    ("base_angular_velocity", 3),
    ("projected_gravity", 3),
    ("joint_position_error", 12),
    ("joint_velocity", 12),
    ("previous_action", 12),
    ("gait_phase", 2),
    ("height_samples", 121),
]
OBSERVATION_DIM = sum(d for _, d in OBSERVATION_LAYOUT)  # 168


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, v: np.ndarray) -> np.ndarray:
        std = np.where(np.asarray(self.std) > 1e-8, self.std, 1.0)
        return (np.asarray(v, dtype=float) - self.mean) / std

    @classmethod
    def identity(cls) -> "Normalizer":
        return cls(mean=np.zeros(OBSERVATION_DIM), std=np.ones(OBSERVATION_DIM))

    @classmethod
    def from_artifact(cls, mean, std) -> "Normalizer":
        return cls(mean=np.asarray(mean, float), std=np.asarray(std, float))


def build(parts: dict[str, np.ndarray], normalizer: Normalizer):
    """Return ``(normalized_observation, ok)``. ``ok`` False ⇒ caller must safe-stop."""
    chunks = []
    for name, dim in OBSERVATION_LAYOUT:
        if name not in parts:
            return None, False
        v = np.asarray(parts[name], dtype=float).reshape(-1)
        if v.shape[0] != dim or not np.all(np.isfinite(v)):
            return None, False
        chunks.append(v)
    obs = np.concatenate(chunks)
    return normalizer.normalize(obs), True
