"""Observation assembly + normalization (P4-M1-T2). Pure numpy, no Isaac Lab / torch.

The observation **order and dimensions are a contract**: deployment (observation_builder.py)
must produce the identical layout (docs/training_method.md, AGENTS.md §4). Keeping the order
here as the single source of truth lets both training and deployment import it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (name, dim) in the exact concatenation order. Sum = 168.
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


class ObservationError(ValueError):
    """Raised when a part is missing or has the wrong dimension."""


def assemble(parts: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate observation parts in the canonical order, validating each dim."""
    chunks = []
    for name, dim in OBSERVATION_LAYOUT:
        if name not in parts:
            raise ObservationError(f"missing observation part: {name}")
        v = np.asarray(parts[name], dtype=float).reshape(-1)
        if v.shape[0] != dim:
            raise ObservationError(f"part {name} has dim {v.shape[0]}, expected {dim}")
        chunks.append(v)
    return np.concatenate(chunks)


@dataclass
class Normalizer:
    """Affine normalizer with a zero-std guard. Stats come from the training artifact."""

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, v: np.ndarray) -> np.ndarray:
        std = np.where(np.asarray(self.std) > 1e-8, self.std, 1.0)
        return (np.asarray(v, dtype=float) - self.mean) / std

    @classmethod
    def identity(cls) -> "Normalizer":
        return cls(mean=np.zeros(OBSERVATION_DIM), std=np.ones(OBSERVATION_DIM))
