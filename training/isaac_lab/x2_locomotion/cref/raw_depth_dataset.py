"""Raw-depth dataset collector (P6-M1-T1).

Roll out the trained Phase 4 height-map (teacher) policy in sim while recording (depth crop,
proprio, teacher action, teacher value, height samples, terrain label) tuples for
teacher-student distillation. BLOCKED to run: requires Isaac Lab + the Phase 4 checkpoint.

The on-disk record schema is defined here so the distillation loader and the collector agree.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

RECORD_FIELDS = (
    "depth", "proprio", "teacher_action", "teacher_value", "height_samples", "terrain_label",
)


@dataclass
class DepthSample:
    depth: np.ndarray            # (1, H, W)
    proprio: np.ndarray          # (proprio_dim,)
    teacher_action: np.ndarray   # (action_dim,)
    teacher_value: float
    height_samples: np.ndarray   # (121,)
    terrain_label: int

    def to_record(self) -> dict:
        return asdict(self)


def collect(checkpoint: str, num_steps: int, out_path: str) -> int:
    """Collect ``num_steps`` samples to ``out_path``. BLOCKED on Isaac Lab + checkpoint."""
    try:
        from isaaclab.app import AppLauncher  # noqa: F401
    except Exception as exc:
        print(f"[raw_depth_dataset] BLOCKED: Isaac Lab not available ({exc}).")
        return 2
    raise NotImplementedError("roll out teacher policy and record samples once Isaac Lab ready")
