"""policy_logger (P5-M1-T4) — full deployment record per roadmap §9.4.

Wraps x2_common.logging_utils.JsonlRecorder to write one JSON line per control cycle with all
required fields. Logging is part of the feature — every real-robot test must produce this
record. Pure helper (no ROS2) so it can be reused offline and unit-tested.
"""
from __future__ import annotations

from x2_common.logging_utils import JsonlRecorder

# Fields required by roadmap §9.4.
REQUIRED_FIELDS = (
    "timestamp", "robot_mode", "terrain_status", "heightmap", "observation",
    "raw_action", "filtered_action", "joint_state", "imu_state",
    "safety_supervisor_state", "operator_command", "stop_reason",
)


class PolicyLogger:
    def __init__(self, path: str):
        self._rec = JsonlRecorder(path)

    def log(self, record: dict) -> None:
        """Write a cycle record, ensuring every required §9.4 field is present (None if absent)."""
        full = {k: record.get(k) for k in REQUIRED_FIELDS}
        full.update(record)  # keep any extra fields too
        self._rec.write(full)

    def close(self) -> None:
        self._rec.close()

    def __enter__(self) -> "PolicyLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
