"""Policy-log analysis helpers (P5-M3-T7) — pure logic, no ROS2.

Summarise a JSONL deployment log (stop reasons, action stats) and compare sim vs real action
traces. Used by tools/analyze_policy_log.py and tools/compare_sim_real.py.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize(records: list[dict]) -> dict:
    """Return counts of stop reasons + how many cycles would have commanded."""
    reasons = Counter(r.get("stop_reason") or r.get("safety_state") for r in records)
    would = sum(1 for r in records if r.get("would_command"))
    return {
        "cycles": len(records),
        "would_command": would,
        "stop_reasons": dict(reasons),
    }


def max_action_divergence(sim_actions, real_actions) -> float:
    """Max abs difference between aligned sim/real action arrays (compare_sim_real)."""
    a = np.asarray(sim_actions, dtype=float)
    b = np.asarray(real_actions, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.max(np.abs(a[:n] - b[:n])))
