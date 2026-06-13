"""ONNX runner helpers (P5-M1-T2) — pure logic, no onnxruntime.

The output-validation logic is pure and unit-testable; the actual ONNX session load/run
lives in onnx_policy_runner.py (needs onnxruntime). Bad output ⇒ safe stop.
"""
from __future__ import annotations

import numpy as np


def validate_output(action, expected_dim: int) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for a policy output: right dim and all finite."""
    a = np.asarray(action, dtype=float).reshape(-1)
    if a.shape[0] != expected_dim:
        return False, f"action dim {a.shape[0]} != expected {expected_dim}"
    if not np.all(np.isfinite(a)):
        return False, "action contains NaN/Inf"
    return True, "ok"


def within_period(inference_seconds: float, policy_period_seconds: float) -> bool:
    """True if inference fit inside the policy period budget (e.g. 20 ms at 50 Hz)."""
    return inference_seconds <= policy_period_seconds
