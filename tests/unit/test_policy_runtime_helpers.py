"""Unit tests for the remaining Phase 5 helpers (ONNX validation, logger, log analysis)."""
import numpy as np
import pytest

from x2_policy_runtime.core.onnx_runner_core import validate_output, within_period
from x2_policy_runtime.core.log_analysis import summarize, max_action_divergence
from x2_policy_runtime.policy_logger import REQUIRED_FIELDS


def test_validate_output_accepts_correct():
    ok, _ = validate_output(np.zeros(12), 12)
    assert ok


def test_validate_output_rejects_wrong_dim():
    ok, reason = validate_output(np.zeros(11), 12)
    assert not ok and "dim" in reason


def test_validate_output_rejects_nonfinite():
    a = np.zeros(12)
    a[0] = np.inf
    ok, reason = validate_output(a, 12)
    assert not ok and "nan" in reason.lower()


def test_within_period():
    assert within_period(0.01, 0.02)
    assert not within_period(0.03, 0.02)


def test_summarize_counts_stop_reasons():
    records = [
        {"safety_state": "ok", "would_command": True},
        {"safety_state": "operator stop requested", "would_command": False},
        {"safety_state": "ok", "would_command": True},
    ]
    s = summarize(records)
    assert s["cycles"] == 3
    assert s["would_command"] == 2
    assert s["stop_reasons"]["ok"] == 2


def test_max_action_divergence():
    sim = [[0.0, 0.0], [0.1, 0.1]]
    real = [[0.0, 0.0], [0.3, 0.1]]
    assert max_action_divergence(sim, real) == pytest.approx(0.2)


def test_required_log_fields_cover_roadmap_9_4():
    for field in ("observation", "raw_action", "filtered_action", "stop_reason",
                  "joint_state", "imu_state", "safety_supervisor_state"):
        assert field in REQUIRED_FIELDS
