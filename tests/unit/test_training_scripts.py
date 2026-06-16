"""Unit tests for pure helpers in the training scripts (no torch / Isaac Lab)."""
import numpy as np

from x2_locomotion.scripts.evaluate_policy import check_graduation, GRADUATION_THRESHOLDS
from x2_locomotion.scripts.export_onnx import numeric_match


def test_graduation_passes_when_all_thresholds_met():
    rates = {k: v + 0.02 for k, v in GRADUATION_THRESHOLDS.items()}
    passed, failures = check_graduation(rates)
    assert passed and failures == []


def test_graduation_fails_on_low_stairs():
    rates = {k: v + 0.02 for k, v in GRADUATION_THRESHOLDS.items()}
    rates["stairs_up"] = 0.5
    passed, failures = check_graduation(rates)
    assert not passed and any("stairs_up" in f for f in failures)


def test_graduation_fails_on_missing_metric():
    rates = dict(GRADUATION_THRESHOLDS)
    del rates["flat"]
    passed, failures = check_graduation(rates)
    assert not passed and any("flat" in f for f in failures)


def test_numeric_match_within_tolerance():
    a = np.zeros(12)
    b = np.full(12, 1e-5)
    ok, diff = numeric_match(a, b, tol=1e-4)
    assert ok and diff == np.float64(1e-5)


def test_numeric_match_exceeds_tolerance():
    ok, diff = numeric_match(np.zeros(12), np.full(12, 1.0), tol=1e-4)
    assert not ok and diff == 1.0


def test_numeric_match_shape_mismatch():
    ok, diff = numeric_match(np.zeros(12), np.zeros(11), tol=1.0)
    assert not ok and diff == float("inf")
