"""Unit tests for pure helpers in the training scripts (no torch / Isaac Lab)."""
import numpy as np

from x2_locomotion.scripts.evaluate_policy import (
    check_graduation, GRADUATION_THRESHOLDS,
    EpisodeOutcome, episode_success, success_rate, DEFAULT_MAX_VEL_ERROR_MPS)
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


def test_episode_success_requires_survival_and_tracking():
    # survived + low error -> success
    assert episode_success(EpisodeOutcome(survived=True, mean_vel_error=0.1))
    # fell -> fail even with perfect tracking
    assert not episode_success(EpisodeOutcome(survived=False, mean_vel_error=0.0))
    # survived but poor tracking -> fail
    assert not episode_success(EpisodeOutcome(survived=True, mean_vel_error=1.0))


def test_episode_success_boundary_inclusive():
    # error exactly at the gate counts as success (<=)
    assert episode_success(EpisodeOutcome(survived=True, mean_vel_error=DEFAULT_MAX_VEL_ERROR_MPS))


def test_success_rate_fraction():
    outcomes = [
        EpisodeOutcome(True, 0.1),   # success
        EpisodeOutcome(True, 0.1),   # success
        EpisodeOutcome(False, 0.1),  # fell
        EpisodeOutcome(True, 0.9),   # poor tracking
    ]
    assert success_rate(outcomes) == 0.5


def test_success_rate_empty_fails_closed():
    assert success_rate([]) == 0.0
