"""Unit tests for pure helpers in the training scripts (no torch / Isaac Lab)."""
import numpy as np

from x2_locomotion.scripts.evaluate_policy import (
    check_graduation, GRADUATION_THRESHOLDS,
    EpisodeOutcome, episode_success, success_rate, split_success_rates,
    is_walk_command, WALK_TRACK_FRACTION, STAND_SPEED_MAX_MPS)
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


def test_standing_under_walk_command_fails():
    # THE bug we fixed: commanded 0.5 m/s but barely moving (0.05) -> NOT a success.
    o = EpisodeOutcome(survived=True, cmd_speed=0.5, achieved_speed=0.05)
    assert is_walk_command(o)
    assert not episode_success(o)


def test_walk_success_when_actually_moving():
    # commanded 0.5, achieved >= 0.5*0.5=0.25 -> success
    assert episode_success(EpisodeOutcome(survived=True, cmd_speed=0.5, achieved_speed=0.30))
    # exactly at the fraction boundary counts (>=)
    assert episode_success(EpisodeOutcome(survived=True, cmd_speed=0.5,
                                          achieved_speed=WALK_TRACK_FRACTION * 0.5))


def test_walk_fails_if_fell_even_if_moving():
    assert not episode_success(EpisodeOutcome(survived=False, cmd_speed=0.5, achieved_speed=0.5))


def test_stand_command_success_requires_staying_still():
    # zero command, staying still -> success
    assert episode_success(EpisodeOutcome(survived=True, cmd_speed=0.0, achieved_speed=0.05))
    # zero command but drifting fast -> fail
    assert not episode_success(EpisodeOutcome(survived=True, cmd_speed=0.0,
                                              achieved_speed=STAND_SPEED_MAX_MPS + 0.1))


def test_split_success_rates_separates_walk_and_stand():
    outcomes = [
        EpisodeOutcome(True, 0.0, 0.02),   # stand: success
        EpisodeOutcome(True, 0.5, 0.40),   # walk: success (moving)
        EpisodeOutcome(True, 0.5, 0.03),   # walk: FAIL (standing under walk cmd)
        EpisodeOutcome(False, 0.5, 0.40),  # walk: FAIL (fell)
    ]
    s = split_success_rates(outcomes)
    assert s["n_stand"] == 1 and s["n_walk"] == 3
    assert s["stand_success"] == 1.0
    assert abs(s["walk_success"] - (1 / 3)) < 1e-9
    assert s["overall"] == 0.5


def test_success_rate_empty_fails_closed():
    assert success_rate([]) == 0.0
