"""Unit tests for x2_common.config_loader."""
import pytest

from x2_common import config_loader as cl


def test_load_repo_config_safe_locomotion():
    cfg = cl.load_config("safe_locomotion")
    assert cfg["command_source"]["name"] == "x2_terrain_safe_locomotion"


def test_load_missing_raises():
    with pytest.raises(cl.ConfigError):
        cl.load_config("does_not_exist_xyz")


def test_get_dotted_key():
    cfg = {"roi": {"x_min_m": 0.0, "x_max_m": 2.0}}
    assert cl.get(cfg, "roi.x_max_m") == 2.0


def test_get_missing_key_raises_without_default():
    with pytest.raises(cl.ConfigError):
        cl.get({"a": 1}, "a.b.c")


def test_get_missing_key_returns_default():
    assert cl.get({"a": 1}, "x.y", default=42) == 42


def test_safety_flag_defaults_false():
    cfg = cl.load_config("safety_limits")
    assert cl.get(cfg, "REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED") is False
