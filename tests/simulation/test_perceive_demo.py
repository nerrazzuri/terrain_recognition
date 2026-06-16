"""Regression test for the MuJoCo perceive-and-stop demo (perception on rendered geometry).

Runs the perception + safe-locomotion cores on a depth cloud ray-cast from a MuJoCo scene and
asserts the perceive-and-stop behaviour on realistic (partially observed) sensor data:
flat -> safe + forward command; obstacles -> unsafe + zero command. Skips without mujoco.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
mujoco_only = pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "training/mujoco/scripts"))


def _run(scene, obstacle_x=1.0):
    import perceive_demo as demo
    from x2_common import config_loader
    from x2_safe_locomotion.core.velocity_policy import VelocityPolicy

    perc = config_loader.load_config("terrain_perception")
    policy = VelocityPolicy.from_dict(config_loader.load_config("safe_locomotion"))
    result, _ = demo.run_scene(scene, perc, policy, obstacle_x=obstacle_x)
    return result


@mujoco_only
def test_flat_ground_walks():
    r = _run("flat")
    assert r["terrain"] == "flat_ground"
    assert r["safe"] is True
    assert r["cmd_v"] > 0.0


@mujoco_only
def test_curb_stops():
    r = _run("curb")
    assert r["safe"] is False
    assert r["cmd_v"] == 0.0


@mujoco_only
def test_stairs_detected_and_stops():
    r = _run("stairs")
    assert r["terrain"] in ("stairs_up", "curb_or_single_step", "unknown_unsafe")
    assert r["safe"] is False
    assert r["cmd_v"] == 0.0


@mujoco_only
def test_approach_flips_from_go_to_stop():
    far = _run("stairs", obstacle_x=2.4)
    near = _run("stairs", obstacle_x=1.0)
    assert far["safe"] is True and far["cmd_v"] > 0.0     # clear far ahead -> walk
    assert near["safe"] is False and near["cmd_v"] == 0.0  # stairs in range -> stop
