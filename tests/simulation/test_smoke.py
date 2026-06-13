"""Simulation smoke tests (P3-M3-T4).

Runnable locally / in CI without a GPU: the pure-logic sim cores are exercised directly, and
the Isaac Lab-dependent configs are imported only if ``isaaclab`` is installed (otherwise
skipped). This guards the sim scaffolding against drift even on machines without Isaac Lab.
"""
import importlib.util

import numpy as np
import pytest

from x2_locomotion.robots import x2_joint_map as jm
from x2_locomotion.tasks.common import terrain_spec as ts
from x2_locomotion.tasks.common import height_samples as hs

_HAS_ISAACLAB = importlib.util.find_spec("isaaclab") is not None
isaac_only = pytest.mark.skipif(not _HAS_ISAACLAB, reason="Isaac Lab not installed")


def test_joint_map_round_trip_integration():
    m = jm.JointMap(jm.aimdk_leg_order())
    v = np.arange(12.0)
    np.testing.assert_allclose(m.to_sim(m.to_aimdk(v)), v)


def test_terrain_curriculum_spans_all_levels():
    names = [ts.level_params(i).name for i in range(ts.num_levels())]
    assert names[0] == "flat" and names[-1] == "mixed"
    assert ts.num_levels() == 7


def test_height_samples_default_observation_size():
    grid = hs.SampleGrid(11, 11, -0.4, 1.2, -0.5, 0.5)
    out = hs.sample_heights(grid, lambda x, y: np.zeros_like(x), (0, 0), 0.0, 0.55)
    assert out.shape == (121,)  # matches training observation height_samples dim


@isaac_only
def test_robot_cfg_builds_when_isaaclab_present():
    from x2_locomotion.robots.x2_robot_cfg import build_robot_cfg, assets_available
    if not assets_available():
        pytest.skip("X2 USD asset not present (P3-M1)")
    cfg = build_robot_cfg()
    assert cfg is not None


@isaac_only
def test_standing_env_cfg_imports_when_isaaclab_present():
    from x2_locomotion.tasks.standing.x2_standing_env_cfg import X2StandingEnvCfg
    assert X2StandingEnvCfg is not None
