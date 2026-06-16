"""MuJoCo model smoke test (secondary sim, P3-M1/P3-M2).

Loads the X2 MJCF (X2_URDF-v1.3.0/x2_ultra.xml) and checks the 12 leg joints are present with
the expected names. Skips when ``mujoco`` is not installed or the meshes are absent (they are
gitignored — run tools/fetch_x2_assets.sh). This validates the placed asset the moment a sim
dependency exists.
"""
import importlib.util
from pathlib import Path

import pytest

from x2_locomotion.robots.x2_joint_map import aimdk_leg_order

_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
_REPO = Path(__file__).resolve().parents[2]
_MJCF = _REPO / "training/mujoco/model/x2_ultra.xml"
_MESHES = _REPO / "training/isaac_lab/assets/meshes"

mujoco_only = pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")


def test_mjcf_file_present():
    # the model text file is committed even though meshes are gitignored
    assert _MJCF.is_file()


def _require_model():
    if not _MESHES.is_dir() or not any(_MESHES.glob("*.STL")):
        pytest.skip("meshes not present (gitignored); run tools/fetch_x2_assets.sh")
    import mujoco
    return mujoco, mujoco.MjModel.from_xml_path(str(_MJCF))


@mujoco_only
def test_mjcf_loads_and_has_leg_joints():
    mujoco, model = _require_model()
    joint_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(model.njnt)}
    for leg_joint in aimdk_leg_order():
        assert leg_joint in joint_names, f"missing {leg_joint} in MJCF"


@mujoco_only
def test_mjcf_has_floating_base_and_expected_dof():
    mujoco, model = _require_model()
    # 31 actuated joints (12 leg + 3 waist + 12 arm + 2 head) + 1 floating base
    assert model.njnt == 32
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.nu == 31
    assert 20.0 < model.body_subtreemass[0] < 80.0   # plausible humanoid mass (~43 kg)


@mujoco_only
def test_model_steps_without_exploding():
    """Spawn-and-step AC (roadmap §7.3): from the default pose under gravity the model must
    stay finite and not blow up."""
    import numpy as np
    mujoco, model = _require_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos)), "qpos went non-finite (model exploded)"
    assert np.all(np.isfinite(data.qvel))
    assert np.max(np.abs(data.qvel)) < 50.0, "implausible velocities (instability)"


@mujoco_only
def test_x2_stands_under_pd():
    """Stage-A standing AC (roadmap §7.3): the PD controller holds the X2 upright for 3 s."""
    if not _MESHES.is_dir() or not any(_MESHES.glob("*.STL")):
        pytest.skip("meshes not present (gitignored); run tools/fetch_x2_assets.sh")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(_REPO / "training/mujoco/scripts"))
    import stand

    rc = stand.run(seconds=3.0, spawn_z=0.0, view=False)
    assert rc == 0, "X2 did not stand stably under PD"
