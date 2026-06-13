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


@mujoco_only
def test_mjcf_loads_and_has_leg_joints():
    if not _MESHES.is_dir() or not any(_MESHES.glob("*.STL")):
        pytest.skip("meshes not present (gitignored); run tools/fetch_x2_assets.sh")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    joint_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(model.njnt)}
    for leg_joint in aimdk_leg_order():
        assert leg_joint in joint_names, f"missing {leg_joint} in MJCF"
