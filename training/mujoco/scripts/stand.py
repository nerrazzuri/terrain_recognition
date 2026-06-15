#!/usr/bin/env python3
"""stand.py — make the X2 stand in MuJoCo with a PD pose controller (secondary-sim bring-up).

Loads the scene (floor + robot), holds the default standing pose with per-joint PD torque
(x2_locomotion.control.pd.PDController), and reports whether the base height stays stable for
the run — i.e. the robot stands rather than collapsing. Validates the default pose + PD gains
before RL training (roadmap §7.3 spawn-and-stand AC), with no GPU/Isaac Lab needed.

Usage:
    python training/mujoco/scripts/stand.py [--seconds 3] [--spawn-z 0.62] [--view]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "training/isaac_lab"))

from x2_locomotion.control.pd import PDController  # noqa: E402
from x2_locomotion.robots.default_pose import DEFAULT_LEG_POSE  # noqa: E402

_SCENE = _REPO / "training/mujoco/model/scene.xml"

# PD gains by joint group (torque mode), tuned so the straight-leg pose stands in MuJoCo.
_KP = {"hip": 400.0, "knee": 400.0, "ankle": 200.0, "waist": 300.0, "arm": 100.0, "head": 30.0}
_KD = {"hip": 15.0, "knee": 15.0, "ankle": 8.0, "waist": 10.0, "arm": 4.0, "head": 1.5}


def _group(joint_name: str) -> str:
    for key in ("hip", "knee", "ankle", "waist", "head"):
        if key in joint_name:
            return key
    if any(a in joint_name for a in ("shoulder", "elbow", "wrist")):
        return "arm"
    return "arm"


def build(model):
    """Return (act_qadr, act_vadr, q_des, kp, kd, eff) arrays indexed by actuator id."""
    import mujoco
    n = model.nu
    qadr = np.zeros(n, dtype=int)
    vadr = np.zeros(n, dtype=int)
    q_des = np.zeros(n)
    kp = np.zeros(n)
    kd = np.zeros(n)
    eff = np.zeros(n)
    for a in range(n):
        jid = model.actuator_trnid[a, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        qadr[a] = model.jnt_qposadr[jid]
        vadr[a] = model.jnt_dofadr[jid]
        q_des[a] = DEFAULT_LEG_POSE.get(jname, 0.0)
        g = _group(jname)
        kp[a], kd[a] = _KP[g], _KD[g]
        eff[a] = float(model.actuator_ctrlrange[a, 1])  # effort limit
    return qadr, vadr, q_des, kp, kd, eff


def run(seconds: float, spawn_z: float, view: bool) -> int:
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    qadr, vadr, q_des, kp, kd, eff = build(model)
    pd = PDController(kp, kd, eff)

    # set the standing pose, then place the base so the soles just rest on the floor
    for a in range(model.nu):
        data.qpos[qadr[a]] = q_des[a]
    data.qpos[2] = 0.0
    mujoco.mj_forward(model, data)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    lowest = min(data.geom_xpos[g, 2] for g in range(model.ngeom) if g != floor_gid)
    auto_z = -lowest + 0.002
    if spawn_z <= 0:
        spawn_z = auto_z
    data.qpos[2] = spawn_z
    mujoco.mj_forward(model, data)

    steps = int(seconds / model.opt.timestep)
    heights = []

    def control():
        q = data.qpos[qadr]
        qd = data.qvel[vadr]
        data.ctrl[:] = pd.torque(q, qd, q_des)

    if view:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                control()
                mujoco.mj_step(model, data)
                heights.append(float(data.qpos[2]))
                viewer.sync()
    else:
        for _ in range(steps):
            control()
            mujoco.mj_step(model, data)
            heights.append(float(data.qpos[2]))

    heights = np.array(heights)
    settled = heights[len(heights) // 2:]          # second half
    base_z = float(settled.mean())
    drop = float(heights[0] - heights[-1])
    finite = bool(np.all(np.isfinite(data.qpos)))
    upright = _torso_upright(model, data)
    stood = finite and upright > 0.85 and base_z > 0.35 and settled.std() < 0.03
    print(f"spawn_z={spawn_z:.2f} -> settled base_z={base_z:.3f} (std {settled.std():.3f}) "
          f"drop={drop:.3f} upright={upright:.2f} finite={finite}")
    print("RESULT:", "STANDS (stable)" if stood else "did not stand stably")
    return 0 if stood else 1


def _torso_upright(model, data) -> float:
    """Cosine of the torso z-axis vs world up (1.0 = perfectly upright)."""
    import mujoco
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if bid < 0:
        bid = 0
    R = data.xmat[bid].reshape(3, 3)
    return float(R[2, 2])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--spawn-z", type=float, default=0.0, help="0 = auto (soles on floor)")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args(argv)
    try:
        import mujoco  # noqa: F401
    except Exception as exc:
        print(f"BLOCKED: mujoco not available ({exc}). pip install mujoco")
        return 2
    return run(args.seconds, args.spawn_z, args.view)


if __name__ == "__main__":
    sys.exit(main())
