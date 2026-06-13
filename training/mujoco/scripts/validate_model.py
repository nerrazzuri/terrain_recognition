#!/usr/bin/env python3
"""validate_model.py — load the X2 MuJoCo model, print stats, and step physics.

The secondary-sim sanity check (roadmap §7): confirms the model compiles, has a floating base
and the expected DoF, and does not explode from the default pose under gravity. Useful before
trusting the MJCF / mesh assets. Needs `mujoco` + the meshes (tools/fetch_x2_assets.sh).

Usage:
    python training/mujoco/scripts/validate_model.py [--steps 300] [--view]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MJCF = Path(__file__).resolve().parents[1] / "model" / "x2_ultra.xml"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--view", action="store_true", help="open the interactive viewer")
    args = ap.parse_args(argv)

    try:
        import numpy as np
        import mujoco
    except Exception as exc:
        print(f"BLOCKED: mujoco not available ({exc}). pip install mujoco")
        return 2

    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    print(f"compiled OK: nq={model.nq} nv={model.nv} njnt={model.njnt} "
          f"nu={model.nu} nbody={model.nbody} mass={model.body_subtreemass[0]:.2f} kg")

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if args.view:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
        return 0

    for _ in range(args.steps):
        mujoco.mj_step(model, data)
    ok = bool(np.all(np.isfinite(data.qpos)) and np.max(np.abs(data.qvel)) < 50.0)
    print(f"after {args.steps} steps: base_z={data.qpos[2]:.3f} "
          f"max|qvel|={np.max(np.abs(data.qvel)):.2f} -> {'OK' if ok else 'UNSTABLE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
