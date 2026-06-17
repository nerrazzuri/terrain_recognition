#!/usr/bin/env python3
"""Full STAND -> WALK demo in our own loop, mirroring the factory STABLE -> MOVE sequence.

  Phase 1 STAND : factory `cpgtelecon` (STAND_DEFAULT) holds a FIRM, STILL stand (no stepping).
  Phase 2 WALK  : once a firm stand is verified+held, hand off to `cpgwalk` (LOCOMOTION_DEFAULT)
                  and ramp the forward command in.

Both factory ONNX policies run inside our own loop (obs builder -> ONNX -> software PD), so this is
the offline rehearsal for the hardware takeover. cpgtelecon runs at 100 Hz, cpgwalk at 50 Hz.

    python training/mujoco/run_stand_walk_mujoco.py --vx 0.6 --video ~/Downloads/x2_stand_walk.mp4
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training" / "isaac_lab"))
from x2_locomotion.cref import factory_cpgwalk as wc      # noqa: E402
from x2_locomotion.cref import factory_cpgtelecon as tc   # noqa: E402

RT = "/home/liang/Projects/X2 Locomotion/0.9.7/agibot/software/mc_param/robot/lx2501_3_t2d5/rl_models"
DEFAULT_WALK = f"{RT}/cpgwalkrun_v25_v2.onnx"
DEFAULT_STAND = f"{RT}/cpgtelecon_v3_fix.onnx"

# stand gate (firm-stand tolerances + factory estimator fall/height thresholds)
BODY_HEIGHT, STAND_HEIGHT_RATIO = 0.652, 0.8
STAND_PITCH, STAND_ROLL, STAND_OMEGA = 0.20, 0.15, 0.5


def quat_to_euler(w, x, y, z):
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--walk_onnx", default=DEFAULT_WALK)
    ap.add_argument("--stand_onnx", default=DEFAULT_STAND)
    ap.add_argument("--vx", type=float, default=0.6)
    ap.add_argument("--stand_seconds", type=float, default=3.0, help="max time allotted to reach a firm stand")
    ap.add_argument("--stand_hold", type=float, default=0.8, help="firm-stand must hold this long to unlock walk")
    ap.add_argument("--walk_seconds", type=float, default=10.0)
    ap.add_argument("--ramp", type=float, default=1.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args(argv)
    for p in (args.walk_onnx, args.stand_onnx):
        if not pathlib.Path(p).exists():
            print(f"[err] ONNX not found: {p}"); return 2

    if args.video:
        import os
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(REPO / "training" / "mujoco" / "model" / "scene.xml"))
    data = mujoco.MjData(model)
    bz = model.joint("floating_base_joint").qposadr[0]
    bd = model.joint("floating_base_joint").dofadr[0]

    # actuator tables (one PD target per actuator; policies overwrite their own joints)
    names = [model.actuator(i).name.replace("motor_", "") for i in range(model.nu)]
    jqa = np.array([model.jnt_qposadr[model.actuator(i).trnid[0]] for i in range(model.nu)])
    jda = np.array([model.jnt_dofadr[model.actuator(i).trnid[0]] for i in range(model.nu)])
    cr = model.actuator_ctrlrange
    idx = {n: i for i, n in enumerate(names)}
    ids_tc = np.array([idx[n] for n in tc.ACTION_SEQ])
    ids_wc = np.array([idx[n] for n in wc.JOINT_ORDER])
    qa_tc = np.array([model.joint(n).qposadr[0] for n in tc.SEQ_OBS])
    da_tc = np.array([model.joint(n).dofadr[0] for n in tc.SEQ_OBS])
    qa_wc = np.array([model.joint(n).qposadr[0] for n in wc.JOINT_ORDER])
    da_wc = np.array([model.joint(n).dofadr[0] for n in wc.JOINT_ORDER])

    stand_pose = dict(zip(wc.JOINT_ORDER, wc.DEFAULT_DOF_POS))
    tgt = np.array([stand_pose.get(n, 0.0) for n in names], dtype=np.float64)
    kp = np.full(model.nu, 40.0)
    kd = np.full(model.nu, 4.0)

    # init standing
    mujoco.mj_resetData(model, data)
    for n, p in zip(wc.JOINT_ORDER, wc.DEFAULT_DOF_POS):
        data.qpos[model.joint(n).qposadr[0]] = p
    data.qpos[bz + 2] = 0.655
    mujoco.mj_forward(model, data)

    # video
    renderer = cam = frames = None
    if args.video:
        renderer = mujoco.Renderer(model, 480, 640)
        cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 120, -15, 3.2
        frames = []

    def apply_pd_and_step(sub):
        for _ in range(sub):
            q = data.qpos[jqa]; qd = data.qvel[jda]
            data.ctrl[:] = np.clip(kp * (tgt - q) - kd * qd, cr[:, 0], cr[:, 1])
            mujoco.mj_step(model, data)

    def grab(phase, k, fps_every):
        if renderer is not None and k % fps_every == 0:
            cam.lookat = data.qpos[bz:bz + 3]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    def firm_stand():
        roll, pitch = quat_to_euler(*data.qpos[bz + 3:bz + 7])[:2]
        h = float(data.qpos[bz + 2])
        w = float(np.linalg.norm(data.qvel[bd + 3:bd + 6]))
        return (abs(pitch) < STAND_PITCH and abs(roll) < STAND_ROLL
                and h >= STAND_HEIGHT_RATIO * BODY_HEIGHT and w < STAND_OMEGA)

    # ---------------- Phase 1: STAND on cpgtelecon (100 Hz) ----------------
    stand = tc.FactoryCpgteleconPolicy(args.stand_onnx)
    kp[ids_tc], kd[ids_tc] = tc.KPS, tc.KDS
    sub_tc = max(1, round(tc.CONTROL_DT / model.opt.timestep))
    stand_since = None
    stood = False
    n_stand = int(args.stand_seconds / tc.CONTROL_DT)
    foot_l = model.body("left_ankle_roll_link").id
    foot_zs = []
    for k in range(n_stand):
        gv = tc.quat_rotate_inverse(*data.qpos[bz + 3:bz + 7])
        om = data.qvel[bd + 3:bd + 6].astype(np.float32)
        obs = tc.build_obs(om, gv, np.zeros(3), np.zeros(4), data.qpos[qa_tc], data.qvel[da_tc],
                           stand._prev_action, tc.ARM_TARGET_DEFAULT, np.zeros(4))
        tgt[ids_tc] = tc.action_to_targets(stand.infer(obs))
        apply_pd_and_step(sub_tc)
        foot_zs.append(data.xpos[foot_l][2])
        grab("stand", k, max(1, round((1 / 25) / tc.CONTROL_DT)))
        t = k * tc.CONTROL_DT
        if firm_stand():
            stand_since = stand_since or t
            if t - stand_since >= args.stand_hold:
                stood = True
                print(f"[stand] firm stand verified at t={t:.2f}s (foot travel so far "
                      f"{np.ptp(foot_zs) * 1000:.1f} mm) -> handing off to cpgwalk")
                break
        else:
            stand_since = None
    print(f"[stand] foot vertical travel during stand = {np.ptp(foot_zs) * 1000:.1f} mm "
          f"({'STILL' if np.ptp(foot_zs) < 0.01 else 'MOVED'})")
    if not stood:
        print("[stand] FAILED to reach a firm stand -> NOT walking.");
        if frames:
            import imageio.v2 as imageio, os
            imageio.mimsave(args.video, frames, fps=25, macro_block_size=None); os._exit(1)
        return 1

    # ---------------- Phase 2: WALK on cpgwalk (50 Hz) ----------------
    walk = wc.FactoryCpgwalkPolicy(args.walk_onnx)
    kp[:], kd[:] = 40.0, 4.0
    kp[ids_wc], kd[ids_wc] = wc.KPS, wc.KDS
    sub_wc = max(1, round(wc.CONTROL_DT / model.opt.timestep))
    x0 = float(data.qpos[bz]); n_walk = int(args.walk_seconds / wc.CONTROL_DT)
    fell = None
    for k in range(n_walk):
        t = k * wc.CONTROL_DT
        cmd_scale = min(1.0, t / max(args.ramp, 1e-6))
        eu = quat_to_euler(*data.qpos[bz + 3:bz + 7])
        om = data.qvel[bd + 3:bd + 6].astype(np.float32)
        cmd = np.array([args.vx, 0, 0, 0], dtype=np.float32) * cmd_scale
        obs = wc.build_obs(om, eu, cmd, data.qpos[qa_wc], data.qvel[da_wc],
                           walk._prev_action, wc.cpg_phase(t))
        tgt[ids_wc] = wc.action_to_targets(walk.infer(obs))
        apply_pd_and_step(sub_wc)
        grab("walk", k, max(1, round((1 / 25) / wc.CONTROL_DT)))
        if data.qpos[bz + 2] < 0.35 and fell is None:
            fell = t
    dx = float(data.qpos[bz]) - x0
    vx = dx / (n_walk * wc.CONTROL_DT)
    print(f"[walk] dx={dx:+.2f} m  mean_vx={vx:+.2f} m/s (cmd {args.vx})"
          + (f"  | FELL at {fell:.1f}s" if fell else "  | stayed up"))
    print(f"[RESULT] {'STAND-STILL then WALKED' if (stood and not fell and vx > 0.2) else 'see logs'}")

    if frames:
        import imageio.v2 as imageio, os
        imageio.mimsave(args.video, frames, fps=25, macro_block_size=None)
        print(f"[video] wrote {len(frames)} frames -> {args.video}")
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
