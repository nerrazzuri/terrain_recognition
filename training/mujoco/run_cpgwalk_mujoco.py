#!/usr/bin/env python3
"""Run the factory `cpgwalk` ONNX inside OUR OWN control loop, in MuJoCo (sim-to-sim).

This is the offline rehearsal for the hardware Way-A test: we reproduce the MC's wrapper
ourselves — build the 65-obs, generate the CPG clock, run `cpgwalkrun_v25_v2.onnx`, map the
action to PD joint targets, and drive the robot. If X2 balances + walks here under *our* loop,
the takeover logic is proven before we ever touch the robot.

    # headless self-check (no viewer): walk forward 8 s, report whether it stayed up
    python training/mujoco/run_cpgwalk_mujoco.py --onnx /path/to/cpgwalkrun_v25_v2.onnx --vx 0.5 --seconds 8

    # with viewer
    python training/mujoco/run_cpgwalk_mujoco.py --onnx ... --vx 0.5 --view

The control math (obs/scales/clock/action mapping) is the unit-tested
`x2_locomotion.cref.factory_cpgwalk` module — the SAME code the ROS2 deploy node uses, so a
pass here is meaningful for hardware.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training" / "isaac_lab"))
from x2_locomotion.cref import factory_cpgwalk as fc  # noqa: E402

MODEL = REPO / "training" / "mujoco" / "model" / "scene.xml"
# default ONNX = the 0.9.7 robot runtime (all robots are 0.9). cpgwalkrun_v25_v2.onnx is
# byte-identical to the 0.8 file, but we point at 0.9.7 so all dev follows the 0.9 line.
DEFAULT_ONNX = ("/home/liang/Projects/X2 Locomotion/0.9.7/agibot/software/"
                "mc_param/robot/lx2501_3_t2d5/rl_models/cpgwalkrun_v25_v2.onnx")

# Strict stand-gate thresholds. Fall + height ratio come from the 0.9.7 runtime
# action_state.yaml `estimator`; the firm-stand tolerances are tighter (must be settled, not just
# "not fallen") so we only start walking from a genuinely stable stand — mirroring STABLE->MOVE.
BODY_HEIGHT = 0.652            # cpgwalk_config body_height
STAND_HEIGHT_RATIO = 0.8       # estimator.stand_height_ratio
FALL_PITCH, FALL_ROLL = 0.7, 0.5   # estimator.fall_pitch / fall_roll
STAND_PITCH, STAND_ROLL, STAND_OMEGA = 0.20, 0.15, 0.5


def quat_to_euler(w, x, y, z):
    """(w,x,y,z) -> roll,pitch,yaw (rad)."""
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--vx", type=float, default=0.5, help="forward velocity command (m/s)")
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--cmd4", type=float, default=0.0, help="4th command dim (gait/mode flag — confirm via rl/debug)")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--settle", type=float, default=0.5,
                    help="min balance-in-place warmup before the stand gate may unlock walking")
    ap.add_argument("--stand_hold", type=float, default=0.5,
                    help="seconds the firm-stand criteria must hold before walking unlocks")
    ap.add_argument("--ramp", type=float, default=1.0, help="seconds to ramp command 0->target")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--video", default=None, help="write an MP4 of the run to this path (headless, EGL)")
    args = ap.parse_args(argv)

    if not pathlib.Path(args.onnx).exists():
        print(f"[err] ONNX not found: {args.onnx}\n      pass --onnx <path to cpgwalkrun_v25_v2.onnx>")
        return 2

    if args.video:
        import os
        os.environ.setdefault("MUJOCO_GL", "egl")   # offscreen render
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    phys_dt = model.opt.timestep
    substeps = max(1, round(fc.CONTROL_DT / phys_dt))

    # name -> qpos/dof/actuator-ctrl indices for the 17 policy joints
    qadr, dadr, cadr = [], [], []
    for name in fc.JOINT_ORDER:
        j = model.joint(name)
        qadr.append(j.qposadr[0]); dadr.append(j.dofadr[0])
        cadr.append(model.actuator(f"motor_{name}").id)
    qadr, dadr, cadr = map(np.array, (qadr, dadr, cadr))
    ctrlrange = model.actuator_ctrlrange[cadr]            # software-PD torque clamp
    kps, kds = fc.KPS, fc.KDS

    # init at the factory default standing pose, spawned near standing equilibrium height
    # (model nominal pelvis is 0.68 m but the default pose settles to ~0.64 m — spawn close to
    # avoid a cosmetic drop transient at t=0)
    mujoco.mj_resetData(model, data)
    data.qpos[qadr] = fc.DEFAULT_DOF_POS
    data.qpos[model.joint("floating_base_joint").qposadr[0] + 2] = 0.655
    mujoco.mj_forward(model, data)

    pol = fc.FactoryCpgwalkPolicy(args.onnx)
    targets = fc.DEFAULT_DOF_POS.copy()
    base_qadr = model.joint("floating_base_joint").qposadr[0]
    base_dadr = model.joint("floating_base_joint").dofadr[0]

    def apply_pd():
        q = data.qpos[qadr]; qd = data.qvel[dadr]
        tau = kps * (targets - q) - kds * qd
        data.ctrl[cadr] = np.clip(tau, ctrlrange[:, 0], ctrlrange[:, 1])

    def step_policy(t, cmd_scale):
        nonlocal targets
        quat = data.qpos[base_qadr + 3: base_qadr + 7]      # w,x,y,z
        euler = quat_to_euler(*quat)
        omega = data.qvel[base_dadr + 3: base_dadr + 6].astype(np.float32)  # base-frame gyro
        cmd = np.array([args.vx, args.vy, args.yaw, args.cmd4], dtype=np.float32) * cmd_scale
        obs = fc.build_obs(omega, euler, cmd, data.qpos[qadr], data.qvel[dadr],
                           pol._prev_action, fc.cpg_phase(t))
        action = pol.infer(obs)
        targets = fc.action_to_targets(action)

    # optional offscreen video recorder (camera follows the pelvis)
    renderer = cam = frames = None
    render_every = fps = None
    if args.video:
        renderer = mujoco.Renderer(model, 480, 640)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 120, -15, 3.2
        frames = []
        fps = 25
        render_every = max(1, round((1.0 / fps) / fc.CONTROL_DT))

    def run_loop(viewer=None):
        n_ctrl = int(args.seconds / fc.CONTROL_DT)
        min_h, fell = 1e9, None
        walk_t0 = walk_x0 = ramp_t0 = stand_since = None
        walk_unlocked = False
        for k in range(n_ctrl):
            t = k * fc.CONTROL_DT
            # --- STRICT STAND GATE (mirrors factory STABLE->MOVE; thresholds from action_state.yaml) ---
            # The policy balances at zero command until a firm, settled, upright stand is HELD; only
            # then does the forward command ramp in. The robot can never start walking from a fall.
            roll, pitch = quat_to_euler(*data.qpos[base_qadr + 3:base_qadr + 7])[:2]
            h = float(data.qpos[base_qadr + 2])
            wbase = float(np.linalg.norm(data.qvel[base_dadr + 3:base_dadr + 6]))
            if not walk_unlocked:
                firm = (t >= args.settle and abs(pitch) < STAND_PITCH and abs(roll) < STAND_ROLL
                        and h >= STAND_HEIGHT_RATIO * BODY_HEIGHT and wbase < STAND_OMEGA)
                stand_since = (stand_since or t) if firm else None
                if stand_since is not None and (t - stand_since) >= args.stand_hold:
                    walk_unlocked, ramp_t0 = True, t
                    print(f"[stand] firm stand verified at t={t:.2f}s -> walk unlocked")
                cmd_scale = 0.0
            else:
                cmd_scale = min(1.0, (t - ramp_t0) / max(args.ramp, 1e-6))
                if walk_t0 is None and cmd_scale >= 1.0:     # measure steady-state speed (post-ramp)
                    walk_t0, walk_x0 = t, float(data.qpos[base_qadr])
            step_policy(t, cmd_scale)
            for _ in range(substeps):
                apply_pd()
                mujoco.mj_step(model, data)
            h = float(data.qpos[base_qadr + 2])
            min_h = min(min_h, h)
            if fell is None and (h < 0.35 or abs(pitch) > FALL_PITCH or abs(roll) > FALL_ROLL):
                fell = t
            if viewer is not None:
                viewer.sync()
            if renderer is not None and k % render_every == 0:
                cam.lookat = data.qpos[base_qadr:base_qadr + 3]      # track the robot
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())
        if walk_x0 is None:                                  # walk never unlocked
            return min_h, fell, 0.0, 0.0, walk_unlocked
        dx = float(data.qpos[base_qadr]) - walk_x0
        dt_walk = (n_ctrl * fc.CONTROL_DT) - walk_t0
        return min_h, fell, dx, (dx / dt_walk if dt_walk > 0 else 0.0), walk_unlocked

    print(f"[cpgwalk-mujoco] dt={fc.CONTROL_DT}s substeps={substeps} cmd=({args.vx},{args.vy},{args.yaw}|{args.cmd4})")
    if args.view:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as v:
            min_h, fell, dx, vx_mean, walk_unlocked = run_loop(v)
    else:
        min_h, fell, dx, vx_mean, walk_unlocked = run_loop()

    final_h = float(data.qpos[base_qadr + 2])
    upright = fell is None and final_h > 0.45
    walked = upright and walk_unlocked and vx_mean >= 0.5 * abs(args.vx) and abs(args.vx) > 0
    if not walk_unlocked and args.vx != 0:
        verdict = "STAND GATE NOT PASSED (walk never unlocked)"
    elif walked:
        verdict = "WALKED (stand-gated)"
    elif upright:
        verdict = "STAYED UP (not walking)"
    else:
        verdict = "FELL"
    print(f"[RESULT] {verdict} | stand_unlocked={walk_unlocked} | final_height={final_h:.3f} "
          f"min_height={min_h:.3f} | forward dx={dx:+.2f} m  mean_vx={vx_mean:+.2f} m/s (cmd {args.vx:+.2f})"
          + (f" | fell at t={fell:.1f}s" if fell else ""))
    print("  Walk unlocks ONLY after a firm stand is verified+held (factory STABLE->MOVE). "
          "height ~0.65=standing, <0.35=fallen.")

    if frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=fps, macro_block_size=None)
        print(f"[video] wrote {len(frames)} frames -> {args.video}")
        import os
        os._exit(0 if (walked or (upright and args.vx == 0)) else 1)  # skip noisy EGL cleanup
    return 0 if (walked or (upright and args.vx == 0)) else 1


if __name__ == "__main__":
    sys.exit(main())
