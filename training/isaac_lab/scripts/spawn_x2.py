"""spawn_x2.py — spawn the X2 in Isaac Sim and check it stands (P3-M2-T5 / P3-M3-T1).

Isaac Lab standalone script: launches Isaac Sim, spawns a ground plane + the X2 articulation
from our x2_robot_cfg, holds the default standing pose with the configured PD actuators, and
reports base-height + uprightness over time plus final foot heights (diagnostics). This is the
Isaac Lab counterpart of the validated MuJoCo training/mujoco/scripts/stand.py.

Run on the GPU box (see training/isaac_lab/SETUP.md):
    python training/isaac_lab/scripts/spawn_x2.py --headless --seconds 5
    python training/isaac_lab/scripts/spawn_x2.py --headless --spawn-z 0.75 --fix-base   # debug
"""
from __future__ import annotations

import argparse

# --- 1. launch the simulator (must happen before importing isaaclab.* assets) ----------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn + stand the X2 in Isaac Sim.")
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--spawn-z", type=float, default=0.0, help="override spawn height (0 = cfg default)")
parser.add_argument("--fix-base", action="store_true", help="debug: pin the base (isolate joints/asset)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- 2. now safe to import the rest -----------------------------------------------------
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from x2_locomotion.robots.x2_robot_cfg import build_robot_cfg, BASE_HEIGHT_M  # noqa: E402


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))

    # ground + light
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/ground", cfg_ground)
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0)
    cfg_light.func("/World/Light", cfg_light)

    # our X2
    robot_cfg = build_robot_cfg().replace(prim_path="/World/Robot")
    spawn_z = args.spawn_z if args.spawn_z > 0 else BASE_HEIGHT_M
    robot_cfg.init_state = robot_cfg.init_state.replace(pos=(0.0, 0.0, spawn_z))
    if args.fix_base:
        robot_cfg.spawn = robot_cfg.spawn.replace(articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True))
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.reset()

    # report the articulation structure
    print(f"[spawn_x2] joints={robot.num_joints} bodies={robot.num_bodies} spawn_z={spawn_z:.2f}")
    default_q = robot.data.default_joint_pos.clone()
    body_names = robot.data.body_names
    foot_ids = [i for i, n in enumerate(body_names) if "ankle_roll" in n]

    dt = sim.get_physics_dt()
    steps = int(args.seconds / dt)
    log_every = max(1, steps // 10)
    print(f"[spawn_x2]  t(s)  base_z  upright")
    base_z_last = spawn_z
    for k in range(steps):
        robot.set_joint_position_target(default_q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        if k % log_every == 0 or k == steps - 1:
            bz = float(robot.data.root_pos_w[0, 2].item())
            qw, qx, qy, qz = (float(v) for v in robot.data.root_quat_w[0].tolist())
            upright = 1.0 - 2.0 * (qx * qx + qy * qy)   # world-z component of body z-axis
            print(f"[spawn_x2] {k*dt:5.2f}  {bz:6.3f}  {upright:6.2f}")
            base_z_last = bz

    # final foot heights (should be ~0 if standing on the floor)
    if foot_ids:
        fz = [float(robot.data.body_pos_w[0, i, 2].item()) for i in foot_ids]
        print(f"[spawn_x2] foot_z (ankle_roll) = {[round(z,3) for z in fz]}")

    qw, qx, qy, qz = (float(v) for v in robot.data.root_quat_w[0].tolist())
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    stood = bool(base_z_last > 0.45 and upright > 0.85)
    print(f"[spawn_x2] RESULT base_z={base_z_last:.3f} upright={upright:.2f} "
          f"-> {'STANDS' if stood else 'did NOT stand'}")
    simulation_app.close()
    return 0 if stood else 1


if __name__ == "__main__":
    raise SystemExit(main())
