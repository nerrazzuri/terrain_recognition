"""spawn_x2.py — spawn the X2 in Isaac Sim and check it stands (P3-M2-T5 / P3-M3-T1).

Isaac Lab standalone script: launches Isaac Sim, spawns a ground plane + the X2 articulation
from our x2_robot_cfg, holds the default standing pose with the configured PD actuators, and
reports base-height stability. This is the Isaac Lab counterpart of the (already validated)
MuJoCo training/mujoco/scripts/stand.py.

Run on the GPU box (see training/isaac_lab/SETUP.md):
    ./isaaclab.sh -p training/isaac_lab/scripts/spawn_x2.py --headless --seconds 5
"""
from __future__ import annotations

import argparse

# --- 1. launch the simulator (must happen before importing isaaclab.* assets) ----------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn + stand the X2 in Isaac Sim.")
parser.add_argument("--seconds", type=float, default=5.0)
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
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.6])

    # ground + light
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg())

    # our X2
    robot_cfg = build_robot_cfg().replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.reset()
    default_q = robot.data.default_joint_pos.clone()

    steps = int(args.seconds / sim.get_physics_dt())
    base_z = []
    for _ in range(steps):
        # hold the default standing pose via the configured PD actuators
        robot.set_joint_position_target(default_q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        base_z.append(float(robot.data.root_pos_w[0, 2].item()))

    z = torch.tensor(base_z)
    settled = z[len(z) // 2:]
    stood = bool(settled.mean() > 0.35 and settled.std() < 0.03)
    print(f"[spawn_x2] spawn_height={BASE_HEIGHT_M:.2f} "
          f"settled_base_z={settled.mean().item():.3f} (std {settled.std().item():.3f}) "
          f"-> {'STANDS' if stood else 'did NOT stand'}")
    simulation_app.close()
    return 0 if stood else 1


if __name__ == "__main__":
    raise SystemExit(main())
