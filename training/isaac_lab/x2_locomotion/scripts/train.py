"""train.py (P4-M1-T1) — PPO training for the X2 (Isaac Lab 2.3 + rsl_rl).

Trains a curriculum-stage policy. Stage A = standing (Isaac-X2-Standing-v0). Instantiates the
env cfg + rsl_rl runner cfg directly (no hydra/registry) for robustness, mirroring the Isaac
Lab rsl_rl training flow.

Run on the GPU box (see training/isaac_lab/SETUP.md):
    python training/isaac_lab/x2_locomotion/scripts/train.py --task standing --num_envs 4096 --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an X2 locomotion policy (rsl_rl PPO).")
parser.add_argument("--task", default="standing", choices=["standing", "flat_walk"],
                    help="curriculum stage (more stages added as they are built)")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

# Self-bootstrap import paths + config dir so the script runs from a fresh shell (env vars do
# not survive a Brev stop/restart). repo = .../training/isaac_lab/x2_locomotion/scripts -> [4]
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "training/isaac_lab"))
sys.path.insert(0, str(_REPO / "ros2_ws/src/x2_common"))
os.environ.setdefault("X2_CONFIG_DIR", str(_REPO / "configs"))

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import x2_locomotion.tasks  # noqa: F401,E402  (registers the gym tasks)
from x2_locomotion.tasks.standing.x2_standing_env_cfg import X2StandingEnvCfg  # noqa: E402
from x2_locomotion.tasks.flat_walk.x2_flat_walk_env_cfg import X2FlatWalkEnvCfg  # noqa: E402
from x2_locomotion.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    X2StandingPPORunnerCfg, X2FlatWalkPPORunnerCfg)

_ENV_CFGS = {"standing": X2StandingEnvCfg, "flat_walk": X2FlatWalkEnvCfg}
_AGENT_CFGS = {"standing": X2StandingPPORunnerCfg, "flat_walk": X2FlatWalkPPORunnerCfg}


def main():
    env_cfg = _ENV_CFGS[args.task]()
    agent_cfg = _AGENT_CFGS[args.task]()
    if args.num_envs:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations:
        agent_cfg.max_iterations = args.max_iterations
    env_cfg.seed = agent_cfg.seed = args.seed
    agent_cfg.device = args.device

    log_dir = os.path.abspath(os.path.join(
        "logs", "rsl_rl", agent_cfg.experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train] task={args.task} num_envs={env_cfg.scene.num_envs} "
          f"iters={agent_cfg.max_iterations} log_dir={log_dir}")

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    runner.save(os.path.join(log_dir, "model_final.pt"))
    print(f"[train] done. checkpoint: {os.path.join(log_dir, 'model_final.pt')}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
