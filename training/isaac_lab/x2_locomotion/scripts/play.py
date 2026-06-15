"""play.py (P4-M4-T1) — roll out a trained X2 checkpoint (Isaac Lab 2.3 + rsl_rl).

Loads a checkpoint and runs the policy for visual inspection. Add --livestream 2 (drop
--headless) to watch in the /viewer tab.

    python training/isaac_lab/x2_locomotion/scripts/play.py --task standing \
        --checkpoint logs/rsl_rl/x2_standing/<run>/model_final.pt --livestream 2 --num_envs 16
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained X2 policy.")
parser.add_argument("--task", default="standing", choices=["standing", "flat_walk"])
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import x2_locomotion.tasks  # noqa: F401,E402
from x2_locomotion.tasks.standing.x2_standing_env_cfg import X2StandingEnvCfg  # noqa: E402
from x2_locomotion.tasks.flat_walk.x2_flat_walk_env_cfg import X2FlatWalkEnvCfg  # noqa: E402
from x2_locomotion.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    X2StandingPPORunnerCfg, X2FlatWalkPPORunnerCfg)

_ENV_CFGS = {"standing": X2StandingEnvCfg, "flat_walk": X2FlatWalkEnvCfg}
_AGENT_CFGS = {"standing": X2StandingPPORunnerCfg, "flat_walk": X2FlatWalkPPORunnerCfg}


def main():
    env_cfg = _ENV_CFGS[args.task]()
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg = _AGENT_CFGS[args.task]()
    agent_cfg.device = args.device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # rsl_rl versions differ: get_observations() may return obs or (obs, extras)
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            step_out = env.step(actions)
            obs = step_out[0]

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
