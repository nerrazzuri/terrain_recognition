"""play.py (P4-M4-T1) — roll out a trained X2 checkpoint (Isaac Lab 2.1 + rsl_rl).

Two modes:
  * --video : headless offscreen render to an MP4 (works on a remote/cloud GPU box with no
    display). Needs --headless --enable_cameras. The MP4 lands next to the checkpoint under
    videos/play/. This is how you "watch it walk" on a server.
  * live    : open a native window / livestream (needs a display or --livestream 2).

    # record a clip on a headless box:
    python .../play.py --task flat_walk --checkpoint .../model_1050.pt \
        --num_envs 16 --video --video_length 300 --headless --enable_cameras
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained X2 policy.")
parser.add_argument("--task", default="standing", choices=["standing", "flat_walk"])
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--video", action="store_true", help="record an MP4 (headless offscreen)")
parser.add_argument("--video_length", type=int, default=300, help="video length in policy steps")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Recording offscreen requires the rendering pipeline.
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# Self-bootstrap import paths + config dir.
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "training/isaac_lab"))
sys.path.insert(0, str(_REPO / "ros2_ws/src/x2_common"))
os.environ.setdefault("X2_CONFIG_DIR", str(_REPO / "configs"))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import x2_locomotion.tasks  # noqa: F401,E402  (registers Isaac-X2-*-v0)
from x2_locomotion.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    X2StandingPPORunnerCfg, X2FlatWalkPPORunnerCfg)

_GYM_ID = {"standing": "Isaac-X2-Standing-v0", "flat_walk": "Isaac-X2-FlatWalk-v0"}
_AGENT_CFGS = {"standing": X2StandingPPORunnerCfg, "flat_walk": X2FlatWalkPPORunnerCfg}


def main():
    task_id = _GYM_ID[args.task]
    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)
    agent_cfg = _AGENT_CFGS[args.task]()
    agent_cfg.device = args.device

    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

    if args.video:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "videos", "play")
        env = gym.wrappers.RecordVideo(
            env, video_folder=out_dir, step_trigger=lambda s: s == 0,
            video_length=args.video_length, disable_logger=True)
        print(f"[play] recording {args.video_length} steps -> {out_dir}", flush=True)

    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    # In --video mode, step a bounded number of times then stop (so the clip is written and the
    # process exits); live mode runs until the window closes.
    max_steps = args.video_length + 20 if args.video else None
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs = env.step(actions)[0]
        step += 1
        if max_steps is not None and step >= max_steps:
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
