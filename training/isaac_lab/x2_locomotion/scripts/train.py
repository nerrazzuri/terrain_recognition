"""train.py (P4-M1-T1) — PPO training entry point.

Launches Isaac Lab + rsl_rl PPO on a chosen curriculum stage, with config from
configs/training_default.yaml (512 envs, height-map version, camera rendering off).

BLOCKED to run: requires Isaac Lab, rsl_rl and a GPU. The argument parsing + config wiring
are real; the sim launch is guarded so a missing Isaac Lab gives a clear message, not a crash.

Usage: python -m x2_locomotion.scripts.train --task standing --max-iterations 1500
"""
from __future__ import annotations

import argparse
import sys

from x2_common import config_loader

_TASKS = {"standing", "flat_walk", "rough", "stairs"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(_TASKS), default="standing")
    ap.add_argument("--max-iterations", type=int, default=1500)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = config_loader.load_config("training_default")
    num_envs = args.num_envs or int(cfg["sim"]["num_envs"])
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    print(f"[train] task={args.task} num_envs={num_envs} seed={seed} "
          f"iters={args.max_iterations}")

    try:
        from isaaclab.app import AppLauncher  # noqa: F401
    except Exception as exc:
        print(f"[train] BLOCKED: Isaac Lab not available ({exc}).")
        print("        Install Isaac Lab + rsl_rl and provide the X2 USD asset (P3-M1).")
        return 2

    # --- with Isaac Lab present, the real pipeline would: -------------------------------
    #   1. AppLauncher(headless=True) to start Isaac Sim
    #   2. build the env cfg for args.task (standing/flat_walk/rough/stairs env cfgs)
    #   3. wrap with rsl_rl OnPolicyRunner using the ppo block from training_default.yaml
    #   4. runner.learn(num_learning_iterations=args.max_iterations)
    #   5. save checkpoints + normalization stats for export_onnx / deployment
    raise NotImplementedError("wire rsl_rl OnPolicyRunner once Isaac Lab is installed")


if __name__ == "__main__":
    sys.exit(main())
