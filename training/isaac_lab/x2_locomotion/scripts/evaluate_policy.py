"""evaluate_policy.py (P4-M4-T1) — success-rate report against the graduation criteria.

Rolls out a trained checkpoint headlessly and computes a success rate, then checks it against
the Phase 4 graduation gate (roadmap §8.11): flat >95%, rough >90%, 5 cm step >90%,
10 cm step >80%, stair-up >80%.

An episode counts as a **success** when the robot (a) survived to the episode time-out without
falling AND (b) tracked the commanded planar velocity within tolerance over the episode.

The pure logic (``check_graduation``, ``episode_success``, ``success_rate``) is unit-tested
with no torch / Isaac Lab. The rollout itself needs Isaac Lab + a trained checkpoint and is
launched lazily inside ``run_rollouts`` so importing this module stays cheap.

    # on the GPU box, venv active:
    python training/isaac_lab/x2_locomotion/scripts/evaluate_policy.py \
        --task flat_walk --checkpoint models/x2_flat_walk_v1/model_1050.pt --episodes 200
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

GRADUATION_THRESHOLDS = {
    "flat": 0.95, "rough": 0.90, "step_5cm": 0.90, "step_10cm": 0.80, "stairs_up": 0.80,
}

# Which graduation condition each curriculum task reports against.
TASK_CONDITION = {"standing": "flat", "flat_walk": "flat", "rough": "rough", "stairs": "stairs_up"}

# Default success gate: stayed upright the whole episode and tracked commanded vel within this.
DEFAULT_MAX_VEL_ERROR_MPS = 0.3


def check_graduation(success_rates: dict[str, float]) -> tuple[bool, list[str]]:
    """Return ``(passed, failures)`` comparing measured rates to the graduation gate."""
    failures = []
    for name, thresh in GRADUATION_THRESHOLDS.items():
        rate = success_rates.get(name)
        if rate is None:
            failures.append(f"{name}: not measured")
        elif rate < thresh:
            failures.append(f"{name}: {rate:.2f} < {thresh:.2f}")
    return (len(failures) == 0), failures


@dataclass(frozen=True)
class EpisodeOutcome:
    """One finished episode. ``survived`` = reached the time-out without falling."""

    survived: bool
    mean_vel_error: float          # mean |commanded - actual| planar velocity over the episode


def episode_success(outcome: EpisodeOutcome,
                    max_vel_error: float = DEFAULT_MAX_VEL_ERROR_MPS) -> bool:
    """A locomotion episode succeeds iff it stayed upright AND tracked the command."""
    return bool(outcome.survived and outcome.mean_vel_error <= max_vel_error)


def success_rate(outcomes: list[EpisodeOutcome],
                 max_vel_error: float = DEFAULT_MAX_VEL_ERROR_MPS) -> float:
    """Fraction of episodes that count as a success. Empty -> 0.0 (fail closed)."""
    if not outcomes:
        return 0.0
    return sum(episode_success(o, max_vel_error) for o in outcomes) / len(outcomes)


def run_rollouts(task: str, checkpoint: str, episodes: int, device: str,
                 num_envs: int = 64) -> list[EpisodeOutcome]:
    """Headless rollout of the trained policy. Returns one EpisodeOutcome per finished episode.

    Lazily launches Isaac Sim so importing this module (for the pure-logic tests) is cheap.
    """
    # --- launch the simulator first, then import the env stack (Isaac Lab ordering rule) ---
    from isaaclab.app import AppLauncher
    _p = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(_p)
    app_launcher = AppLauncher(_p.parse_args(["--headless", "--device", device]))
    simulation_app = app_launcher.app

    import torch  # noqa: E402
    from rsl_rl.runners import OnPolicyRunner  # noqa: E402
    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

    import x2_locomotion.tasks  # noqa: F401,E402  (registers tasks)
    from x2_locomotion.tasks.standing.x2_standing_env_cfg import X2StandingEnvCfg  # noqa: E402
    from x2_locomotion.tasks.flat_walk.x2_flat_walk_env_cfg import X2FlatWalkEnvCfg  # noqa: E402
    from x2_locomotion.agents.rsl_rl_ppo_cfg import (  # noqa: E402
        X2StandingPPORunnerCfg, X2FlatWalkPPORunnerCfg)

    env_cfgs = {"standing": X2StandingEnvCfg, "flat_walk": X2FlatWalkEnvCfg}
    agent_cfgs = {"standing": X2StandingPPORunnerCfg, "flat_walk": X2FlatWalkPPORunnerCfg}

    env_cfg = env_cfgs[task]()
    env_cfg.scene.num_envs = num_envs
    agent_cfg = agent_cfgs[task]()
    agent_cfg.device = device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=device)

    robot = env.unwrapped.scene["robot"]
    cmd_mgr = env.unwrapped.command_manager

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    err_sum = torch.zeros(num_envs, device=device)
    steps = torch.zeros(num_envs, device=device)
    outcomes: list[EpisodeOutcome] = []

    while len(outcomes) < episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

            cmd = cmd_mgr.get_command("base_velocity")           # (N,3) lin_x, lin_y, ang_z
            actual = robot.data.root_lin_vel_b[:, :2]            # (N,2) planar body velocity
            err = torch.linalg.norm(cmd[:, :2] - actual, dim=-1)
            err_sum += err
            steps += 1.0

            # time_outs (truncation) = survived; a done without time_out = fell.
            time_out = extras.get("time_outs")
            if time_out is None:
                time_out = torch.zeros_like(dones, dtype=torch.bool)
            done_idx = torch.nonzero(dones, as_tuple=False).flatten()
            for i in done_idx.tolist():
                n = max(steps[i].item(), 1.0)
                outcomes.append(EpisodeOutcome(
                    survived=bool(time_out[i].item()),
                    mean_vel_error=float(err_sum[i].item() / n)))
                err_sum[i] = 0.0
                steps[i] = 0.0

    env.close()
    simulation_app.close()
    return outcomes[:episodes]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="flat_walk", choices=["standing", "flat_walk"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_vel_error", type=float, default=DEFAULT_MAX_VEL_ERROR_MPS)
    args = ap.parse_args(argv)

    print(f"[evaluate] task={args.task} checkpoint={args.checkpoint} episodes={args.episodes}")
    try:
        import isaaclab.app  # noqa: F401
    except Exception as exc:  # pragma: no cover - only on a box without Isaac Lab
        print(f"[evaluate] BLOCKED: Isaac Lab not available ({exc}).")
        return 2

    outcomes = run_rollouts(args.task, args.checkpoint, args.episodes,
                            args.device, args.num_envs)
    rate = success_rate(outcomes, args.max_vel_error)
    condition = TASK_CONDITION.get(args.task, args.task)
    survived = sum(o.survived for o in outcomes)
    print(f"[evaluate] {condition}: success_rate={rate:.3f} "
          f"(survived {survived}/{len(outcomes)}; vel-err gate {args.max_vel_error} m/s)")

    # Single-task gate check: graduation spans every terrain, so this only fills the one
    # condition this task covers; run each task to complete the full gate.
    passed, failures = check_graduation({condition: rate})
    print(f"[evaluate] {condition} threshold "
          f"{'PASS' if not any(condition in f for f in failures) else 'FAIL'} "
          f"(gate={GRADUATION_THRESHOLDS.get(condition)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
