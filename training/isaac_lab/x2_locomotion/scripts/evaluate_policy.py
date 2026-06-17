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

# Success classification thresholds. These distinguish *walking* from *standing*: at the low
# command speeds used here (≤ ~1 m/s) a plain velocity-error gate is useless because a robot
# standing still scores a small error against a small command. So a WALK command requires the
# robot to actually MOVE (achieve a fraction of the commanded speed), while a STAND command
# requires it to stay still.
WALK_CMD_THRESHOLD_MPS = 0.1   # commanded planar speed above this => "walk" episode
WALK_TRACK_FRACTION = 0.5      # walk success: achieved >= this * commanded speed
STAND_SPEED_MAX_MPS = 0.15     # stand success: achieved planar speed stays below this


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
    """One finished episode.

    ``survived``       reached the episode time-out without falling.
    ``cmd_speed``      mean commanded planar speed (m/s) over the episode.
    ``achieved_speed`` mean achieved planar speed (m/s) over the episode.
    """

    survived: bool
    cmd_speed: float
    achieved_speed: float


def is_walk_command(o: EpisodeOutcome) -> bool:
    return o.cmd_speed > WALK_CMD_THRESHOLD_MPS


def episode_success(o: EpisodeOutcome) -> bool:
    """Walk episode: upright AND actually moving toward the command. Stand episode: upright AND
    actually staying still. A standing robot under a walk command now correctly FAILS."""
    if not o.survived:
        return False
    if is_walk_command(o):
        return o.achieved_speed >= WALK_TRACK_FRACTION * o.cmd_speed
    return o.achieved_speed <= STAND_SPEED_MAX_MPS


def success_rate(outcomes: list[EpisodeOutcome]) -> float:
    """Fraction of episodes that count as a success. Empty -> 0.0 (fail closed)."""
    if not outcomes:
        return 0.0
    return sum(episode_success(o) for o in outcomes) / len(outcomes)


def split_success_rates(outcomes: list[EpisodeOutcome]) -> dict[str, float]:
    """Break results into walk vs stand so 'stands but never walks' can't hide in an aggregate."""
    walk = [o for o in outcomes if is_walk_command(o)]
    stand = [o for o in outcomes if not is_walk_command(o)]

    def _rate(lst):
        return sum(episode_success(o) for o in lst) / len(lst) if lst else float("nan")

    return {
        "overall": success_rate(outcomes),
        "walk_success": _rate(walk),
        "stand_success": _rate(stand),
        "n_walk": len(walk),
        "n_stand": len(stand),
        "mean_walk_cmd_speed": sum(o.cmd_speed for o in walk) / len(walk) if walk else 0.0,
        "mean_walk_achieved_speed": sum(o.achieved_speed for o in walk) / len(walk) if walk else 0.0,
    }


def run_rollouts(task: str, checkpoint: str, episodes: int, device: str,
                 num_envs: int = 64, condition: str = "flat") -> list[EpisodeOutcome]:
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

    cmd_sum = torch.zeros(num_envs, device=device)       # accumulated commanded planar speed
    ach_sum = torch.zeros(num_envs, device=device)       # accumulated achieved planar speed
    steps = torch.zeros(num_envs, device=device)
    outcomes: list[EpisodeOutcome] = []

    # Step budget guards against an env that never terminates (≈1200 policy steps/episode).
    max_steps = max(episodes // max(num_envs, 1) + 2, 4) * 1500
    print(f"[evaluate] rolling out (target {episodes} episodes, {num_envs} envs, "
          f"budget {max_steps} steps)...", flush=True)
    step = 0
    while len(outcomes) < episodes and step < max_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
        step += 1

        cmd = cmd_mgr.get_command("base_velocity")           # (N,3) lin_x, lin_y, ang_z
        actual = robot.data.root_lin_vel_b[:, :2]            # (N,2) planar body velocity
        cmd_sum += torch.linalg.norm(cmd[:, :2], dim=-1)     # commanded planar speed
        ach_sum += torch.linalg.norm(actual, dim=-1)         # achieved planar speed
        steps += 1.0

        # time_outs (truncation) = survived to the time limit; a done without it = fell.
        time_out = extras.get("time_outs") if isinstance(extras, dict) else None
        if time_out is None:
            time_out = torch.zeros_like(dones, dtype=torch.bool)
        for i in torch.nonzero(dones, as_tuple=False).flatten().tolist():
            n = max(steps[i].item(), 1.0)
            outcomes.append(EpisodeOutcome(
                survived=bool(time_out[i].item()),
                cmd_speed=float(cmd_sum[i].item() / n),
                achieved_speed=float(ach_sum[i].item() / n)))
            cmd_sum[i] = 0.0
            ach_sum[i] = 0.0
            steps[i] = 0.0
        if len(outcomes) and len(outcomes) % 25 == 0:
            print(f"[evaluate]   {len(outcomes)}/{episodes} episodes...", flush=True)

    outcomes = outcomes[:episodes]
    # Print the result BEFORE closing the sim app — simulation_app.close() can hard-exit the
    # process, so anything after it (incl. a print back in main) may never run.
    s = split_success_rates(outcomes)
    gate = GRADUATION_THRESHOLDS.get(condition)
    verdict = "PASS" if s["overall"] >= (gate or 0.0) else "FAIL"
    print(f"[evaluate] RESULT {condition}: overall={s['overall']:.3f} -> {verdict} (gate={gate}) | "
          f"WALK {s['walk_success']:.3f} (n={s['n_walk']}, "
          f"cmd~{s['mean_walk_cmd_speed']:.2f} achieved~{s['mean_walk_achieved_speed']:.2f} m/s) | "
          f"STAND {s['stand_success']:.3f} (n={s['n_stand']}) | "
          f"survived {sum(o.survived for o in outcomes)}/{len(outcomes)}", flush=True)
    if s["n_walk"] and s["mean_walk_achieved_speed"] < WALK_TRACK_FRACTION * s["mean_walk_cmd_speed"]:
        print("[evaluate] WARNING: walk-commanded robots are barely moving — policy is standing, "
              "not walking.", flush=True)

    env.close()
    simulation_app.close()
    return outcomes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="flat_walk", choices=["standing", "flat_walk"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    print(f"[evaluate] task={args.task} checkpoint={args.checkpoint} episodes={args.episodes}")
    try:
        import isaaclab.app  # noqa: F401
    except Exception as exc:  # pragma: no cover - only on a box without Isaac Lab
        print(f"[evaluate] BLOCKED: Isaac Lab not available ({exc}).")
        return 2

    # run_rollouts prints the RESULT line itself (before it closes the sim app, which can
    # hard-exit the process). Graduation spans every terrain — this only fills the one
    # condition this task covers; run each task to complete the full gate.
    condition = TASK_CONDITION.get(args.task, args.task)
    run_rollouts(args.task, args.checkpoint, args.episodes,
                 args.device, args.num_envs, condition)
    return 0


if __name__ == "__main__":
    sys.exit(main())
