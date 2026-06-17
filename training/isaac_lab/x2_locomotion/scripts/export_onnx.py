"""export_onnx.py (P4-M4-T3/T4) — export the trained actor to ONNX and validate numerically.

Stage A/B were trained with rsl_rl's built-in MLP actor-critic (``RslRlPpoActorCriticCfg`` in
agents/rsl_rl_ppo_cfg.py), NOT the custom ActorCritic in tasks/common/network.py (that one is
wired in from Stage E when privileged observations arrive). So we reconstruct the rsl_rl
ActorCritic, load the checkpoint, and export its **actor** (observation -> action), then check
ONNX vs PyTorch on random vectors within tolerance (AGENTS.md §4: sim-to-real determinism).

``numeric_match`` is pure and unit-tested. The export + ONNX-runtime comparison need torch,
rsl_rl and onnxruntime installed (run on the GPU box venv).

    python training/isaac_lab/x2_locomotion/scripts/export_onnx.py \
        --checkpoint models/x2_flat_walk_v1/model_1050.pt --out models/x2_flat_walk_v1/policy.onnx
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from x2_common import config_loader

# Actor architecture — must match agents/rsl_rl_ppo_cfg.py:X2StandingPPORunnerCfg.policy.
# (rsl_rl stores no arch metadata in the checkpoint, so we state it here as the source of truth.)
ACTOR_HIDDEN_DIMS = [256, 256, 128]
CRITIC_HIDDEN_DIMS = [256, 256, 128]
ACTIVATION = "elu"
ACTION_DIM = 12


def numeric_match(a, b, tol: float) -> tuple[bool, float]:
    """Return ``(within_tol, max_abs_diff)`` between two arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False, float("inf")
    max_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    return (max_diff <= tol), max_diff


def export(checkpoint: str, out_path: str, num_test_vectors: int = 32) -> int:
    cfg = config_loader.load_config("training_default")
    tol = float(cfg["export"]["numeric_tolerance"])
    try:
        import torch
        import onnxruntime
        from rsl_rl.modules import ActorCritic
        from x2_locomotion.tasks.common.observations import OBSERVATION_DIM
    except Exception as exc:
        print(f"[export_onnx] BLOCKED: torch/onnxruntime/rsl_rl not available ({exc}).")
        return 2

    # Infer dims straight from the checkpoint — the trained env's obs width is the source of
    # truth (it may differ from observations.OBSERVATION_DIM if the obs terms changed; e.g. the
    # Stage-B policy is 206-dim = joint_pos/vel over all 31 joints, vs the 168-dim contract).
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    obs_dim = int(state["actor.0.weight"].shape[1])
    critic_obs_dim = int(state["critic.0.weight"].shape[1])
    actor_layers = sorted(int(k.split(".")[1]) for k in state
                          if k.startswith("actor.") and k.endswith(".weight"))
    action_dim = int(state[f"actor.{actor_layers[-1]}.weight"].shape[0])
    if obs_dim != OBSERVATION_DIM:
        print(f"[export_onnx] NOTE: checkpoint obs_dim={obs_dim} != deployment contract "
              f"OBSERVATION_DIM={OBSERVATION_DIM} — Phase 5 observation_builder must match {obs_dim}.")

    # 1. Rebuild the rsl_rl actor-critic (dims from the checkpoint) and load the trained weights.
    actor_critic = ActorCritic(
        num_actor_obs=obs_dim, num_critic_obs=critic_obs_dim, num_actions=action_dim,
        actor_hidden_dims=ACTOR_HIDDEN_DIMS, critic_hidden_dims=CRITIC_HIDDEN_DIMS,
        activation=ACTIVATION)
    actor_critic.load_state_dict(state)
    actor_critic.eval()
    # nn.Sequential: obs -> mean action (empirical_normalization=False, so no normalizer layer)
    actor = actor_critic.actor

    # 2. Export the actor to ONNX (dynamic batch).
    dummy = torch.zeros(1, obs_dim)
    torch.onnx.export(
        actor, dummy, out_path,
        input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17)
    print(f"[export_onnx] wrote {out_path} (obs_dim={obs_dim}, action_dim={action_dim})")

    # 3. Validate ONNX vs PyTorch on random observations.
    sess = onnxruntime.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for _ in range(num_test_vectors):
        x = np.random.randn(1, obs_dim).astype(np.float32)
        onnx_out = sess.run(None, {"obs": x})[0]
        with torch.no_grad():
            torch_out = actor(torch.from_numpy(x)).numpy()
        ok, diff = numeric_match(onnx_out, torch_out, tol)
        worst = max(worst, diff)
        if not ok:
            print(f"[export_onnx] FAIL: ONNX/PyTorch diff {diff:.2e} > tol {tol:.2e}")
            return 1
    print(f"[export_onnx] OK: {num_test_vectors} vectors within tol {tol:.2e} "
          f"(worst diff {worst:.2e})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="policy.onnx")
    ap.add_argument("--num_test_vectors", type=int, default=32)
    args = ap.parse_args(argv)
    return export(args.checkpoint, args.out, args.num_test_vectors)


if __name__ == "__main__":
    sys.exit(main())
