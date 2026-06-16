"""export_onnx.py (P4-M4-T3/T4) — export the trained actor to ONNX and validate numerically.

Exports ActorCritic.act (observation -> action) to ONNX, then validates ONNX vs PyTorch on
random test vectors within tolerance (AGENTS.md §4: determinism for sim-to-real).

``numeric_match`` is pure and unit-tested. The export + ONNX-runtime comparison need torch
and onnxruntime installed (BLOCKED to run here: torch absent).

Usage: python -m x2_locomotion.scripts.export_onnx --checkpoint <run>/model.pt --out policy.onnx
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from x2_common import config_loader


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
        import torch  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception as exc:
        print(f"[export_onnx] BLOCKED: torch/onnxruntime not available ({exc}).")
        return 2

    # --- with torch present: ----------------------------------------------------------
    # Stage A/B use rsl_rl's built-in MLP actor (RslRlPpoActorCriticCfg), NOT the custom
    # ActorCritic from network.py.  The custom ActorCritic (height_encoder + proprio_encoder)
    # is wired in from Stage E onwards when privileged observations are added.
    #
    # For Stage A/B checkpoints (model_final.pt saved by rsl_rl OnPolicyRunner):
    #   1. Load checkpoint: ckpt = torch.load(checkpoint)
    #   2. Instantiate runner actor: actor is an MLP with dims [OBS_DIM, 256, 256, 128, 12]
    #      where OBS_DIM = 168 (observations.OBSERVATION_DIM).
    #      from x2_locomotion.agents.rsl_rl_ppo_cfg import X2StandingPPORunnerCfg
    #      policy_cfg = X2StandingPPORunnerCfg().policy
    #      from rsl_rl.modules import ActorCritic as RslActorCritic
    #      model = RslActorCritic(OBS_DIM, OBS_DIM, 12, policy_cfg)
    #   3. model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    #   4. dummy_obs = torch.zeros(1, 168)  # OBS_DIM = observations.OBSERVATION_DIM
    #   5. torch.onnx.export(model.actor, dummy_obs, out_path,
    #                        input_names=["obs"], output_names=["action"],
    #                        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}})
    #   6. Run onnxruntime on random obs vectors; compare to model.actor(obs) via numeric_match
    #   7. fail if any vector exceeds tol
    raise NotImplementedError("export + validate once torch/onnxruntime are installed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="policy.onnx")
    args = ap.parse_args(argv)
    return export(args.checkpoint, args.out)


if __name__ == "__main__":
    sys.exit(main())
