"""train_cref.py (P6-M3-T2/T3/T4) — distill then fine-tune the raw-depth CReF policy.

Pipeline: collect dataset (raw_depth_dataset) -> distill student to teacher (distillation) ->
fine-tune with PPO. BLOCKED to run: requires torch + Isaac Lab + the Phase 4 teacher checkpoint.

Usage: python -m x2_locomotion.scripts.train_cref --teacher <run>/model.pt --phase distill
"""
from __future__ import annotations

import argparse
import sys

_PHASES = {"collect", "distill", "finetune"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="Phase 4 height-map checkpoint")
    ap.add_argument("--phase", choices=sorted(_PHASES), default="distill")
    ap.add_argument("--steps", type=int, default=100_000)
    args = ap.parse_args(argv)
    print(f"[train_cref] phase={args.phase} teacher={args.teacher} steps={args.steps}")
    try:
        import torch  # noqa: F401
    except Exception as exc:
        print(f"[train_cref] BLOCKED: torch not available ({exc}).")
        return 2
    raise NotImplementedError(
        "Start only after the Phase 4 height-map policy works (roadmap §10.1). "
        "Wire collect/distill/finetune once torch + Isaac Lab + teacher checkpoint exist.")


if __name__ == "__main__":
    sys.exit(main())
