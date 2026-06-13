"""play.py (P4-M4-T1) — roll out a trained checkpoint in sim for visual inspection.

BLOCKED to run: requires Isaac Lab + a trained checkpoint. See evaluate_policy.py for the
quantitative success-rate report.

Usage: python -m x2_locomotion.scripts.play --task stairs --checkpoint <run>/model.pt
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="flat_walk")
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args(argv)
    print(f"[play] task={args.task} checkpoint={args.checkpoint}")
    try:
        from isaaclab.app import AppLauncher  # noqa: F401
    except Exception as exc:
        print(f"[play] BLOCKED: Isaac Lab not available ({exc}).")
        return 2
    raise NotImplementedError("load checkpoint and roll out once Isaac Lab is installed")


if __name__ == "__main__":
    sys.exit(main())
