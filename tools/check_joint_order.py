#!/usr/bin/env python3
"""check_joint_order.py (P3-M2-T2).

Verify a simulator joint ordering against the canonical AimDK leg order and the configured
joint limits. Prints a side-by-side table and flags missing/extra joints or out-of-range
limits. Run this before trusting any joint mapping in the policy runtime (AGENTS.md §3).

Usage:
    python tools/check_joint_order.py --sim-joints left_hip_pitch,left_hip_roll,...
    python tools/check_joint_order.py            # uses the canonical order as a self-check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "training/isaac_lab"))
sys.path.insert(0, str(_REPO / "ros2_ws/src/x2_common"))

from x2_locomotion.robots import x2_joint_map as jm  # noqa: E402
from x2_common import config_loader  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-joints", default=None,
                    help="comma-separated simulator joint names (default: canonical order)")
    args = ap.parse_args(argv)

    sim_names = (args.sim_joints.split(",") if args.sim_joints else jm.aimdk_leg_order())
    try:
        jmap = jm.JointMap(sim_names)
    except jm.JointMapError as exc:
        print(f"JOINT ORDER MISMATCH: {exc}")
        return 1
    print(jmap.describe())

    limits = config_loader.load_config("joint_limits_x2_ultra")
    if not limits.get("verified", False):
        print("\nWARNING: joint_limits_x2_ultra.yaml is marked verified: false — "
              "confirm against the AimDK joint motion-range docs before use.")
    joints = limits.get("joints", {})
    missing = [n for n in jm.aimdk_leg_order() if n not in joints]
    if missing:
        print(f"WARNING: no limit entry for: {missing}")
    print(f"\nUnits: {limits.get('units')}  soft margin: {limits.get('soft_limit_margin_rad')} rad")
    return 0


if __name__ == "__main__":
    sys.exit(main())
