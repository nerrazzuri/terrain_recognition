#!/usr/bin/env python3
"""Validate the factory cpgwalk teacher against a real robot recording (P4-M6-T2 follow-up).

Two modes:

  # 1) INSPECT — works on any aimrt/ros2 .mcap; run this FIRST on a new recording to learn the
  #    exact topic names, message types, rates, and field names (esp. for joint state/command + IMU).
  python tools/validate_factory_teacher.py inspect <bag.mcap>

  # 2) VALIDATE — reconstruct the 65-obs at each timestep, run cpgwalk.onnx, and compare the
  #    predicted action to the RECORDED joint command; grid-search the CPG phase (freq, offset)
  #    that best reproduces the recording. Low residual => our teacher/obs reconstruction is faithful.
  python tools/validate_factory_teacher.py validate <bag.mcap> --onnx <cpgwalkrun.onnx>

The FIELD_MAP below is the one thing to finalize after `inspect` reveals the real message layout.
Decoding uses mcap-ros2-support (schemas are embedded in the bag). Obs/action logic is reused
from x2_locomotion.cref.factory_cpgwalk (already unit-tested).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

# --- obs/action contract (validated module) ---
sys.path.insert(0, __file__.rsplit("/tools/", 1)[0] + "/training/isaac_lab")
from x2_locomotion.cref import factory_cpgwalk as fc  # noqa: E402

# ---------------------------------------------------------------------------------------------
# FIELD_MAP — FILL IN after running `inspect` on the real walking recording. Maps each obs input
# to (topic, attribute-path). Joints must be reordered into fc.JOINT_ORDER (17-DoF) by name.
# Leaving a value None just means that obs block stays zero (still useful for a partial check).
FIELD_MAP = {
    "imu_topic": None,          # e.g. "/aima/hal/imu/chest/state"
    "imu_angvel_attr": None,    # e.g. "angular_velocity"  (.x/.y/.z)
    "imu_euler_attr": None,     # e.g. "orientation" (quat -> euler) or an rpy field
    "joint_state_topic": None,  # e.g. "/aima/hal/joint/leg/state"
    "joint_state_name_attr": None,   # e.g. "name" (list of joint names)
    "joint_state_pos_attr": None,    # e.g. "position"
    "joint_state_vel_attr": None,    # e.g. "velocity"
    "joint_cmd_topic": None,    # e.g. "/aima/hal/joint/leg/command"
    "joint_cmd_pos_attr": None, # commanded position target
    "vel_cmd_topic": None,      # e.g. "/aima/mc/locomotion/velocity"
    "vel_cmd_attrs": None,      # e.g. ("forward_velocity","lateral_velocity","angular_velocity")
}
# ---------------------------------------------------------------------------------------------


def inspect(path: str) -> int:
    from mcap_ros2.reader import read_ros2_messages
    first, count, t0, t1 = {}, defaultdict(int), {}, {}
    for m in read_ros2_messages(path):
        tp = m.channel.topic
        count[tp] += 1
        t1[tp] = m.log_time
        if tp not in first:
            first[tp] = (m.schema.name, m.ros_msg)
            t0[tp] = m.log_time
    print(f"=== {path} : {len(count)} topics ===")
    for tp in sorted(count):
        sch, msg = first[tp]
        span = t1[tp] - t0[tp]
        dt = span.total_seconds() if hasattr(span, "total_seconds") else span / 1e9
        rate = (count[tp] - 1) / dt if dt > 0 else 0.0
        print(f"\n{tp}\n  type={sch}  msgs={count[tp]}  ~{rate:.1f} Hz")
        for a in [x for x in dir(msg) if not x.startswith("_")][:25]:
            try:
                v = getattr(msg, a)
                if callable(v):
                    continue
                desc = (f"len={len(v)}" if hasattr(v, "__len__") and not isinstance(v, str)
                        else repr(v)[:60])
                print(f"    .{a}: {desc}")
            except Exception:
                pass
    return 0


def _extract(path):
    """Pull the mapped topics into aligned arrays. Returns dict of time-series (None if unmapped)."""
    from mcap_ros2.reader import read_ros2_messages
    fm = FIELD_MAP
    needed = {fm[k] for k in ("imu_topic", "joint_state_topic", "joint_cmd_topic", "vel_cmd_topic")} - {None}
    if not needed:
        print("[validate] FIELD_MAP not filled in — run `inspect` first and set the topics/fields.")
        return None
    rows = defaultdict(list)
    for m in read_ros2_messages(path, topics=list(needed)):
        rows[m.channel.topic].append((m.log_time, m.ros_msg))
    return rows


def validate(path: str, onnx: str) -> int:
    rows = _extract(path)
    if rows is None:
        return 2
    # NOTE: with FIELD_MAP filled, build per-timestep 65-obs via fc.build_obs (reordering joints
    # into fc.JOINT_ORDER), then grid-search CPG phase (freq w, offset phi) minimizing
    # || cpgwalk(obs_with_q(w,phi)) - recorded_joint_command ||. Report best (w,phi) + residual.
    pol = fc.FactoryCpgwalkPolicy(onnx)  # noqa: F841 (used once obs assembled)
    print("[validate] extraction OK; obs-assembly + CPG-phase search run once FIELD_MAP is set "
          "from the real recording. (Skeleton in place — finalize field mapping tomorrow.)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect"); pi.add_argument("mcap")
    pv = sub.add_parser("validate"); pv.add_argument("mcap"); pv.add_argument("--onnx", required=True)
    args = ap.parse_args(argv)
    return inspect(args.mcap) if args.cmd == "inspect" else validate(args.mcap, args.onnx)


if __name__ == "__main__":
    sys.exit(main())
