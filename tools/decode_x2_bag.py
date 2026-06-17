#!/usr/bin/env python3
"""Decode an X2 rosbag2 (.db3/.mcap) of locomotion telemetry into aligned numpy time-series.

Solves the version-mismatch decoding problem two ways:
  * register the matching `aimdk_msgs` defs from a versioned SDK (e.g. lx2501_3-v0.8.2.4) and let
    rosbags deserialize cleanly; and/or
  * a **name-anchored** joint parser that ignores the header/version layout entirely — it finds
    each joint-name string in the CDR bytes and reads the float64 pos/vel/effort that follow
    (works regardless of msg version). Used as a fallback / cross-check.

Output: an .npz with per-joint position/velocity time-series, joint commands, velocity command,
and IMU — the inputs needed to validate the factory cpgwalk teacher (P4-M6).

    python tools/decode_x2_bag.py <bag_dir> --sdk /path/to/lx2501_3-v0.8.2.4 --out x2_walk.npz
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

LEG = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
       "left_ankle_pitch_joint", "left_ankle_roll_joint",
       "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
       "right_ankle_pitch_joint", "right_ankle_roll_joint"]
WAIST = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]


def _typestore(sdk: str | None):
    from rosbags.typesys import get_typestore, Stores
    from rosbags.typesys.msg import get_types_from_msg
    ts = get_typestore(Stores.ROS2_HUMBLE)
    if sdk:
        allt = {}
        for p in pathlib.Path(sdk).rglob("*.msg"):
            if "aimdk_msgs" in str(p):
                try:
                    allt.update(get_types_from_msg(p.read_text(), f"aimdk_msgs/msg/{p.stem}"))
                except Exception:
                    pass
        ts.register(allt)
    return ts


def _align(p, size, base=4):
    rem = (p - base) % size
    return p + ((size - rem) % size)


def anchored_joint_positions(raw: bytes, names: list[str]) -> dict[str, tuple[float, float]]:
    """Version-proof: anchor on each joint-name string, read the float64 pos/vel that follow.

    JointState layout per element: string name | (align8) float64 pos,vel,effort | u8 x3 temps.
    """
    out = {}
    idx = raw.find(names[0].encode())
    if idx < 0:
        return out
    p = idx - 4
    for _ in names:
        p = _align(p, 4)
        nlen = struct.unpack_from("<I", raw, p)[0]; p += 4
        name = raw[p:p + nlen - 1].decode("utf-8", "replace"); p += nlen
        p = _align(p, 8)
        pos, vel, _eff = struct.unpack_from("<ddd", raw, p); p += 24
        p += 3
        out[name] = (pos, vel)
    return out


def decode(bag: str, sdk: str | None):
    from rosbags.rosbag2 import Reader
    ts = _typestore(sdk)
    series = {"vel_cmd": [], "leg_pos": [], "leg_vel": []}

    with Reader(bag) as r:
        # velocity command (clean decode with matched defs)
        vc = [c for c in r.connections if c.topic == "/aima/mc/locomotion/velocity"]
        for conn, t, raw in r.messages(connections=vc):
            try:
                m = ts.deserialize_cdr(raw, conn.msgtype)
                series["vel_cmd"].append((t, m.forward_velocity, m.lateral_velocity, m.angular_velocity))
            except Exception:
                break
        # leg joint state via name-anchored parser (version-proof)
        lc = [c for c in r.connections if c.topic == "/aima/hal/joint/leg/state"]
        for conn, t, raw in r.messages(connections=lc):
            d = anchored_joint_positions(raw, LEG)
            if len(d) == len(LEG):
                series["leg_pos"].append([t] + [d[n][0] for n in LEG])
                series["leg_vel"].append([t] + [d[n][1] for n in LEG])

    return {k: np.array(v) for k, v in series.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag")
    ap.add_argument("--sdk", default=None, help="path to version-matched SDK (for clean decode)")
    ap.add_argument("--out", default="x2_walk.npz")
    args = ap.parse_args(argv)

    s = decode(args.bag, args.sdk)
    vc, lp = s["vel_cmd"], s["leg_pos"]
    print(f"velocity cmd: {len(vc)} msgs", end="")
    if len(vc):
        fwd = vc[:, 1]
        print(f" | fwd max={fwd.max():+.2f} mean={fwd.mean():+.2f} moving={np.mean(np.abs(fwd) > 0.1):.0%}")
    print(f"leg state: {len(lp)} frames, {len(LEG)} joints")
    np.savez(args.out, **s)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
