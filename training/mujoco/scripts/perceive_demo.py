#!/usr/bin/env python3
"""perceive_demo.py — run the terrain perception + safe-stop pipeline on MuJoCo geometry.

Instead of a hand-built point cloud, this renders a depth cloud by ray-casting a forward-down
RGB-D sensor (head height) into a MuJoCo scene, then runs the REAL perception cores
(height map -> ground/slope/stair/gap -> classifier) and the safe-locomotion velocity policy.
It prints, per scenario, the terrain type, safety decision, and the velocity the safe adapter
would command. An "approach" sweep shows the command ramp to zero as X2 nears an obstacle.

No GPU / Isaac Lab / ROS2 needed (mj_ray is pure geometry, runs headless).

Usage:
    python training/mujoco/scripts/perceive_demo.py
    python training/mujoco/scripts/perceive_demo.py --scene stairs --plot out.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for p in ("ros2_ws/src/x2_common", "ros2_ws/src/x2_terrain_perception",
          "ros2_ws/src/x2_safe_locomotion", "training/isaac_lab"):
    sys.path.insert(0, str(_REPO / p))

from x2_common import config_loader  # noqa: E402
from x2_terrain_perception.core import (  # noqa: E402
    heightmap as hm, stairs as st, gaps as gp, slope as sl, ground_plane as gpl,
    roughness as rg, classifier as cl, grid_msg)
from x2_safe_locomotion.core.velocity_policy import VelocityPolicy  # noqa: E402

config_loader.os.environ.setdefault("X2_CONFIG_DIR", str(_REPO / "configs"))


# ----------------------------------------------------------------------------- scenes
def scene_xml(kind: str, obstacle_x: float = 1.0) -> str:
    """Build a MuJoCo scene: infinite floor + an obstacle in front of the robot (+x)."""
    geoms = ['<geom name="floor" type="plane" size="0 0 0.05"/>']
    x = obstacle_x
    if kind == "flat":
        pass
    elif kind == "curb":                       # single 0.15 m step
        geoms.append(f'<geom type="box" pos="{x + 0.4:.3f} 0 0.075" size="0.6 0.8 0.075"/>')
    elif kind == "stairs":                      # 4 steps, rise 0.15, tread 0.30
        rise, tread, n = 0.15, 0.30, 4
        for i in range(1, n + 1):
            front = x + (i - 1) * tread
            back = x + n * tread + 0.6
            cx = (front + back) / 2.0
            hx = (back - front) / 2.0
            geoms.append(f'<geom type="box" pos="{cx:.3f} 0 {i*rise/2:.3f}" '
                         f'size="{hx:.3f} 0.8 {i*rise/2:.3f}"/>')
    elif kind == "platform":                    # tall 0.40 m platform edge
        geoms.append(f'<geom type="box" pos="{x + 0.6:.3f} 0 0.2" size="0.8 0.8 0.2"/>')
    else:
        raise ValueError(f"unknown scene {kind}")
    return f'<mujoco><worldbody>{"".join(geoms)}</worldbody></mujoco>'


# ----------------------------------------------------------------------- depth sensor
def raycast_cloud(model, data, sensor_h=1.10, tilt_deg=32.0,
                  fov_h_deg=70.0, fov_v_deg=55.0, nu=64, nv=48) -> np.ndarray:
    """Cast a forward-down RGB-D grid from the head and return hit points in the base frame.

    Base frame: origin at the floor under the sensor, +x forward, +z up (robot faces +x).
    Misses (no geom hit) are simply absent -> those map cells stay unknown (occlusion).
    """
    import mujoco
    origin = np.array([0.0, 0.0, sensor_h])
    tilt = np.radians(tilt_deg)
    us = np.radians(np.linspace(-fov_h_deg / 2, fov_h_deg / 2, nu))
    vs = np.radians(np.linspace(-fov_v_deg / 2, fov_v_deg / 2, nv))
    gid = np.zeros(1, dtype=np.int32)
    pts = []
    for v in vs:
        el = tilt + v                          # downward elevation
        for u in us:
            d = np.array([np.cos(el) * np.cos(u), np.cos(el) * np.sin(u), -np.sin(el)])
            d /= np.linalg.norm(d)
            dist = mujoco.mj_ray(model, data, origin, d, None, 1, -1, gid)
            if dist < 0 or dist > 4.0:
                continue
            pts.append(origin + dist * d)       # world == base (robot at origin, +x fwd)
    return np.array(pts, dtype=float) if pts else np.empty((0, 3))


# --------------------------------------------------------------------------- pipeline
def perceive(cloud: np.ndarray, perc: dict, policy: VelocityPolicy):
    """Run the perception + decision cores on a base-frame cloud. Returns a result dict."""
    cfg = hm.HeightMapConfig.from_dict(perc["heightmap"])
    grid = hm.HeightMap(cfg, decay=0.5)
    grid.update(cloud, measurement_confidence=1.0)
    heights, conf, _ = grid.to_arrays()
    _, h2d, c2d, xpos = grid_msg.grid_from_flat(
        cfg.width, cfg.height, cfg.resolution_m, cfg.origin_x_m, cfg.origin_y_m, heights, conf)

    # Forward profile over the central y band, keeping only OBSERVED columns; align the
    # confidence profile to the SAME columns (else unobserved columns drag confidence down).
    c = h2d.shape[0] // 2
    band_h = h2d[max(0, c - 3): c + 3, :]
    band_c = c2d[max(0, c - 3): c + 3, :]
    with np.errstate(invalid="ignore"):
        prof_all = np.nanmean(band_h, axis=0)
        cprof_all = band_c.mean(axis=0)
    valid = ~np.isnan(prof_all)
    px, pz = xpos[valid], prof_all[valid]
    cprof = cprof_all[valid]

    stair_res = st.detect_stairs(px, pz, st.StairParams.from_dict(perc["stair_detector"]))
    gap_res = gp.detect_gap(px, pz, cprof, gp.GapParams.from_dict(perc["gap_detector"]))

    ys, xs = np.nonzero(np.isfinite(h2d))
    if xs.size >= 3:
        wpts = np.column_stack([cfg.origin_x_m + (xs + 0.5) * cfg.resolution_m,
                                cfg.origin_y_m + (ys + 0.5) * cfg.resolution_m, h2d[ys, xs]])
        plane = gpl.fit_plane_ransac(wpts, 0.03, seed=0)
        slope_deg = sl.slope_angle_deg(plane.normal)
        slope_dir = sl.slope_direction(plane.normal,
                                       float(perc["slope_detector"]["slope_threshold_deg"]))
        overall = float(np.clip(np.nanmean(cprof) if cprof.size else 0.0, 0.0, 1.0))
    else:
        slope_deg, slope_dir, overall = 0.0, "none", 0.0

    step_h = rg.max_single_step_m(pz) if not stair_res.stairs_detected else 0.0
    inp = cl.ClassifierInputs(
        overall_confidence=overall, slope_angle_deg=slope_deg, slope_direction=slope_dir,
        roughness_m=rg.roughness_m(h2d), single_step_height_m=step_h,
        max_obstacle_height_m=float(np.nanmax(h2d)) if np.isfinite(h2d).any() else 0.0,
        stairs=stair_res, gap=gap_res)
    out = cl.classify(inp, cl.ClassifierParams.from_dict(perc))

    cmd_v = policy.safe_velocity(0.10, out.terrain_type, out.safe_to_continue)
    return {"terrain": out.terrain_type, "safe": out.safe_to_continue, "reason": out.reason,
            "cmd_v": cmd_v, "n_points": int(cloud.shape[0]),
            "first_step_m": stair_res.first_step_distance_m if stair_res.stairs_detected else None}


def run_scene(kind: str, perc, policy, obstacle_x=1.0):
    import mujoco
    model = mujoco.MjModel.from_xml_string(scene_xml(kind, obstacle_x))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cloud = raycast_cloud(model, data)
    return perceive(cloud, perc, policy), cloud


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None, help="flat|curb|stairs|platform (default: all)")
    ap.add_argument("--sweep", action="store_true", help="approach sweep toward stairs")
    ap.add_argument("--plot", default=None, help="save the height map of --scene to a PNG")
    args = ap.parse_args(argv)
    try:
        import mujoco  # noqa: F401
    except Exception as exc:
        print(f"BLOCKED: mujoco not available ({exc}). pip install mujoco")
        return 2

    perc = config_loader.load_config("terrain_perception")
    policy = VelocityPolicy.from_dict(config_loader.load_config("safe_locomotion"))

    scenes = [args.scene] if args.scene else ["flat", "curb", "stairs", "platform"]
    print(f"{'scene':10} {'terrain_type':20} {'safe':5} {'cmd_v(m/s)':10} pts   reason")
    print("-" * 92)
    for kind in scenes:
        r, cloud = run_scene(kind, perc, policy)
        print(f"{kind:10} {r['terrain']:20} {str(r['safe']):5} {r['cmd_v']:<10.3f} "
              f"{r['n_points']:<5} {r['reason']}")
        if args.plot and kind == (args.scene or "stairs"):
            _save_plot(cloud, perc, args.plot)

    if args.sweep:
        print("\nApproach sweep — stairs moving from far to near (X2 at origin):")
        print(f"{'obstacle_x':12} {'terrain_type':20} {'safe':5} {'cmd_v(m/s)':10}")
        print("-" * 56)
        for ox in [2.4, 2.0, 1.6, 1.2, 1.0, 0.8, 0.6]:
            r, _ = run_scene("stairs", perc, policy, obstacle_x=ox)
            print(f"{ox:<12.2f} {r['terrain']:20} {str(r['safe']):5} {r['cmd_v']:<10.3f}")
    return 0


def _save_plot(cloud, perc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = hm.HeightMapConfig.from_dict(perc["heightmap"])
    grid = hm.HeightMap(cfg, decay=0.5)
    grid.update(cloud, 1.0)
    h = np.ma.masked_invalid(grid.height_grid_2d())
    plt.figure(figsize=(6, 3))
    plt.imshow(h, origin="lower", aspect="auto",
               extent=[cfg.origin_x_m, cfg.origin_x_m + cfg.width * cfg.resolution_m,
                       cfg.origin_y_m, cfg.origin_y_m + cfg.height * cfg.resolution_m])
    plt.colorbar(label="height (m)"); plt.xlabel("forward x (m)"); plt.ylabel("lateral y (m)")
    plt.title("MuJoCo-rendered height map (masked = unknown)")
    plt.tight_layout(); plt.savefig(path, dpi=110); print(f"saved {path}")


if __name__ == "__main__":
    sys.exit(main())
