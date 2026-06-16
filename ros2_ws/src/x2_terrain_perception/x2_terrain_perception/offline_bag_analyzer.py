"""offline_bag_analyzer (P1-M4-T3).

Replays a recorded rosbag2 of point clouds through the perception cores **offline** (no live
robot, no ROS graph spin) and prints the per-frame terrain classification, so detection can
be validated on recorded scenes (roadmap §5.6 / P1-M4-T6). Logging is part of the feature:
results are also written as JSON lines.

Usage:
    python -m x2_terrain_perception.offline_bag_analyzer <bag_dir> \
        [--topic /x2/terrain/points_processed] [--out logs/bag_analysis.jsonl]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from x2_common import config_loader
from x2_common.logging_utils import JsonlRecorder, get_logger

from .core import heightmap as hm, traversability as tv, stairs as st, gaps, slope, \
    ground_plane, classifier as cl, roughness, grid_msg

log = get_logger("offline_bag_analyzer")


def _read_clouds(bag_dir: str, topic: str):
    """Yield (timestamp_ns, Nx3 float array) for each PointCloud2 on ``topic``."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs_py import point_cloud2 as pc2
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"ROS2 + rosbag2_py required to read bags: {exc}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""))
    while reader.has_next():
        tname, data, ts = reader.read_next()
        if tname != topic:
            continue
        msg = deserialize_message(data, PointCloud2)
        pts = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True)], dtype=float)
        yield ts, pts


def analyze(bag_dir: str, topic: str, out_path: str) -> int:
    perc = config_loader.load_config("terrain_perception")
    cfg = hm.HeightMapConfig.from_dict(perc["heightmap"])
    stair_p = st.StairParams.from_dict(perc["stair_detector"])
    gap_p = gaps.GapParams.from_dict(perc["gap_detector"])
    cls_p = cl.ClassifierParams.from_dict(perc)
    slope_thresh = float(perc["slope_detector"]["slope_threshold_deg"])
    max_step = float(perc["stair_detector"]["max_rise_m"])

    frames = 0
    with JsonlRecorder(out_path) as rec:
        for ts, pts in _read_clouds(bag_dir, topic):
            grid = hm.HeightMap(cfg, decay=0.5)
            grid.update(pts, 1.0)
            h2d, c2d = grid.height_grid_2d(), grid.confidence_grid_2d()
            heights, conf, _ = grid.to_arrays()
            _, h2dm, c2dm, xpos = grid_msg.grid_from_flat(
                cfg.width, cfg.height, cfg.resolution_m, cfg.origin_x_m, cfg.origin_y_m,
                heights, conf)
            px, pz = st.forward_height_profile(h2dm, xpos)
            cprof = np.nanmean(c2dm[c2dm.shape[0] // 2 - 3: c2dm.shape[0] // 2 + 3, :],
                               axis=0)[: px.shape[0]]
            stair_res = st.detect_stairs(px, pz, stair_p)
            gap_res = gaps.detect_gap(px, pz, cprof, gap_p)

            ys, xs = np.nonzero(np.isfinite(h2dm))
            if xs.size >= 3:
                wpts = np.column_stack([
                    cfg.origin_x_m + (xs + 0.5) * cfg.resolution_m,
                    cfg.origin_y_m + (ys + 0.5) * cfg.resolution_m, h2dm[ys, xs]])
                plane = ground_plane.fit_plane_ransac(wpts, 0.03, seed=0)
                slope_deg = slope.slope_angle_deg(plane.normal)
                slope_dir = slope.slope_direction(plane.normal, slope_thresh)
                overall = float(np.clip(np.nanmean(cprof) if cprof.size else 0.0, 0, 1))
            else:
                slope_deg, slope_dir, overall = 0.0, "none", 0.0

            inp = cl.ClassifierInputs(
                overall_confidence=overall, slope_angle_deg=slope_deg,
                slope_direction=slope_dir, roughness_m=roughness.roughness_m(h2dm),
                single_step_height_m=roughness.max_single_step_m(pz)
                if not stair_res.stairs_detected else 0.0,
                max_obstacle_height_m=float(np.nanmax(h2dm)) if np.isfinite(h2dm).any() else 0.0,
                stairs=stair_res, gap=gap_res)
            out = cl.classify(inp, cls_p)
            rec.write({
                "stamp_ns": int(ts), "terrain_type": out.terrain_type,
                "confidence": out.confidence, "safe_to_continue": out.safe_to_continue,
                "reason": out.reason, "stairs_detected": stair_res.stairs_detected,
                "gap_detected": gap_res.gap_detected})
            log.info(f"[{frames}] {out.terrain_type} ({out.confidence:.2f}) — {out.reason}")
            frames += 1

    log.info(f"analyzed {frames} frames -> {out_path}")
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay a rosbag through the perception cores.")
    ap.add_argument("bag_dir")
    ap.add_argument("--topic", default="/x2/terrain/points_processed")
    ap.add_argument("--out", default="logs/perception/bag_analysis.jsonl")
    args = ap.parse_args(argv)
    n = analyze(args.bag_dir, args.topic, args.out)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
