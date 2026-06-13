"""terrain_classifier node (P1-M3-T4) — the perception integrator.

Subscribes to the height map and the stair estimate, runs the slope / gap / roughness /
single-step cores plus the classification decision logic, and publishes
x2_terrain_msgs/TerrainStatus. Perception only — it never sends a locomotion command; the
safe-locomotion package consumes this status.

Confidence and decision policy follow roadmap §5.4 and are unit-tested in core.classifier.
"""
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node

from x2_terrain_msgs.msg import TerrainGrid, StairEstimate, TerrainStatus
from x2_common import config_loader
from x2_common.qos_profiles import latched_qos

from .core import classifier as cl
from .core import gaps, slope, ground_plane, roughness, grid_msg
from .core.stairs import StairResult, forward_height_profile


class TerrainClassifier(Node):
    def __init__(self):
        super().__init__("terrain_classifier")
        perc = config_loader.load_config("terrain_perception")
        self._cls_params = cl.ClassifierParams.from_dict(perc)
        self._gap_params = gaps.GapParams.from_dict(perc["gap_detector"])
        self._slope_thresh = float(perc["slope_detector"]["slope_threshold_deg"])
        self._last_stair = StairResult(False, "none", 0.0, 0.0, 0.0, 0, 0.0, 0.0)

        self.create_subscription(
            TerrainGrid, "/x2/terrain/heightmap", self._on_grid, latched_qos())
        self.create_subscription(
            StairEstimate, "/x2/terrain/stair_estimate", self._on_stair, latched_qos())
        self.pub = self.create_publisher(
            TerrainStatus, "/x2/terrain/status", latched_qos())

    def _on_stair(self, msg: StairEstimate):
        self._last_stair = StairResult(
            msg.stairs_detected, msg.direction, msg.confidence, msg.rise_m, msg.tread_m,
            msg.visible_step_count, msg.first_step_distance_m, msg.recommended_stop_distance_m)

    def _on_grid(self, msg: TerrainGrid):
        cfg, h2d, c2d, xpos = grid_msg.grid_from_flat(
            msg.width, msg.height, msg.resolution_m, msg.origin_x_m, msg.origin_y_m,
            msg.height_m, msg.confidence)
        px, pz = forward_height_profile(h2d, xpos)
        # central-band confidence profile, aligned with pz's valid columns
        cprof = np.nanmean(c2d[c2d.shape[0] // 2 - 3: c2d.shape[0] // 2 + 3, :], axis=0)
        valid = ~np.isnan(np.nanmean(np.where(c2d > 0, h2d, np.nan), axis=0))
        cprof = cprof[valid] if cprof.shape[0] == valid.shape[0] else cprof[: px.shape[0]]

        gap_res = gaps.detect_gap(px, pz, cprof, self._gap_params)

        # Ground plane / slope from known cells reconstructed as a small point set.
        pts = self._grid_points(cfg, h2d)
        if pts.shape[0] >= 3:
            plane = ground_plane.fit_plane_ransac(pts, distance_thresh=0.03, seed=0)
            slope_deg = slope.slope_angle_deg(plane.normal)
            slope_dir = slope.slope_direction(plane.normal, self._slope_thresh)
            overall_conf = float(np.clip(np.nanmean(cprof) if cprof.size else 0.0, 0.0, 1.0))
        else:
            slope_deg, slope_dir, overall_conf = 0.0, "none", 0.0

        rough = roughness.roughness_m(h2d)
        # single step = a forward jump consistent with a curb but not a full staircase
        from .core.roughness import max_single_step_m
        step_h = max_single_step_m(pz) if not self._last_stair.stairs_detected else 0.0
        max_obs = float(np.nanmax(h2d)) if np.isfinite(h2d).any() else 0.0

        inp = cl.ClassifierInputs(
            overall_confidence=overall_conf, slope_angle_deg=slope_deg,
            slope_direction=slope_dir, roughness_m=rough, single_step_height_m=step_h,
            max_obstacle_height_m=max_obs, stairs=self._last_stair, gap=gap_res)
        out = cl.classify(inp, self._cls_params)

        status = TerrainStatus()
        status.header = msg.header
        status.terrain_type = out.terrain_type
        status.confidence = float(out.confidence)
        status.slope_angle_deg = float(out.slope_angle_deg)
        status.max_obstacle_height_m = float(out.max_obstacle_height_m)
        status.estimated_step_height_m = float(out.estimated_step_height_m)
        status.estimated_step_depth_m = float(out.estimated_step_depth_m)
        status.gap_width_m = float(out.gap_width_m)
        status.safe_to_continue = bool(out.safe_to_continue)
        status.reason = out.reason
        self.pub.publish(status)

    @staticmethod
    def _grid_points(cfg, h2d) -> np.ndarray:
        ys, xs = np.nonzero(np.isfinite(h2d))
        if xs.size == 0:
            return np.empty((0, 3))
        wx = cfg.origin_x_m + (xs + 0.5) * cfg.resolution_m
        wy = cfg.origin_y_m + (ys + 0.5) * cfg.resolution_m
        wz = h2d[ys, xs]
        return np.column_stack([wx, wy, wz])


def main(args=None):
    rclpy.init(args=args)
    node = TerrainClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
