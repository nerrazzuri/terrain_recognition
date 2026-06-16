"""slope_detector node (P1-M2-T4).

Standalone debug node: derives slope angle + up/down direction from the height map and
publishes them on debug topics. The classifier consumes slope internally; this node is for
live inspection. Perception only.
"""
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

from x2_terrain_msgs.msg import TerrainGrid
from x2_common import config_loader
from x2_common.qos_profiles import latched_qos

from .core import ground_plane, slope, grid_msg


class SlopeDetector(Node):
    def __init__(self):
        super().__init__("slope_detector")
        perc = config_loader.load_config("terrain_perception")
        self._thresh = float(perc["slope_detector"]["slope_threshold_deg"])
        self.create_subscription(
            TerrainGrid, "/x2/terrain/heightmap", self._on_grid, latched_qos())
        self.angle_pub = self.create_publisher(Float32, "/x2/terrain/slope_angle_deg", 10)
        self.dir_pub = self.create_publisher(String, "/x2/terrain/slope_direction", 10)

    def _on_grid(self, msg: TerrainGrid):
        cfg, h2d, _, _ = grid_msg.grid_from_flat(
            msg.width, msg.height, msg.resolution_m, msg.origin_x_m, msg.origin_y_m,
            msg.height_m, msg.confidence)
        ys, xs = np.nonzero(np.isfinite(h2d))
        if xs.size < 3:
            return
        wx = cfg.origin_x_m + (xs + 0.5) * cfg.resolution_m
        wy = cfg.origin_y_m + (ys + 0.5) * cfg.resolution_m
        pts = np.column_stack([wx, wy, h2d[ys, xs]])
        fit = ground_plane.fit_plane_ransac(pts, 0.03, seed=0)
        self.angle_pub.publish(Float32(data=float(slope.slope_angle_deg(fit.normal))))
        self.dir_pub.publish(String(data=slope.slope_direction(fit.normal, self._thresh)))


def main(args=None):
    rclpy.init(args=args)
    node = SlopeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
