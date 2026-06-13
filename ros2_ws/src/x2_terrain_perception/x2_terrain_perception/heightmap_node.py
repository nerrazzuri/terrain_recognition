"""heightmap_node (P1-M2-T3).

Subscribes to the processed point cloud, accumulates a robot-centred elevation map with
time decay (core.heightmap.HeightMap), fills per-cell traversability (core.traversability),
and publishes x2_terrain_msgs/TerrainGrid. Perception only — no motion.

The map maths is fully unit-tested in core.heightmap; this node is the ROS2 wrapper.
"""
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from x2_terrain_msgs.msg import TerrainGrid
from x2_terrain_msgs.srv import ResetTerrainMap
from x2_common import config_loader
from x2_common.qos_profiles import sensor_qos, latched_qos

from .core import heightmap as hm
from .core import traversability as tv

try:
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:  # pragma: no cover
    pc2 = None


class HeightmapNode(Node):
    def __init__(self):
        super().__init__("heightmap_node")
        perc = config_loader.load_config("terrain_perception")
        self._cfg = hm.HeightMapConfig.from_dict(perc["heightmap"])
        self._map = hm.HeightMap(self._cfg, decay=0.5)
        self._max_step = float(perc["stair_detector"]["max_rise_m"])
        rate = float(perc["heightmap"]["update_rate_hz"])

        self.sub = self.create_subscription(
            PointCloud2, "/x2/terrain/points_processed", self._on_cloud, sensor_qos())
        self.pub = self.create_publisher(TerrainGrid, "/x2/terrain/heightmap", latched_qos())
        self.create_service(ResetTerrainMap, "/x2/terrain/reset_map", self._on_reset)
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info(
            f"heightmap_node: {self._cfg.width}x{self._cfg.height} @ {self._cfg.resolution_m} m")

    def _on_cloud(self, msg: PointCloud2):
        if pc2 is None:
            return
        pts = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True)], dtype=float)
        if pts.size:
            self._map.update(pts, measurement_confidence=1.0)

    def _publish(self):
        heights, conf, _ = self._map.to_arrays()
        trav = tv.estimate(
            self._map.height_grid_2d(), self._map.confidence_grid_2d(),
            self._cfg.resolution_m, self._max_step).reshape(-1)
        msg = TerrainGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.resolution_m = self._cfg.resolution_m
        msg.width = self._cfg.width
        msg.height = self._cfg.height
        msg.origin_x_m = self._cfg.origin_x_m
        msg.origin_y_m = self._cfg.origin_y_m
        msg.height_m = heights.tolist()
        msg.confidence = conf.tolist()
        msg.traversability = trav.astype(np.uint8).tolist()
        self.pub.publish(msg)

    def _on_reset(self, request, response):
        if request.clear_history:
            self._map = hm.HeightMap(self._cfg, decay=0.5)
            response.success = True
            response.message = "terrain map cleared"
        else:
            response.success = False
            response.message = "no-op (clear_history was false)"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = HeightmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
