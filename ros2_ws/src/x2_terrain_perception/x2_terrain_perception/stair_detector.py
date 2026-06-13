"""stair_detector node (P1-M3-T1).

Subscribes to the height map, extracts a forward profile, runs the tested stair-detection
core, and publishes x2_terrain_msgs/StairEstimate. Perception only.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node

from x2_terrain_msgs.msg import TerrainGrid, StairEstimate
from x2_common import config_loader
from x2_common.qos_profiles import latched_qos

from .core import stairs as st
from .core import grid_msg


class StairDetector(Node):
    def __init__(self):
        super().__init__("stair_detector")
        perc = config_loader.load_config("terrain_perception")
        self._params = st.StairParams.from_dict(perc["stair_detector"])
        self.sub = self.create_subscription(
            TerrainGrid, "/x2/terrain/heightmap", self._on_grid, latched_qos())
        self.pub = self.create_publisher(
            StairEstimate, "/x2/terrain/stair_estimate", latched_qos())

    def _on_grid(self, msg: TerrainGrid):
        _, h2d, _, xpos = grid_msg.grid_from_flat(
            msg.width, msg.height, msg.resolution_m, msg.origin_x_m, msg.origin_y_m,
            msg.height_m, msg.confidence)
        px, pz = st.forward_height_profile(h2d, xpos)
        res = st.detect_stairs(px, pz, self._params)
        out = StairEstimate()
        out.header = msg.header
        out.stairs_detected = res.stairs_detected
        out.direction = res.direction
        out.confidence = float(res.confidence)
        out.rise_m = float(res.rise_m)
        out.tread_m = float(res.tread_m)
        out.visible_step_count = int(res.visible_step_count)
        out.first_step_distance_m = float(res.first_step_distance_m)
        out.recommended_stop_distance_m = float(res.recommended_stop_distance_m)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = StairDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
