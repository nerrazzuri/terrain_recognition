"""traversability_estimator node (P1-M3-T3).

Standalone debug node: recomputes the per-cell traversability grid from the height map and
republishes it as a UInt8MultiArray for inspection. heightmap_node already fills the
TerrainGrid.traversability field for normal operation; this node is for tuning/debug.
"""
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

from x2_terrain_msgs.msg import TerrainGrid
from x2_common import config_loader
from x2_common.qos_profiles import latched_qos

from .core import traversability as tv, grid_msg


class TraversabilityEstimator(Node):
    def __init__(self):
        super().__init__("traversability_estimator")
        perc = config_loader.load_config("terrain_perception")
        self._max_step = float(perc["stair_detector"]["max_rise_m"])
        self.create_subscription(
            TerrainGrid, "/x2/terrain/heightmap", self._on_grid, latched_qos())
        self.pub = self.create_publisher(
            UInt8MultiArray, "/x2/terrain/traversability", latched_qos())

    def _on_grid(self, msg: TerrainGrid):
        cfg, h2d, c2d, _ = grid_msg.grid_from_flat(
            msg.width, msg.height, msg.resolution_m, msg.origin_x_m, msg.origin_y_m,
            msg.height_m, msg.confidence)
        trav = tv.estimate(h2d, c2d, cfg.resolution_m, self._max_step)
        self.pub.publish(UInt8MultiArray(data=trav.reshape(-1).astype(np.uint8).tolist()))


def main(args=None):
    rclpy.init(args=args)
    node = TraversabilityEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
