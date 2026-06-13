"""visualization_node (P1-M4-T2 companion).

Subscribes to the height map and republishes it as a nav_msgs/OccupancyGrid so it can be
viewed live in RViz (occupancy = normalised height; unknown cells = -1). Perception only.
"""
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

from x2_terrain_msgs.msg import TerrainGrid
from x2_common.qos_profiles import latched_qos


class VisualizationNode(Node):
    def __init__(self):
        super().__init__("visualization_node")
        self.create_subscription(
            TerrainGrid, "/x2/terrain/heightmap", self._on_grid, latched_qos())
        self.pub = self.create_publisher(
            OccupancyGrid, "/x2/terrain/heightmap_viz", latched_qos())

    def _on_grid(self, msg: TerrainGrid):
        heights = np.asarray(msg.height_m, dtype=float)
        conf = np.asarray(msg.confidence, dtype=float)
        known = conf > 0.0
        occ = np.full(heights.shape, -1, dtype=np.int8)  # unknown -> -1
        if known.any():
            hk = heights[known]
            lo, hi = float(hk.min()), float(hk.max())
            span = max(1e-6, hi - lo)
            scaled = np.clip((heights - lo) / span * 100.0, 0, 100).astype(np.int8)
            occ[known] = scaled[known]
        grid = OccupancyGrid()
        grid.header = msg.header
        grid.info.resolution = msg.resolution_m
        grid.info.width = msg.width
        grid.info.height = msg.height
        grid.info.origin.position.x = msg.origin_x_m
        grid.info.origin.position.y = msg.origin_y_m
        grid.data = occ.tolist()
        self.pub.publish(grid)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
