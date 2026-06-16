"""gap_detector node (P1-M3-T2).

Standalone debug node: runs gap/drop-off detection on the height map and publishes a
boolean + reason string. The classifier consumes gaps internally; this node is for live
inspection. Fail safe: unknown regions ahead are reported unsafe. Perception only.
"""
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from x2_terrain_msgs.msg import TerrainGrid
from x2_common import config_loader
from x2_common.qos_profiles import latched_qos

from .core import gaps, grid_msg
from .core.stairs import forward_height_profile


class GapDetector(Node):
    def __init__(self):
        super().__init__("gap_detector")
        perc = config_loader.load_config("terrain_perception")
        self._params = gaps.GapParams.from_dict(perc["gap_detector"])
        self.create_subscription(
            TerrainGrid, "/x2/terrain/heightmap", self._on_grid, latched_qos())
        self.flag_pub = self.create_publisher(Bool, "/x2/terrain/gap_detected", 10)
        self.reason_pub = self.create_publisher(String, "/x2/terrain/gap_reason", 10)

    def _on_grid(self, msg: TerrainGrid):
        _, h2d, c2d, xpos = grid_msg.grid_from_flat(
            msg.width, msg.height, msg.resolution_m, msg.origin_x_m, msg.origin_y_m,
            msg.height_m, msg.confidence)
        px, pz = forward_height_profile(h2d, xpos)
        # Build confidence profile over the same central band with the same valid-column mask
        # that forward_height_profile applies internally (drops all-NaN columns).
        n_y = h2d.shape[0]
        c = n_y // 2
        band_h = h2d[max(0, c - 3): min(n_y, c + 3), :]
        band_c = c2d[max(0, c - 3): min(n_y, c + 3), :]
        with np.errstate(invalid="ignore"):
            valid_cols = ~np.all(np.isnan(band_h), axis=0)
        cprof = np.nanmean(band_c, axis=0)[valid_cols]
        res = gaps.detect_gap(px, pz, cprof, self._params)
        self.flag_pub.publish(Bool(data=bool(res.gap_detected or res.unknown_ahead)))
        self.reason_pub.publish(String(data=res.reason))


def main(args=None):
    rclpy.init(args=args)
    node = GapDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
