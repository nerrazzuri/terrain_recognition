#!/usr/bin/env python3
"""visualize_heightmap.py (P1-M4-T2).

Live matplotlib view of /x2/terrain/heightmap (TerrainGrid). Unknown cells (confidence 0)
are shown masked so they are visibly distinct from flat ground. Read-only / perception only.

Usage:
    python tools/visualize_heightmap.py            # live, needs ROS2 + the perception stack
    python tools/visualize_heightmap.py --demo     # synthetic demo, no ROS2 required
"""
from __future__ import annotations

import argparse
import sys


def _show(height_2d, conf_2d, resolution_m, block):
    import numpy as np
    import matplotlib.pyplot as plt

    masked = np.ma.masked_where(conf_2d <= 0.0, height_2d)
    plt.clf()
    plt.imshow(masked, origin="lower", aspect="equal",
               extent=[0, height_2d.shape[1] * resolution_m,
                       -height_2d.shape[0] * resolution_m / 2,
                       height_2d.shape[0] * resolution_m / 2])
    plt.colorbar(label="height (m)")
    plt.title("X2 height map (masked = unknown)")
    plt.xlabel("forward x (m)")
    plt.ylabel("lateral y (m)")
    plt.pause(0.001)
    if block:
        plt.show()


def demo():
    import numpy as np
    h = np.zeros((40, 50))
    h[:, 30:] = 0.15            # a step at x ~ 1.2 m
    conf = np.ones((40, 50))
    conf[:, 45:] = 0.0          # an unknown band
    _show(h, conf, 0.04, block=True)


def live():
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from x2_terrain_msgs.msg import TerrainGrid
    import matplotlib.pyplot as plt

    class Viz(Node):
        def __init__(self):
            super().__init__("visualize_heightmap")
            self.create_subscription(TerrainGrid, "/x2/terrain/heightmap", self._cb, 1)

        def _cb(self, msg):
            h = np.asarray(msg.height_m, float).reshape(msg.height, msg.width)
            c = np.asarray(msg.confidence, float).reshape(msg.height, msg.width)
            _show(h, c, msg.resolution_m, block=False)

    rclpy.init()
    node = Viz()
    plt.ion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic demo, no ROS2")
    args = ap.parse_args(argv)
    if args.demo:
        demo()
    else:
        live()
    return 0


if __name__ == "__main__":
    sys.exit(main())
