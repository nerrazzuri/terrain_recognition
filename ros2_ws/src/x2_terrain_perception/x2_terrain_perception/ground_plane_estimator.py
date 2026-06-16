"""ground_plane_estimator node (P1-M2-T2).

Standalone debug node: fits a ground plane to the processed cloud and publishes the plane
normal + slope angle on debug topics. The classifier also computes this internally; this node
is for live inspection / tuning. Perception only.
"""
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32

from x2_common import config_loader
from x2_common.qos_profiles import sensor_qos

from .core import ground_plane, slope

try:
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:  # pragma: no cover
    pc2 = None


class GroundPlaneEstimator(Node):
    def __init__(self):
        super().__init__("ground_plane_estimator")
        perc = config_loader.load_config("terrain_perception")
        gcfg = perc["ground_plane_estimator"]
        self._dist = float(gcfg["ransac_distance_thresh_m"])
        self._iters = int(gcfg["ransac_max_iterations"])
        self.create_subscription(
            PointCloud2, "/x2/terrain/points_processed", self._on_cloud, sensor_qos())
        self.normal_pub = self.create_publisher(Vector3Stamped, "/x2/terrain/ground_normal", 10)
        self.slope_pub = self.create_publisher(Float32, "/x2/terrain/slope_deg", 10)

    def _on_cloud(self, msg: PointCloud2):
        if pc2 is None:
            return
        pts = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True)], dtype=float)
        if pts.shape[0] < 3:
            return
        fit = ground_plane.fit_plane_ransac(pts, self._dist, self._iters, seed=0)
        v = Vector3Stamped()
        v.header = msg.header
        v.vector.x, v.vector.y, v.vector.z = map(float, fit.normal)
        self.normal_pub.publish(v)
        self.slope_pub.publish(Float32(data=float(slope.slope_angle_deg(fit.normal))))


def main(args=None):
    rclpy.init(args=args)
    node = GroundPlaneEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
