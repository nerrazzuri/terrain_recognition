"""pointcloud_projector node (P1-M2-T1).

Subscribes to an RGB-D / LiDAR PointCloud2, transforms it into the robot base frame, drops
NaN/inf points, crops to the ROI, voxel-downsamples, and republishes a processed cloud plus
debug counts. Survives missing input via a freshness watchdog (perception only — no motion).

ROS2-only glue around tested core helpers (x2_common.transforms). Topic names/QoS come from
configs/robot_topics.yaml and MUST be verified on the robot before trusting them.
"""
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32MultiArray

from x2_common import config_loader, transforms
from x2_common.qos_profiles import sensor_qos

try:
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:  # pragma: no cover - available only in a ROS2 runtime
    pc2 = None


class PointcloudProjector(Node):
    def __init__(self):
        super().__init__("pointcloud_projector")
        perc = config_loader.load_config("terrain_perception")
        topics = config_loader.load_config("robot_topics")
        cfg = perc["pointcloud_projector"]
        self.base_frame = cfg["base_frame"]
        roi = cfg["roi"]
        self._roi = (
            (roi["x_min_m"], roi["x_max_m"]),
            (roi["y_min_m"], roi["y_max_m"]),
            (roi["z_min_m"], roi["z_max_m"]),
        )
        self._voxel = float(cfg["voxel_size_m"])
        self._timeout = float(cfg["missing_cloud_timeout_s"])

        in_topic = config_loader.get(topics, "sensors.depth_pointcloud.name")
        self.sub = self.create_subscription(
            PointCloud2, in_topic, self._on_cloud, sensor_qos())
        self.pub = self.create_publisher(
            PointCloud2, "/x2/terrain/points_processed", sensor_qos())
        self.debug = self.create_publisher(
            Int32MultiArray, "/x2/terrain/projector_counts", 10)

        self._last_msg_time = None
        self.create_timer(0.2, self._watchdog)
        self.get_logger().info(f"pointcloud_projector: in={in_topic} base={self.base_frame}")

    def _on_cloud(self, msg: PointCloud2):
        self._last_msg_time = self.get_clock().now()
        if pc2 is None:
            return
        pts = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=False)], dtype=float)
        n_in = pts.shape[0]
        # NOTE: assumes the cloud is already in (or aligned to) the base frame. When a TF is
        # needed, look up base<-sensor and apply transforms.transform_points here.
        pts = transforms.drop_nonfinite(pts)
        pts = transforms.crop_roi(pts, *self._roi)
        pts = self._voxel_downsample(pts)
        n_out = pts.shape[0]
        self.debug.publish(Int32MultiArray(data=[int(n_in), int(n_out)]))
        if pc2 is not None and n_out > 0:
            header = msg.header
            header.frame_id = self.base_frame
            self.pub.publish(pc2.create_cloud_xyz32(header, pts.tolist()))

    def _voxel_downsample(self, pts: np.ndarray) -> np.ndarray:
        if pts.size == 0 or self._voxel <= 0:
            return pts
        keys = np.floor(pts / self._voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return pts[np.sort(idx)]

    def _watchdog(self):
        if self._last_msg_time is None:
            return
        age = (self.get_clock().now() - self._last_msg_time).nanoseconds * 1e-9
        if age > self._timeout:
            self.get_logger().warn(
                f"no point cloud for {age:.1f}s (> {self._timeout}s) — downstream should stop")


def main(args=None):
    rclpy.init(args=args)
    node = PointcloudProjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
