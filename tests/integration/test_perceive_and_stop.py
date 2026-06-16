"""Integration test: the perceive-and-stop pipeline as a live ROS2 graph (roadmap §13 demo).

Builds the real nodes (heightmap -> stair_detector -> terrain_classifier -> safety_supervisor
-> velocity_adapter) with an rclpy executor, publishes a synthetic processed point cloud, and
asserts end-to-end behaviour:

  * a flat-ground cloud  -> terrain classified safe, adapter commands forward > 0 (dry-run);
  * a stairs-ahead cloud -> terrain classified unsafe, supervisor stops, adapter zeros velocity
                            with the stop reason coming from terrain perception.

Requires a built + sourced workspace (x2_terrain_msgs, rclpy); skipped otherwise so the unit
suite still runs without ROS2.
"""
import importlib.util
import os

import numpy as np
import pytest

_HAS_ROS = (importlib.util.find_spec("rclpy") is not None
            and importlib.util.find_spec("x2_terrain_msgs") is not None
            and importlib.util.find_spec("sensor_msgs_py") is not None)
pytestmark = pytest.mark.skipif(not _HAS_ROS, reason="ROS2 workspace not built/sourced")

# Ensure config discovery works when imported from the installed packages.
os.environ.setdefault(
    "X2_CONFIG_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "configs"))


def _make_cloud(scenario: str):
    """Synthetic base-frame point cloud over the ROI (x 0..2, y -0.4..0.4)."""
    pts = []
    for x in np.arange(0.05, 2.0, 0.02):
        for y in np.arange(-0.4, 0.4, 0.02):
            if scenario == "flat":
                z = 0.0
            elif scenario == "stairs":
                # flat run, then 0.15 m rise / 0.30 m tread staircase from x=0.9
                z = 0.0 if x < 0.9 else 0.15 * (int((x - 0.9) / 0.30) + 1)
            else:
                z = 0.0
            pts.append((float(x), float(y), float(z)))
    return np.array(pts, dtype=float)


def _run_scenario(scenario: str, seconds: float = 4.0):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, Imu
    from sensor_msgs_py import point_cloud2 as pc2
    from std_msgs.msg import Bool
    from geometry_msgs.msg import Twist, TwistStamped
    from x2_terrain_msgs.msg import TerrainStatus, SafetyDecision

    from x2_terrain_perception.heightmap_node import HeightmapNode
    from x2_terrain_perception.stair_detector import StairDetector
    from x2_terrain_perception.terrain_classifier import TerrainClassifier
    from x2_safe_locomotion.safety_supervisor import SafetySupervisor
    from x2_safe_locomotion.velocity_adapter import VelocityAdapter

    class Harness(Node):
        def __init__(self):
            super().__init__("test_harness")
            from x2_common.qos_profiles import sensor_qos, latched_qos
            self.cloud_pub = self.create_publisher(
                PointCloud2, "/x2/terrain/points_processed", sensor_qos())
            self.imu_pub = self.create_publisher(Imu, "/aima/hal/imu/chest/state", sensor_qos())
            self.estop_pub = self.create_publisher(Bool, "/x2/operator/estop", latched_qos())
            self.src_pub = self.create_publisher(
                Bool, "/x2/safe_locomotion/source_registered", latched_qos())
            self.desired_pub = self.create_publisher(
                Twist, "/x2/safe_locomotion/desired_velocity", 10)
            self.status = None
            self.decision = None
            self.cmd = None
            self.create_subscription(TerrainStatus, "/x2/terrain/status",
                                     lambda m: setattr(self, "status", m), latched_qos())
            self.create_subscription(SafetyDecision, "/x2/terrain/safety_decision",
                                     lambda m: setattr(self, "decision", m), latched_qos())
            self.create_subscription(
                TwistStamped, "/x2/safe_locomotion/debug",
                lambda m: setattr(self, "cmd", m), 10)
            self._pts = _make_cloud(scenario)
            self.create_timer(0.1, self._tick)

        def _tick(self):
            now = self.get_clock().now().to_msg()
            header_frame = "base_link"
            from std_msgs.msg import Header
            h = Header(stamp=now, frame_id=header_frame)
            self.cloud_pub.publish(pc2.create_cloud_xyz32(h, self._pts.tolist()))
            imu = Imu()
            imu.header.stamp = now
            imu.orientation.w = 1.0     # upright
            self.imu_pub.publish(imu)
            self.estop_pub.publish(Bool(data=False))
            self.src_pub.publish(Bool(data=False))     # dry-run: source not registered
            d = Twist()
            d.linear.x = 0.10                          # operator wants to walk forward
            self.desired_pub.publish(d)

    rclpy.init()
    nodes = [HeightmapNode(), StairDetector(), TerrainClassifier(),
             SafetySupervisor(), VelocityAdapter()]
    harness = Harness()
    ex = SingleThreadedExecutor()
    for n in nodes + [harness]:
        ex.add_node(n)
    try:
        end = harness.get_clock().now().nanoseconds + int(seconds * 1e9)
        while harness.get_clock().now().nanoseconds < end:
            ex.spin_once(timeout_sec=0.05)
        return harness.status, harness.decision, harness.cmd
    finally:
        for n in nodes + [harness]:
            n.destroy_node()
        rclpy.shutdown()


def test_flat_ground_is_safe_and_moves():
    status, decision, cmd = _run_scenario("flat")
    assert status is not None, "no terrain status published"
    assert status.terrain_type in ("flat_ground", "rough_ground"), status.terrain_type
    assert status.safe_to_continue is True
    assert decision is not None and decision.stop is False, getattr(decision, "reason", None)
    assert cmd is not None and cmd.twist.linear.x > 0.0   # walking forward (dry-run)


def test_stairs_ahead_classified_unsafe_and_stops():
    status, decision, cmd = _run_scenario("stairs")
    assert status is not None, "no terrain status published"
    # stairs must NOT be considered safe to continue
    assert status.safe_to_continue is False, status.terrain_type
    assert status.terrain_type in ("stairs_up", "curb_or_single_step",
                                   "unknown_unsafe", "gap_or_hole"), status.terrain_type
    assert decision is not None and decision.stop is True
    # the stop is terrain-driven (IMU/command were kept fresh), and velocity is zero
    assert cmd is not None and abs(cmd.twist.linear.x) < 1e-6
