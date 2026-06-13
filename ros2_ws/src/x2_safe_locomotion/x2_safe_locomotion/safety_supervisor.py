"""safety_supervisor node (P2-M2-T2).

Monitors terrain-status freshness, IMU freshness, command freshness, body tilt, terrain
type, operator e-stop and robot mode, and publishes a SafetyDecision. The velocity adapter
must obey ``stop``. Fail closed: any missing critical input → stop (decision logic is the
unit-tested core.supervisor). Logs the stop reason.
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from x2_terrain_msgs.msg import TerrainStatus, SafetyDecision
from x2_common import config_loader
from x2_common.qos_profiles import sensor_qos, latched_qos
from x2_common.safety_limits import FreshnessWatchdog

from .core import supervisor as sup


class SafetySupervisor(Node):
    def __init__(self):
        super().__init__("safety_supervisor")
        safe = config_loader.load_config("safe_locomotion")
        ss = safe["safety_supervisor"]
        self._max_roll = float(ss["max_roll_deg"])
        self._max_pitch = float(ss["max_pitch_deg"])
        self._terrain_wd = FreshnessWatchdog(float(ss["terrain_status_timeout_s"]))
        self._imu_wd = FreshnessWatchdog(float(ss["imu_timeout_s"]))
        self._cmd_wd = FreshnessWatchdog(float(ss["command_timeout_s"]))

        self._t_terrain = None
        self._t_imu = None
        self._t_cmd = None
        self._terrain_type = "unknown_unsafe"
        self._safe_to_continue = False
        self._roll = 0.0
        self._pitch = 0.0
        self._estop = False

        self.create_subscription(TerrainStatus, "/x2/terrain/status", self._on_terrain, latched_qos())
        self.create_subscription(Imu, "/aima/hal/imu/chest/state", self._on_imu, sensor_qos())
        self.create_subscription(Bool, "/x2/operator/estop", self._on_estop, latched_qos())
        self.create_subscription(Bool, "/x2/safe_locomotion/command_heartbeat",
                                 self._on_cmd, 10)
        self.pub = self.create_publisher(
            SafetyDecision, "/x2/terrain/safety_decision", latched_qos())
        self.create_timer(0.05, self._tick)  # 20 Hz

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_terrain(self, msg: TerrainStatus):
        self._t_terrain = self._now()
        self._terrain_type = msg.terrain_type
        self._safe_to_continue = msg.safe_to_continue

    def _on_imu(self, msg: Imu):
        self._t_imu = self._now()
        q = msg.orientation
        # roll/pitch from quaternion
        sinr = 2 * (q.w * q.x + q.y * q.z)
        cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
        self._roll = math.degrees(math.atan2(sinr, cosr))
        sinp = 2 * (q.w * q.y - q.z * q.x)
        self._pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))

    def _on_estop(self, msg: Bool):
        self._estop = msg.data

    def _on_cmd(self, msg: Bool):
        self._t_cmd = self._now()

    def _tick(self):
        now = self._now()
        state = sup.SupervisorState(
            terrain_fresh=self._terrain_wd.is_fresh(self._t_terrain, now),
            imu_fresh=self._imu_wd.is_fresh(self._t_imu, now),
            command_fresh=self._cmd_wd.is_fresh(self._t_cmd, now),
            roll_deg=self._roll, pitch_deg=self._pitch,
            max_roll_deg=self._max_roll, max_pitch_deg=self._max_pitch,
            terrain_type=self._terrain_type, safe_to_continue=self._safe_to_continue,
            operator_estop=self._estop, robot_mode_ok=True)
        stop, reason = sup.evaluate_stop(state)

        out = SafetyDecision()
        out.header.stamp = self.get_clock().now().to_msg()
        out.stop = bool(stop)
        out.reason = reason
        out.max_forward_velocity = 0.0
        out.perception_fresh = state.terrain_fresh
        out.imu_fresh = state.imu_fresh
        out.operator_estop = state.operator_estop
        self.pub.publish(out)
        if stop:
            self.get_logger().warn(f"STOP: {reason}", throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = SafetySupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
