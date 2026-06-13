"""command_smoother node (P2-M1-T3).

Standalone smoother: ramps a raw velocity command within acceleration limits and republishes
it. The velocity_adapter already smooths internally; this node is available for pipelines that
keep smoothing as a separate stage. Emergency-stop input forces an immediate zero.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from x2_common import config_loader
from x2_common.qos_profiles import command_qos

from .core.smoother import CommandSmoother


class CommandSmootherNode(Node):
    def __init__(self):
        super().__init__("command_smoother")
        safe = config_loader.load_config("safe_locomotion")
        cs = safe["command_smoother"]
        self._smoother = CommandSmoother(
            float(cs["max_forward_accel_mps2"]), float(cs["max_yaw_accel_radps2"]))
        self._period = 0.05
        self._target = (0.0, 0.0)
        self._estop = False
        self.create_subscription(Twist, "/x2/safe_locomotion/raw_velocity", self._on_raw, 10)
        self.create_subscription(Bool, "/x2/operator/estop", self._on_estop, command_qos())
        self.pub = self.create_publisher(
            Twist, "/x2/safe_locomotion/smoothed_velocity", command_qos())
        self.create_timer(self._period, self._tick)

    def _on_raw(self, msg: Twist):
        self._target = (msg.linear.x, msg.angular.z)

    def _on_estop(self, msg: Bool):
        self._estop = msg.data

    def _tick(self):
        if self._estop:
            cmd = self._smoother.emergency_stop()
        else:
            cmd = self._smoother.step(self._target[0], self._target[1], self._period)
        out = Twist()
        out.linear.x = cmd.forward
        out.angular.z = cmd.yaw
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CommandSmootherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
