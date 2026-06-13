"""emergency_stop_node (P2-M2-T3).

Operator manual emergency stop — highest priority. Latches an e-stop boolean on
``/x2/operator/estop`` that the supervisor, adapter and smoother all obey (immediate zero).
Once engaged it stays engaged until an explicit reset, so a transient input cannot silently
re-enable motion.

Inputs:
- ``/x2/operator/estop_trigger`` (Bool true) — engage the e-stop.
- ``/x2/operator/estop_reset`` (Bool true) — clear it (deliberate operator action).
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from x2_common.qos_profiles import latched_qos


class EmergencyStopNode(Node):
    def __init__(self):
        super().__init__("emergency_stop_node")
        self._engaged = False
        self.create_subscription(Bool, "/x2/operator/estop_trigger", self._on_trigger, latched_qos())
        self.create_subscription(Bool, "/x2/operator/estop_reset", self._on_reset, latched_qos())
        self.pub = self.create_publisher(Bool, "/x2/operator/estop", latched_qos())
        self.create_timer(0.05, self._tick)  # republish so late subscribers always see state
        self._publish()

    def _on_trigger(self, msg: Bool):
        if msg.data and not self._engaged:
            self._engaged = True
            self.get_logger().error("EMERGENCY STOP ENGAGED")
            self._publish()

    def _on_reset(self, msg: Bool):
        if msg.data and self._engaged:
            self._engaged = False
            self.get_logger().warn("emergency stop cleared by operator")
            self._publish()

    def _publish(self):
        self.pub.publish(Bool(data=self._engaged))

    def _tick(self):
        self._publish()


def main(args=None):
    rclpy.init(args=args)
    node = EmergencyStopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
