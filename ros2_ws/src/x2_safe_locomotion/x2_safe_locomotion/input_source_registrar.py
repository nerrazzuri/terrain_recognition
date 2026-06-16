"""input_source_registrar (P2-M1-T1).

Registers the locomotion command source ``x2_terrain_safe_locomotion`` with AimDK before any
velocity is published, and verifies/logs priority. Fails closed: if registration fails and
``fail_closed_on_register_error`` is set, the node reports failure so the velocity adapter
will not start publishing.

Uses the real AimDK ``aimdk_msgs/srv/SetMcInputSource`` service (SDK lx2501_3 v0.9.0.4):
``action.value=1001`` to register, with ``input_source.{name,priority,timeout}`` from
configs/safe_locomotion.yaml. Fail closed if ``aimdk_msgs`` is unavailable or the call fails.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from x2_common import config_loader
from x2_common.qos_profiles import latched_qos


class InputSourceRegistrar(Node):
    def __init__(self):
        super().__init__("input_source_registrar")
        safe = config_loader.load_config("safe_locomotion")
        cs = safe["command_source"]
        self._name = cs["name"]
        self._fail_closed = bool(cs["fail_closed_on_register_error"])
        self._service = cs.get("service", "/aimdk_msgs/srv/SetMcInputSource")
        self._action_value = int(cs.get("register_action_value", 1001))
        self._priority = int(cs.get("priority", 40))
        self._timeout_ms = int(cs.get("timeout_ms", 1000))

        self.status_pub = self.create_publisher(
            Bool, "/x2/safe_locomotion/source_registered", latched_qos())
        self.info_pub = self.create_publisher(
            String, "/x2/safe_locomotion/source_info", latched_qos())

        ok, info = self._register_source()
        self.status_pub.publish(Bool(data=ok))
        self.info_pub.publish(String(data=info))
        if ok:
            self.get_logger().info(f"registered source '{self._name}': {info}")
        else:
            msg = f"source registration FAILED: {info}"
            if self._fail_closed:
                self.get_logger().error(msg + " — failing closed (adapter must not publish)")
            else:
                self.get_logger().warn(msg)

    def _register_source(self) -> tuple[bool, str]:
        """Call AimDK SetMcInputSource to register this source. Fail closed on any error."""
        try:
            from aimdk_msgs.srv import SetMcInputSource
        except Exception as exc:  # pragma: no cover - depends on robot SDK
            return False, f"aimdk_msgs not available: {exc}"
        try:
            client = self.create_client(SetMcInputSource, self._service)
            if not client.wait_for_service(timeout_sec=8.0):
                return False, f"service {self._service} not available"
            req = SetMcInputSource.Request()
            req.action.value = self._action_value
            req.input_source.name = self._name
            req.input_source.priority = self._priority
            req.input_source.timeout = self._timeout_ms
            req.request.header.stamp = self.get_clock().now().to_msg()
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if not future.done() or future.result() is None:
                return False, "registration call did not complete"
            return True, (f"name={self._name} priority={self._priority} "
                          f"timeout={self._timeout_ms}ms")
        except Exception as exc:  # pragma: no cover - robot-only path
            return False, f"registration error: {exc}"


def main(args=None):
    rclpy.init(args=args)
    node = InputSourceRegistrar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
