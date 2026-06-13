"""input_source_registrar (P2-M1-T1).

Registers the locomotion command source ``x2_terrain_safe_locomotion`` with AimDK before any
velocity is published, and verifies/loggs priority. Fails closed: if registration fails and
``fail_closed_on_register_error`` is set, the node reports failure so the velocity adapter
will not start publishing.

The concrete AimDK registration call depends on the installed SDK and is intentionally
isolated in ``register_source`` — wire it to the real API and verify against the robot
(AGENTS.md §3). Until then it returns False (fail closed), never a false success.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from x2_common import config_loader
from x2_common.qos_profiles import latched_qos


def register_source(name: str) -> tuple[bool, str]:
    """Attempt to register the command source with AimDK.

    Returns ``(success, info)``. Replace the body with the real AimDK call. Fail closed:
    any error (including the SDK being absent) returns ``False``.
    """
    try:
        import aimdk  # noqa: F401  (real SDK; not present in this dev env)
    except Exception as exc:  # pragma: no cover - depends on robot SDK
        return False, f"AimDK not available / source registration not wired: {exc}"
    # TODO: call the real AimDK source-registration API here and return its result.
    return False, "register_source not implemented for installed SDK"


class InputSourceRegistrar(Node):
    def __init__(self):
        super().__init__("input_source_registrar")
        safe = config_loader.load_config("safe_locomotion")
        self._name = safe["command_source"]["name"]
        self._fail_closed = bool(safe["command_source"]["fail_closed_on_register_error"])

        self.status_pub = self.create_publisher(
            Bool, "/x2/safe_locomotion/source_registered", latched_qos())
        self.info_pub = self.create_publisher(
            String, "/x2/safe_locomotion/source_info", latched_qos())

        ok, info = register_source(self._name)
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
