"""motion_state_monitor node (P2-M2-T1).

Tracks robot mode/state plus the freshness of command and perception inputs, and publishes a
simple boolean health summary the supervisor/operator can watch. Read-only — issues no
commands.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from x2_terrain_msgs.msg import TerrainStatus
from x2_common import config_loader
from x2_common.qos_profiles import latched_qos
from x2_common.safety_limits import FreshnessWatchdog


class MotionStateMonitor(Node):
    def __init__(self):
        super().__init__("motion_state_monitor")
        safe = config_loader.load_config("safe_locomotion")
        ss = safe["safety_supervisor"]
        self._terrain_wd = FreshnessWatchdog(float(ss["terrain_status_timeout_s"]))
        self._cmd_wd = FreshnessWatchdog(float(ss["command_timeout_s"]))
        self._t_terrain = None
        self._t_cmd = None

        self.create_subscription(TerrainStatus, "/x2/terrain/status", self._on_terrain, latched_qos())
        self.create_subscription(Bool, "/x2/safe_locomotion/command_heartbeat", self._on_cmd, 10)
        self.health_pub = self.create_publisher(Bool, "/x2/safe_locomotion/inputs_healthy", 10)
        self.detail_pub = self.create_publisher(String, "/x2/safe_locomotion/state_detail", 10)
        self.create_timer(0.1, self._tick)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_terrain(self, msg: TerrainStatus):
        self._t_terrain = self._now()

    def _on_cmd(self, msg: Bool):
        self._t_cmd = self._now()

    def _tick(self):
        now = self._now()
        terrain_ok = self._terrain_wd.is_fresh(self._t_terrain, now)
        cmd_ok = self._cmd_wd.is_fresh(self._t_cmd, now)
        healthy = terrain_ok and cmd_ok
        self.health_pub.publish(Bool(data=healthy))
        self.detail_pub.publish(String(data=f"terrain_fresh={terrain_ok} command_fresh={cmd_ok}"))


def main(args=None):
    rclpy.init(args=args)
    node = MotionStateMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
