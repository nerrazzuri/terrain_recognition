"""velocity_adapter node (P2-M1-T2) — the Phase 2 integrator.

Converts a desired forward velocity into a *safe* one: caps speed by terrain type, stops
before stairs/gaps/unknown, smooths the command, and zeros it when perception is stale or the
safety supervisor says stop. Publishes a per-cycle heartbeat the supervisor uses for command
freshness.

Defaults to **dry-run** (``configs/safe_locomotion.yaml: dry_run.enabled``) — it publishes
only to the debug topic and never to the real velocity topic until dry-run is disabled AND
the command source is registered. Velocity-level only; this node never climbs anything.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Bool

from x2_terrain_msgs.msg import TerrainStatus, SafetyDecision
from x2_common import config_loader
from x2_common.qos_profiles import latched_qos, command_qos
from x2_common.safety_limits import FreshnessWatchdog

from .core.velocity_policy import VelocityPolicy
from .core.smoother import CommandSmoother


class VelocityAdapter(Node):
    def __init__(self):
        super().__init__("velocity_adapter")
        safe = config_loader.load_config("safe_locomotion")
        topics = config_loader.load_config("robot_topics")
        self._policy = VelocityPolicy.from_dict(safe)
        cs = safe["command_smoother"]
        self._smoother = CommandSmoother(
            float(cs["max_forward_accel_mps2"]), float(cs["max_yaw_accel_radps2"]))
        self._dry_run = bool(safe["dry_run"]["enabled"])
        self._terrain_wd = FreshnessWatchdog(
            float(safe["safety_supervisor"]["terrain_status_timeout_s"]))
        self._safety_wd = FreshnessWatchdog(
            float(safe["safety_supervisor"]["terrain_status_timeout_s"]))
        self._period = 0.05

        self._desired_forward = 0.0
        self._terrain_type = "unknown_unsafe"
        self._safe_to_continue = False
        self._t_terrain = None
        self._t_safety = None
        self._supervisor_stop = True       # fail closed until we hear otherwise
        self._source_registered = False

        self.create_subscription(TerrainStatus, "/x2/terrain/status", self._on_terrain, latched_qos())
        self.create_subscription(SafetyDecision, "/x2/terrain/safety_decision",
                                 self._on_safety, latched_qos())
        self.create_subscription(Twist, "/x2/safe_locomotion/desired_velocity",
                                 self._on_desired, 10)
        self.create_subscription(Bool, "/x2/safe_locomotion/source_registered",
                                 self._on_source, latched_qos())

        self._debug_pub = self.create_publisher(
            TwistStamped, safe["dry_run"]["debug_topic"], command_qos())
        # Real publisher uses the AimDK McLocomotionVelocity message (SDK v0.9.0.4). Created
        # lazily only when going live so dry-run never needs aimdk_msgs.
        real_topic = config_loader.get(topics, "locomotion.velocity.name")
        self._real_topic = real_topic
        self._real_pub = None
        self._McLocomotionVelocity = None
        self._heartbeat = self.create_publisher(Bool, "/x2/safe_locomotion/command_heartbeat", 10)

        self.create_timer(self._period, self._tick)
        mode = "DRY-RUN (debug only)" if self._dry_run else "LIVE"
        self.get_logger().info(f"velocity_adapter started in {mode}; real topic={real_topic}")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_terrain(self, msg: TerrainStatus):
        self._t_terrain = self._now()
        self._terrain_type = msg.terrain_type
        self._safe_to_continue = msg.safe_to_continue

    def _on_safety(self, msg: SafetyDecision):
        self._t_safety = self._now()
        self._supervisor_stop = msg.stop

    def _on_desired(self, msg: Twist):
        self._desired_forward = msg.linear.x

    def _on_source(self, msg: Bool):
        self._source_registered = msg.data

    def _tick(self):
        self._heartbeat.publish(Bool(data=True))
        now = self._now()
        terrain_fresh = self._terrain_wd.is_fresh(self._t_terrain, now)
        safety_fresh = self._safety_wd.is_fresh(self._t_safety, now)

        # Fail closed: any missing critical input → immediate hard stop, not a ramp.
        hard_stop = not terrain_fresh or not safety_fresh or self._supervisor_stop
        if hard_stop:
            target_forward = 0.0
        else:
            target_forward = self._policy.safe_velocity(
                self._desired_forward, self._terrain_type, self._safe_to_continue)

        if hard_stop:
            cmd = self._smoother.emergency_stop()
        elif target_forward <= 0.0:
            cmd = self._smoother.step(0.0, 0.0, self._period)
        else:
            cmd = self._smoother.step(target_forward, 0.0, self._period)

        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.twist.linear.x = cmd.forward
        stamped.twist.angular.z = cmd.yaw
        self._debug_pub.publish(stamped)

        # Only ever touch the real robot when explicitly live AND source registered.
        if not self._dry_run and self._source_registered:
            self._publish_real(cmd)

    def _publish_real(self, cmd):
        """Publish the AimDK McLocomotionVelocity command. Created lazily on first live use."""
        if self._real_pub is None:
            try:
                from aimdk_msgs.msg import McLocomotionVelocity
            except Exception as exc:  # pragma: no cover - robot-only path
                self.get_logger().error(
                    f"aimdk_msgs unavailable — cannot publish live velocity: {exc}",
                    throttle_duration_sec=5.0)
                return
            self._McLocomotionVelocity = McLocomotionVelocity
            self._real_pub = self.create_publisher(
                McLocomotionVelocity, self._real_topic, command_qos())
        out = self._McLocomotionVelocity()
        out.forward_velocity = float(cmd.forward)
        out.lateral_velocity = 0.0
        out.angular_velocity = float(cmd.yaw)
        self._real_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
