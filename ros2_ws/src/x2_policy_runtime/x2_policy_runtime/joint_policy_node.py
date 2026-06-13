"""joint_policy_node — the Phase 5/6 policy integrator. GATED.

Pipeline: observation_builder -> onnx_policy_runner -> action_filter -> policy_safety_supervisor.

SAFETY GATE (SAFETY.md / roadmap §14): while ``REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED`` is
false (the default), this node runs **dry-run only** — it publishes a PolicyDebug message
showing what it *would* command and NEVER publishes a leg joint command. The real leg-command
publisher is only created when the flag is true. Even then, the safety supervisor cuts output
on any unsafe condition and missing observations safe-stop.

The numeric core (observation builder, action filter, supervisor) is all unit-tested.
"""
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node

from x2_terrain_msgs.msg import PolicyDebug
from x2_common import config_loader
from x2_common.qos_profiles import command_qos

from .core import observation_builder as ob
from .core import action_filter as af
from .core import policy_supervisor as ps

_LEG_COMMAND_TOPIC = "/aima/hal/joint/leg/command"  # FORBIDDEN while flag is false


class JointPolicyNode(Node):
    def __init__(self):
        super().__init__("joint_policy_node")
        safety = config_loader.load_config("safety_limits")
        joint_limits = config_loader.load_config("joint_limits_x2_ultra")
        from x2_locomotion.robots.x2_joint_map import aimdk_leg_order  # type: ignore
        order = aimdk_leg_order()

        self._approved = bool(safety["REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED"])
        self._filter = af.ActionFilter.from_configs(joint_limits, safety, order)
        self._max_tilt = float(np.radians(safety["orientation_limits_deg"]["max_pitch"]))
        self._normalizer = ob.Normalizer.identity()   # load real stats from training artifact
        self._prev_action = np.zeros(len(order))

        self.debug_pub = self.create_publisher(PolicyDebug, "/x2/policy/debug", command_qos())
        # Real leg-command publisher is created ONLY when explicitly approved.
        self._leg_pub = None
        if self._approved:
            self.get_logger().error(
                "REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED is TRUE — leg commands ENABLED. "
                "Ensure the documented safety review passed (SAFETY.md).")
            # from sensor_msgs.msg import JointState  # real message TBD per AimDK
            # self._leg_pub = self.create_publisher(JointState, _LEG_COMMAND_TOPIC, command_qos())
        else:
            self.get_logger().info(
                "policy runtime in DRY-RUN (approval flag false): PolicyDebug only, "
                "no leg command will be published.")

        # NOTE: subscriptions to real obs inputs + ONNX runner are wired when a policy.onnx
        # and the verified sensor topics exist (Phase 4 export + AGENTS.md §3 topic check).
        self.create_timer(0.02, self._tick)  # 50 Hz

    def _tick(self):
        # Placeholder obs (all unknown) until real inputs are wired -> builder safe-stops.
        parts = {name: np.zeros(dim) for name, dim in ob.OBSERVATION_LAYOUT}
        obs, ok_obs = ob.build(parts, self._normalizer)

        raw_action = np.zeros(len(self._prev_action))
        if ok_obs:
            # raw_action = onnx_runner.infer(obs)  -- wired with a real policy.onnx
            pass
        filtered, ok_filter = self._filter.filter(raw_action, self._prev_action, dt=0.02)

        state = ps.PolicyState(
            roll=0.0, pitch=0.0, max_tilt=self._max_tilt,
            joint_fresh=False, imu_fresh=False,   # no real inputs yet -> fail closed
            inference_ok=ok_obs, action_finite=ok_filter,
            target_in_limits=ok_filter, operator_stop=False, base_stable=True)
        cut, reason = ps.evaluate(state)

        would_command = ok_obs and ok_filter and not cut
        msg = PolicyDebug()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.observation = obs.astype(float).tolist() if obs is not None else []
        msg.raw_action = raw_action.astype(float).tolist()
        msg.filtered_action = filtered.astype(float).tolist()
        msg.would_command = bool(would_command)
        msg.safety_state = reason
        self.debug_pub.publish(msg)

        if would_command:
            self._prev_action = filtered
        # Real publish path — only ever reached when approved AND a publisher exists.
        if self._approved and self._leg_pub is not None and would_command:
            self.get_logger().warn("publishing leg command (APPROVED)", throttle_duration_sec=2.0)
            # self._leg_pub.publish(<JointState built from filtered targets>)


def main(args=None):
    rclpy.init(args=args)
    node = JointPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
