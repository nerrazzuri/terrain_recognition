#!/usr/bin/env python3
"""Deploy the factory `cpgwalk` ONNX in OUR control loop on the real X2 (hardware Way-A).

Mirrors training/mujoco/run_cpgwalk_mujoco.py — SAME control math (x2_locomotion.cref
factory_cpgwalk) — but reads live robot state and publishes joint commands. Built for a
**gantry/harness bring-up**: it starts in HOLD (no policy), ramps stiffness in, has a state
watchdog, joint-limit clamps, and an e-stop. Read docs/hardware_bringup_cpgwalk.md before running.

Modes (set live):
  HOLD  (default)        : PD-hold the neutral standing pose. Proves takeover + PD with no gait.
  RUN   (enable=true)    : run cpgwalk; command from /cpgwalk/cmd_vel (Twist).
Safety:
  * stiffness ramps 0->1 over `ramp_s` on every HOLD->RUN and on startup
  * watchdog: stale state/imu (> `state_timeout_s`) => force HOLD
  * every commanded position clamped to JOINT_LIMITS (from the X2 MJCF / SDK)
  * /cpgwalk/estop (Bool true) => latch HOLD with low stiffness until node restart

Run ON the robot (best timing) or a wired laptop on the same ROS_DOMAIN_ID. Needs onnxruntime
+ aimdk_msgs on the PYTHONPATH/ROS env.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data

from aimdk_msgs.msg import JointCommandArray, JointCommand, JointStateArray
from aimdk_msgs.msg import McLocomotionVelocity  # noqa: F401  (optional: subscribe to MC vel)
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

REPO = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "training" / "isaac_lab"))
from x2_locomotion.cref import factory_cpgwalk as fc  # noqa: E402

# Per-joint position limits (rad) — from x2_ultra.xml ranges, with a small safety margin.
JOINT_LIMITS = {
    "left_hip_pitch_joint": (-2.70, 2.55), "left_hip_roll_joint": (-0.23, 2.90),
    "left_hip_yaw_joint": (-1.68, 3.43), "left_knee_joint": (0.0, 2.40),
    "left_ankle_pitch_joint": (-0.80, 0.45), "left_ankle_roll_joint": (-0.26, 0.26),
    "right_hip_pitch_joint": (-2.70, 2.55), "right_hip_roll_joint": (-2.90, 0.23),
    "right_hip_yaw_joint": (-3.43, 1.68), "right_knee_joint": (0.0, 2.40),
    "right_ankle_pitch_joint": (-0.80, 0.45), "right_ankle_roll_joint": (-0.26, 0.26),
    "waist_yaw_joint": (-3.43, 2.38), "waist_pitch_joint": (-0.31, 0.31),
    "waist_roll_joint": (-0.48, 0.48),
    "left_shoulder_pitch_joint": (-3.08, 2.04), "right_shoulder_pitch_joint": (-3.08, 2.04),
}
# Neutral standing pose for the 17 policy joints == factory default_dof_pos.
NEUTRAL = dict(zip(fc.JOINT_ORDER, fc.DEFAULT_DOF_POS))
KP = dict(zip(fc.JOINT_ORDER, fc.KPS))
KD = dict(zip(fc.JOINT_ORDER, fc.KDS))
LEG = fc.JOINT_ORDER[0:12]
WAIST = fc.JOINT_ORDER[12:15]
ARM = fc.JOINT_ORDER[15:17]   # L/R shoulder pitch only (cpgwalk uses these for balance)

RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)


def quat_to_euler(x, y, z, w):
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


class CpgwalkDeploy(Node):
    def __init__(self):
        super().__init__("cpgwalk_deploy")
        self.declare_parameter("onnx", "")
        self.declare_parameter("imu_topic", "/aima/hal/imu/torso/state")  # use_chest_imu:false
        self.declare_parameter("ramp_s", 3.0)
        self.declare_parameter("state_timeout_s", 0.1)
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("imu_roll_offset", 0.0)
        self.declare_parameter("imu_pitch_offset", -0.015)
        # --- strict STAND-before-WALK gate (mirrors factory STABLE->MOVE; thresholds from
        #     action_state.yaml `estimator`). No walk command is applied until a firm stand is
        #     verified and held, so the robot can never start walking from a lying/unstable state.
        self.declare_parameter("stand_pitch_max", 0.20)   # rad, firm-stand upright tolerance
        self.declare_parameter("stand_roll_max", 0.15)    # rad
        self.declare_parameter("stand_omega_max", 0.50)   # rad/s, must be settled
        self.declare_parameter("stand_hold_s", 1.0)       # must hold the above this long
        self.declare_parameter("fall_pitch", 0.70)        # action_state.yaml estimator.fall_pitch
        self.declare_parameter("fall_roll", 0.50)         # action_state.yaml estimator.fall_roll
        onnx = self.get_parameter("onnx").value
        if not onnx or not pathlib.Path(onnx).exists():
            raise RuntimeError("set --ros-args -p onnx:=/path/to/cpgwalkrun_v25_v2.onnx")

        self.policy = fc.FactoryCpgwalkPolicy(onnx)
        self.enabled = False
        self.estopped = False
        self.cmd = np.zeros(4, dtype=np.float32)
        self.kp_scale = 0.0
        self.t_run = 0.0
        self.walk_ok = False          # set True only after a firm stand is verified
        self.stand_since = None       # time the firm-stand criteria first held continuously
        self.stand_pitch = float(self.get_parameter("stand_pitch_max").value)
        self.stand_roll = float(self.get_parameter("stand_roll_max").value)
        self.stand_omega = float(self.get_parameter("stand_omega_max").value)
        self.stand_hold_s = float(self.get_parameter("stand_hold_s").value)
        self.fall_pitch = float(self.get_parameter("fall_pitch").value)
        self.fall_roll = float(self.get_parameter("fall_roll").value)
        self.dt = 1.0 / float(self.get_parameter("rate_hz").value)
        self.ramp_s = float(self.get_parameter("ramp_s").value)
        self.timeout = float(self.get_parameter("state_timeout_s").value)
        self.roll_off = float(self.get_parameter("imu_roll_offset").value)
        self.pitch_off = float(self.get_parameter("imu_pitch_offset").value)

        self.pos: dict[str, float] = {}
        self.vel: dict[str, float] = {}
        self.omega = np.zeros(3, dtype=np.float32)
        self.euler = np.zeros(3, dtype=np.float32)
        self.t_state = self.t_imu = -1e9

        for tp in ("/aima/hal/joint/leg/state", "/aima/hal/joint/waist/state",
                   "/aima/hal/joint/arm/state"):
            self.create_subscription(JointStateArray, tp, self._on_state, qos_profile_sensor_data)
        imu_topic = self.get_parameter("imu_topic").value
        self.create_subscription(Imu, imu_topic, self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Bool, "/cpgwalk/enable", self._on_enable, 10)
        self.create_subscription(Bool, "/cpgwalk/estop", self._on_estop, 10)
        self.create_subscription(Twist, "/cpgwalk/cmd_vel", self._on_cmd, 10)

        self.pub_leg = self.create_publisher(JointCommandArray, "/aima/hal/joint/leg/command", RELIABLE)
        self.pub_waist = self.create_publisher(JointCommandArray, "/aima/hal/joint/waist/command", RELIABLE)
        self.pub_arm = self.create_publisher(JointCommandArray, "/aima/hal/joint/arm/command", RELIABLE)

        self.create_timer(self.dt, self._tick)
        self.get_logger().info(f"cpgwalk_deploy up. imu={imu_topic}. Mode=HOLD. "
                               f"Enable with: ros2 topic pub -1 /cpgwalk/enable std_msgs/Bool '{{data: true}}'")

    # --- callbacks ---
    def _on_state(self, msg: JointStateArray):
        for j in msg.joints:
            self.pos[j.name] = j.position
            self.vel[j.name] = j.velocity
        self.t_state = self._now()

    def _on_imu(self, msg: Imu):
        w = msg.angular_velocity
        self.omega = np.array([w.x, w.y, w.z], dtype=np.float32)
        q = msg.orientation
        self.euler = quat_to_euler(q.x, q.y, q.z, q.w)
        self.euler[0] -= self.roll_off
        self.euler[1] -= self.pitch_off
        self.t_imu = self._now()

    def _on_enable(self, msg: Bool):
        if self.estopped:
            return
        if msg.data and not self.enabled:
            self.t_run, self.kp_scale = 0.0, 0.0  # restart ramp + clock
            self.walk_ok, self.stand_since = False, None  # re-verify a firm stand every time
            self.policy.reset()
            self.get_logger().info("STAND phase: balancing at zero command; walk stays locked "
                                   "until a firm stand is verified.")
        self.enabled = bool(msg.data)

    def _on_estop(self, msg: Bool):
        if msg.data:
            self.estopped, self.enabled = True, False
            self.get_logger().warn("E-STOP latched: holding pose at low stiffness. Restart node to clear.")

    def _on_cmd(self, msg: Twist):
        self.cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z, 0.0], dtype=np.float32)

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    # --- control loop ---
    def _tick(self):
        now = self._now()
        fresh = (now - self.t_state) < self.timeout and (now - self.t_imu) < self.timeout
        roll, pitch = float(self.euler[0]), float(self.euler[1])

        # hard fall guard (factory estimator thresholds) — always active while enabled
        if self.enabled and (abs(pitch) > self.fall_pitch or abs(roll) > self.fall_roll):
            self.enabled, self.walk_ok = False, False
            self.get_logger().error(f"FALL guard tripped (pitch={pitch:+.2f} roll={roll:+.2f}) -> HOLD")
        if self.enabled and not fresh:
            self.enabled, self.walk_ok = False, False
            self.get_logger().warn("Watchdog: stale state/imu -> HOLD.")

        run = self.enabled and not self.estopped and fresh
        if run:
            self.t_run += self.dt
            self.kp_scale = min(1.0, self.t_run / max(self.ramp_s, 1e-6))

            # STRICT STAND GATE: verify a firm, settled, upright stand is HELD before unlocking walk.
            if not self.walk_ok:
                firm = (abs(pitch) < self.stand_pitch and abs(roll) < self.stand_roll
                        and float(np.linalg.norm(self.omega)) < self.stand_omega
                        and self.kp_scale >= 1.0)          # stiffness fully ramped in
                if firm:
                    self.stand_since = self.stand_since or now
                    if (now - self.stand_since) >= self.stand_hold_s:
                        self.walk_ok = True
                        self.get_logger().info("STAND verified firm -> WALK unlocked.")
                else:
                    self.stand_since = None                # reset the hold timer on any wobble

            # command is zero until the stand is verified (can never walk from lying/unstable)
            cmd = self.cmd if self.walk_ok else np.zeros(4, dtype=np.float32)
            dof_pos = np.array([self.pos.get(n, NEUTRAL[n]) for n in fc.JOINT_ORDER], dtype=np.float32)
            dof_vel = np.array([self.vel.get(n, 0.0) for n in fc.JOINT_ORDER], dtype=np.float32)
            obs = fc.build_obs(self.omega, self.euler, cmd, dof_pos, dof_vel,
                               self.policy._prev_action, fc.cpg_phase(self.t_run))
            action = self.policy.infer(obs)
            targets = dict(zip(fc.JOINT_ORDER, fc.action_to_targets(action)))
            kp_scale = self.kp_scale
        else:
            targets = dict(NEUTRAL)                          # HOLD neutral pose
            kp_scale = 0.3 if self.estopped else 1.0

        self._publish(self.pub_leg, LEG, targets, kp_scale)
        self._publish(self.pub_waist, WAIST, targets, kp_scale)
        self._publish(self.pub_arm, ARM, targets, kp_scale)

    def _publish(self, pub, names, targets, kp_scale):
        arr = JointCommandArray()
        for n in names:
            lo, hi = JOINT_LIMITS[n]
            jc = JointCommand()
            jc.name = n
            jc.position = float(np.clip(targets[n], lo, hi))
            jc.velocity = 0.0
            jc.effort = 0.0
            jc.stiffness = float(KP[n] * kp_scale)
            jc.damping = float(KD[n])
            arr.joints.append(jc)
        pub.publish(arr)


def main():
    rclpy.init()
    node = CpgwalkDeploy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
