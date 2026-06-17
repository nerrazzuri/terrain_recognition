#!/usr/bin/env python3
"""Deploy the factory stand+walk policies in OUR control loop on the real X2 (hardware Way-A).

Mirrors training/mujoco/run_stand_walk_mujoco.py and the factory STABLE -> MOVE sequence:
  STAND phase : `cpgtelecon` (STAND_DEFAULT) holds a FIRM, STILL stand @100 Hz (no stepping).
  WALK  phase : after a firm stand is verified+held, hand off to `cpgwalk` (LOCOMOTION) @50 Hz.

A walk command is NEVER applied until the stand is verified — the robot can't start from a
lying/unstable state. Built for a gantry/harness bring-up: stiffness ramp, state watchdog,
per-joint limit clamps, hard fall guard (factory estimator thresholds), latching e-stop.
Read docs/hardware_bringup_cpgwalk.md before running. Run ON the robot (best timing) or a wired
laptop on the same ROS_DOMAIN_ID. Needs onnxruntime + aimdk_msgs.

Control: /cpgwalk/enable (Bool) start; /cpgwalk/cmd_vel (Twist) forward speed; /cpgwalk/estop (Bool).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data

from aimdk_msgs.msg import JointCommandArray, JointCommand, JointStateArray
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

REPO = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "training" / "isaac_lab"))
from x2_locomotion.cref import factory_cpgwalk as wc       # noqa: E402
from x2_locomotion.cref import factory_cpgtelecon as tc    # noqa: E402

# Published joint groups (full lower body + arms so nothing is left un-commanded).
LEG = wc.JOINT_ORDER[0:12]
WAIST = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
ARM = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
       "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
       "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"]
ALL_J = LEG + WAIST + ARM

# Per-joint position limits (rad) — from x2_ultra.xml, small safety margin. Unlisted -> generous.
JOINT_LIMITS = {
    "left_hip_pitch_joint": (-2.70, 2.55), "left_hip_roll_joint": (-0.23, 2.90),
    "left_hip_yaw_joint": (-1.68, 3.43), "left_knee_joint": (0.0, 2.40),
    "left_ankle_pitch_joint": (-0.80, 0.45), "left_ankle_roll_joint": (-0.26, 0.26),
    "right_hip_pitch_joint": (-2.70, 2.55), "right_hip_roll_joint": (-2.90, 0.23),
    "right_hip_yaw_joint": (-3.43, 1.68), "right_knee_joint": (0.0, 2.40),
    "right_ankle_pitch_joint": (-0.80, 0.45), "right_ankle_roll_joint": (-0.26, 0.26),
    "waist_yaw_joint": (-3.43, 2.38), "waist_pitch_joint": (-0.31, 0.31), "waist_roll_joint": (-0.48, 0.48),
}
# Neutral standing pose: legs from cpgwalk default; waist 0; arms a mild hang.
NEUTRAL = {n: 0.0 for n in ALL_J}
NEUTRAL.update(dict(zip(wc.JOINT_ORDER, wc.DEFAULT_DOF_POS)))
# cpgtelecon (14 action) and cpgwalk (17) per-joint gains
KP_TC = dict(zip(tc.ACTION_SEQ, tc.KPS)); KD_TC = dict(zip(tc.ACTION_SEQ, tc.KDS))
KP_WC = dict(zip(wc.JOINT_ORDER, wc.KPS)); KD_WC = dict(zip(wc.JOINT_ORDER, wc.KDS))
KP_HOLD, KD_HOLD = 40.0, 4.0

RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)


class StandWalkDeploy(Node):
    def __init__(self):
        super().__init__("cpgwalk_deploy")
        self.declare_parameter("onnx", "")              # cpgwalk (walk)
        self.declare_parameter("stand_onnx", "")        # cpgtelecon (stand)
        self.declare_parameter("imu_topic", "/aima/hal/imu/torso/state")
        self.declare_parameter("ramp_s", 3.0)
        self.declare_parameter("state_timeout_s", 0.1)
        self.declare_parameter("stand_pitch_max", 0.20)
        self.declare_parameter("stand_roll_max", 0.15)
        self.declare_parameter("stand_omega_max", 0.50)
        self.declare_parameter("stand_hold_s", 1.0)
        self.declare_parameter("fall_pitch", 0.70)      # estimator.fall_pitch
        self.declare_parameter("fall_roll", 0.50)       # estimator.fall_roll
        self.declare_parameter("imu_roll_offset", 0.0)
        self.declare_parameter("imu_pitch_offset", -0.03)
        walk_onnx = self.get_parameter("onnx").value
        stand_onnx = self.get_parameter("stand_onnx").value
        if not (walk_onnx and pathlib.Path(walk_onnx).exists()
                and stand_onnx and pathlib.Path(stand_onnx).exists()):
            raise RuntimeError("set -p onnx:=<cpgwalkrun_v25_v2.onnx> -p stand_onnx:=<cpgtelecon_v3_fix.onnx>")

        self.walk = wc.FactoryCpgwalkPolicy(walk_onnx)
        self.stand = tc.FactoryCpgteleconPolicy(stand_onnx)
        self.base_dt = tc.CONTROL_DT                    # 100 Hz base; cpgwalk decimated to 50 Hz
        self.phase = "HOLD"                             # HOLD | STAND | WALK | SAFE
        self.enabled = self.estopped = False
        self.cmd = np.zeros(4, dtype=np.float32)
        self.kp_scale = 0.0
        self.t_phase = 0.0
        self.stand_since = None
        self.tick = 0
        self.hold_pose = dict(NEUTRAL)                  # captured at enable so held joints don't jerk

        self.ramp_s = float(self.get_parameter("ramp_s").value)
        self.timeout = float(self.get_parameter("state_timeout_s").value)
        self.stand_pitch = float(self.get_parameter("stand_pitch_max").value)
        self.stand_roll = float(self.get_parameter("stand_roll_max").value)
        self.stand_omega = float(self.get_parameter("stand_omega_max").value)
        self.stand_hold_s = float(self.get_parameter("stand_hold_s").value)
        self.fall_pitch = float(self.get_parameter("fall_pitch").value)
        self.fall_roll = float(self.get_parameter("fall_roll").value)
        self.roll_off = float(self.get_parameter("imu_roll_offset").value)
        self.pitch_off = float(self.get_parameter("imu_pitch_offset").value)

        self.pos: dict[str, float] = {}
        self.vel: dict[str, float] = {}
        self.omega = np.zeros(3, dtype=np.float32)
        self.euler = np.zeros(3, dtype=np.float32)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # x,y,z,w
        self.t_state = self.t_imu = -1e9

        for tp in ("/aima/hal/joint/leg/state", "/aima/hal/joint/waist/state", "/aima/hal/joint/arm/state"):
            self.create_subscription(JointStateArray, tp, self._on_state, qos_profile_sensor_data)
        self.create_subscription(Imu, self.get_parameter("imu_topic").value, self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Bool, "/cpgwalk/enable", self._on_enable, 10)
        self.create_subscription(Bool, "/cpgwalk/estop", self._on_estop, 10)
        self.create_subscription(Twist, "/cpgwalk/cmd_vel", self._on_cmd, 10)

        self.pub_leg = self.create_publisher(JointCommandArray, "/aima/hal/joint/leg/command", RELIABLE)
        self.pub_waist = self.create_publisher(JointCommandArray, "/aima/hal/joint/waist/command", RELIABLE)
        self.pub_arm = self.create_publisher(JointCommandArray, "/aima/hal/joint/arm/command", RELIABLE)
        self.create_timer(self.base_dt, self._tick)
        self.get_logger().info("stand+walk deploy up @100Hz. Mode=HOLD. enable -> STAND(cpgtelecon) "
                               "-> (firm stand) -> WALK(cpgwalk).")

    # ---- callbacks ----
    def _on_state(self, msg: JointStateArray):
        for j in msg.joints:
            self.pos[j.name] = j.position; self.vel[j.name] = j.velocity
        self.t_state = self._now()

    def _on_imu(self, msg: Imu):
        w = msg.angular_velocity
        self.omega = np.array([w.x, w.y, w.z], dtype=np.float32)
        q = msg.orientation
        self.quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
        self.euler = _quat_to_euler(q.x, q.y, q.z, q.w)
        self.euler[0] -= self.roll_off; self.euler[1] -= self.pitch_off
        self.t_imu = self._now()

    def _on_enable(self, msg: Bool):
        if self.estopped:
            return
        if msg.data and not self.enabled:
            self.phase, self.t_phase, self.kp_scale, self.stand_since = "STAND", 0.0, 0.0, None
            self.stand.reset(); self.walk.reset()
            self.hold_pose = {n: self.pos.get(n, NEUTRAL[n]) for n in ALL_J}  # don't jerk held joints
            self.get_logger().info("STAND phase (cpgtelecon): holding still; walk locked until firm stand.")
        self.enabled = bool(msg.data)
        if not msg.data:
            self.phase = "HOLD"

    def _on_estop(self, msg: Bool):
        if msg.data:
            self.estopped, self.enabled, self.phase = True, False, "SAFE"
            self.get_logger().warn("E-STOP latched: HOLD at low stiffness. Restart node to clear.")

    def _on_cmd(self, msg: Twist):
        self.cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z, 0.0], dtype=np.float32)

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    # ---- control loop @100 Hz ----
    def _tick(self):
        self.tick += 1
        now = self._now()
        fresh = (now - self.t_state) < self.timeout and (now - self.t_imu) < self.timeout
        roll, pitch = float(self.euler[0]), float(self.euler[1])

        if self.enabled and (abs(pitch) > self.fall_pitch or abs(roll) > self.fall_roll):
            self.enabled, self.phase = False, "HOLD"
            self.get_logger().error(f"FALL guard (pitch={pitch:+.2f} roll={roll:+.2f}) -> HOLD")
        if self.enabled and not fresh:
            self.enabled, self.phase = False, "HOLD"
            self.get_logger().warn("Watchdog: stale state/imu -> HOLD.")

        targets = dict(self.hold_pose if self.phase in ("STAND", "WALK") else NEUTRAL)
        kp = {n: KP_HOLD for n in ALL_J}
        kd = {n: KD_HOLD for n in ALL_J}
        kp_scale = 1.0

        if self.phase in ("STAND", "WALK") and not self.estopped and fresh:
            self.t_phase += self.base_dt
            self.kp_scale = min(1.0, self.t_phase / max(self.ramp_s, 1e-6))
            kp_scale = self.kp_scale

            if self.phase == "STAND":
                gv = tc.quat_rotate_inverse(self.quat[3], self.quat[0], self.quat[1], self.quat[2])
                dof = np.array([self.pos.get(n, tc.DEFAULT_DOF_POS_23[i]) for i, n in enumerate(tc.SEQ_OBS)], dtype=np.float32)
                dv = np.array([self.vel.get(n, 0.0) for n in tc.SEQ_OBS], dtype=np.float32)
                obs = tc.build_obs(self.omega, gv, np.zeros(3), np.zeros(4), dof, dv,
                                   self.stand._prev_action, tc.ARM_TARGET_DEFAULT, np.zeros(4))
                act = tc.action_to_targets(self.stand.infer(obs))
                for n, v in zip(tc.ACTION_SEQ, act):
                    targets[n] = float(v); kp[n], kd[n] = KP_TC[n], KD_TC[n]
                # firm-stand verification -> hand off to WALK
                firm = (abs(pitch) < self.stand_pitch and abs(roll) < self.stand_roll
                        and float(np.linalg.norm(self.omega)) < self.stand_omega and self.kp_scale >= 1.0)
                self.stand_since = (self.stand_since or now) if firm else None
                if self.stand_since and (now - self.stand_since) >= self.stand_hold_s:
                    self.phase, self.t_phase = "WALK", 0.0
                    self.walk.reset()
                    self.get_logger().info("STAND verified firm -> WALK (cpgwalk) unlocked.")

            elif self.phase == "WALK":
                if self.tick % 2 == 0:                  # cpgwalk @ 50 Hz (decimate the 100 Hz base)
                    t = self.t_phase
                    cmd = self.cmd * min(1.0, t / max(self.ramp_s, 1e-6))
                    dof = np.array([self.pos.get(n, NEUTRAL[n]) for n in wc.JOINT_ORDER], dtype=np.float32)
                    dv = np.array([self.vel.get(n, 0.0) for n in wc.JOINT_ORDER], dtype=np.float32)
                    obs = wc.build_obs(self.omega, self.euler, cmd, dof, dv,
                                       self.walk._prev_action, wc.cpg_phase(t))
                    self._walk_targets = dict(zip(wc.JOINT_ORDER, wc.action_to_targets(self.walk.infer(obs))))
                for n, v in getattr(self, "_walk_targets", {}).items():
                    targets[n] = float(v); kp[n], kd[n] = KP_WC[n], KD_WC[n]
        elif self.estopped:
            kp_scale = 0.3

        self._publish(self.pub_leg, LEG, targets, kp, kd, kp_scale)
        self._publish(self.pub_waist, WAIST, targets, kp, kd, kp_scale)
        self._publish(self.pub_arm, ARM, targets, kp, kd, kp_scale)

    def _publish(self, pub, names, targets, kp, kd, kp_scale):
        arr = JointCommandArray()
        for n in names:
            lo, hi = JOINT_LIMITS.get(n, (-3.2, 3.2))
            jc = JointCommand()
            jc.name = n
            jc.position = float(np.clip(targets[n], lo, hi))
            jc.velocity = 0.0
            jc.effort = 0.0
            jc.stiffness = float(kp[n] * kp_scale)
            jc.damping = float(kd[n])
            arr.joints.append(jc)
        pub.publish(arr)


def _quat_to_euler(x, y, z, w):
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def main():
    rclpy.init()
    node = StandWalkDeploy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
