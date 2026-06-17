"""Factory cpgwalk policy adapter (P4-M6-T2).

Reproduces the robot's factory CPG walking policy I/O contract (see
docs/factory_cpgwalk_contract.md): a 65-dim observation, frame-stacked x10 -> [1,650], through
the `cpgwalkrun` ONNX -> [1,17] action (joint position offsets). Used as the **distillation
teacher** for the stair policy and as a reference walker.

Obs assembly + action->target mapping are pure numpy (unit-testable). ONNX inference is lazy
(needs onnxruntime + the model file from the robot runtime, not committed here).

CAVEAT: the CPG phase obs (`q`, [61:65]) is generated inside the closed factory MC; we
approximate it with a 2-oscillator clock. Exact phase/command scaling must be confirmed in sim
against recorded robot state before trusting this as a faithful teacher.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 17-DoF order: obs `seq` == `action_seq` (factory_cpgwalk_contract.md).
JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
]
ACTION_DIM = 17
OBS_DIM = 65
FRAME_STACK = 10

ACTION_SCALE = np.array(
    [0.5, 0.5, 0.5, 0.5, 0.2, 0.02, 0.5, 0.5, 0.5, 0.5, 0.2, 0.02, 0.2, 0.2, 0.2, 0.2, 0.2])
DEFAULT_DOF_POS = np.array(
    [-0.248, 0., 0., 0.5303, -0.2823, 0., -0.248, 0., 0., 0.5303, -0.2823, 0., 0., 0., 0., 0., 0.])
KPS = np.array([120., 120., 120., 150., 40., 30., 120., 120., 120., 150., 40., 30., 160., 80., 80., 80., 80.])
KDS = np.array([5., 5., 5., 5., 3., 2., 5., 5., 5., 5., 3., 2., 5., 5., 5., 4., 4.])

OBS_CLIP = 18.0
ACTION_CLIP = 18.0
CONTROL_DT = 0.02          # 50 Hz
CPG_T = 0.6                # CPG cycle time (s)

# obs_scales (CPGWalkConfig.obs_scales)
S_ANG_VEL, S_QUAT, S_LIN_VEL, S_DOF_POS, S_DOF_VEL = 1.0, 1.0, 2.0, 1.0, 0.05

# obs slice boundaries (CPGWalkConfig.obs_index)
SL_OMEGA = slice(0, 3)
SL_EULER = slice(3, 6)
SL_CMD = slice(6, 10)      # 4 dims: vx, vy, yaw, + 1 (gait/height/stand flag — confirm)
SL_POS = slice(10, 27)
SL_VEL = slice(27, 44)
SL_ACT = slice(44, 61)
SL_Q = slice(61, 65)       # CPG phase


def build_obs(imu_omega, imu_euler, command4, dof_pos, dof_vel, prev_action, cpg_q) -> np.ndarray:
    """Assemble one 65-dim observation frame, scaled + clipped per the factory contract.

    dof_pos is relative to DEFAULT_DOF_POS (caller passes raw joint pos; we subtract default).
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[SL_OMEGA] = np.asarray(imu_omega, dtype=np.float32) * S_ANG_VEL
    obs[SL_EULER] = np.asarray(imu_euler, dtype=np.float32) * S_QUAT
    obs[SL_CMD] = np.asarray(command4, dtype=np.float32) * S_LIN_VEL
    obs[SL_POS] = (np.asarray(dof_pos, dtype=np.float32) - DEFAULT_DOF_POS) * S_DOF_POS
    obs[SL_VEL] = np.asarray(dof_vel, dtype=np.float32) * S_DOF_VEL
    obs[SL_ACT] = np.asarray(prev_action, dtype=np.float32)
    obs[SL_Q] = np.asarray(cpg_q, dtype=np.float32)
    return np.clip(obs, -OBS_CLIP, OBS_CLIP)


def cpg_phase(t: float, cycle_t: float = CPG_T) -> np.ndarray:
    """Approximate the 4-dim CPG phase `q`: two antiphase oscillators (L, R) as (sin, cos).

    NOTE: an approximation of the factory CPG — confirm against recorded robot obs.
    """
    ph = 2.0 * np.pi * (t / cycle_t)
    return np.array([np.sin(ph), np.cos(ph), np.sin(ph + np.pi), np.cos(ph + np.pi)], dtype=np.float32)


def action_to_targets(action) -> np.ndarray:
    """Map a 17-dim policy action to PD joint position targets."""
    a = np.clip(np.asarray(action, dtype=np.float32), -ACTION_CLIP, ACTION_CLIP)
    return DEFAULT_DOF_POS + a * ACTION_SCALE


@dataclass
class FactoryCpgwalkPolicy:
    """Loads the factory cpgwalk ONNX and runs it with a 10-frame stacked 65-obs history."""

    onnx_path: str

    def __post_init__(self):
        import onnxruntime as ort
        self._sess = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self._in = self._sess.get_inputs()[0].name
        self._hist = np.zeros((FRAME_STACK, OBS_DIM), dtype=np.float32)
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

    def reset(self):
        self._hist[:] = 0.0
        self._prev_action[:] = 0.0

    def infer(self, obs65: np.ndarray) -> np.ndarray:
        """Push one obs frame, run the stacked policy, return the 17-dim action."""
        self._hist = np.roll(self._hist, -1, axis=0)
        self._hist[-1] = obs65
        x = self._hist.reshape(1, FRAME_STACK * OBS_DIM).astype(np.float32)
        action = self._sess.run(None, {self._in: x})[0].reshape(ACTION_DIM)
        self._prev_action = action
        return action
