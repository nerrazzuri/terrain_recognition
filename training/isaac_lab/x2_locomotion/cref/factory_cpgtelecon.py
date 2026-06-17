"""Factory `cpgtelecon` policy adapter — the STAND_DEFAULT controller (P4-M6).

Unlike `cpgwalk` (a walker that marches even at zero command), `cpgtelecon` is the factory's
**stand / teleop balancer**: at near-zero command it holds a still, firm stand (the CPG is frozen
below `stand_cmd_threshold`). We run it for the STAND phase so the robot stands still before
walking — mirroring the factory `STABLE`/`STAND_DEFAULT` state.

Contract from mc_param/.../rl/cpgtelecon_config.yaml + rl_models/cpgtelecon_v3_fix.onnx:
  obs 85 (x10 frame-stack -> [1,850]) -> action 14 (12 legs + waist_pitch + waist_roll).
  100 Hz. Uses **projected gravity** (not euler) and pose/arm-target obs blocks.

Obs assembly is pure numpy (unit-testable); ONNX inference is lazy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 23-joint obs order (dof_pos / dof_vel blocks)
SEQ_OBS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]
# 14-joint action order (12 legs + waist pitch/roll)
ACTION_SEQ = SEQ_OBS[0:12] + ["waist_pitch_joint", "waist_roll_joint"]

OBS_DIM = 85
ACTION_DIM = 14
Q_DIM = 23
FRAME_STACK = 10
CONTROL_DT = 0.01          # 100 Hz
CPG_T = 0.85
OBS_CLIP = ACTION_CLIP = 18.0

ACTION_SCALE = np.full(14, 0.5)
# default_dof_pos for the 14 action joints (from cpgtelecon_config default_dof_pos)
DEFAULT_DOF_POS = np.array([-0.235, 0., 0., 0.5, -0.265, 0., -0.235, 0., 0., 0.5, -0.265, 0., 0., 0.])
# default for the 23 obs joints
DEFAULT_DOF_POS_23 = np.array(
    [-0.235, 0., 0., 0.5, -0.265, 0., -0.235, 0., 0., 0.5, -0.265, 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
KPS = np.array([100., 100., 100., 150., 40., 30., 100., 100., 100., 150., 40., 30., 40., 40.])
KDS = np.array([4., 3., 3., 5., 3., 2., 4., 3., 3., 5., 3., 2., 5., 5.])
# default arm target (8 = L/R x [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow])
ARM_TARGET_DEFAULT = np.array([0.4, 0.0, 0.0, -1.2, 0.4, 0.0, 0.0, -1.2])

# obs_scales
S_LIN_VEL, S_DOF_POS, S_DOF_VEL, S_ANG_VEL = 2.0, 1.0, 0.05, 1.0

# obs slices (CPGTeleConConfig.obs_index)
SL_OMEGA = slice(0, 3)       # pelvis angular velocity
SL_GVEC = slice(3, 6)        # projected gravity vector
SL_CMD = slice(6, 9)         # vx, vy, yaw
SL_POSCMD = slice(9, 13)     # pose command (roll, pitch, yaw, height) — 0 for a neutral stand
SL_POS = slice(13, 36)       # 23 dof pos (rel default)
SL_VEL = slice(36, 59)       # 23 dof vel
SL_ACT = slice(59, 73)       # 14 prev action
SL_ARM = slice(73, 81)       # 8 arm target pos
SL_Q = slice(81, 85)         # 4 CPG phase


def quat_rotate_inverse(w, x, y, z, v=(0.0, 0.0, -1.0)) -> np.ndarray:
    """Rotate world vector v into the body frame (projected gravity for upright = [0,0,-1])."""
    q = np.array([w, x, y, z], dtype=np.float64)
    qv = q[1:]
    v = np.asarray(v, dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * (2.0 * w)
    c = qv * (2.0 * np.dot(qv, v))
    return (a - b + c).astype(np.float32)


def build_obs(omega, gvec, command3, pos_command4, dof_pos23, dof_vel23,
              prev_action14, arm_target8, cpg_q4) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[SL_OMEGA] = np.asarray(omega, dtype=np.float32) * S_ANG_VEL
    obs[SL_GVEC] = np.asarray(gvec, dtype=np.float32)
    obs[SL_CMD] = np.asarray(command3, dtype=np.float32) * S_LIN_VEL
    obs[SL_POSCMD] = np.asarray(pos_command4, dtype=np.float32)
    obs[SL_POS] = (np.asarray(dof_pos23, dtype=np.float32) - DEFAULT_DOF_POS_23) * S_DOF_POS
    obs[SL_VEL] = np.asarray(dof_vel23, dtype=np.float32) * S_DOF_VEL
    obs[SL_ACT] = np.asarray(prev_action14, dtype=np.float32)
    obs[SL_ARM] = np.asarray(arm_target8, dtype=np.float32)
    obs[SL_Q] = np.asarray(cpg_q4, dtype=np.float32)
    return np.clip(obs, -OBS_CLIP, OBS_CLIP)


def action_to_targets(action) -> np.ndarray:
    a = np.clip(np.asarray(action, dtype=np.float32), -ACTION_CLIP, ACTION_CLIP)
    return DEFAULT_DOF_POS + a * ACTION_SCALE


@dataclass
class FactoryCpgteleconPolicy:
    """Loads cpgtelecon ONNX; runs with a 10-frame stacked 85-obs history. Stand = zero command."""

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

    def infer(self, obs85: np.ndarray) -> np.ndarray:
        self._hist = np.roll(self._hist, -1, axis=0)
        self._hist[-1] = obs85
        x = self._hist.reshape(1, FRAME_STACK * OBS_DIM).astype(np.float32)
        action = self._sess.run(None, {self._in: x})[0].reshape(ACTION_DIM)
        self._prev_action = action
        return action
