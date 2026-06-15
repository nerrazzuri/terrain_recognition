"""PD joint controller — pure numpy, sim-agnostic.

torque = kp*(q_des - q) + kd*(qdot_des - qdot), clamped to the per-joint effort limit. Used
by the MuJoCo standing controller (and any torque-mode actuator path). Gains/limits come from
configs, never hardcoded in the caller.
"""
from __future__ import annotations

import numpy as np


class PDController:
    def __init__(self, kp, kd, effort_limit):
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.effort_limit = np.asarray(effort_limit, dtype=float)
        if not (self.kp.shape == self.kd.shape == self.effort_limit.shape):
            raise ValueError("kp, kd, effort_limit must have the same shape")

    def torque(self, q, qdot, q_des, qdot_des=None) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        q_des = np.asarray(q_des, dtype=float)
        if qdot_des is None:
            qdot_des = np.zeros_like(qdot)
        else:
            qdot_des = np.asarray(qdot_des, dtype=float)
        if not (q.shape == qdot.shape == q_des.shape == self.kp.shape):
            raise ValueError("state vectors must match the controller dimension")
        tau = self.kp * (q_des - q) + self.kd * (qdot_des - qdot)
        return np.clip(tau, -self.effort_limit, self.effort_limit)
