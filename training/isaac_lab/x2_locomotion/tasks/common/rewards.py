"""Reward components (P4-M2-T1). Pure numpy, no Isaac Lab / torch.

Each component is a small pure function so it can be unit-tested and logged separately
(roadmap §8.9). The Isaac Lab env's reward manager calls these per step and sums them with
configured weights.
"""
from __future__ import annotations

import numpy as np


def velocity_tracking(cmd_vel, actual_vel, sigma: float = 0.25) -> float:
    """Exponential velocity-tracking reward in [0, 1]; 1 at perfect tracking."""
    err = np.asarray(cmd_vel, float) - np.asarray(actual_vel, float)
    return float(np.exp(-float(err @ err) / (sigma * sigma)))


def torso_stability(roll: float, pitch: float, ang_vel) -> float:
    """Negative penalty for tilt + angular velocity; larger (less negative) is better."""
    ang = np.asarray(ang_vel, float)
    return float(-(roll * roll + pitch * pitch) - 0.05 * float(ang @ ang))


def base_height_reward(height: float, target: float, tol: float = 0.05) -> float:
    """Reward for keeping the base near the target height."""
    return float(np.exp(-((height - target) ** 2) / (tol * tol)))


def foot_clearance(swing_height, target_clearance: float) -> float:
    """Reward swing feet clearing the terrain near the target clearance."""
    sh = np.asarray(swing_height, float)
    return float(-np.sum((sh - target_clearance) ** 2))


def foot_slip_penalty(contact_mask, foot_planar_vel) -> float:
    """Penalise horizontal foot velocity while in contact."""
    mask = np.asarray(contact_mask, float)
    vel = np.asarray(foot_planar_vel, float)
    return float(np.sum(mask * np.sum(vel * vel, axis=-1)))


def action_rate_penalty(action, prev_action) -> float:
    """Penalise large action changes (smoothness). 0 when unchanged."""
    d = np.asarray(action, float) - np.asarray(prev_action, float)
    return float(d @ d)


def energy_penalty(torque, joint_vel) -> float:
    """Penalise mechanical power |tau . qdot|."""
    return float(np.abs(np.asarray(torque, float) @ np.asarray(joint_vel, float)))


def joint_limit_penalty(q, q_min, q_max, margin: float = 0.05) -> float:
    """Penalise joints within ``margin`` of (or past) their limits."""
    q = np.asarray(q, float)
    lo = np.asarray(q_min, float) + margin
    hi = np.asarray(q_max, float) - margin
    over = np.maximum(0.0, q - hi) + np.maximum(0.0, lo - q)
    return float(over @ over)


def foothold_quality(landing_unsafe_mask) -> float:
    """Penalise feet landing in unsafe cells (near stair edge / gap / unknown)."""
    return float(-np.sum(np.asarray(landing_unsafe_mask, float)))
