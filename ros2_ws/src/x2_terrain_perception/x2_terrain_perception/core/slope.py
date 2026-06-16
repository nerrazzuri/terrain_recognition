"""Slope angle and up/down direction from a ground-plane normal — pure logic.

Forward is +x in the base frame. A plane that rises with +x is ``slope_up``.
"""
from __future__ import annotations

import math

import numpy as np


def slope_angle_deg(normal: np.ndarray) -> float:
    """Angle (degrees) between the plane normal and vertical. 0 = flat."""
    n = np.asarray(normal, dtype=float)
    nrm = np.linalg.norm(n)
    if nrm < 1e-9:
        return 0.0
    nz = abs(n[2]) / nrm
    nz = min(1.0, max(-1.0, nz))
    return math.degrees(math.acos(nz))


def forward_gradient(normal: np.ndarray) -> float:
    """dz/dx implied by the plane normal (positive = rises ahead)."""
    nx, _, nz = np.asarray(normal, dtype=float)
    if abs(nz) < 1e-9:
        return 0.0
    return -nx / nz


def slope_direction(normal: np.ndarray, threshold_deg: float = 6.0) -> str:
    """Classify slope as ``"up"`` / ``"down"`` / ``"none"`` w.r.t. forward (+x)."""
    if slope_angle_deg(normal) < threshold_deg:
        return "none"
    grad = forward_gradient(normal)
    if grad > 0:
        return "up"
    if grad < 0:
        return "down"
    return "none"
