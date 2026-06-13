"""Standard QoS profiles for the stack.

Sensor streams (clouds, IMU, images) use best-effort + keep-last; commands and latched
state use reliable. ``rclpy`` is imported lazily so this module loads in non-ROS contexts
(the profile *factory* functions only need rclpy when actually called).
"""
from __future__ import annotations

from typing import Any


def sensor_qos(depth: int = 5) -> Any:
    """Best-effort, volatile, keep-last — for high-rate sensor data (clouds, IMU, images)."""
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def command_qos(depth: int = 10) -> Any:
    """Reliable, volatile, keep-last — for command topics (e.g. velocity)."""
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def latched_qos(depth: int = 1) -> Any:
    """Reliable + transient-local — for latched/last-value state (maps, static config)."""
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )
