"""x2_common — shared utilities for the X2 terrain locomotion stack.

Modules here must not perform robot I/O of their own. ROS2-specific helpers import
``rclpy`` lazily so that pure-logic modules (config_loader, transforms, safety_limits,
time_sync) remain importable — and unit-testable — without a ROS2 install.
"""

__all__ = [
    "config_loader",
    "joint_map",
    "logging_utils",
    "safety_limits",
    "time_sync",
    "transforms",
    "qos_profiles",
    "topic_discovery",
]

__version__ = "0.0.0"
