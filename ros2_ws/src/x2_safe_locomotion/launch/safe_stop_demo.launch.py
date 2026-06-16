"""safe_stop_demo launch (P2-M3-T6).

Brings up the full perceive-and-stop demo in DRY-RUN by default (no real velocity sent):

    perception:  pointcloud_projector -> heightmap_node -> stair_detector -> terrain_classifier
    locomotion:  emergency_stop_node, safety_supervisor, velocity_adapter (dry-run)

Demo flow (roadmap §13): flat ground -> live height map -> place box/curb/stair -> X2
classifies unsafe terrain -> send a slow forward desired velocity -> adapter slows and stops
before the obstacle -> the safety_decision log shows the stop reason.

Set dry_run.enabled: false in configs/safe_locomotion.yaml ONLY for a sanctioned on-robot
test, with the command source registered and an operator e-stop ready (SAFETY.md / §6.5).
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    perception = "x2_terrain_perception"
    locomotion = "x2_safe_locomotion"
    return LaunchDescription([
        Node(package=perception, executable="pointcloud_projector", name="pointcloud_projector"),
        Node(package=perception, executable="heightmap_node", name="heightmap_node"),
        Node(package=perception, executable="stair_detector", name="stair_detector"),
        Node(package=perception, executable="terrain_classifier", name="terrain_classifier"),
        Node(package=perception, executable="visualization_node", name="visualization_node"),
        Node(package=locomotion, executable="emergency_stop_node", name="emergency_stop_node"),
        Node(package=locomotion, executable="safety_supervisor", name="safety_supervisor"),
        Node(package=locomotion, executable="motion_state_monitor", name="motion_state_monitor"),
        Node(package=locomotion, executable="velocity_adapter", name="velocity_adapter"),
    ])
