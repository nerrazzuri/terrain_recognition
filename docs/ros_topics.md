# ROS Topics

> ⚠️ **Every topic name and QoS below is provisional** and taken from the roadmap. It MUST
> be verified on the real robot with `ros2 topic list` and `ros2 topic info -v` before any
> node trusts it (AGENTS.md §3). Record verification results in the "Verified" column.

Run the helper scripts to (re)generate verification data:

```bash
tools/check_topics.sh    # ros2 topic list + presence check against configs/robot_topics.yaml
tools/check_qos.sh       # ros2 topic info -v for each expected topic
```

## Inputs — sensors (subscribe)

| Topic | Type (expected) | Used by | Verified |
|-------|-----------------|---------|:--------:|
| `/aima/hal/sensor/rgbd_head_front/depth_image` | `sensor_msgs/Image` | perception | ☐ |
| `/aima/hal/sensor/rgbd_head_front/depth_pointcloud` | `sensor_msgs/PointCloud2` | pointcloud_projector | ☐ |
| `/aima/hal/sensor/rgbd_head_front/depth_camera_info` | `sensor_msgs/CameraInfo` | perception | ☐ |
| `/aima/hal/sensor/rgbd_head_front/imu` | `sensor_msgs/Imu` | ground_plane_estimator | ☐ |
| `/aima/hal/sensor/lidar_chest_front/lidar_pointcloud` | `sensor_msgs/PointCloud2` | pointcloud_projector | ☐ |
| `/aima/hal/sensor/lidar_chest_front/imu` | `sensor_msgs/Imu` | — | ☐ |
| `/aima/hal/imu/chest/state` | (verify) | safety_supervisor | ☐ |
| `/aima/hal/imu/torso/state` | (verify) | safety_supervisor | ☐ |

Prefer point cloud / compressed streams over raw cross-unit camera subscriptions when
bandwidth is high.

## Internal — terrain messages (this stack)

| Topic | Type | Publisher | Subscriber |
|-------|------|-----------|------------|
| `/x2/terrain/heightmap` | `x2_terrain_msgs/TerrainGrid` | heightmap_node | classifier, safe_locomotion |
| `/x2/terrain/status` | `x2_terrain_msgs/TerrainStatus` | terrain_classifier | safe_locomotion |
| `/x2/terrain/stair_estimate` | `x2_terrain_msgs/StairEstimate` | stair_detector | safe_locomotion |
| `/x2/terrain/safety_decision` | `x2_terrain_msgs/SafetyDecision` | safety_supervisor | (logging, operator UI) |
| `/x2/policy/debug` | `x2_terrain_msgs/PolicyDebug` | policy_runtime | (logging) |

## Outputs — locomotion (publish)

| Topic | Type (expected) | Publisher | Notes |
|-------|-----------------|-----------|-------|
| `/aima/mc/locomotion/velocity` | (verify) | velocity_adapter | source `x2_terrain_safe_locomotion` must be registered first |
| `/aima/hal/joint/leg/command` | (verify) | **FORBIDDEN** while approval flag is false | leg joint policy — gated |

Velocity command fields (verify): `source`, `forward_velocity`, `lateral_velocity`,
`angular_velocity`.

## References

- AimDK X2 index: <https://x2-aimdk.agibot.com/en/latest/index.html>
- Locomotion Control: <https://x2-aimdk.agibot.com/en/latest/Interface/control_mod/locomotion.html>
- Joint Control: <https://x2-aimdk.agibot.com/en/latest/Interface/control_mod/joint_control.html>
- Sensor Interfaces: <https://x2-aimdk.agibot.com/en/latest/Interface/hal/sensor.html>
