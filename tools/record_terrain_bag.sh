#!/usr/bin/env bash
# record_terrain_bag.sh (P1-M4-T1) — record the sensor streams needed to replay the terrain
# perception pipeline offline: depth image, depth cloud, LiDAR cloud, IMU, TF, joint state.
#
# Topic names are pulled from configs/robot_topics.yaml but MUST be verified on the robot
# first (tools/check_topics.sh). Logging is part of the feature — name each bag by scene.
#
# Usage: tools/record_terrain_bag.sh <scene_name> [configs/robot_topics.yaml]
#   e.g. tools/record_terrain_bag.sh stairs_up
set -euo pipefail

SCENE="${1:?usage: record_terrain_bag.sh <scene_name> [config]}"
CFG="${2:-configs/robot_topics.yaml}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="bags/terrain_${SCENE}_${STAMP}"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: ros2 not found. Source your ROS2 + AimDK environment first." >&2
  exit 2
fi

# Pull a few key sensor topic names from the config (verify they exist first).
get() { grep -A2 "^[[:space:]]*$1:" "$CFG" | grep -oE 'name:[[:space:]]*/[^[:space:]]+' | head -1 | awk '{print $2}'; }

TOPICS=()
for key in depth_image depth_pointcloud lidar_pointcloud rgbd_imu imu_chest imu_torso; do
  t="$(get "$key" || true)"
  [[ -n "${t:-}" ]] && TOPICS+=("$t")
done
# Always try to capture TF and joint states if present.
TOPICS+=("/tf" "/tf_static" "/joint_states")

echo "Recording scene '${SCENE}' -> ${OUT}"
printf '  %s\n' "${TOPICS[@]}"
echo "(Ctrl-C to stop.)"
mkdir -p bags
exec ros2 bag record -o "$OUT" "${TOPICS[@]}"
