#!/usr/bin/env bash
# check_qos.sh — print verbose type/QoS info for each expected topic so QoS can be matched
# (AGENTS.md §3). Run after check_topics.sh.
#
# Usage: tools/check_qos.sh [configs/robot_topics.yaml]
set -euo pipefail

CFG="${1:-configs/robot_topics.yaml}"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: ros2 not found. Source your ROS2 + AimDK environment first." >&2
  exit 2
fi
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG" >&2
  exit 2
fi

mapfile -t EXPECTED < <(grep -oE 'name:[[:space:]]*/[^[:space:]]+' "$CFG" | awk '{print $2}' | sort -u)

for t in "${EXPECTED[@]}"; do
  echo "############################################################"
  echo "# $t"
  echo "############################################################"
  ros2 topic info -v "$t" 2>&1 || echo "  (not available)"
  echo
done
