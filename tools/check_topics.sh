#!/usr/bin/env bash
# check_topics.sh — verify the topics listed in configs/robot_topics.yaml actually exist
# on the running robot (AGENTS.md §3: never trust a roadmap topic name; verify it).
#
# Usage: tools/check_topics.sh [configs/robot_topics.yaml]
# Exit non-zero if any expected sensor/locomotion topic is missing.
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

echo "== Live topics =="
ros2 topic list | sort

echo
echo "== Checking expected topics from $CFG =="
# Extract every 'name: /...' value from the YAML (sensors + terrain + locomotion).
mapfile -t EXPECTED < <(grep -oE 'name:[[:space:]]*/[^[:space:]]+' "$CFG" | awk '{print $2}' | sort -u)

LIVE="$(ros2 topic list)"
missing=0
for t in "${EXPECTED[@]}"; do
  if grep -qxF "$t" <<<"$LIVE"; then
    echo "  OK      $t"
  else
    echo "  MISSING $t"
    missing=$((missing + 1))
  fi
done

echo
if [[ "$missing" -gt 0 ]]; then
  echo "RESULT: $missing expected topic(s) missing. Update configs/robot_topics.yaml or"
  echo "        investigate the robot before trusting these names."
  exit 1
fi
echo "RESULT: all expected topics present. Remember to also verify TYPES/QoS with check_qos.sh."
