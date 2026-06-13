#!/usr/bin/env python3
"""replay_bag_heightmap.py (P1-M4-T3).

Thin CLI wrapper around x2_terrain_perception.offline_bag_analyzer: replays a recorded
rosbag2 of point clouds through the perception cores offline and writes a JSONL report.

Usage:
    python tools/replay_bag_heightmap.py <bag_dir> [--topic ...] [--out ...]
"""
import sys
from pathlib import Path

# Make the perception package importable when run from the repo without a colcon install.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "ros2_ws/src/x2_terrain_perception"))
sys.path.insert(0, str(_REPO / "ros2_ws/src/x2_common"))

from x2_terrain_perception.offline_bag_analyzer import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
