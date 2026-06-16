#!/usr/bin/env bash
# fetch_x2_assets.sh — repopulate the X2 mesh STL files (gitignored, ~112 MB) from the
# official URDF zip into training/isaac_lab/assets/meshes/ (the mujoco model symlinks to it).
#
# Usage: tools/fetch_x2_assets.sh /path/to/X2_URDF-v1.3.0.zip
set -euo pipefail

ZIP="${1:?usage: fetch_x2_assets.sh <X2_URDF-*.zip>}"
DEST="training/isaac_lab/assets/meshes"

if [[ ! -f "$ZIP" ]]; then
  echo "ERROR: zip not found: $ZIP" >&2
  exit 2
fi
mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
unzip -q "$ZIP" -d "$TMP"
SRC="$(dirname "$(find "$TMP" -name 'x2_ultra.urdf' | head -1)")"
cp -v "$SRC"/meshes/*.STL "$DEST"/
echo "Repopulated $(ls "$DEST"/*.STL | wc -l) meshes into $DEST"
