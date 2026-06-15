#!/usr/bin/env bash
# brev_bootstrap.sh — one-shot setup for an Isaac Lab GPU box (NVIDIA Brev / A6000).
#
# Use as the Brev Launchable "startup script", or run once after first SSH. Idempotent: safe
# to re-run. It installs Isaac Lab, clones this repo, repopulates the X2 meshes, and (if a USD
# converter is available) converts the URDF -> USD. Does NOT need the physical robot.
#
# Before running, stage:
#   ~/X2_URDF-v1.3.0.zip            (for the gitignored meshes; upload via Brev/scp)
#   GIT_URL                          (set below; PAT only needed if the repo is private)
#
# After it finishes, validate with:
#   cd ~/IsaacLab && ./isaaclab.sh -p ~/terrain_recognition/training/isaac_lab/scripts/spawn_x2.py --headless
set -euo pipefail

GIT_URL="${GIT_URL:-https://github.com/nerrazzuri/terrain_recognition.git}"
REPO="${REPO:-$HOME/terrain_recognition}"
URDF_ZIP="${URDF_ZIP:-$HOME/X2_URDF-v1.3.0.zip}"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
VENV="${VENV:-$HOME/env_isaaclab}"
ISAACSIM_VER="${ISAACSIM_VER:-4.5.0}"

log() { echo -e "\n=== $* ==="; }

log "0/6 sanity"
nvidia-smi || { echo "no GPU visible — wrong instance?"; exit 1; }
df -h / | tail -1

log "1/6 apt deps"
sudo apt-get update -y
sudo apt-get install -y git python3-venv build-essential unzip cmake

log "2/6 python venv + Isaac Sim ($ISAACSIM_VER)"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install "isaacsim[all,extscache]==${ISAACSIM_VER}" --extra-index-url https://pypi.nvidia.com

log "3/6 Isaac Lab"
if [[ ! -d "$ISAACLAB" ]]; then
  git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB"
fi
( cd "$ISAACLAB" && ./isaaclab.sh --install )

log "4/6 clone this repo"
if [[ ! -d "$REPO/.git" ]]; then
  git clone "$GIT_URL" "$REPO"
else
  ( cd "$REPO" && git pull --ff-only || true )
fi

log "5/6 repopulate X2 meshes"
if [[ -f "$URDF_ZIP" ]]; then
  ( cd "$REPO" && bash tools/fetch_x2_assets.sh "$URDF_ZIP" )
else
  echo "WARNING: $URDF_ZIP not found — upload it and run: tools/fetch_x2_assets.sh <zip>"
fi

log "6/6 URDF -> USD (optional; needs Isaac Lab converter)"
CONV="$ISAACLAB/scripts/tools/convert_urdf.py"
URDF="$REPO/training/isaac_lab/assets/x2_ultra_simple_collision.urdf"
USD="$REPO/training/isaac_lab/assets/x2.usd"
if [[ -f "$CONV" && -f "$URDF" && -d "$REPO/training/isaac_lab/assets/meshes" ]]; then
  ( cd "$ISAACLAB" && ./isaaclab.sh -p "$CONV" "$URDF" "$USD" --merge-joints --headless ) \
    && echo "converted -> $USD" || echo "USD conversion skipped/failed (do it manually per SETUP.md)"
else
  echo "skipping USD conversion (converter or meshes missing) — see SETUP.md §3"
fi

cat <<EOF

DONE. To validate the X2 spawns and stands in Isaac Sim:

  source $VENV/bin/activate
  export PYTHONPATH=$REPO/training/isaac_lab:$REPO/ros2_ws/src/x2_common:\$PYTHONPATH
  export X2_CONFIG_DIR=$REPO/configs
  export X2_USD_PATH=$USD
  cd $ISAACLAB && ./isaaclab.sh -p $REPO/training/isaac_lab/scripts/spawn_x2.py --headless --seconds 5
EOF
