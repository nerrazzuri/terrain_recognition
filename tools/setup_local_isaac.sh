#!/usr/bin/env bash
# setup_local_isaac.sh — one-shot LOCAL Isaac Sim + Isaac Lab install for viewing X2 policies.
#
# Target: Ubuntu 22.04, system Python 3.10, RTX GPU (e.g. RTX 4060 8GB) — viewing only, not training.
# Bakes in every fix discovered during the vast.ai bring-up:
#   - Isaac Sim 4.5.0 (needs Python 3.10)
#   - setuptools<80 + flatdict --no-build-isolation (setuptools 81 dropped pkg_resources)
#   - Isaac Lab pinned to v2.1.0 (the release that pairs with Isaac Sim 4.5)
#   - LD_LIBRARY_PATH pointing at all bundled Isaac Sim .so dirs (physx/USD/carb)
#   - URDF -> USD conversion (Isaac Lab 2.1 URDF importer needs joint-drive gains; USD avoids that)
# No sudo required (all needed system libs verified present on a desktop Ubuntu 22.04).
set -euo pipefail

REPO="${REPO:-/home/liang/Projects/terrain_recognition}"
VENV="${VENV:-$HOME/isaac_env}"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
URDF_ZIP="${URDF_ZIP:-$HOME/Downloads/X2_URDF-v1.3.0.zip}"

log() { echo -e "\n=== $* ==="; }

log "0/7 sanity"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 --version
df -h "$HOME" | tail -1

log "1/7 create venv ($VENV) from system python 3.10 (no python3.10-venv/sudo needed)"
# ensurepip is missing on this box (python3.10-venv not installed); create without pip
# and bootstrap pip via get-pip.py so we need no sudo.
rm -rf "$VENV"
python3 -m venv --without-pip "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py
python -m pip install --upgrade pip

log "2/7 install Isaac Sim 4.5.0 (~15 GB, the long step)"
python -m pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

log "3/7 setuptools<80 + flatdict fix (so Isaac Lab core installs)"
python -m pip install "setuptools<80" wheel
python -m pip install --no-build-isolation flatdict==4.0.1

log "4/7 Isaac Lab v2.1.0"
[ -d "$ISAACLAB" ] || git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB"
( cd "$ISAACLAB" && git checkout v2.1.0 && ./isaaclab.sh --install || true )
# belt-and-suspenders: ensure the core extensions are importable
( cd "$ISAACLAB" && python -m pip install -e source/isaaclab \
    -e source/isaaclab_assets -e source/isaaclab_rl -e source/isaaclab_tasks )

log "5/7 persist LD_LIBRARY_PATH + X2 env into the venv activate script"
ISAACSIM_DIR="$VENV/lib/python3.10/site-packages/isaacsim"
LIBS="$(find "$ISAACSIM_DIR" -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | tr '\n' ':')"
# idempotent: strip any prior block we added before re-adding
sed -i '/# --- Isaac Sim runtime env (setup_local_isaac.sh) ---/,/# --- end Isaac Sim runtime env ---/d' "$VENV/bin/activate"
cat >> "$VENV/bin/activate" <<EOF
# --- Isaac Sim runtime env (setup_local_isaac.sh) ---
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBS}\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/training/isaac_lab:$REPO/ros2_ws/src/x2_common:\${PYTHONPATH:-}"
export X2_CONFIG_DIR="$REPO/configs"
export X2_USD_PATH="$REPO/training/isaac_lab/assets/x2.usd"
# --- end Isaac Sim runtime env ---
EOF
# re-source so THIS shell has the new vars for the steps below
# shellcheck disable=SC1091
source "$VENV/bin/activate"

log "6/7 repopulate X2 meshes from $URDF_ZIP"
if [ -f "$URDF_ZIP" ]; then
  ( cd "$REPO" && bash tools/fetch_x2_assets.sh "$URDF_ZIP" )
else
  echo "WARNING: $URDF_ZIP not found — upload it then run tools/fetch_x2_assets.sh <zip>"
fi

log "7/7 convert URDF -> USD"
( cd "$ISAACLAB" && ./isaaclab.sh -p scripts/tools/convert_urdf.py \
    "$REPO/training/isaac_lab/assets/x2_ultra_simple_collision.urdf" \
    "$REPO/training/isaac_lab/assets/x2.usd" --merge-joints --headless )

cat <<EOF

############################################################
DONE. To watch the trained X2 walk (native window):

  source $VENV/bin/activate
  cd $ISAACLAB
  ./isaaclab.sh -p $REPO/training/isaac_lab/x2_locomotion/scripts/play.py \\
      --task flat_walk \\
      --checkpoint $REPO/models/x2_flat_walk_v1/model_1050.pt \\
      --num_envs 4

(8 GB VRAM: keep --num_envs small. First launch compiles shaders — be patient.)
############################################################
EOF
