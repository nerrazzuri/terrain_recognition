# Isaac Lab Setup — X2 Terrain Locomotion (NVIDIA Brev / A6000)

Repo-specific runbook to get from a fresh GPU instance to "X2 spawns and stands in Isaac Sim"
and the first RL training run. Target: **NVIDIA Brev Launchable, 1× RTX A6000 (48 GB)**,
Ubuntu 22.04. The A6000 (Ampere) is well-supported by Isaac Sim — no Blackwell driver caveats.

> Phase gate: this is Phase 3/4 (sim only, no hardware, no real leg policy). The approval flag
> stays false. Nothing here touches the robot.

---

## 0. Instance checklist

- GPU: A6000 48 GB ✅ (any RTX with ≥ 16 GB works; 48 GB gives big env headroom + Phase 6)
- Driver: **≥ 535** (Brev images ship a compatible driver). Verify: `nvidia-smi`
- Disk: **≥ 80 GB** free (Isaac Sim + Isaac Lab + caches are large)
- CUDA: 12.x (bundled with the Isaac Sim pip packages)
- Headless: yes — we run Isaac Sim with `--headless` (no display needed)

```bash
nvidia-smi                      # confirm A6000 + driver
df -h /                         # confirm >= 80 GB free
python3 --version               # expect 3.10
```

---

## 1. Install Isaac Lab (pip path — recommended on Brev)

Isaac Sim 4.5+/2025 installs from pip; Isaac Lab sits on top. Use a venv.

```bash
sudo apt-get update && sudo apt-get install -y git python3-venv build-essential
python3 -m venv ~/env_isaaclab && source ~/env_isaaclab/bin/activate
pip install --upgrade pip

# Isaac Sim (CUDA 12) — pulls the renderer + PhysX
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && ./isaaclab.sh --install        # installs isaaclab + rsl_rl
```

Smoke-test Isaac Sim itself (downloads assets on first run; accept the EULA):

```bash
cd ~/IsaacLab && ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

If that prints simulation steps and exits cleanly, Isaac Sim works on the box.

> Alternative (container path): use the `nvcr.io/nvidia/isaac-lab` container if Brev offers an
> Isaac Lab launchable — then skip the pip install and `git clone` only this repo.

---

## 2. Clone this repo + place the X2 assets

```bash
git clone https://github.com/nerrazzuri/terrain_recognition.git ~/terrain_recognition
cd ~/terrain_recognition
```

The mesh STLs are **gitignored** (~112 MB), so a fresh clone has the URDF/MJCF text but no
meshes. Get them onto the box one of two ways:

```bash
# (a) copy the URDF zip up from your machine, then repopulate:
#     scp X2_URDF-v1.3.0.zip <brev-host>:~/   (or use Brev's file upload)
tools/fetch_x2_assets.sh ~/X2_URDF-v1.3.0.zip

# (b) or scp the whole training/isaac_lab/assets/meshes/ directory up directly.
ls training/isaac_lab/assets/meshes/*.STL | wc -l    # expect 45
```

---

## 3. Convert the URDF → USD (P3-M1-T2)

Isaac Lab can import URDF directly, but a one-time USD conversion is faster and more stable:

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    ~/terrain_recognition/training/isaac_lab/assets/x2_ultra_simple_collision.urdf \
    ~/terrain_recognition/training/isaac_lab/assets/x2.usd \
    --merge-joints --headless

# point our robot cfg at the produced USD
export X2_USD_PATH=~/terrain_recognition/training/isaac_lab/assets/x2.usd
```

`x2_robot_cfg.assets_available()` will now return true via the USD path. (If you skip this,
it falls back to the URDF importer using `X2_URDF_PATH`.)

---

## 4. Validate: X2 spawns and stands in Isaac Sim (P3-M2-T5 / P3-M3-T1)

```bash
cd ~/IsaacLab
export PYTHONPATH=~/terrain_recognition/training/isaac_lab:~/terrain_recognition/ros2_ws/src/x2_common:$PYTHONPATH
export X2_CONFIG_DIR=~/terrain_recognition/configs
./isaaclab.sh -p ~/terrain_recognition/training/isaac_lab/scripts/spawn_x2.py --headless --seconds 5
```

Expect: the X2 spawns from our `x2_robot_cfg`, stands under PD for a few seconds, and the
script reports a stable base height (the Isaac Lab counterpart of the MuJoCo `stand.py` result
we already validated: base ≈ 0.67 m, upright). This confirms the asset, joint order, default
pose, and PD gains transfer to Isaac Sim.

---

## 5. Train (Phase 4)

> **Status:** `scripts/train.py` wires args + config and launches Isaac Lab; the manager-based
> RL **environment** (connecting our tested observation/reward/termination/curriculum cores to
> Isaac Lab's managers) is the next implementation step — do it once §4 confirms the asset is
> good in Isaac Sim. Then:

```bash
export PYTHONPATH=~/terrain_recognition/training/isaac_lab:$PYTHONPATH
./isaaclab.sh -p ~/terrain_recognition/training/isaac_lab/x2_locomotion/scripts/train.py \
    --task standing --num-envs 4096
```

A6000 (48 GB) comfortably runs 4096+ envs for the height-map policy (camera off). Curriculum:
standing → flat_walk → rough → stairs. Checkpoints + ONNX export (`export_onnx.py`) follow the
roadmap §8.12 / §8.11 graduation gate.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `isaacsim` import fails / GLX errors | run with `--headless`; ensure driver ≥ 535 (`nvidia-smi`) |
| Out of disk during install | need ≥ 80 GB; clear `~/.cache`, resize the Brev volume |
| `x2_common` / config not found | set `PYTHONPATH` + `X2_CONFIG_DIR` as in §4 |
| Meshes missing on convert | run `tools/fetch_x2_assets.sh <zip>` (§2) |
| Robot explodes on spawn | check `X2_USD_PATH`; try the simple-collision URDF; lower PD in `x2_actuator_cfg` |

See [README.md](README.md) for the training layout and [../../docs/training_method.md](../../docs/training_method.md) for the method.
