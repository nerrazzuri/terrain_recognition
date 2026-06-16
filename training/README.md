# Training (Phases 3–4, 6)

Simulation model + RL training for X2 locomotion. **Primary: Isaac Lab** (GPU-parallel PPO);
secondary: MuJoCo (model bring-up / debugging).

## Do you need Isaac Lab or MuJoCo?

- **Phases 1–2** (terrain perception + safe stop): **no simulator** — pure ROS2/Python.
- **Phase 3+**: yes. **Isaac Lab is required for training** (512–1024 parallel envs need an
  NVIDIA GPU + Isaac Sim). MuJoCo is optional and useful for validating the URDF/contacts
  before committing to Isaac.

## Status

The **pure-logic cores are implemented and unit-tested** (no GPU needed):
- `robots/x2_joint_map.py` — sim↔AimDK joint ordering (+ `tools/check_joint_order.py`)
- `tasks/common/terrain_spec.py` — progressive terrain levels 0–6
- `tasks/common/height_samples.py` — 11×11 height-sample observation extraction

The **Isaac Lab configs are scaffolded** and `import isaaclab` — they are **blocked** until:
1. **Isaac Lab is installed**, and
2. the **X2 robot assets exist** (`assets/x2.usd`, meshes) — P3-M1-T1/T2: extract URDF/MJCF +
   meshes + mass/inertia/limits from the Agibot SDK and convert to USD. Set `X2_USD_PATH` or
   drop the file at `training/isaac_lab/assets/x2.usd`.

The training scripts (`scripts/train.py`, `play.py`, `export_onnx.py`, …) and the reward /
observation / termination / curriculum / domain-randomization modules are implemented against
the Isaac Lab + rsl_rl APIs; **running them is blocked on the above**.

## Layout

```
isaac_lab/
  assets/                       x2.usd / meshes  (P3-M1 — NOT in repo, extract from SDK)
  x2_locomotion/
    robots/   x2_joint_map.py  x2_actuator_cfg.py  x2_robot_cfg.py
    tasks/
      common/   observations.py rewards.py terminations.py curriculum.py
                terrain_generator.py terrain_spec.py height_samples.py
                domain_randomization.py network.py
      standing/ flat_walk/ rough_terrain/ stairs/   env cfgs (Stages A–G)
    scripts/  train.py play.py evaluate_policy.py inspect_observation.py export_onnx.py
mujoco/                          secondary sim (model bring-up)
```

## Running (once Isaac Lab + assets are present)

```bash
python -m x2_locomotion.scripts.train  --task standing   # Stage A
python -m x2_locomotion.scripts.play   --task flat_walk
python -m x2_locomotion.scripts.export_onnx --checkpoint <run>/model.pt
```

Graduation criteria and the curriculum are in [../docs/training_method.md](../docs/training_method.md).
