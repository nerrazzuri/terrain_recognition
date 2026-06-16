# X2 Terrain Locomotion

Terrain-aware locomotion and stair-climbing research stack for the **Agibot X2 / X2 Ultra**
humanoid robot.

> ⚠️ **This is a real humanoid-robot project, not an SDK demo.** Hardware can be damaged
> and people can be hurt. Read [SAFETY.md](SAFETY.md) before running anything against a
> real robot. The non-negotiable safety rules in [AGENTS.md](AGENTS.md) override everything.

## What this is

A staged stack that takes X2 from *seeing* terrain to *safely traversing* it:

1. **Terrain awareness** — classify terrain ahead (flat / rough / slope / curb / stairs /
   gap / platform / unknown) from RGB-D + LiDAR + IMU. Perception only.
2. **Safe SDK locomotion** — use the AimDK velocity API to walk slowly and **stop before
   unsafe terrain**. No stair climbing.
3. **X2 simulation model** — Isaac Lab (primary) / MuJoCo (secondary) X2 environment.
4. **RL locomotion training** — PPO height-map policy through a standing → walking → rough
   → step → stairs curriculum.
5. **Sim-to-real deployment** — ONNX export, staged hardware bring-up under strict safety.
6. **CReF raw-depth policy** — upgrade the height-map policy to raw-depth recurrent fusion.

The recommended **first win** is *terrain detection + safe stop* — the robot should learn
to say "stairs detected, unsafe to proceed" before it ever tries to climb.

## Repository layout

```
docs/        design + protocol docs (architecture, ros_topics, data_contracts, …)
ros2_ws/     ROS2 workspace
  src/x2_common              shared library (config, QoS, transforms, safety limits, …)
  src/x2_terrain_msgs        custom messages (TerrainGrid, TerrainStatus, StairEstimate, …)
  src/x2_terrain_perception  perception pipeline (Phase 1)
  src/x2_safe_locomotion     safe velocity adapter + safety supervisor (Phase 2)
  src/x2_policy_runtime       ONNX policy runtime (Phase 5)
training/    Isaac Lab / MuJoCo training code (Phases 3–4, 6)
tools/       rosbag recording, topic/QoS checks, visualisation, log analysis
configs/     all thresholds, ROIs, limits, gains (configs over constants)
tests/       unit / integration / simulation / hardware_dry_run
```

See [x2_terrain_stair_climbing_roadmap.md](x2_terrain_stair_climbing_roadmap.md) for the
full plan and [TASKS.md](TASKS.md) for the living task tracker.

## Getting started

This is early-stage scaffolding. The current focus is **Phase 0 (foundation)** and the
**First Sprint** (perceive terrain & stop safely) — see [TASKS.md](TASKS.md).

### Building the ROS2 workspace

```bash
cd ros2_ws
colcon build
source install/setup.bash
```

### Running tests

```bash
pytest tests/unit            # pure-logic unit tests (no ROS2 / sim required)
```

## Documentation map

| Doc | Purpose |
|-----|---------|
| [ROADMAP.md](ROADMAP.md) | Pointer to the source-of-truth roadmap |
| [SAFETY.md](SAFETY.md) | Non-negotiable safety rules (read first) |
| [TASKS.md](TASKS.md) | Living task tracker, per-task IDs and status |
| [AGENTS.md](AGENTS.md) | Rules for working in this repo |
| [docs/architecture.md](docs/architecture.md) | System architecture |
| [docs/ros_topics.md](docs/ros_topics.md) | Topic inventory (must be verified on robot) |
| [docs/data_contracts.md](docs/data_contracts.md) | Message field contracts |
| [docs/training_method.md](docs/training_method.md) | RL training method |
| [docs/real_robot_test_protocol.md](docs/real_robot_test_protocol.md) | Hardware test protocol |
| [docs/sim_to_real_checklist.md](docs/sim_to_real_checklist.md) | Sim-to-real go/no-go |
| [docs/known_risks.md](docs/known_risks.md) | Known risks register |
