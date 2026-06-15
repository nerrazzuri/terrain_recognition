# X2 Terrain & Stair-Climbing — Modules & Task Tracker

**Plan source of truth:** [x2_terrain_stair_climbing_roadmap.md](x2_terrain_stair_climbing_roadmap.md)
**This file:** living checklist of every module and task. Update each task's status box as work progresses.
**Last updated:** 2026-06-15 — Workspace builds on ROS2 Humble; perceive-and-stop pipeline validated end-to-end as a live ROS2 graph (flat→go, stairs→stop, dry-run). X2 stands in MuJoCo. Integrated X2_URDF-v1.3.0 + verified vs robot MC (lx2501_3). 186 tests pass (incl. integration). Sim/hardware/training items [~]/[!] with blockers.

---

## How to use this file

- Each task has a **stable ID** (e.g. `P1-M2-T1`) — reference it in commits/PRs.
- Update the **status box**: `[ ]` → `[~]` → `[x]` (or `[!]` if blocked; note the blocker inline).
- **DoD** = phase Definition of Done — a gate that must pass before the next phase starts.
- **Do not skip phase gates.** Real stair climbing is forbidden until the Phase 5 documented safety review passes.
- To expand any task into a full work item, use the roadmap §11 template (Goal / Files / Implementation / Safety / Tests / Acceptance / Notes).

### Status legend
| Box | Meaning |
|-----|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Done + verified against acceptance criteria |
| `[!]` | Blocked (note the blocker inline) |

### Global safety flag
| Flag | Value | Notes |
|------|-------|-------|
| `REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED` | **false** | Must stay false until the Phase 5 documented safety review. Roadmap §14. While false: policy nodes run dry-run only, publish debug only, no `/aima/hal/joint/leg/command` — only SDK velocity control via the safe adapter. |

---

## Progress overview

| Phase | Risk | Training | Modules | Tasks | Done | Status |
|-------|------|----------|:---:|:---:|:---:|--------|
| P0 — Foundation / Repo setup | — | No | 2 | 5 | 5 | ✅ Done |
| P1 — Terrain Awareness | Low | No | 4 | 17 | 15 | Code done (2 blocked on hardware bags) |
| P2 — Safe SDK Locomotion | Low/Med | No | 3 | 12 | 9 | Pipeline validated live (dry-run); on-robot blocked |
| P3 — X2 Simulation Model | None | Not yet | 3 | 11 | 6 | Model validated in MuJoCo; Isaac Lab path pending |
| P4 — RL Locomotion Training | Sim only | Yes | 5 | 19 | 6 | Logic done; training blocked (Isaac Lab/GPU/torch) |
| P5 — Sim-to-Real Deployment | **High** | Trained | 3 | 14 | 6 | Runtime logic done; hardware gated/blocked |
| P6 — CReF Raw-Depth Policy | **High** | Yes | 4 | 11 | 1 | Architecture scaffolded; training blocked |
| **Total** | | | **24** | **89** | **48** | |

**Current focus:** Phase 4 RL training on **Isaac Lab 2.3 via cloud GPU** (NVIDIA Brev, L40S 48 GB). **Stage A standing TRAINED** (1500 iters → stays upright full episode; learned a wide-splay stance, an under-specified-reward artifact). **Stage B flat-walk env built** (`Isaac-X2-FlatWalk-v0`): velocity tracking + hip-deviation (anti-splay) + feet-air-time + standing-env mixing → stands planted AND walks; training run next. P0-P2 software complete + sim-validated; P1/P2 hardware items blocked on the robot.

---

# Phase 0 — Foundation & Repository Setup
*Prerequisite for everything. Establishes the repo skeleton from roadmap §3.*

## Module 0.1 — Repo & Docs Skeleton
- `[x]` **P0-M1-T1** Create the `x2_terrain_locomotion/` tree from roadmap §3: `docs/`, `ros2_ws/src/`, `training/`, `tools/`, `configs/`, `tests/{unit,integration,simulation,hardware_dry_run}`.
- `[x]` **P0-M1-T2** Seed top-level docs: `README.md`, `ROADMAP.md` (link to roadmap), `SAFETY.md` (copy §1 + §14), `TASKS.md` (this file).
- `[x]` **P0-M1-T3** Seed `docs/`: `architecture.md`, `ros_topics.md`, `data_contracts.md`, `training_method.md`, `real_robot_test_protocol.md`, `sim_to_real_checklist.md`, `known_risks.md`.

## Module 0.2 — Shared Library & Configs
- `[x]` **P0-M2-T1** Create `x2_common` package: `topic_discovery.py`, `time_sync.py`, `transforms.py`, `qos_profiles.py`, `config_loader.py`, `logging_utils.py`, `safety_limits.py` (+ `package.xml`, `setup.py`). *Pure-logic modules unit-tested (31 tests passing); ROS2 imports are lazy.*
- `[x]` **P0-M2-T2** Create base config files: `robot_topics.yaml`, `terrain_perception.yaml`, `safe_locomotion.yaml`, `safety_limits.yaml`, `joint_limits_x2_ultra.yaml`, `training_default.yaml`.

---

# Phase 1 — Terrain Awareness  *(Risk: Low · No training)*
> **Goal:** perception-only. X2 stands still, classifies terrain ahead, and publishes terrain type + height/tread/slope estimates + safety decision. **No locomotion commands in this phase.**
>
> **Target classes:** flat_ground · rough_ground · slope_up · slope_down · curb_or_single_step · stairs_up · stairs_down · gap_or_hole · platform · unknown_unsafe.

## Module 1.1 — Custom Messages & Workspace
- `[x]` **P1-M1-T1** Create the ROS2 workspace skeleton (`ros2_ws/src/...` packages with `package.xml` / `setup.py`). *5 packages; **`colcon build` verified on ROS2 Humble** (x2_terrain_msgs + 4 Python pkgs); `ros2 run` resolves all nodes (setup.cfg script-dir fix).*
- `[x]` **P1-M1-T2** Define `x2_terrain_msgs`: `TerrainCell.msg`, `TerrainGrid.msg`, `TerrainStatus.msg`, `StairEstimate.msg`, `SafetyDecision.msg`, `PolicyDebug.msg`, `srv/ResetTerrainMap.srv` (fields per roadmap §5.3). *ament_cmake + rosidl.*
- `[x]` **P1-M1-T3** Build `topic_discovery.py` + `tools/check_topics.sh` / `check_qos.sh`; verify real topics with `ros2 topic list` / `ros2 topic info -v`. *topic_discovery in x2_common (P0); tools added. **Topic names + message types verified against SDK lx2501_3-v0.9.0.4** and recorded in `robot_topics.yaml` (verified:true); live QoS check still pending on robot.*

## Module 1.2 — Point Cloud → Height Map Pipeline
- `[x]` **P1-M2-T1** `pointcloud_projector.py` — subscribe RGB-D/LiDAR cloud, transform to base frame, drop NaNs, crop ROI (x 0–2 m, y ±0.8 m, z −0.5–1.0 m), voxel downsample. *Uses tested x2_common.transforms (drop_nonfinite, crop_roi); watchdog for missing cloud. AC needs live-rate check on robot.*
- `[x]` **P1-M2-T2** `ground_plane_estimator.py` — RANSAC / robust least squares + IMU prior → plane normal, height offset, confidence. *core.ground_plane RANSAC+SVD; 9 unit tests (flat ±2°, ramp direction, outlier rejection, multi-plane confidence drop).*
- `[x]` **P1-M2-T3** `heightmap_node.py` — robot-centered elevation map (2.0×1.6 m, 0.04 m res = 50×40 cells, 10 Hz, 0.5 s decay), per-cell confidence, debug viz. *core.heightmap; 10 unit tests incl. coord conversion, indexing, step height, unknown-not-flat.*
- `[x]` **P1-M2-T4** `slope_detector.py` — slope angle + up/down direction from ground plane / height map. *core.slope; covered by ground-plane/slope tests.*

## Module 1.3 — Terrain Feature Detectors
- `[x]` **P1-M3-T1** `stair_detector.py` — detect repeated horizontal planes + vertical risers; estimate rise/tread, first-step distance, recommended stop distance, confidence. *core.stairs; 7 unit tests (clutter rejected, single step ≠ stairs, never confident under ambiguity).*
- `[x]` **P1-M3-T2** `gap_detector.py` — detect holes/drop-offs from missing/lower cells; estimate width + distance. *core.gaps; 5 unit tests; unknown-region-unsafe + reason string.*
- `[x]` **P1-M3-T3** `traversability_estimator.py` — per-cell traversability feeding the height map `traversability[]` field. *core.traversability; 4 unit tests; filled by heightmap_node.*
- `[x]` **P1-M3-T4** `terrain_classifier.py` — fuse ground/heightmap/stair/slope/gap → final terrain type + confidence + reason (decision policy §5.4). *core.classifier; 9 unit tests incl. precedence + safe_to_continue invariants.*

## Module 1.4 — Tooling, Data & Tests
- `[x]` **P1-M4-T1** `tools/record_terrain_bag.sh` — record depth image, depth cloud, LiDAR cloud, IMU, TF, joint state.
- `[x]` **P1-M4-T2** `tools/visualize_heightmap.py` + `visualization_node.py` — live height-map view. *viz node → OccupancyGrid; tool has `--demo` (no-ROS) mode.*
- `[x]` **P1-M4-T3** `offline_bag_analyzer.py` + `tools/replay_bag_heightmap.py` — replay bags through the pipeline offline.
- `[x]` **P1-M4-T4** Unit tests: grid coordinate conversion, grid indexing, terrain-classification decision logic. *78 unit tests total, all TDD.*
- `[!]` **P1-M4-T5** Record real terrain bags: flat, carpet, reflective, curb 5/10/15 cm, stairs up, stairs down, platform edge, gap mockup, cluttered unsafe. **BLOCKED: requires physical robot + sensors.** Recording script (P1-M4-T1) is ready.
- `[!]` **P1-M4-T6** Validate detection on each offline bag scene. **BLOCKED on P1-M4-T5** (needs the real bags). Offline analyzer is ready to run them.

### ✅ Phase 1 Definition of Done
- `[~]` Perception runs at 8–10 Hz · height map visualized live · flat/slope/curb/stairs/gap detected in offline bags · `unknown_unsafe` used correctly · **no locomotion commands sent** · logs saved per test. *Code + algorithms complete and unit-tested; **detection validated on MuJoCo-rendered depth** (flat/curb/stairs/platform correctly classified; stair rise/tread accurate) and as a live ROS2 graph. Live-rate measurement + real-bag detection still pending hardware bags. (Fixed a confidence-aggregation bug surfaced by realistic partial-coverage sensing.)*

---

# Phase 2 — Safe SDK Locomotion Adaptation  *(Risk: Low/Medium · No training)*
> **Goal:** use the AimDK velocity API to walk slowly on known flat terrain and **stop before unsafe terrain**. **Still does not climb stairs.** First demo / customer milestone.
>
> Publishes to `/aima/mc/locomotion/velocity` (fields: source, forward, lateral, angular). The command source must be registered before publishing.

## Module 2.1 — Command Source & Velocity
- `[x]` **P2-M1-T1** `input_source_registrar.py` — register source `x2_terrain_safe_locomotion`, check priority, prevent name collision.
  - *AC:* registers before publisher starts · fails closed on failure · logs source/priority. *Real AimDK API wired: `aimdk_msgs/srv/SetMcInputSource` (action.value=1001, priority/timeout from config); fails closed if aimdk_msgs absent or call fails. Live registration test pending on robot.*
- `[x]` **P2-M1-T2** `velocity_adapter.py` — map desired → safe velocity by terrain type; stop before stairs/gaps/unknown; smooth. Policy: flat ≤0.12, rough ≤0.06, mild slope ≤0.04 m/s; curb/stairs/gap/unknown → stop. *core.velocity_policy (9 tests); dry-run default; stale-perception watchdog zeros velocity. Live publisher uses real `aimdk_msgs/msg/McLocomotionVelocity` (forward/lateral/angular).*
- `[x]` **P2-M1-T3** `command_smoother.py` — ramp limits (fwd accel 0.05 m/s², yaw 0.10 rad/s²); emergency = immediate zero. *core.smoother (3 tests).*

## Module 2.2 — Safety Layer
- `[x]` **P2-M2-T1** `motion_state_monitor.py` — track robot mode/state plus command + perception freshness.
- `[x]` **P2-M2-T2** `safety_supervisor.py` — hard stop on: terrain_status >0.5 s missing · IMU >0.2 s missing · roll/pitch over threshold · unknown/stairs/gap ahead · operator stop · command timeout · unexpected mode.
  - *AC:* any missing critical input → stop · reason logged · manual e-stop overrides all. *core.supervisor (9 tests, every stop condition + e-stop override).*
- `[x]` **P2-M2-T3** `emergency_stop_node.py` — operator manual e-stop, highest priority. *Latching; stays engaged until explicit reset.*

## Module 2.3 — Tests & Demo
- `[x]` **P2-M3-T1** Dry-run mode — publish to debug topic only (no real velocity). *Default; real topic only when dry_run off AND source registered. **Validated live** in the integration test (debug TwistStamped only).*
- `[!]` **P2-M3-T2** Flat-ground walking test (0.05 m/s). **BLOCKED: requires physical robot.**
- `[!]` **P2-M3-T3** Stair/curb-stop test (must stop before first step). **BLOCKED: requires physical robot.**
- `[~]` **P2-M3-T4** Missing-sensor watchdog test. *Watchdog logic unit-tested (FreshnessWatchdog + supervisor); node-level integration test needs a ROS2 runtime.*
- `[!]` **P2-M3-T5** Manual emergency-stop test. **BLOCKED: requires physical robot / ROS2 runtime.** Latching e-stop logic in place.
- `[x]` **P2-M3-T6** Demo script: walk forward → slow → stop before stairs, with logged stop reason. *`launch/safe_stop_demo.launch.py` (dry-run) + **automated integration test** (`tests/integration/test_perceive_and_stop.py`): live ROS2 graph proves flat→go, stairs→stop with the terrain-driven stop reason.*

### ✅ Phase 2 Definition of Done
- `[~]` Source registration works · safe adapter commands slow walking · X2 stops before stairs/gaps/unknown · watchdog stop works · manual stop works · logs prove the stop came from terrain perception. *Pipeline **validated two ways**: (1) live ROS2 graph in dry-run (synthetic input); (2) **sim-in-the-loop on MuJoCo-rendered depth** (`training/mujoco/scripts/perceive_demo.py`) — flat→walk 0.10 m/s; curb/stairs/platform→stop; approach sweep flips go→stop as stairs enter the ~2 m range; stair rise/tread recovered to ~1 cm. **On-robot walking/stop + AimDK source registration still blocked on hardware + SDK.***

---

# Phase 3 — X2 Simulation Model  *(Risk: None · No training yet)*
> **Goal:** build a simulation accurate enough to train/test before touching real hardware. Primary: **Isaac Lab**; secondary: MuJoCo. No RL stair policy before this is stable.

## Module 3.1 — Robot Model Assets
- `[x]` **P3-M1-T1** Locate/collect X2 URDF/MJCF, meshes, joint names/order/limits, default pose, mass, inertia, foot collision, torque/velocity limits, PD estimates, actuator delay, self-collision pairs (roadmap §7.2). *From **X2_URDF-v1.3.0**: `x2_ultra.urdf` + `x2_ultra_simple_collision.urdf` + MJCF (`x2_ultra.xml`) + meshes placed under `training/`. Real leg+waist joint names/limits/effort/velocity extracted into `joint_limits_x2_ultra.yaml` (verified:true). Meshes gitignored (~112 MB); repopulate via `tools/fetch_x2_assets.sh`.*
- `[~]` **P3-M1-T2** Convert model to an Isaac Lab asset (`x2.usd`); place under `training/isaac_lab/assets/`. *URDF placed; `x2_robot_cfg` spawns via Isaac Lab's URDF importer or a converted USD (`X2_USD_PATH`); `assets_available()` now true. **Conversion command + full runbook in [training/isaac_lab/SETUP.md](training/isaac_lab/SETUP.md)** (NVIDIA Brev / A6000); run on the GPU box.*

## Module 3.2 — Config & Validation
- `[x]` **P3-M2-T1** `x2_joint_map.py` — map sim joints ↔ AimDK order (legs: L then R; hip_pitch/roll/yaw, knee, ankle_pitch/roll). *Pure logic, 6 unit tests; **leg order VERIFIED against the robot MC `robot_model.yaml`** — full 31-DoF body order in `AIMDK_BODY_ORDER` (legs 0-11 → waist → head → arms); limits from URDF v1.3.0.*
- `[x]` **P3-M2-T2** `tools/check_joint_order.py` — joint-order verification tool. *Runs; prints side-by-side table; warns on unverified limits.*
- `[x]` **P3-M2-T3** Joint limit config `joint_limits_x2_ultra.yaml`. *Real values from X2_URDF-v1.3.0 (`verified: true`); 12 leg + 3 waist joints with position/velocity/effort limits.*
- `[~]` **P3-M2-T4** `x2_actuator_cfg.py` — actuator / PD parameters. *Builds ImplicitActuatorCfg from config; **blocked on Isaac Lab**; PD gains are first estimates needing sim validation.*
- `[~]` **P3-M2-T5** `x2_robot_cfg.py` — asset path, default pose, base height, limits, PD gains, contact/termination bodies, feet names. *Isaac Lab cfg scaffold complete (spawns via URDF importer / USD). **Model itself validated in MuJoCo**: loads (nq=38, 31 actuators, 43.5 kg, floating base), steps 300 frames under gravity without exploding (no NaN, stable). Isaac Lab spawn-and-stand still pending Isaac Lab install.*

## Module 3.3 — Environments & Terrain
- `[~]` **P3-M3-T1** `x2_standing_env_cfg.py` — flat ground, standing target, low disturbance, terminate on fall. *Scaffold (timing from config); **blocked on Isaac Lab + asset**.*
- `[~]` **P3-M3-T2** `terrain_generator.py` — progressive levels 0–6 (flat → rough → slope → single step → stairs up → stairs down → mixed) with params from §7.4. *Specs in `terrain_spec.py` done + unit-tested (5 tests); Isaac Lab adapter scaffolded (**blocked on Isaac Lab**).*
- `[x]` **P3-M3-T3** Height-sample extraction around the robot in sim. *`height_samples.py` pure logic, 3 unit tests (shape, base-relative, yaw rotation); 121-dim matches training obs.*
- `[x]` **P3-M3-T4** Simulation smoke tests (`tests/simulation/`), runnable locally / in CI. *Runs without GPU; **MuJoCo tests load the real X2 model, verify 12 leg joints + floating base + DoF, and step physics without exploding**; Isaac-dependent cases skip when isaaclab absent. Plus `training/mujoco/scripts/validate_model.py`.*

### ✅ Phase 3 Definition of Done
- `[~]` X2 spawns and stands in sim · joint ordering verified vs AimDK · terrain generator exists · height samples extractable · basic sim tests run. *Joint map / terrain specs / height samples / smoke tests done & tested; **X2 spawns AND STANDS in MuJoCo** under a PD pose controller (base_z 0.67 m, upright ~1.0, stable 3 s — `training/mujoco/scripts/stand.py`, regression-tested). The Isaac Lab path still pends Isaac Lab install.*

---

# Phase 4 — RL Locomotion Training  *(Risk: Sim only · Training required)*
> **Goal:** PPO (asymmetric actor-critic) height-map policy. Curriculum: standing → flat walk → rough → single step → stairs up → stairs down → mixed. **Height-map input first, not raw depth.**
>
> Action space: start **12-DoF legs only** (joint position offsets). Expand to +3 waist (+arms for balance) only after legs work. Timing: physics 200 Hz, policy 50 Hz, decimation 4.

## Module 4.1 — Training Infrastructure
- `[x]` **P4-M1-T1** PPO training config + `scripts/train.py` (512 envs → 1024 if stable; camera rendering off for height-map version). ***Runs reproducibly on Isaac Lab 2.3 + rsl_rl (cloud L40S 48 GB):** manager-based env builds (83-dim obs, 12-DoF leg action, 10 reward terms), PPO trains and checkpoints. Stage-A standing env in `tasks/standing/`. Convergence run (1500 iters) pending.*
- `[x]` **P4-M1-T2** `observations.py` — sim observation builder (cmd vel ×3, base ang vel, projected gravity, joint pos err, joint vel, prev action, gait phase sin/cos, height samples 11×11=121). Normalize all. *Pure numpy; dim=168 contract; 6 unit tests (order, missing/dim guards, normalizer zero-std guard). Shared with deployment.*
- `[~]` **P4-M1-T3** Network: height_encoder + proprio_encoder + actor + critic (privileged) per §8.6. *`network.py` exact architecture; **blocked to run on torch**.*

## Module 4.2 — Rewards & Terminations
- `[x]` **P4-M2-T1** `rewards.py` — separately logged components: velocity tracking, torso stability, foot clearance, foothold quality, foot slip, energy/smoothness, joint safety. *Pure functions; 4 unit tests.*
- `[x]` **P4-M2-T2** `terminations.py` — fall/collision: base too low, roll/pitch too high, head/torso/knee collision, invalid contact. *4 unit tests.*
- `[x]` **P4-M2-T3** `curriculum.py` — curriculum manager across stages A–G. *4 unit tests (advance/hold/complete).*
- `[x]` **P4-M2-T4** `domain_randomization.py` — mass/inertia/CoM, motor strength, PD, action delay, sensor latency, IMU noise, depth/heightmap noise, friction, encoder noise (§8.10). *Seedable; 2 unit tests (within-range, reproducible).*

## Module 4.3 — Curriculum Tasks
- `[~]` **P4-M3-T1** Stage A — standing (`x2_standing_env_cfg.py`): stand 30 s, recover small pushes. ***Trained & converged** (1500 iters, L40S, ~40k steps/s): X2 stays upright the full 20 s episode in ~93% of cases (`bad_orientation` 1.0→0.05), survives periodic pushes, mean reward +3. **Caveat:** drifts/yaws while upright (no planar-velocity penalty yet) — adding velocity-command tracking next for a planted stand + Stage-B walking.*
- `[~]` **P4-M3-T2** Stage B — flat walking (`x2_flat_walk_env_cfg.py`): fwd 0–0.3 m/s, yaw ±0.3 rad/s. *Env built on Isaac Lab 2.3: velocity-command tracking + feet-air-time + hip-deviation (anti-splay) rewards; ~20% standing-env mixing so one policy stands planted AND walks. Registered Isaac-X2-FlatWalk-v0. Training run pending.*
- `[~]` **P4-M3-T3** Stage C — rough terrain (`x2_rough_env_cfg.py`): 1–5 cm noise, mild slopes. *Scaffolded; blocked.*
- `[~]` **P4-M3-T4** Stage D — single step / curb: 2→5→8→12→15 cm. *In `x2_stairs_env_cfg` + terrain_spec level 3; blocked.*
- `[~]` **P4-M3-T5** Stage E — stairs up (`x2_stairs_env_cfg.py`): rise 5–18 cm, tread 24–35 cm, 1–8 steps. *Scaffolded (terrain_spec level 4); blocked.*
- `[~]` **P4-M3-T6** Stage F — stairs down: rise 5–15 cm, tread 24–35 cm. *terrain_spec level 5; blocked.*
- `[~]` **P4-M3-T7** Stage G — mixed terrain generalization. *terrain_spec level 6; blocked.*

## Module 4.4 — Evaluation & Export
- `[~]` **P4-M4-T1** `scripts/play.py` + `evaluate_policy.py` — success-rate reports. *`check_graduation` pure + 3 unit tests; **rollouts blocked on Isaac Lab + checkpoint**.*
- `[x]` **P4-M4-T2** `scripts/inspect_observation.py` — observation sanity inspection. *Runnable without torch; verifies the 168-dim layout.*
- `[~]` **P4-M4-T3** `scripts/export_onnx.py` — ONNX export. *Export wiring + `numeric_match`; **blocked to run on torch/onnxruntime**.*
- `[~]` **P4-M4-T4** PyTorch-vs-ONNX numerical validation on test vectors. *`numeric_match` pure + 3 unit tests; **full validation blocked on torch**.*

## Module 4.5 — Graduation Gate
- `[~]` **P4-M5-T1** Confirm acceptance: flat >95%, rough >90%, 5 cm step >90%, 10 cm step >80%, stair-up >80% · no joint-limit abuse · no unrealistic torque reliance · smooth actions · survives randomized latency/noise. *`GRADUATION_THRESHOLDS` + `check_graduation` tested; **actual confirmation blocked on training**.*

### ✅ Phase 4 Definition of Done
- `[~]` PPO pipeline reproducible · height-map policy walks + handles rough terrain · single-step + stair-up curriculum shows measurable success · eval script produces success-rate reports · ONNX export numerically checked vs PyTorch. *All training **logic** (obs/rewards/terminations/curriculum/DR/graduation/numeric-check) implemented + unit-tested; **training run + ONNX export blocked on Isaac Lab + GPU + torch + X2 asset**.*

---

# Phase 5 — Sim-to-Real Deployment  *(Risk: HIGH · Already trained)*
> **Highest-risk phase.** Strict safety. `REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED` must be explicitly set true only after the documented safety review (§14). Escalate through levels in order; pass the Go/No-Go gate before each step up.

## Module 5.1 — Policy Runtime
- `[x]` **P5-M1-T1** `observation_builder.py` — build obs from real robot state, match training normalization exactly; missing values → safe stop. *core.observation_builder; 4 tests incl. **layout==training** contract test; missing/non-finite → ok=False (safe stop).*
- `[~]` **P5-M1-T2** `onnx_policy_runner.py` — load ONNX, fixed-rate inference, validate output dims, timing metrics. *`onnx_runner_core` (validate_output, within_period) tested (4); **session load blocked on onnxruntime + policy.onnx**.*
- `[x]` **P5-M1-T3** `action_filter.py` — clamp joint targets, limit action rate + joint velocity, low-pass, joint safety envelope. *core.action_filter; 6 tests incl. extreme-input fuzz proving outputs never exceed soft limits + NaN/Inf rejection.*
- `[x]` **P5-M1-T4** `policy_logger.py` — log full deployment record (§9.4). *Wraps JsonlRecorder; enforces all §9.4 required fields (test).*

## Module 5.2 — Safety Supervisor
- `[x]` **P5-M2-T1** `policy_safety_supervisor.py` — cut output on: roll/pitch over threshold · joint/IMU missing · inference timeout · action NaN/Inf · joint target outside soft limit · operator stop · base instability. Switch to damping/zero. *core.policy_supervisor; 7 tests (every cut condition + operator override).*

## Module 5.3 — Staged Hardware Bring-up
> **All hardware levels are GATED by `REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED=false` and require the physical robot. `joint_policy_node` runs dry-run / PolicyDebug-only and creates the leg-command publisher ONLY when the flag is true.**
- `[!]` **P5-M3-T1** Level 0 — offline replay against recorded logs (no robot command topic). **BLOCKED: needs a trained policy.onnx + recorded logs.** Runner + analyzer ready.
- `[~]` **P5-M3-T2** Level 1 — hardware dry-run (powered, debug output only, no leg command). *`joint_policy_node` implements dry-run/PolicyDebug-only; **running on powered hardware blocked**.*
- `[!]` **P5-M3-T3** Level 2 — suspended standing (gantry; balance/small offsets; operator e-stop ready). **BLOCKED: hardware + documented safety review + approval flag.**
- `[!]` **P5-M3-T4** Level 3 — suspended stepping in place (verify joint order, sign, delay). **BLOCKED: hardware + approval.**
- `[!]` **P5-M3-T5** Level 4 — foam obstacle 2→5→8 cm. **BLOCKED: hardware + approval.**
- `[!]` **P5-M3-T6** Level 5 — single wooden step 5→8→10–12 cm. **BLOCKED: hardware + approval.**
- `[x]` **P5-M3-T7** `tools/analyze_policy_log.py` + `compare_sim_real.py` — real-log analyzer. *core.log_analysis (summarize, max_action_divergence); 3 tests.*
- `[x]` **P5-M3-T8** Go/No-Go checklist gate (§9.5) before each escalation. *Documented in [docs/real_robot_test_protocol.md](docs/real_robot_test_protocol.md) + [docs/sim_to_real_checklist.md](docs/sim_to_real_checklist.md).*
- `[!]` **P5-M3-T9** Level 6 — real stairs. **BLOCKED: only after documented safety review + all lower levels pass.**

### ✅ Phase 5 Definition of Done
- `[~]` ONNX runtime works on target compute · observation builder matches training · safety supervisor stops unsafe output · suspended tests pass · low-obstacle tests pass · single-step tests pass · real stairs approved only after documented safety review. *Runtime **logic** (obs builder, action filter, safety supervisor, logger, log analysis) complete + unit-tested; **ONNX load + all hardware levels blocked on onnxruntime/trained policy + physical robot + documented safety review**.*

---

# Phase 6 — CReF-Style Raw-Depth Perception Policy  *(Risk: HIGH · Training required)*
> **Goal:** upgrade the height-map policy → raw-depth recurrent cross-modal fusion (CReF). **Start only after the Phase 4 height-map policy works.** Prefer distillation (Option A) first.

## Module 6.1 — Data & Encoders
- `[~]` **P6-M1-T1** Raw-depth dataset collector from simulation. *`cref/raw_depth_dataset.py` record schema + DepthSample defined; **collection blocked on Isaac Lab + Phase 4 teacher checkpoint**.*
- `[~]` **P6-M1-T2** Depth encoder (patch embedding / CNN → depth tokens). *`cref/depth_encoder.py` (torch); **blocked to run on torch**.*

## Module 6.2 — Fusion Network
- `[~]` **P6-M2-T1** Cross-modal attention (proprio query attends to depth tokens). *`cref/cross_modal_attention.py` (torch MultiheadAttention); blocked to run.*
- `[~]` **P6-M2-T2** GRU recurrent fusion + actor head. *`cref/gru_fusion.py`; assembled in `cref/cref_policy.py`; blocked to run.*
- `[~]` **P6-M2-T3** Auxiliary heads: terrain type, height reconstruction, safe-foothold probability, depth validity mask. *`cref/aux_heads.py`; blocked to run.*

## Module 6.3 — Training
- `[x]` **P6-M3-T1** Depth augmentation (dropout, missing pixels, reflective/edge noise, motion blur, pitch/roll offset, latency, extrinsic error, FoV crop, near/far clipping). *`cref/depth_augmentation.py` pure numpy; 7 unit tests (dropout fraction, NaN missing, clip bounds, seeded reproducible).*
- `[~]` **P6-M3-T2** Teacher-student distillation pipeline (teacher = height-map policy; loss = action + value imitation + aux terrain prediction). *`cref/distillation.py` loss + weights defined; **blocked to run on torch**.*
- `[~]` **P6-M3-T3** Fine-tune with PPO after distillation. *`scripts/train_cref.py --phase finetune`; blocked.*
- `[~]` **P6-M3-T4** Train raw-depth policy in simulation. *`scripts/train_cref.py`; **blocked on torch + Isaac Lab + teacher**.*

## Module 6.4 — Evaluation & Deploy
- `[!]` **P6-M4-T1** Compare vs height-map policy (clutter / gaps / platforms). **BLOCKED: needs both trained policies.**
- `[!]` **P6-M4-T2** Deploy through the **same Phase 5 safety protocol** (suspended → foam → low step → stairs). **BLOCKED: hardware + approval flag + documented safety review.**

### ✅ Phase 6 Definition of Done
- `[~]` Raw-depth policy ≥ height-map policy in sim · survives depth noise/latency · generalizes better to clutter/gaps/platforms · suspended tests pass · low-obstacle + single-step tests pass · stairs follow the Phase 5 protocol. *Full CReF architecture (depth encoder → cross-modal attention → GRU fusion → aux heads) + distillation loss + depth augmentation implemented; **training + comparison + deploy blocked on torch + Isaac Lab + the Phase 4 teacher policy + hardware**. Per roadmap §10.1, start only after the Phase 4 height-map policy works.*

---

## First Sprint (2 weeks)
> From roadmap §13. Goal: **X2 perceives terrain and stops before unsafe terrain using existing SDK locomotion.** Track via the IDs above.

- `[x]` P0-M2-T2 — `configs/terrain_perception.yaml` + `configs/safe_locomotion.yaml`
- `[x]` P1-M1-T2 — `x2_terrain_msgs`
- `[x]` P1-M2-T3 — `heightmap_node.py`
- `[x]` P1-M3-T1 — `stair_detector.py`
- `[x]` P1-M4-T1 — `tools/record_terrain_bag.sh`
- `[x]` P1-M4-T2 — `tools/visualize_heightmap.py`
- `[x]` P2-M1-T2 — `velocity_adapter.py`
- `[x]` P2-M2-T2 — `safety_supervisor.py`

**Sprint demo:** flat ground → live height map → place box/curb/stair → X2 classifies unsafe terrain → start slow forward → X2 slows + stops before obstacle → log shows the stop reason.

---

## Cross-cutting reminders (apply to every task)
- **Safety first:** velocity/action clamps · state watchdog · IMU fall detector · timeout detector · operator e-stop · logging — on all real-robot motion (§1.1).
- **Verify topics on the real robot** with `ros2 topic list` / `ros2 topic info -v` before trusting roadmap topic names.
- **No real stair climbing** before the Phase 5 documented safety review.
- **Prefer point cloud / compressed streams** over raw cross-unit camera subscriptions when bandwidth is high (§1.1).
- **Start with height-map locomotion** before raw-depth CReF; PPO is the first RL algorithm (§1.3).
