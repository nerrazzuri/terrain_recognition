# AGENTS.md — Rules for working in this repo

**Project:** Agibot X2 / X2 Ultra terrain-aware locomotion & stair-climbing research stack.
**Plan source of truth:** [x2_terrain_stair_climbing_roadmap.md](x2_terrain_stair_climbing_roadmap.md)
**Living task tracker:** [TASKS.md](TASKS.md)
**Stack:** Python · ROS2 · AimDK · Isaac Lab (primary sim) / MuJoCo (secondary) · PyTorch + PPO · ONNX runtime.

> This is a real humanoid-robot project, not an SDK demo. Hardware can be damaged and people can be hurt. Safety rules below are non-negotiable.

---

## 1. Non-negotiable safety rules (roadmap §1, §14)

- **Real low-level leg policy is forbidden by default.** Treat `REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED = false` as the standing state. While false:
  - policy nodes run **dry-run only** and may publish **debug topics only**;
  - **never** publish to `/aima/hal/joint/leg/command` (or any leg joint command topic);
  - only SDK velocity-level control via the safe velocity adapter is allowed.
- **No real stair climbing** until the Phase 5 documented safety review passes. Phases 1–2 must not climb anything.
- **All real-robot motion** must have: velocity/action clamps · state watchdog · IMU fall detector · timeout detector · operator emergency stop · logging enabled. Do not write motion code that omits any of these.
- **Fail closed.** If a required input is missing/stale (perception, IMU, command freshness), the safe action is to **stop**, never to continue.
- **Don't skip phase gates.** Each phase has a Definition of Done in TASKS.md; do not start a later phase before the prior gate passes. Height-map locomotion comes before raw-depth CReF.

---

## 2. Workflow rules

- **Before coding any task:** find it in [TASKS.md](TASKS.md) by its ID (e.g. `P1-M2-T1`). If it isn't there, add it before starting.
- **Update status boxes** as work moves: `[ ]` → `[~]` → `[x]` (or `[!]` blocked, with the blocker noted inline). Also bump the per-phase **Done** count in the progress-overview table.
- **The roadmap is the plan;** TASKS.md is the checklist. If the two disagree, the roadmap wins — fix TASKS.md to match (or raise it).
- **To expand a task into a full work item**, use the roadmap §11 template: Goal / Files / Implementation / Safety / Tests / Acceptance / Notes.
- **Reference the task ID in commit messages and PRs** (e.g. `P1-M2-T3: heightmap_node initial build`).
- **Follow the repo layout** in roadmap §3. New code goes in the package/dir that already owns that responsibility; don't invent parallel structures.

---

## 3. Robot-interface rules

- **Verify topics on the real robot** with `ros2 topic list` and `ros2 topic info -v` before trusting any topic name from the roadmap — names/QoS must be confirmed, not assumed.
- **Register the locomotion command source** (`x2_terrain_safe_locomotion`) before publishing any velocity command.
- **Prefer point cloud / compressed streams** over raw cross-compute-unit camera subscriptions when bandwidth is high.
- **Joint order is not assumed.** Verify joint names, left/right order, and rad-vs-degree units against AimDK; load joint limits from config, never hardcode them in policy runtime.

---

## 4. Code & testing conventions

- **Language:** Python. ROS2 packages use `package.xml` + `setup.py` and build with `colcon build`; source `install/setup.bash` before running nodes.
- **Configs over constants:** thresholds, ROIs, limits, gains live in `configs/*.yaml`, not in source.
- **Tests:** `pytest`. Unit tests for pure logic (grid/coordinate conversion, classification decisions, action filters); sim tests under `tests/simulation/`; hardware checks under `tests/hardware_dry_run/`. Add/maintain tests for code you change.
- **Determinism for sim-to-real:** observation ordering and normalization must match between training and deployment exactly; validate ONNX output numerically against the PyTorch reference.
- **Logging is part of the feature**, not an afterthought — perception tests and every real-robot test must produce saved logs (see roadmap §9.4 for the required deployment log fields).

---

## 5. When in doubt

- If a request would publish real leg-joint commands, climb real stairs, or disable a safety check while the approval flag is false — **stop and ask.** Do not proceed on assumed authorization.
- If a topic, joint, or limit named in the roadmap can't be verified on the actual robot/SDK, flag it rather than guessing.
