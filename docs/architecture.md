# Architecture

System architecture for the X2 terrain-aware locomotion stack. Source of truth for the plan
is [../x2_terrain_stair_climbing_roadmap.md](../x2_terrain_stair_climbing_roadmap.md).

## Design principles

- **Modular by mode.** Perception-only, velocity-control, and joint-policy modes can run
  independently. A failure or absence in one layer never silently enables a riskier layer.
- **Configs over constants.** Thresholds, ROIs, limits, and gains live in `configs/*.yaml`,
  loaded via `x2_common.config_loader` — never hardcoded in node source.
- **Fail closed.** Missing or stale inputs resolve to *stop*, not *continue*.
- **Deterministic sim-to-real.** Observation ordering and normalization are identical
  between training and deployment; ONNX output is validated numerically against PyTorch.

## Layered data flow

```
   sensors (RGB-D, LiDAR, IMU)                         AimDK / robot HAL
        │                                                     ▲
        ▼                                                     │
┌──────────────────────┐                                      │
│  x2_terrain_perception│  Phase 1 — perception only          │
│  pointcloud_projector │                                      │
│  ground_plane_estimator                                      │
│  heightmap_node       │── /x2/terrain/heightmap ─┐           │
│  slope/stair/gap det. │── /x2/terrain/stair_estimate         │
│  terrain_classifier   │── /x2/terrain/status ────┤           │
└──────────────────────┘                           │           │
                                                    ▼           │
                                      ┌──────────────────────┐  │
                                      │  x2_safe_locomotion   │  │  Phase 2
                                      │  velocity_adapter     │  │  velocity only
                                      │  command_smoother     │  │
                                      │  safety_supervisor    │──┘ /aima/mc/locomotion/velocity
                                      │  emergency_stop_node  │     (source registered first)
                                      └──────────────────────┘
                                                    ▲
                                      ┌──────────────────────┐
                                      │  x2_policy_runtime    │  Phases 5–6
                                      │  observation_builder  │  joint policy
                                      │  onnx_policy_runner   │  GATED by approval flag;
                                      │  action_filter        │  dry-run / debug only
                                      │  policy_safety_superv. │  until Phase 5 review
                                      └──────────────────────┘
```

`x2_common` is the shared library used by every node (config loading, QoS profiles, frame
transforms, time sync, structured logging, safety-limit checks). `x2_terrain_msgs` defines
the custom message contracts that link perception → locomotion.

## Packages

| Package | Phase | Responsibility |
|---------|:-----:|----------------|
| `x2_common` | 0 | Shared utilities; no robot I/O of its own. |
| `x2_terrain_msgs` | 1 | Custom message + service definitions. |
| `x2_terrain_perception` | 1 | Cloud → height map → terrain classification. |
| `x2_safe_locomotion` | 2 | Terrain-gated velocity adapter + safety layer. |
| `x2_policy_runtime` | 5–6 | ONNX policy execution under safety supervision. |

## Timing targets

| Stage | Rate |
|-------|------|
| Perception pipeline | 8–10 Hz |
| Physics (sim) | 200 Hz |
| Policy (sim + deploy) | 50 Hz (decimation 4) |

## Safety-critical invariants

- The joint-policy layer (`x2_policy_runtime`) must not publish to any leg-joint command
  topic while `REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED` is false. See [../SAFETY.md](../SAFETY.md).
- `velocity_adapter` must command zero within the watchdog timeout when perception is stale.
- Every real-robot motion path carries: velocity/action clamps, state watchdog, IMU fall
  detector, timeout detector, operator e-stop, logging.
