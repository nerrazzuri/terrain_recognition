# Factory `cpgwalk` policy contract (P4-M6-T1)

Decoded from the robot runtime `mc_param/robot/lx2501_3_t2d5/rl/cpgwalk_config.yaml` +
`rl_models/cpgwalkrun_v25_v2.onnx`. This is the interface our **stair policy must match** so it
can (a) be distilled from the factory gait and (b) drive `/aima/hal/joint/*/command` cleanly.

## Model I/O (ONNX)
- **Input** `input`: `[1, 650]` float = **65-dim obs × 10 frame-stack** (flattened, most-recent-last assumed).
- **Output** `output`: `[1, 17]` float = 17-DoF **action** (joint position offsets, see scaling below).
- Same I/O for `cpgstep_v1.onnx` (the blind step policy).

## 17-DoF joint order (obs `seq` == `action_seq`)
```
0  left_hip_pitch    1  left_hip_roll    2  left_hip_yaw    3  left_knee
4  left_ankle_pitch  5  left_ankle_roll  6  right_hip_pitch 7  right_hip_roll
8  right_hip_yaw     9  right_knee      10 right_ankle_pitch 11 right_ankle_roll
12 waist_yaw        13 waist_pitch      14 waist_roll       15 left_shoulder_pitch
16 right_shoulder_pitch
```
(legs 0–11, waist 12–14, shoulder pitch 15–16 — shoulders used for balance.)

## 65-dim observation layout (per-frame; from `CPGWalkConfig.obs_index`)
| slice | dims | content | scale (`obs_scales`) |
|-------|------|---------|----------------------|
| [0:3]   | 3  | `imu_omega` base angular velocity | ang_vel 1.0 |
| [3:6]   | 3  | `imu_euler` roll/pitch/yaw         | quat 1.0 |
| [6:10]  | 4  | `command` (vx, vy, yaw, + 1 — likely gait/height/stand flag; confirm) | lin_vel 2.0 |
| [10:27] | 17 | `pos` joint positions (rel default) | dof_pos 1.0 |
| [27:44] | 17 | `dof_vel` joint velocities | dof_vel 0.05 |
| [44:61] | 17 | `action` previous action | — |
| [61:65] | 4  | `q` **CPG phase / oscillator state** (the gait clock) | — |
Total = 3+3+4+17+17+17+4 = **65**. Stacked ×10 → 650.

## Action → joint targets
`target_dof_pos = default_dof_pos + action * action_scale`, tracked by PD (`kps`/`kds`).
- `action_scale` = `[0.5,0.5,0.5,0.5,0.2,0.02, 0.5,0.5,0.5,0.5,0.2,0.02, 0.2,0.2,0.2, 0.2,0.2]`
- `default_dof_pos` = `[-0.248,0,0,0.5303,-0.2823,0, -0.248,0,0,0.5303,-0.2823,0, 0,0,0, 0,0]`
- `kps` = `[120,120,120,150,40,30, 120,120,120,150,40,30, 160,80,80, 80,80]`
- `kds` = `[5,5,5,5,3,2, 5,5,5,5,3,2, 5,5,5, 4,4]`
- `obs_clip = action_clip = 18.0`
- control `dt = 0.02 s` → **50 Hz** (the `# 10Hz` comment is stale)

## CPG gait params (`CPGWalkConfig.cpgwalk`)
`body_height 0.652`, `swing_height 0.07`, `ankle_height 0.071`, `cpg_t 0.6`, gait modes
(`run_mode`, `step_mode`), `min_velx_command 0.1`, yaw thresholds, etc. The policy is
**CPG-guided**: a central pattern generator supplies the rhythmic phase (`q`, obs [61:65]) and
the network outputs corrective offsets — this is why the factory gait is clean/rhythmic.

## Why this matters for us (the limp)
Our from-scratch v3 walker had **no CPG / gait clock** (our `gait_phase` obs was zeros), so it
found an effective but **asymmetric/limping** gait. To get a natural gait we should either feed
a CPG/clock phase like the factory does, or — preferably — **distill the factory `cpgwalk`**
(which already encodes a good gait) and only learn the terrain/stair delta on top.

## Recovering the exact CPG phase `q` (ground truth) — two confirmed ways
The CPG phase was the one piece generated *inside* the sealed MC. The runtime resolves it:
1. **It's an open-loop clock with known params, not feedback-driven.** `cpg_t: 0.6` s (active gait 2;
   `gait_1_t 0.87`, `gait_2_t 0.6`, `change_cpg_t 1`). So `q` is a deterministic time oscillator we
   reconstruct exactly. Crucially, when we run our OWN loop we *generate* the clock — no need to sync to
   the robot's instantaneous phase; the policy follows whatever consistent 0.6 s clock we feed it.
2. **The robot publishes the policy debug stream:** `/aima/mc/rl/debug` @ **50 Hz** (`use_debug: true`,
   `PublisherManager` → `rl_debug`). Recording it during an MC-driven walk very likely captures the real
   65-obs incl. `q[61:65]`. **Action: record `/aima/mc/rl/debug` (+ `/aima/mc_debug_f64`) next robot
   session**, decode, read true CPG/obs, validate our reconstruction (supersedes joint-command matching).

## Obs/action filtering — required for faithful LIVE replay (`CPGWalkConfig.filter`)
The MC low-pass filters obs + actions before/after the policy. To run cpgwalk *live* in our loop we must
replicate: `imu_vel_cutoff 95`, `imu_rpy_cutoff 95`, per-joint `q_vel_cutoff` + `action_cutoff` arrays,
`filter_x 0.6`, `T 0.5`, command `max_acceleration/deceleration 2.0`. (Less critical for sim-time
distillation, but mandatory for hardware Way-A replay.)

## Implications for the stair policy (Module 4.6)
- Match this **17-DoF action** + **PD scaling** so our output maps 1:1 to `/aima/hal/joint/*/command`.
- Reproduce the **65-obs builder** (incl. a CPG phase) to run the factory policy as a **teacher**.
- For stairs, **append a height-map block** to the obs (the one thing the factory lacks) and
  fine-tune the distilled gait to use it for foot placement.
