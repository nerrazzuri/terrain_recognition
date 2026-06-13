# X2 Terrain Understanding & Stair-Climbing Roadmap

**Project:** Agibot X2 / X2 Ultra terrain-aware locomotion and stair-climbing research stack  
**Target reader:** coding agent / robotics implementation engineer  
**Status:** planning document for task generation  
**Recommended first demo:** X2 detects stairs/gaps/unsafe terrain and stops safely before them  
**Recommended long-term goal:** trained perceptive locomotion policy that can climb low obstacles, curbs, and stairs after simulation validation

---

## 0. Executive Verdict

This is a serious humanoid locomotion project, not a simple SDK app.

The public AimDK interface gives access to useful pieces:

- velocity-level locomotion command through `/aima/mc/locomotion/velocity`;
- RGB-D depth, RGB-D point cloud, LiDAR point cloud, and IMU data;
- joint command/state topics for leg, waist, arm, and head groups.

However, the SDK does **not** provide complete intelligent stair-climbing behavior as a one-line API. The safest route is a staged architecture:

1. terrain awareness;
2. safe SDK locomotion adaptation;
3. X2 simulator construction;
4. reinforcement-learning locomotion training;
5. sim-to-real deployment;
6. advanced CReF-style raw-depth perceptive locomotion.

Do **not** attempt direct real-stair climbing before simulation and safety validation. The first commercial-looking milestone should be terrain detection + safe stop. The robot should first learn to say: **“stairs detected, unsafe to proceed”**, not immediately climb.

---

## 1. Core Constraints and Assumptions

### 1.1 Hard safety constraints

- No real stair climbing during Phase 1 or Phase 2.
- No direct low-level leg joint policy on a free-standing robot until:
  - simulator policy is stable;
  - ONNX export is verified;
  - joint ordering is verified;
  - safety supervisor is active;
  - robot is suspended or physically protected;
  - emergency stop is tested.
- No raw camera subscriptions across compute units when bandwidth is high. Prefer point cloud or compressed streams when possible.
- All real-robot motion must have:
  - velocity/action clamps;
  - state watchdog;
  - IMU fall detector;
  - timeout detector;
  - operator emergency stop;
  - logging enabled.

### 1.2 Engineering assumptions

- X2 runs ROS2 and AimDK-compatible interfaces.
- The coding agent can access the robot workspace or a cloned development repo.
- Exact topic availability must be verified using ROS2 discovery on the target robot.
- Robot model files may exist inside the extracted Agibot folder, but mass/inertia/contact/actuator details may still need validation.
- Training will likely happen on a PC with NVIDIA GPU. RTX 4060 is acceptable for smaller experiments but not ideal for massive parallel simulation.

### 1.3 Research assumptions

- Start with height-map / elevation-map locomotion before raw-depth CReF.
- PPO is the first RL algorithm to implement.
- CReF-style raw-depth cross-modal recurrent fusion is Phase 6, after height-map policy works.
- The long-term stack should be modular so perception-only, velocity-control, and joint-policy modes can run separately.

---

## 2. Six-Phase Plan Overview

| Phase | Name | Main Goal | Robot Risk | Training Needed? | Primary Output |
|---|---|---:|---:|---:|---|
| 1 | Terrain Awareness | Detect terrain using RGB-D/LiDAR/IMU | Low | No | height map + terrain classification |
| 2 | Safe SDK Locomotion | Slow/stop based on terrain using existing locomotion API | Low/Medium | No | safe velocity adapter |
| 3 | X2 Simulation Model | Build validated X2 simulation | None | Not yet | Isaac Lab/MuJoCo X2 environment |
| 4 | RL Locomotion Training | Train walking → rough terrain → steps → stairs | Simulation only | Yes | PPO height-map policy |
| 5 | Sim-to-Real Deployment | Export ONNX and test on real X2 safely | High | Already trained | real robot suspended/low-step trials |
| 6 | Advanced CReF-Style Policy | Upgrade to raw-depth recurrent fusion | High | Yes | raw-depth perceptive locomotion policy |

---

## 3. Recommended Repository Structure

```text
x2_terrain_locomotion/
  README.md
  ROADMAP.md
  SAFETY.md
  TASKS.md

  docs/
    architecture.md
    ros_topics.md
    data_contracts.md
    training_method.md
    real_robot_test_protocol.md
    sim_to_real_checklist.md
    known_risks.md

  ros2_ws/
    src/
      x2_common/
        package.xml
        setup.py
        x2_common/
          topic_discovery.py
          time_sync.py
          transforms.py
          qos_profiles.py
          config_loader.py
          logging_utils.py
          safety_limits.py

      x2_terrain_msgs/
        msg/
          TerrainCell.msg
          TerrainGrid.msg
          TerrainStatus.msg
          StairEstimate.msg
          SafetyDecision.msg
          PolicyDebug.msg
        srv/
          ResetTerrainMap.srv

      x2_terrain_perception/
        package.xml
        setup.py
        x2_terrain_perception/
          heightmap_node.py
          pointcloud_projector.py
          ground_plane_estimator.py
          stair_detector.py
          gap_detector.py
          slope_detector.py
          terrain_classifier.py
          traversability_estimator.py
          visualization_node.py
          offline_bag_analyzer.py

      x2_safe_locomotion/
        package.xml
        setup.py
        x2_safe_locomotion/
          input_source_registrar.py
          velocity_adapter.py
          motion_state_monitor.py
          safety_supervisor.py
          emergency_stop_node.py
          command_smoother.py

      x2_policy_runtime/
        package.xml
        setup.py
        x2_policy_runtime/
          observation_builder.py
          onnx_policy_runner.py
          action_filter.py
          joint_policy_node.py
          policy_safety_supervisor.py
          policy_logger.py

  training/
    README.md
    isaac_lab/
      assets/
        x2.urdf
        x2.usd
        meshes/
      x2_locomotion/
        robots/
          x2_robot_cfg.py
          x2_joint_map.py
          x2_actuator_cfg.py
        tasks/
          standing/
            x2_standing_env_cfg.py
          flat_walk/
            x2_flat_walk_env_cfg.py
          rough_terrain/
            x2_rough_env_cfg.py
          stairs/
            x2_stairs_env_cfg.py
          common/
            observations.py
            rewards.py
            terminations.py
            curriculum.py
            terrain_generator.py
            domain_randomization.py
        scripts/
          train.py
          play.py
          export_onnx.py
          evaluate_policy.py
          inspect_observation.py

    mujoco/
      model/
      scripts/

  tools/
    record_rosbag.sh
    check_topics.sh
    check_qos.sh
    check_joint_order.py
    visualize_heightmap.py
    replay_bag_heightmap.py
    analyze_policy_log.py
    compare_sim_real.py

  configs/
    robot_topics.yaml
    terrain_perception.yaml
    safe_locomotion.yaml
    training_default.yaml
    safety_limits.yaml
    joint_limits_x2_ultra.yaml

  tests/
    unit/
    integration/
    simulation/
    hardware_dry_run/
```

---

## 4. Global Source References for Implementation

Use these sources as the baseline references. Coding agent should verify against the installed SDK and firmware on the actual robot.

### 4.1 AimDK / X2 references

- AimDK X2 index: `https://x2-aimdk.agibot.com/en/latest/index.html`
- Locomotion Control: `https://x2-aimdk.agibot.com/en/latest/Interface/control_mod/locomotion.html`
- Joint Control: `https://x2-aimdk.agibot.com/en/latest/Interface/control_mod/joint_control.html`
- Sensor Interfaces: `https://x2-aimdk.agibot.com/en/latest/Interface/hal/sensor.html`
- Joint Motion Range: `https://x2-aimdk.agibot.com/en/latest/about_agibot_X2/joint_name_and_limit.html`

### 4.2 Research references

- CReF: Cross-modal and Recurrent Fusion for Depth-conditioned Humanoid Locomotion: `https://arxiv.org/abs/2603.29452`
- PIM: Learning Humanoid Locomotion with Perceptive Internal Model: `https://arxiv.org/abs/2411.14386`
- PolygMap / PolyMap-style stair plane mapping: `https://arxiv.org/abs/2510.12346`
- Sparse LiDAR elevation mapping / unsafe stepping penalty paper: `https://arxiv.org/abs/2603.07928`

---

## 5. Phase 1 — Terrain Awareness

### 5.1 Goal

Build perception-only capability. X2 should observe terrain and classify what is in front of it without moving autonomously.

Target classifications:

```text
flat_ground
rough_ground
slope_up
slope_down
curb_or_single_step
stairs_up
stairs_down
gap_or_hole
platform
unknown_unsafe
```

The first real demo should be:

> X2 stands still, looks at stairs/curb/gap, and publishes terrain type, estimated height, estimated tread/depth, slope angle, and safety decision.

### 5.2 Inputs

Verify exact topics on robot using `ros2 topic list` and `ros2 topic info -v`.

Expected useful topics:

```text
/aima/hal/sensor/rgbd_head_front/depth_image
/aima/hal/sensor/rgbd_head_front/depth_pointcloud
/aima/hal/sensor/rgbd_head_front/depth_camera_info
/aima/hal/sensor/rgbd_head_front/imu
/aima/hal/sensor/lidar_chest_front/lidar_pointcloud
/aima/hal/sensor/lidar_chest_front/imu
/aima/hal/imu/chest/state
/aima/hal/imu/torso/state
```

### 5.3 Outputs

Create custom ROS2 messages.

#### `/x2/terrain/heightmap`

Suggested fields:

```text
std_msgs/Header header
float32 resolution_m
uint32 width
uint32 height
float32 origin_x_m
float32 origin_y_m
float32[] height_m
float32[] confidence
uint8[] traversability
```

#### `/x2/terrain/status`

Suggested fields:

```text
std_msgs/Header header
string terrain_type
float32 confidence
float32 slope_angle_deg
float32 max_obstacle_height_m
float32 estimated_step_height_m
float32 estimated_step_depth_m
float32 gap_width_m
bool safe_to_continue
string reason
```

#### `/x2/terrain/stair_estimate`

Suggested fields:

```text
std_msgs/Header header
bool stairs_detected
string direction
float32 confidence
float32 rise_m
float32 tread_m
uint32 visible_step_count
float32 first_step_distance_m
float32 recommended_stop_distance_m
```

### 5.4 Main modules

#### `pointcloud_projector.py`

Responsibilities:

- Subscribe to RGB-D point cloud and/or LiDAR point cloud.
- Transform point cloud into robot base frame.
- Filter invalid/NaN points.
- Crop region of interest.
- Downsample voxel grid.
- Output normalized point cloud for height-map construction.

Suggested ROI:

```text
x forward: 0.0 m to 2.0 m
y lateral: -0.8 m to +0.8 m
z vertical: -0.5 m to +1.0 m
```

Acceptance criteria:

- Processes at least 8 Hz from 10 Hz sensor input.
- Drops invalid points cleanly.
- Publishes debug count of input/output points.
- Does not crash when point cloud is missing for 2 seconds.

#### `ground_plane_estimator.py`

Responsibilities:

- Estimate local ground plane using RANSAC or robust least squares.
- Use IMU orientation as prior when available.
- Reject outliers.
- Provide plane normal, height offset, and confidence.

Acceptance criteria:

- On flat ground, estimated slope stays within ±2 degrees.
- With a 5–10 degree ramp, slope estimate is directionally correct.
- On stairs, plane confidence should drop or identify multi-plane structure.

#### `heightmap_node.py`

Responsibilities:

- Build robot-centered local elevation map.
- Fuse recent frames with time decay.
- Maintain confidence per cell.
- Publish height map and debug visualization.

Suggested first version:

```text
map_size_x: 2.0 m
map_size_y: 1.6 m
resolution: 0.04 m
width: 50 cells
height: 40 cells
update_rate: 10 Hz
history_window: 0.5 s
```

Acceptance criteria:

- Height map is stable when robot is stationary.
- A 10–15 cm object/step appears at roughly correct height.
- Unknown cells are marked unknown, not flat.
- Unit tests cover coordinate conversion and grid indexing.

#### `stair_detector.py`

Responsibilities:

- Detect repeated horizontal planes and vertical riser edges.
- Estimate stair rise and tread.
- Estimate distance to first step.
- Publish stair confidence.

Detection methods for first version:

1. Extract height map row bands in forward direction.
2. Smooth height profile along x-axis.
3. Find step-like discontinuities.
4. Estimate repeated rise/tread pattern.
5. Validate with confidence score.

Acceptance criteria:

- Detects clear stairs from stationary view.
- Rejects random clutter as stairs if repeated structure is missing.
- Reports first-step distance within acceptable tolerance for safe stopping.
- Does not publish `safe_to_continue=true` when stair confidence is uncertain.

#### `gap_detector.py`

Responsibilities:

- Detect holes/gaps/drop-offs from missing or lower-height cells.
- Estimate gap width and distance.

Acceptance criteria:

- Detects simulated/open gap in front of robot.
- Treats unknown region as unsafe unless confidence is high.
- Publishes reason string for safety decision.

#### `terrain_classifier.py`

Responsibilities:

- Combine ground plane, height map, stair detector, slope detector, and gap detector.
- Output final terrain type.
- Provide confidence and reason.

Decision policy example:

```text
if confidence_low:
    terrain_type = unknown_unsafe
elif gap_detected:
    terrain_type = gap_or_hole
elif stairs_detected and rise > threshold:
    terrain_type = stairs_up or stairs_down
elif single_step_detected:
    terrain_type = curb_or_single_step
elif abs(slope_angle) > threshold:
    terrain_type = slope_up or slope_down
elif roughness > threshold:
    terrain_type = rough_ground
else:
    terrain_type = flat_ground
```

### 5.5 Data recording tasks

Create scripts:

```text
tools/record_terrain_bag.sh
tools/replay_bag_heightmap.py
tools/visualize_heightmap.py
```

Record scenes:

```text
flat floor
carpet floor
reflective floor
single box/curb 5 cm
single box/curb 10 cm
single box/curb 15 cm
stairs up
stairs down
platform edge
gap/hole mockup
cluttered unsafe scene
```

Each bag should include:

```text
RGB-D depth image
RGB-D point cloud
LiDAR point cloud
IMU
TF if available
robot joint state if available
```

### 5.6 Phase 1 Definition of Done

Phase 1 is done when:

- Terrain perception runs at 8–10 Hz.
- Height map is visualized live.
- Flat, slope, curb, stairs, and gap can be detected in offline bags.
- `unknown_unsafe` is used correctly for low-confidence scenes.
- No locomotion command is sent by this phase.
- Logs are saved for every perception test.

---

## 6. Phase 2 — Safe SDK Locomotion Adaptation

### 6.1 Goal

Use existing AimDK velocity locomotion safely. The robot should walk slowly on flat known terrain and stop before unsafe terrain.

This phase still does **not** climb stairs.

### 6.2 Inputs

Subscribe to:

```text
/x2/terrain/status
/x2/terrain/stair_estimate
/x2/terrain/heightmap
/aima/hal/imu/chest/state
/aima/hal/imu/torso/state
robot mode/state topic or service if available
```

### 6.3 Output

Publish to:

```text
/aima/mc/locomotion/velocity
```

Expected command fields:

```text
source
forward_velocity
lateral_velocity
angular_velocity
```

The command source must be registered before locomotion commands are sent.

### 6.4 Main modules

#### `input_source_registrar.py`

Responsibilities:

- Register custom locomotion source name.
- Verify current input source priority.
- Prevent source name collision.

Suggested source name:

```text
x2_terrain_safe_locomotion
```

Acceptance criteria:

- Registration succeeds before velocity publisher starts.
- Node fails closed if registration fails.
- Logs current source and priority information if available.

#### `velocity_adapter.py`

Responsibilities:

- Convert user or autonomous desired velocity into safe velocity.
- Reduce speed based on terrain type.
- Stop before stairs/gaps/unknown terrain.
- Smooth command changes.

Suggested first policy:

```text
flat_ground: allow up to 0.12 m/s forward
rough_ground: allow up to 0.06 m/s forward
slope_up/down: allow up to 0.04 m/s, if mild
curb_or_single_step: stop
stairs_up/down: stop
gap_or_hole: stop
unknown_unsafe: stop
```

Important: start threshold may require command above the robot's minimum movement threshold, but do not use aggressive velocities. Tune based on firmware behavior.

Acceptance criteria:

- Robot can walk slowly on flat ground.
- Robot slows down and stops before detected stairs.
- Velocity command is smooth, not jerky.
- If perception data stops, velocity becomes zero within watchdog timeout.

#### `safety_supervisor.py`

Responsibilities:

- Monitor IMU, robot state, perception freshness, command freshness, and operator stop.
- Enforce hard stop.
- Publish safety reason.

Stop conditions:

```text
terrain_status missing > 0.5 s
IMU missing > 0.2 s
roll angle over threshold
pitch angle over threshold
unknown terrain ahead
stairs/gap ahead
operator stop requested
command timeout
robot mode unexpected
```

Acceptance criteria:

- Any missing critical input causes stop.
- Safety reason is visible in logs.
- Manual emergency stop overrides all commands.

#### `command_smoother.py`

Responsibilities:

- Apply ramp limits to velocity commands.
- Avoid sudden start/stop unless emergency.

Suggested limits:

```text
max_forward_accel: 0.05 m/s^2 initially
max_yaw_accel: 0.10 rad/s^2 initially
emergency_stop: immediate zero command
```

### 6.5 Test protocol

Test order:

1. Dry-run mode: publish to debug topic only.
2. Robot standing: zero velocity only.
3. Robot walking in open flat area: 0.05 m/s.
4. Static obstacle/box ahead: should stop.
5. Mock stair/curb ahead: should stop.
6. Real stairs ahead: should stop before reaching first step.

### 6.6 Phase 2 Definition of Done

Phase 2 is done when:

- Custom input source registration works.
- Safe velocity adapter can command slow walking.
- X2 stops before stairs/gaps/unknown terrain.
- Watchdog stop works.
- Manual stop works.
- Logs prove that stop decision came from terrain perception.

This is the first good customer/demo milestone.

---

## 7. Phase 3 — X2 Simulation Model

### 7.1 Goal

Build a simulation environment accurate enough to train and test locomotion policies before touching the real robot.

Recommended primary platform:

```text
Isaac Lab
```

Optional secondary platform:

```text
MuJoCo
```

### 7.2 Required robot model assets

Collect/verify:

```text
URDF or MJCF
USD conversion if using Isaac Lab
mesh files
joint names
joint order
joint limits
default standing pose
mass values
inertia tensors
foot collision geometry
motor torque limits
velocity limits
PD stiffness/damping estimates
actuator delay estimate
self-collision pairs
```

### 7.3 Critical validation tasks

#### `x2_joint_map.py`

Map simulation joints to AimDK joint order.

For legs, expected group order should be verified:

```text
left leg first, then right leg
hip_pitch
hip_roll
hip_yaw
knee
ankle_pitch
ankle_roll
```

Acceptance criteria:

- Joint names are printed and compared between sim and robot.
- Left/right order is verified.
- Radian/degree units are verified.
- Joint limits are loaded from config, not hardcoded in policy runtime.

#### `x2_robot_cfg.py`

Define:

```text
asset path
default pose
initial base height
joint limits
PD gains
actuator parameters
contact bodies
termination bodies
feet names
```

Acceptance criteria:

- Robot spawns without exploding.
- Robot stands in default pose under gravity with stable PD.
- Feet contact geometry is correct.
- No major mesh/collision mismatch.

#### `x2_standing_env_cfg.py`

Create first training/test environment:

```text
flat ground
standing pose target
low disturbance
termination on fall
```

Acceptance criteria:

- Robot stands for 30 seconds in sim with fixed PD targets.
- Contact forces are reasonable.
- Base height and orientation are stable.

### 7.4 Terrain generator

Implement terrain generator in progressive levels:

```text
level_0_flat
level_1_rough_low
level_2_slope
level_3_single_step
level_4_stairs_up
level_5_stairs_down
level_6_mixed
```

Terrain parameters:

```text
roughness_height_m: 0.01 to 0.05
slope_deg: 3 to 12
single_step_height_m: 0.02 to 0.15
stair_rise_m: 0.05 to 0.18
stair_tread_m: 0.24 to 0.35
friction: 0.4 to 1.2
```

### 7.5 Phase 3 Definition of Done

Phase 3 is done when:

- X2 model spawns and stands in simulation.
- Joint ordering is verified against AimDK.
- Terrain generator exists.
- Height samples can be extracted around robot in simulation.
- Basic simulation tests run in CI or local test command.

No RL stair policy should start before this phase is stable.

---

## 8. Phase 4 — RL Locomotion Training

### 8.1 Goal

Train a locomotion policy in simulation using PPO. Start with standing and flat walking, then rough terrain, single steps, and finally stairs.

This phase should use height-map/elevation-map input first, not raw depth.

### 8.2 Training algorithm

Use:

```text
PPO with asymmetric actor-critic
```

Actor receives realistic deployable observations. Critic can receive privileged simulation information during training.

### 8.3 First action space

Start with lower body only:

```text
12 actions:
left_leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
right_leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
```

Action meaning:

```text
joint position target offset from nominal pose
```

Do not start with full 29-DoF body policy unless the 12-DoF leg policy works.

### 8.4 Later action space

After lower body policy works:

```text
12 leg joints
+ 3 waist joints
+ optional arms for balance
```

Arms can initially be fixed or controlled by a simple balance pattern.

### 8.5 Observation design

Recommended height-map policy observation:

```text
command_forward_velocity
command_lateral_velocity
command_yaw_velocity
base_angular_velocity
projected_gravity_vector
joint_position_error
joint_velocity
previous_action
gait_phase_sin
gait_phase_cos
height_samples
```

Suggested height samples:

```text
96 samples around robot
or 11 x 11 grid = 121 samples
range x: -0.4 m to +1.2 m
range y: -0.5 m to +0.5 m
values: terrain height relative to base or feet
```

Normalize all observations.

### 8.6 Network architecture

Start simple:

```text
height_encoder:
  Linear(height_dim, 128)
  ELU
  Linear(128, 64)
  ELU

proprio_encoder:
  Linear(proprio_dim, 256)
  ELU
  Linear(256, 128)
  ELU

actor:
  concat(height_latent, proprio_latent)
  Linear(192, 256)
  ELU
  Linear(256, 256)
  ELU
  Linear(256, action_dim)

critic:
  Linear(privileged_obs_dim, 512)
  ELU
  Linear(512, 256)
  ELU
  Linear(256, 1)
```

### 8.7 Training frequency

Suggested:

```text
physics_dt: 0.005 s       # 200 Hz physics
policy_dt: 0.02 s         # 50 Hz policy
control_decimation: 4
```

For RTX 4060:

```text
num_envs_initial: 512
num_envs_later: 1024 if stable
network: small MLP first
camera rendering: disabled for height-map version
```

### 8.8 Curriculum

Do not jump directly to stairs.

#### Stage A — Standing

Goal:

```text
stand for 30 seconds
recover from small pushes
keep torso upright
```

#### Stage B — Flat walking

Commands:

```text
forward_velocity: 0.0 to 0.3 m/s
lateral_velocity: 0.0 initially
yaw_velocity: -0.3 to +0.3 rad/s
```

Goal:

```text
stable walking
low foot slip
reasonable energy
```

#### Stage C — Rough terrain

Terrain:

```text
height noise: 1 to 5 cm
small bumps
mild slopes
```

Goal:

```text
stable torso
foot clearance
no tripping
```

#### Stage D — Single step / curb

Terrain:

```text
step height: 2 cm → 5 cm → 8 cm → 12 cm → 15 cm
```

Goal:

```text
lift swing foot over edge
place foot on top plane
avoid shin/toe collision
```

#### Stage E — Stairs up

Terrain:

```text
rise: 5 to 18 cm
tread: 24 to 35 cm
step count: 1 to 8
```

Goal:

```text
continuous ascent
safe foothold
no backward fall
```

#### Stage F — Stairs down

Terrain:

```text
rise: 5 to 15 cm
tread: 24 to 35 cm
```

Goal:

```text
controlled touchdown
low impact
no forward fall
```

#### Stage G — Mixed terrain

Terrain:

```text
flat → rough → curb → stairs → platform → gap edge
```

Goal:

```text
generalization
no unsafe stepping
```

### 8.9 Reward design

Implement reward components separately with logging.

#### Velocity tracking

```text
reward forward/lateral/yaw velocity tracking
penalize excessive velocity error
```

#### Torso stability

```text
penalize roll/pitch
penalize high angular velocity
reward target base height range
```

#### Foot clearance

```text
reward swing foot clearing terrain height
penalize toe/shin collision
```

#### Foothold quality

```text
reward foot landing on supportable area
penalize landing near stair edge
penalize landing in gap/unknown cell
```

#### Foot slip

```text
when foot in contact, penalize horizontal foot velocity
```

#### Energy and smoothness

```text
penalize torque
penalize joint acceleration
penalize action rate
penalize large action delta
```

#### Joint safety

```text
penalize joint near limit
terminate if joint exceeds safe bound
```

#### Falls and collisions

Terminate if:

```text
base height too low
torso roll/pitch too high
head/torso collision
knee collision
invalid contact state
```

### 8.10 Domain randomization

Mandatory for sim-to-real transfer.

Randomize:

```text
body mass ±5-10%
link inertia ±5-10%
center of mass offset
motor strength ±10-20%
PD stiffness/damping ±10-20%
action delay 1-3 policy steps
sensor latency 10-60 ms
IMU noise and bias
depth/heightmap noise
terrain friction 0.4-1.2
foot friction
terrain height noise
joint encoder noise
```

### 8.11 Training acceptance criteria

A policy can graduate from simulation only if:

- flat walking success rate > 95%;
- rough terrain success rate > 90%;
- single 5 cm step success rate > 90%;
- single 10 cm step success rate > 80%;
- stair-up simulation success rate > 80% before any real stair testing;
- no frequent joint-limit abuse;
- no unrealistic high torque reliance;
- action signals are smooth;
- policy survives randomized latency and noise.

### 8.12 Phase 4 Definition of Done

Phase 4 is done when:

- PPO training pipeline runs reproducibly.
- Height-map policy can walk and handle rough terrain in simulation.
- Single-step and stair-up curriculum shows measurable success.
- Policy evaluation script produces success-rate reports.
- ONNX export is possible and numerically checked against PyTorch output.

---

## 9. Phase 5 — Sim-to-Real Deployment

### 9.1 Goal

Deploy trained policy to real X2 under strict safety conditions.

This is the highest-risk phase.

### 9.2 Required runtime modules

#### `observation_builder.py`

Responsibilities:

- Build policy observation from real robot state.
- Match training normalization exactly.
- Handle missing values safely.

Acceptance criteria:

- Observation vector dimension equals training config.
- Observation ordering is tested.
- Normalization stats loaded from exported training artifact.
- Missing sensor values cause safe stop.

#### `onnx_policy_runner.py`

Responsibilities:

- Load ONNX policy.
- Run inference at fixed frequency.
- Validate output dimensions.
- Provide timing metrics.

Acceptance criteria:

- Inference time is below policy period budget.
- Output matches PyTorch reference within tolerance on test vectors.
- Bad ONNX output triggers safe stop.

#### `action_filter.py`

Responsibilities:

- Clamp joint targets.
- Limit action rate.
- Limit joint velocity.
- Apply low-pass filter.
- Enforce joint safety envelope.

Acceptance criteria:

- No output exceeds configured joint limits.
- No sudden action spike passes through.
- Filter can be unit-tested with extreme inputs.

#### `policy_safety_supervisor.py`

Responsibilities:

- Monitor IMU, joint state, policy timing, action magnitude, and operator stop.
- Cut policy output on unsafe state.
- Optionally switch to damping/zero command behavior.

Stop conditions:

```text
roll/pitch over safe threshold
joint state missing
IMU missing
policy inference timeout
action NaN/Inf
joint target outside soft limit
operator stop
base instability detected
```

### 9.3 Deployment order

#### Level 0 — Offline replay

- Run policy against recorded robot logs.
- Do not connect to robot command topic.
- Verify observation/action shape.

#### Level 1 — Hardware dry-run

- Robot powered but no leg command output.
- Policy publishes only debug output.
- Compare expected joint targets.

#### Level 2 — Suspended standing

- Robot suspended/gantry required.
- Policy controls only standing balance or very small target offsets.
- Emergency stop operator ready.

#### Level 3 — Suspended stepping in place

- Small leg movement only.
- No forward locomotion.
- Verify joint order, sign, and delay.

#### Level 4 — Foam obstacle

- 2 cm foam first.
- Then 5 cm.
- Then 8 cm only if previous level is safe.

#### Level 5 — Single low wooden step

- 5 cm single step.
- Then 8 cm.
- Then 10-12 cm.

#### Level 6 — Real stairs

Only after repeated successful lower-risk tests.

### 9.4 Real robot logging

Every deployment test must log:

```text
timestamp
robot mode
terrain status
heightmap or sampled heights
policy observation
policy raw action
filtered action
joint state
IMU state
safety supervisor state
operator command
stop reason
```

### 9.5 Real robot go/no-go checklist

No-go if:

```text
joint order not verified
safety supervisor disabled
emergency stop not tested
policy output not filtered
robot not suspended for first test
logs not recording
operator not present
battery/power unstable
perception confidence low
latency too high
```

### 9.6 Phase 5 Definition of Done

Phase 5 is done when:

- ONNX runtime works on target compute.
- Observation builder matches training.
- Safety supervisor stops unsafe output.
- Suspended tests pass.
- Low obstacle tests pass.
- Single-step tests pass.
- Real stair testing is approved only after documented safety review.

---

## 10. Phase 6 — Advanced CReF-Style Raw-Depth Perception Policy

### 10.1 Goal

Upgrade from height-map/elevation-map policy to a raw-depth perceptive locomotion policy inspired by CReF.

This should happen only after the height-map policy works. Do not start here.

### 10.2 Why not start with CReF first?

Raw-depth end-to-end locomotion is harder to debug. Failures could come from:

```text
bad depth stream
bad camera calibration
bad depth encoder
bad recurrent state
bad reward
bad contact model
bad sim-to-real depth noise
bad action mapping
```

Height maps are easier to inspect and debug. CReF-style policy is an upgrade, not the foundation.

### 10.3 Inputs

```text
forward RGB-D depth image or depth crop
proprioception
command velocity
previous action
optional height samples as auxiliary input
```

### 10.4 Network architecture concept

```text
depth image
  → patch embedding or CNN encoder
  → depth tokens

proprioception
  → proprio encoder
  → proprio query

cross-modal attention
  proprio query attends to depth tokens

fusion block
  combine proprio feature and depth-aware feature

GRU recurrent memory
  temporal integration

actor head
  joint target action
```

### 10.5 Training method

Start from the Phase 4 height-map policy.

Options:

#### Option A — Distillation

Train raw-depth policy to imitate height-map policy first.

```text
teacher: trained height-map policy
student: raw-depth CReF-style policy
loss: action imitation + value imitation + auxiliary terrain prediction
```

Then fine-tune with RL.

#### Option B — Direct RL

Train raw-depth policy directly with PPO.

This is harder and slower because depth rendering and augmentation are expensive.

Recommended: **Option A first**.

### 10.6 Depth augmentation

Randomize:

```text
depth dropout
missing pixels
reflective noise
edge noise
motion blur equivalent
camera pitch/roll offset
latency
extrinsic calibration error
field-of-view crop
near/far clipping
```

### 10.7 Auxiliary tasks

To stabilize raw-depth learning, add auxiliary heads during training:

```text
terrain type prediction
height sample reconstruction
safe foothold probability
depth validity mask prediction
```

Remove or disable auxiliary heads for deployment if not needed.

### 10.8 Phase 6 Definition of Done

Phase 6 is done when:

- Raw-depth policy matches or beats height-map policy in simulation.
- Policy survives depth noise and latency.
- Policy generalizes to clutter/gaps/platforms better than height-map policy.
- Real robot suspended tests pass.
- Low obstacle and single-step tests pass.
- Stair testing follows same Phase 5 safety protocol.

---

## 11. Coding Agent Task Generation Template

Use this format to generate tasks.

```markdown
## Task: <short name>

### Phase
<phase number and name>

### Goal
<what this task achieves>

### Files to create/modify
- path/to/file.py
- path/to/config.yaml
- path/to/test.py

### Implementation details
- detail 1
- detail 2
- detail 3

### Safety constraints
- constraint 1
- constraint 2

### Tests
- unit test 1
- integration test 1
- dry-run test 1

### Acceptance criteria
- criterion 1
- criterion 2
- criterion 3

### Notes
<any SDK topic, command, or dependency notes>
```

---

## 12. Initial Task Backlog

### Phase 1 backlog

1. Create ROS2 workspace skeleton.
2. Create `x2_terrain_msgs` custom messages.
3. Create topic discovery script.
4. Create point cloud subscriber.
5. Create point cloud projector to base frame.
6. Create height-map builder.
7. Create height-map visualization tool.
8. Create ground plane estimator.
9. Create stair detector.
10. Create gap detector.
11. Create terrain classifier.
12. Create rosbag recording scripts.
13. Create offline replay analyzer.
14. Write unit tests for grid conversion and terrain classification.
15. Record first real terrain bags.

### Phase 2 backlog

1. Create custom input source registrar.
2. Create safe velocity adapter.
3. Create command smoother.
4. Create safety supervisor.
5. Create debug/dry-run mode.
6. Create flat-ground walking test.
7. Create stair-stop test.
8. Create missing-sensor watchdog test.
9. Create manual emergency stop test.
10. Create demo script: walk forward and stop before stairs.

### Phase 3 backlog

1. Locate X2 URDF/MJCF/USD and meshes.
2. Convert model to Isaac Lab asset.
3. Create joint map.
4. Create joint limit config.
5. Create actuator config.
6. Create default standing pose.
7. Create flat-ground standing environment.
8. Validate contact geometry.
9. Create terrain generator.
10. Create simulation smoke tests.

### Phase 4 backlog

1. Create PPO training config.
2. Create observation builder for simulation.
3. Create reward components.
4. Create standing training task.
5. Create flat walking task.
6. Create rough terrain task.
7. Create single-step task.
8. Create stair-up task.
9. Create stair-down task.
10. Add curriculum manager.
11. Add domain randomization.
12. Add policy evaluation report.
13. Add ONNX export.
14. Add PyTorch-vs-ONNX validation.

### Phase 5 backlog

1. Create real robot observation builder.
2. Create ONNX runtime node.
3. Create action filter.
4. Create policy safety supervisor.
5. Create hardware dry-run mode.
6. Create suspended standing test.
7. Create suspended stepping test.
8. Create foam obstacle test protocol.
9. Create single low-step test protocol.
10. Create real robot log analyzer.

### Phase 6 backlog

1. Create raw depth dataset collector from simulation.
2. Create depth encoder.
3. Create cross-modal attention module.
4. Create recurrent GRU fusion module.
5. Create teacher-student distillation pipeline.
6. Add auxiliary terrain prediction heads.
7. Add depth augmentation.
8. Train raw-depth policy in simulation.
9. Compare against height-map policy.
10. Deploy through same Phase 5 safety protocol.

---

## 13. First Sprint Recommendation

Do not start with training.

Start with this 2-week sprint:

### Sprint Goal

X2 can perceive terrain and stop before unsafe terrain using existing SDK locomotion.

### Sprint Deliverables

```text
ros2_ws/src/x2_terrain_msgs
ros2_ws/src/x2_terrain_perception/heightmap_node.py
ros2_ws/src/x2_terrain_perception/stair_detector.py
ros2_ws/src/x2_safe_locomotion/velocity_adapter.py
ros2_ws/src/x2_safe_locomotion/safety_supervisor.py
tools/record_terrain_bag.sh
tools/visualize_heightmap.py
configs/terrain_perception.yaml
configs/safe_locomotion.yaml
```

### Sprint demo

1. Place X2 facing flat ground.
2. Show live height map.
3. Place box/curb/stair in front.
4. X2 publishes stair/curb/unsafe terrain classification.
5. Start slow forward velocity.
6. X2 slows down and stops before obstacle.
7. Log shows exact stop reason.

This is the correct first win.

---

## 14. Non-Negotiable Safety Rule

No coding agent should generate or run a real-robot low-level leg joint stair-climbing command unless the task explicitly states:

```text
REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED=true
```

Default must be:

```text
REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED=false
```

If false:

- policy nodes may run in dry-run only;
- policy outputs may publish debug messages only;
- no `/aima/hal/joint/leg/command` publishing is allowed;
- only SDK velocity-level control is allowed, and only through the safe velocity adapter.

---

## 15. Final Implementation Strategy

The project should move like this:

```text
1. Make X2 see terrain.
2. Make X2 stop safely.
3. Build the X2 simulator.
4. Train flat/rough locomotion.
5. Train steps/stairs in simulation.
6. Export and test under suspension.
7. Test foam and low steps.
8. Only then attempt real stairs.
9. Upgrade to raw-depth CReF-style policy after height-map policy works.
```

This sequence is slower at the beginning but much faster overall because it avoids destroying hardware and avoids debugging everything at once.
