# Training Method

RL locomotion training method (Phase 4), summarising roadmap §8. Height-map policy first;
raw-depth CReF (Phase 6) is an upgrade, not the foundation.

## Algorithm

- **PPO with asymmetric actor-critic.** Actor sees realistic deployable observations; critic
  may see privileged simulation state during training.

## Action space

- **Start: 12-DoF legs only** — joint *position target offsets* from the nominal pose.
  - left leg, then right leg; each: `hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll`.
- Expand to `+3 waist (+arms for balance)` only **after** the 12-DoF leg policy works. Do
  not start with the full 29-DoF body.

## Observation (height-map policy)

Normalize **all** components. Ordering must be identical between training and deployment.

```
command_forward_velocity        (1)
command_lateral_velocity        (1)
command_yaw_velocity            (1)
base_angular_velocity           (3)
projected_gravity_vector        (3)
joint_position_error            (12)
joint_velocity                  (12)
previous_action                 (12)
gait_phase_sin                  (1)
gait_phase_cos                  (1)
height_samples                  (121 = 11×11 grid)
```

Height samples span x ∈ [−0.4, +1.2] m, y ∈ [−0.5, +0.5] m, value = terrain height relative
to base/feet.

## Network (start simple, roadmap §8.6)

```
height_encoder : Linear(height_dim,128) ELU Linear(128,64) ELU
proprio_encoder: Linear(proprio_dim,256) ELU Linear(256,128) ELU
actor          : concat(height_latent, proprio_latent) → Linear(192,256) ELU Linear(256,256) ELU Linear(256,action_dim)
critic         : Linear(privileged_dim,512) ELU Linear(512,256) ELU Linear(256,1)
```

## Timing

```
physics_dt 0.005 s (200 Hz) · policy_dt 0.02 s (50 Hz) · control_decimation 4
num_envs 512 → 1024 if stable · camera rendering OFF for height-map version
```

## Curriculum (do not jump to stairs)

| Stage | Terrain | Goal |
|:-----:|---------|------|
| A | standing | stand 30 s, recover small pushes |
| B | flat walk | fwd 0–0.3 m/s, yaw ±0.3 rad/s, low slip |
| C | rough | 1–5 cm noise, mild slopes, no tripping |
| D | single step / curb | 2→5→8→12→15 cm, clear edge, no shin/toe hit |
| E | stairs up | rise 5–18 cm, tread 24–35 cm, 1–8 steps |
| F | stairs down | rise 5–15 cm, controlled touchdown |
| G | mixed | generalisation, no unsafe stepping |

## Rewards (logged separately)

velocity tracking · torso stability · foot clearance · foothold quality (penalise landing
near stair edge / in gap / unknown cell) · foot slip · energy & smoothness (torque, joint
accel, action rate/delta) · joint safety. Terminate on: base too low, roll/pitch too high,
head/torso/knee collision, invalid contact.

## Domain randomization (mandatory for transfer)

mass/inertia/CoM · motor strength ±10–20% · PD ±10–20% · action delay 1–3 steps · sensor
latency 10–60 ms · IMU noise/bias · depth/heightmap noise · friction 0.4–1.2 · encoder noise.

## Graduation criteria (roadmap §8.11)

flat > 95% · rough > 90% · 5 cm step > 90% · 10 cm step > 80% · stair-up > 80% (sim) · no
joint-limit abuse · no unrealistic torque reliance · smooth actions · survives randomized
latency/noise.

## Export

ONNX export required; validate ONNX output numerically against the PyTorch reference on test
vectors before any deployment (see [sim_to_real_checklist.md](sim_to_real_checklist.md)).
