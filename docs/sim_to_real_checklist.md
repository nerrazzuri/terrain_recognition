# Sim-to-Real Checklist

Gate for moving a trained policy from simulation to hardware (Phase 5, roadmap §9). Pair
with [real_robot_test_protocol.md](real_robot_test_protocol.md) and [../SAFETY.md](../SAFETY.md).

## Determinism (must all pass before any hardware step)

- [ ] Observation vector dimension equals the training config.
- [ ] Observation **ordering** matches training exactly (tested, not assumed).
- [ ] Normalization stats loaded from the exported training artifact (not re-derived).
- [ ] ONNX output matches the PyTorch reference within tolerance on saved test vectors.
- [ ] Inference time is below the policy period budget (50 Hz ⇒ < 20 ms).

## Joint interface

- [ ] Joint names printed and compared sim ↔ robot.
- [ ] Left/right order verified.
- [ ] Radian vs degree units verified.
- [ ] Joint limits loaded from `configs/joint_limits_x2_ultra.yaml`, not hardcoded.

## Safety layer (must be active)

- [ ] `action_filter` clamps joint targets; no output exceeds configured limits.
- [ ] Action rate / joint velocity limited; no spike passes through (unit-tested w/ extremes).
- [ ] `policy_safety_supervisor` cuts output on: roll/pitch over threshold, joint/IMU
      missing, inference timeout, action NaN/Inf, target outside soft limit, operator stop,
      base instability — switching to damping/zero.
- [ ] Missing sensor values ⇒ safe stop in the observation builder.

## Approval flag

- [ ] `REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED` is set `true` **only** after the documented
      safety review, and only for the approved test session.

## Escalation

Follow Levels 0 → 6 from [real_robot_test_protocol.md](real_robot_test_protocol.md). Pass the
Go/No-Go checklist before each escalation. Real stairs (Level 6) only after documented safety
review + all lower levels pass.
