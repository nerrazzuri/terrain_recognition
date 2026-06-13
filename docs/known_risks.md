# Known Risks

Living risk register. Add rows as risks are discovered; reference the relevant task ID.

## Legend

Severity: Low / Med / High. Status: Open / Mitigating / Closed.

## Register

| # | Risk | Phase | Sev | Mitigation | Status |
|---|------|:-----:|:---:|------------|:------:|
| R1 | Topic names / QoS in the roadmap differ from the real robot | 1–2 | Med | Verify with `tools/check_topics.sh` / `check_qos.sh` before trusting any name; record in `ros_topics.md` | Open |
| R2 | Joint order / units (rad vs deg, L/R) wrong ⇒ dangerous joint commands | 3,5 | High | `x2_joint_map.py` + `tools/check_joint_order.py`; limits from config; verify before Level 2 | Open |
| R3 | Stale perception treated as flat ⇒ robot walks into a hazard | 1–2 | High | Fail closed: unknown cells marked unknown; watchdog zeros velocity on stale terrain/IMU | Open |
| R4 | Reflective / transparent floors corrupt depth ⇒ phantom terrain | 1 | Med | Record reflective-floor bags; confidence gating; LiDAR cross-check | Open |
| R5 | Sim-to-real gap: policy unstable on hardware | 4–5 | High | Domain randomization; ONNX vs PyTorch numeric check; suspended bring-up; staged levels | Open |
| R6 | ONNX export diverges numerically from PyTorch | 4 | Med | Validate on test vectors (`P4-M4-T4`) before deploy | Open |
| R7 | Inference exceeds policy period ⇒ control jitter | 5 | Med | Timing metrics in `onnx_policy_runner`; bad/late output ⇒ safe stop | Open |
| R8 | High-bandwidth raw camera subscription saturates cross-unit link | 1 | Med | Prefer point cloud / compressed streams | Open |
| R9 | Operator e-stop not wired / not tested before motion | 2,5 | High | Go/No-Go checklist blocks motion until e-stop tested | Open |
| R10 | Premature real stair climbing before safety review | 5–6 | High | Phase gates; approval flag default false; SAFETY.md | Open |
| R11 | RTX 4060 too small for large parallel sim ⇒ slow/unstable training | 4 | Low | Start 512 envs, height-map (no camera render); scale only if stable | Open |
| R12 | Robot model mass/inertia/contact unvalidated ⇒ bad sim transfer | 3 | Med | Validate assets in `P3-M1-T1`; spawn/stand check in `x2_robot_cfg` | Open |
