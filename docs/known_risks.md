# Known Risks

Living risk register. Add rows as risks are discovered; reference the relevant task ID.

## Legend

Severity: Low / Med / High. Status: Open / Mitigating / Closed.

## Register

| # | Risk | Phase | Sev | Mitigation | Status |
|---|------|:-----:|:---:|------------|:------:|
| R1 | Topic names / QoS in the roadmap differ from the real robot | 1–2 | Med | Verify with `tools/check_topics.sh` / `check_qos.sh` before trusting any name; record in `ros_topics.md` | Open |
| R2 | Joint order / units (rad vs deg, L/R) wrong ⇒ dangerous joint commands | 3,5 | High | **VERIFIED** vs robot MC `robot_model.yaml`: legs(12 L-then-R) → waist(3) → head(2) → arms(14) = 31. `x2_joint_map.AIMDK_BODY_ORDER` matches; limits from URDF v1.3.0. Live `/aima/hal/joint/leg/state` index check still recommended before Level 2 | Mitigating |
| R13 | Roadmap says "29-DoF" but the real robot is **31-DoF** (7 joints/arm: shoulder×3, elbow, wrist×3) — the "x2_31dof" model | 3-6 | Low | We drive only the 12 legs (+3 waist later); `AIMDK_BODY_ORDER`/MJCF/URDF all agree on 31. Roadmap §8.4 number is an estimate; no code impact | Closed |
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
| R14 | **No passive-stable stand.** Disabling the MC leaves the robot in `JOINT_DEFAULT` ("Position Control Stand, joints locked") = stiff joints at the standing pose but **no active balance ⇒ topples fwd/back even rigid** (confirmed on hardware, 0.8). (`STAND_DEFAULT` = the auto-balancing stand; `LOCOMOTION_DEFAULT` = walk.) A custom joint-level policy runs in/from `JOINT_DEFAULT` and therefore owns **continuous active balance from t=0** with **no MC fallback**; the balancing-mode→`JOINT_DEFAULT` handoff instant is the sharpest fall risk. | 5–6 | **High** | Gantry mandatory for ALL bring-up (incl. standing). Confirm: in `JOINT_DEFAULT`, does publishing `/aima/hal/joint/*/command` override the stiff hold (the likely external-control entry point, per `motocontrol.py`)? Staged suspended levels (roadmap §9) before any free-standing; confirm handoff + `dcu_leg_safe` limits with AgiBot. | Open |
| R15 | Low-level joint control = **full whole-body takeover**: the MC exposes only velocity (blind, MC balances) or raw joint (`/aima/hal/joint/*/command` → `hal_ethercat`, PD via sent kp/kd) — **no "MC balances while we steer" mode.** **CONFIRMED via SDK `motocontrol.py`**: publish `JointCommandArray{name,position,velocity,effort,stiffness=kp,damping=kd}` to `/aima/hal/joint/leg/command`; HAL does raw PD, **no balance/safety wrapper** (the example just oscillates one knee). No perception/footstep interface exists (terrain grep empty; teleop RPC is upper-body only). So a perception-aware stair policy must be a complete balance+walk+climb controller replacing the MC. | 5–6 | **High** | Treat the stair policy as a full locomotion policy, not an add-on. Software safety envelope (from `motocontrol.py`, matches `idle.yaml`): per-joint pos limits (knee 0.026–2.12, hip_pitch ±2.49, ankle_pitch −0.80–0.45, ankle_roll ±0.26 rad), Ruckig smoothing @2 ms w/ vel≤1.0 acc≤1.0 jerk≤25. **Still open (HW/AgiBot):** (a) MC↔external arbitration — example doesn't stop MC or call `SetMcInputSource`, so unknown if publish overrides the `JOINT_DEFAULT` hold or fights it; (b) `dcu_leg_safe` is firmware (`X2_DDCU_SAFE_V2.0.0`), thresholds not in readable config. Prove takeover+balance on a gantry before training investment. | Mitigating |
