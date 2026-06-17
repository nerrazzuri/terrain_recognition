# Handoff prompt — paste this into the laptop's Claude Code (robot session)

> Copy everything in the box below as your first message to the laptop Claude Code instance.
> Fill in the two blanks (`<ROBOT_SSH>` and the ONNX transfer) before/at the start.

---

```
You are helping me run a HARDWARE bring-up on an Agibot X2 humanoid today. The robot is on a
GANTRY/HARNESS. Safety is the priority — a 43 kg humanoid with no passive balance; if our control
is wrong it falls. Do NOT improvise robot motion. Follow the committed runbook exactly and stop at
every go/no-go gate for my confirmation.

## Context (read these first, in order)
This repo implements "Way A": running Agibot's factory RL policies inside OUR OWN control loop via
low-level joint commands, as the foundation for a future stair-climbing policy. Everything follows
robot software version 0.9.x.
  1. docs/hardware_bringup_cpgwalk.md   <- THE runbook for today. Follow it step by step.
  2. docs/known_risks.md (esp. R14, R15) <- why this is dangerous; the MC-arbitration unknown.
  3. docs/factory_cpgwalk_contract.md    <- the policy contracts + STAND_DEFAULT(cpgtelecon)->
                                            LOCOMOTION(cpgwalk) sequencing + estimator thresholds.
Validated in sim already: the robot stands STILL on cpgtelecon, then walks on cpgwalk (see
training/mujoco/run_stand_walk_mujoco.py). The deploy node reproduces that on hardware.

## Two goals today
A) RECORD THE GROUND TRUTH (zero risk, robot stays under the factory MC) — do this FIRST.
B) RUN OUR LOOP ON THE ROBOT (harness) — staged, only if A is done and gates pass.

## Robot architecture (TWO boards — read carefully)
The X2 has two onboard computers on an internal network; cross-board ROS 2 is the INTENDED design:
- **Development Computing Unit = 10.0.1.41** — runs SDK / OUR code. Has ROS 2 Humble + aimdk_msgs +
  all prereqs preinstalled and participates in the robot's ROS 2 network. **Run everything here.**
- **Motion Control Unit (PC1) = 10.0.1.40** — runs the MC/HAL and stores the factory runtime+ONNX.
  **Building/running secondary-development code here is PROHIBITED by Agibot (safety). Do NOT.**
All control interfaces (joint command/state, IMU, locomotion, and the MC services SetMcInputSource /
SetMcAction) are exposed over ROS 2 to the dev unit (.41). The SDK examples run on .41 and drive the
robot on .40 — so our deploy node on .41 talking to the MC/HAL on .40 works by design.

## Environment / transfer (do this before anything)
- This repo: `git clone`/`git pull` it onto the **dev unit (.41)** (only our code needs to move; the
  dev unit already has ROS 2 + aimdk_msgs). Ensure `onnxruntime` + `numpy` are in its python; if not,
  ask me before installing.
- ONNX models live on the **main board (.40)** at
  `/agibot/software/mc_param/robot/lx2501_3_t2d5/rl_models/{cpgwalkrun_v25_v2,cpgtelecon_v3_fix}.onnx`.
  Our node runs on .41, so the files must be local to .41. FIRST check whether .41 already has them
  (`find /agibot -name cpgtelecon_v3_fix.onnx` on .41); if not, copy internally (cross-board ssh
  works): `scp 10.0.1.40:/agibot/software/mc_param/robot/lx2501_3_t2d5/rl_models/{cpgwalkrun_v25_v2,cpgtelecon_v3_fix}.onnx ~/onnx/`.
- SSH: I connect to the dev unit; you operate there.
      ROBOT_SSH = <ROBOT_SSH>          # e.g. ssh root@10.0.1.41 — ask me if blank

## Goal A — record the CPG / observation ground truth
Follow docs/hardware_bringup_cpgwalk.md "Part A". Run on the **dev unit (.41)** (ROS 2 env already
set up; it sees the robot topics over the network):
  ros2 topic info /aima/mc/rl/debug --verbose     # capture the message TYPE; tell me
  cd ~ && ros2 bag record -o cpg_debug_$(date +%H%M%S) \
    /aima/mc/rl/debug /aima/mc_debug_f64 \
    /aima/hal/joint/leg/state /aima/hal/joint/waist/state /aima/hal/joint/arm/state \
    /aima/hal/imu/torso/state /aima/hal/imu/chest/state /aima/mc/locomotion/velocity
Have me drive a ~20-30 s walk under the normal MC, then stop the recording. scp the bag back here.
Then (Goal A done) decode it: `python3 tools/validate_factory_teacher.py inspect <bag>` to learn the
rl/debug layout, and read off the 65-obs incl. the CPG phase q[61:65]; report what you find. This is
a guaranteed win even if we never get to Goal B.

## Goal B — run our loop on the robot (HARNESS REQUIRED, staged)
Follow docs/hardware_bringup_cpgwalk.md "Part C" precisely. Critical points:
- C1 ARBITRATION (the sanctioned mechanism — most important): the MC uses PRIORITY-BASED input-source
  arbitration (SDK doc 5.1.3). The proper takeover is NOT "kill the MC" — it is: register our own
  input source via the `SetMcInputSource` service (action ADD, a unique name, priority above the
  active source; RC=80/VR=70/app=60/voice=50/pnc=40), then publish. Study the SDK examples "6.1.6
  Register custom input source" + "6.1.9 Joint motor control" (and keyboard-control 6.1.10) for the
  canonical sequence, and likely a `SetMcAction` to a mode that yields the legs. VERIFY with
  `ros2 topic info /aima/hal/joint/leg/command --verbose` that only our node effectively drives the
  legs, and FIRST prove it with the single-joint example on the harness. If the MC still wins, STOP
  and tell me. (Note: our source has a timeout — if our node dies, the MC re-arbitrates, a safety net.)
- The deploy node enforces a STRICT stand-before-walk gate (cpgtelecon stands still, then cpgwalk
  walks). It will NOT walk until a firm stand is verified. Launch (on the robot, using the on-robot
  ONNX paths):
      ros2 run x2_policy_runtime cpgwalk_deploy --ros-args \
        -p onnx:=/agibot/software/mc_param/robot/lx2501_3_t2d5/rl_models/cpgwalkrun_v25_v2.onnx \
        -p stand_onnx:=/agibot/software/mc_param/robot/lx2501_3_t2d5/rl_models/cpgtelecon_v3_fix.onnx \
        -p imu_topic:=/aima/hal/imu/torso/state
  Then: enable (std_msgs/Bool true) -> watch log for "STAND verified firm -> WALK unlocked" ->
  only then publish /cpgwalk/cmd_vel (start vx 0.3). Abort with /cpgwalk/estop true or Ctrl-C.
- Stop at EVERY gate (C1, C2, C3, C4) and get my explicit OK before proceeding. Keep the harness
  taut. The hardware e-stop is the real safety net.

## How to work
- Verify before claiming success: show me actual command output, don't assume.
- At each gate, summarize what you observed and ask "proceed?" — wait for me.
- If anything is unexpected (publisher conflict, dcu_leg_safe trip, wobble, stale topics), STOP and
  report; do not push through. Prefer the safe option every time.
- Keep notes; at the end, commit any decoded data / findings and push.
```

---

### Notes for me (Liang), not for the laptop instance
- No ONNX transfer needed — the models already live on the robot at
  `/agibot/software/mc_param/robot/lx2501_3_t2d5/rl_models/`. Only the repo code needs to reach the
  robot (git clone/pull or scp), plus `onnxruntime` in the robot's python.
- Have the robot's SSH string ready, the gantry rigged and load-tested, and the e-stop tested.
- Goal A needs no harness and is the safe guaranteed win — prioritize it if time/safety is tight.
