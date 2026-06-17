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

## Environment / transfer (do this before anything)
- This repo is already cloned here. `git pull` to be current.
- The factory ONNX models are NOT in git (large binaries). I will transfer them; confirm these two
  files exist and tell me their path:
      cpgwalkrun_v25_v2.onnx   (cpgwalk / LOCOMOTION)
      cpgtelecon_v3_fix.onnx   (cpgtelecon / STAND_DEFAULT)
  If missing, STOP and ask me to copy the 0.9.7 runtime `rl_models/` folder over.
- The robot's onboard computer has ROS 2 + aimdk_msgs. We will operate on the robot over SSH:
      ROBOT_SSH = <ROBOT_SSH>          # e.g. ssh -p 22 user@192.168.x.x  — ask me if blank
  Plan: run RECORDING and the DEPLOY NODE on the robot's onboard computer (best 50/100 Hz timing).
  Copy this repo + the two ONNX files to the robot with scp. Ensure `onnxruntime` + `numpy` are
  installed in the robot's python; if not, ask me before installing.

## Goal A — record the CPG / observation ground truth
Follow docs/hardware_bringup_cpgwalk.md "Part A". In short, over SSH on the robot:
  source /agibot/software/housekeeper/bin/setup.bash
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
- C1 ARBITRATION (UNRESOLVED — most important): the factory MC normally publishes
  /aima/hal/joint/leg/command. We must release that authority first. Verify with
  `ros2 topic info /aima/hal/joint/leg/command --verbose` that ONLY our node publishes before any
  policy runs. First prove control with the SINGLE-JOINT SDK example (motocontrol.py) on the harness.
  If the MC keeps publishing, STOP and tell me — do not fight it.
- The deploy node enforces a STRICT stand-before-walk gate (cpgtelecon stands still, then cpgwalk
  walks). It will NOT walk until a firm stand is verified. Launch (on the robot):
      ros2 run x2_policy_runtime cpgwalk_deploy --ros-args \
        -p onnx:=<cpgwalkrun_v25_v2.onnx> -p stand_onnx:=<cpgtelecon_v3_fix.onnx> \
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
- Transfer the two ONNX files (and ideally the whole `0.9.7/.../rl_models/` folder) to the laptop or
  straight to the robot before starting.
- Have the robot's SSH string ready, the gantry rigged and load-tested, and the e-stop tested.
- Goal A needs no harness and is the safe guaranteed win — prioritize it if time/safety is tight.
