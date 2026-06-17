# Hardware bring-up — record CPG ground truth + run cpgwalk in our own loop (0.9)

**Target:** X2 on a **gantry/harness**, software version **0.9.x** (all robots are 0.9).
**Goal of the session, in order of risk:**
1. **Part A — record the policy debug log** (zero risk; robot stays under the factory MC). This
   captures the *true* 65-obs incl. the CPG phase, which validates our whole control pipeline.
2. **Part B — decode it** (offline).
3. **Part C — run cpgwalk in OUR loop on the robot** (harness mandatory), staged HOLD → balance → walk.

> ⚠️ **Safety preconditions for Part C:** harness taut and load-tested; hardware e-stop within reach
> and TESTED before any motion; area clear; one person on e-stop, one on the keyboard. The robot has
> **no passive balance** under our control (R14) — if our loop is wrong it falls. Do Part A first so the
> session is a guaranteed win even if we stop before Part C.

---

## Part A — Record the CPG / obs ground truth (no risk, MC stays in control)

1. SSH to the robot and **source the 0.9 ROS 2 env** (so `aimdk_msgs` types resolve):
   ```bash
   ssh <robot>
   source /agibot/software/housekeeper/bin/setup.bash    # provides aimdk_msgs in ros2cli
   ```
2. Confirm the topics exist and learn the `rl/debug` type:
   ```bash
   ros2 topic list | grep -E "rl/debug|mc_debug|imu|joint/(leg|waist|arm)/state|locomotion/velocity"
   ros2 topic info  /aima/mc/rl/debug --verbose      # note the message TYPE + publisher count
   ros2 topic hz    /aima/mc/rl/debug                # expect ~50 Hz
   ```
3. **Record** (write to `~` — `/agibot/data` is not writable). Start recording BEFORE walking:
   ```bash
   cd ~
   ros2 bag record -o cpg_debug_$(date +%H%M%S) \
     /aima/mc/rl/debug /aima/mc_debug_f64 \
     /aima/hal/joint/leg/state /aima/hal/joint/waist/state /aima/hal/joint/arm/state \
     /aima/hal/imu/torso/state /aima/hal/imu/chest/state \
     /aima/mc/locomotion/velocity
   ```
4. With the robot **walking under the normal MC** (operator drives a forward walk, a turn, then
   stop) for **~20–30 s**, then `Ctrl-C` the recorder.
5. Copy the bag off the robot:
   ```bash
   scp -r <robot>:~/cpg_debug_* "/home/liang/Projects/terrain_recognition/data/"   # or any local dir
   ```
✅ **Gate A:** bag exists, `/aima/mc/rl/debug` has ~50 Hz × ~25 s ≈ 1250+ msgs. Session is already a win.

---

## Part B — Decode it (offline, on the laptop)

```bash
cd /home/liang/Projects/terrain_recognition
# 1) learn the rl/debug schema (field names / array layout)
python3 tools/validate_factory_teacher.py inspect data/cpg_debug_XXXXXX        # if mcap
#   (db3 bag: use tools/decode_x2_bag.py with --sdk pointing at the 0.9 SDK)
```
What we're looking for in `/aima/mc/rl/debug`: the 65-float observation per step. Read off
`q[61:65]` (the CPG phase) and confirm our `factory_cpgwalk.cpg_phase()` (0.6 s clock) matches its
frequency/shape, and that the 4th command dim `[9]` behaves as expected. Update
`x2_locomotion.cref.factory_cpgwalk` if anything differs. This closes the last sim-vs-real gap.

---

## Part C — Run cpgwalk in OUR loop on the robot (HARNESS REQUIRED)

### C0. Put the run machine in place — the DEVELOPMENT computing unit (10.0.1.41)
Run our node on the **dev unit (.41)** — NOT the MC unit (.40, prohibited). The dev unit already has
ROS 2 Humble + aimdk_msgs and is on the robot's ROS 2 network. Get our code + the ONNX onto .41:
```bash
# on .41: clone/pull this repo; ensure onnxruntime+numpy in its python.
# ONNX live on the MC unit (.40); copy them to .41 if absent (internal ssh works):
mkdir -p ~/onnx && scp 10.0.1.40:/agibot/software/mc_param/robot/lx2501_3_t2d5/rl_models/{cpgwalkrun_v25_v2,cpgtelecon_v3_fix}.onnx ~/onnx/
export WALK_ONNX=~/onnx/cpgwalkrun_v25_v2.onnx
export STAND_ONNX=~/onnx/cpgtelecon_v3_fix.onnx
```

### C1. STOP THE MC to obtain control authority (low-level joint control)
We do low-level joint control, so per Agibot's joint example (SDK 6.1.9) the native MC must be stopped.
**On PC1 (10.0.1.40)** — this admin command is allowed there:
```bash
aima em stop-app mc       # MC out of the loop. Recover anytime with: aima em start-app mc
ros2 topic info /aima/hal/joint/leg/command --verbose   # confirm the MC is no longer publishing
```
⚠️ The instant the MC stops there is **NO factory balance/estimator/fall-detect** — our node becomes
the entire controller. So Goal A (rl/debug) must already be recorded (it dies with the MC), the robot
must be on a **taut gantry**, and our node must be ready to take over immediately (next step).
**First prove control with ONE joint** (SDK `motocontrol.py`, oscillate a knee) and confirm
`dcu_leg_safe` doesn't trip before running the full policy. If anything is off, STOP.

✅ **Gate C1:** MC stopped; only OUR process publishes `/aima/hal/joint/leg/command`; one joint obeys us.

### C2. STAND on cpgtelecon under our loop (no walk yet)
```bash
ros2 run x2_policy_runtime cpgwalk_deploy --ros-args \
  -p onnx:="$WALK_ONNX" -p stand_onnx:="$STAND_ONNX" \
  -p imu_topic:=/aima/hal/imu/torso/state -p ramp_s:=3.0
ros2 topic pub -1 /cpgwalk/enable std_msgs/Bool '{data: true}'   # enter STAND phase
```
The node holds the robot in a **still stand on cpgtelecon** (stiffness ramping in over 3 s), with a
watchdog, per-joint limit clamps, and fall guard. On the harness, confirm a stable, NON-stepping
stand under our commands. Keep the harness taking some weight.

✅ **Gate C2:** robot holds the neutral pose smoothly under our loop; no oscillation, no limit clamps firing.

### C3. Firm stand FIRST — walk is hard-locked until verified (factory STABLE → MOVE)
The factory enters `STABLE`/`STAND_DEFAULT` (held by `cpgtelecon`) and only then transitions to
`MOVE`/`LOCOMOTION_DEFAULT` (`cpgwalk`). **Our node enforces the same as a strict gate** — on enable
it runs the policy at **zero command** and will **not apply any walk command until a firm stand is
verified and held** (`stand_hold_s`, default 1 s):
```bash
ros2 topic pub -1 /cpgwalk/cmd_vel geometry_msgs/Twist '{}'          # vx=0
ros2 topic pub -1 /cpgwalk/enable  std_msgs/Bool '{data: true}'      # STAND phase
# watch the log for:  "STAND verified firm -> WALK unlocked."
```
Firm-stand criteria (node params; defaults): `|pitch|<0.20`, `|roll|<0.15` rad, base `|omega|<0.5`
rad/s, stiffness fully ramped. A **hard fall guard** (`fall_pitch 0.7`, `fall_roll 0.5` from the
runtime `estimator`) forces HOLD if exceeded. The robot can never start walking from a lying/unstable
state — even if a `cmd_vel` is already being published, it stays at zero until the stand is verified.

✅ **Gate C3:** log shows `WALK unlocked`, and the robot stays upright/balanced for ≥30 s. If it never
unlocks, it is NOT standing firmly enough — do not force it; investigate posture/IMU first.

### C4. First steps (small forward command)
0.9 needs forward command ≥ `min_velx_command` = **0.3** to walk:
```bash
ros2 topic pub -r 10 /cpgwalk/cmd_vel geometry_msgs/Twist '{linear: {x: 0.3}}'
```
Expect a slow walk-in-place / forward shuffle on the harness (like MuJoCo `vx=0.4` → ~0.26 m/s).
Increase toward 0.5–0.6 only if clean.

### Abort at any time
- `ros2 topic pub -1 /cpgwalk/estop std_msgs/Bool '{data: true}'` → node latches HOLD at low stiffness.
- Kill the node (Ctrl-C) → publishing stops.
- Hardware e-stop → cuts motor power. **The harness is the real safety net.**

---

## Notes / known gaps to watch
- **IMU:** config has `use_chest_imu: false` → we default to `/aima/hal/imu/torso/state`. If `ros2 topic
  list` shows a different spelling (e.g. `torse`), pass it via `-p imu_topic:=`.
- **Obs filters:** the MC low-pass filters obs/actions (`CPGWalkConfig.filter`). Our node does not yet —
  acceptable for low-speed harness bring-up; add them (see `docs/factory_cpgwalk_contract.md`) before
  faster walking. Real IMU is noisier than sim, so this matters more on hardware.
- **4th command dim / CPG waveform:** approximate until Part B confirms them from `rl/debug`.
- This whole exercise is **Way A** (run the factory policy in our loop). It proves the takeover; it is
  still blind. Stairs come later via **Way B** (distill + add camera terrain).
