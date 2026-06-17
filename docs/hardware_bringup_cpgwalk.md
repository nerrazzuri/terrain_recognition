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

### C0. Put the run machine in place
Run the node **on the robot's onboard computer** (best 50 Hz timing) or a **wired laptop on the same
`ROS_DOMAIN_ID`**. It needs: this repo, `onnxruntime`, `aimdk_msgs`, and the 0.9.7 ONNX:
```bash
export ONNX="/home/liang/Projects/X2 Locomotion/0.9.7/agibot/software/mc_param/robot/lx2501_3_t2d5/rl_models/cpgwalkrun_v25_v2.onnx"
source /agibot/software/housekeeper/bin/setup.bash      # aimdk_msgs
```

### C1. Confirm + claim the leg-command authority  (the R14 arbitration step — do this carefully)
The MC normally publishes `/aima/hal/joint/leg/command`. Two publishers fighting = a fall.
```bash
ros2 topic info /aima/hal/joint/leg/command --verbose    # how many publishers NOW? (expect MC)
```
**Before trusting cpgwalk, prove arbitration with ONE joint** using the SDK example on the harness:
suspend MC leg control (operator: set MC to a damping/passive state, or stop the MC module via the
process manager), re-check the publisher count is 0, then run the SDK `motocontrol.py` (oscillates one
knee). If the knee tracks our command and `dcu_leg_safe` doesn't trip → arbitration is ours. **If the
MC keeps publishing, STOP** — resolve how to release it (ask AgiBot) before going further.

✅ **Gate C1:** only OUR process publishes `/aima/hal/joint/leg/command`; one joint obeys us on the harness.

### C2. HOLD the neutral pose under our PD (no policy yet)
```bash
ros2 run x2_policy_runtime cpgwalk_deploy --ros-args \
  -p onnx:="$ONNX" -p imu_topic:=/aima/hal/imu/torso/state -p ramp_s:=3.0
```
The node starts in **HOLD**: it PD-holds the factory neutral stand pose, stiffness ramping in over 3 s,
with a watchdog (stale state → HOLD) and per-joint limit clamps. On the harness, confirm the robot
settles into a stable standing pose under our commands. Keep the harness taking some weight.

✅ **Gate C2:** robot holds the neutral pose smoothly under our loop; no oscillation, no limit clamps firing.

### C3. Balance in place (policy, zero velocity)
```bash
ros2 topic pub -1 /cpgwalk/cmd_vel geometry_msgs/Twist '{}'          # vx=0
ros2 topic pub -1 /cpgwalk/enable  std_msgs/Bool '{data: true}'      # RUN
```
Policy now runs with a zero command (it should make tiny balancing motions, no stepping — like the
MuJoCo `vx=0` case). Watch IMU/posture.

✅ **Gate C3:** stays upright and balanced under the policy for ≥30 s.

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
