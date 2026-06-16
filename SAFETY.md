# SAFETY

> This is a real humanoid-robot project, not an SDK demo. **Hardware can be damaged and
> people can be hurt.** The rules below are non-negotiable. They are copied from roadmap
> §1 (Core Constraints) and §14 (Non-Negotiable Safety Rule); the roadmap is the source of
> truth — see [x2_terrain_stair_climbing_roadmap.md](x2_terrain_stair_climbing_roadmap.md).

---

## Global safety flag

| Flag | Default | Meaning |
|------|---------|---------|
| `REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED` | **false** | Must stay false until the Phase 5 *documented* safety review. |

While the flag is **false** (the standing state):

- Policy nodes run **dry-run only** and may publish **debug topics only**.
- **Never** publish to `/aima/hal/joint/leg/command` (or any leg joint command topic).
- Only SDK velocity-level control through the **safe velocity adapter** is allowed.

No coding agent should generate or run a real-robot low-level leg-joint stair-climbing
command unless the task explicitly states `REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED=true`.

---

## 1. Hard safety constraints (roadmap §1.1)

- **No real stair climbing** during Phase 1 or Phase 2.
- **No direct low-level leg-joint policy** on a free-standing robot until *all* of:
  - simulator policy is stable;
  - ONNX export is verified;
  - joint ordering is verified;
  - safety supervisor is active;
  - robot is suspended or physically protected;
  - emergency stop is tested.
- **No raw camera subscriptions across compute units** when bandwidth is high. Prefer point
  cloud or compressed streams.
- **All real-robot motion** must have, with no exceptions:
  - velocity / action clamps;
  - state watchdog;
  - IMU fall detector;
  - timeout detector;
  - operator emergency stop;
  - logging enabled.

Do not write motion code that omits any of these.

---

## 2. Fail closed

If a required input is missing or stale — perception, IMU, command freshness — the safe
action is to **stop**. Never continue on missing data. Unknown terrain is treated as unsafe
until proven otherwise.

---

## 3. Don't skip phase gates

Each phase has a Definition of Done in [TASKS.md](TASKS.md). Do not start a later phase
before the prior gate passes. Height-map locomotion comes before raw-depth CReF.

The staged hardware bring-up order (roadmap §9.3) must be followed in sequence, passing the
Go/No-Go checklist (roadmap §9.5) before each escalation:

```
Level 0  offline replay (no robot command topic)
Level 1  hardware dry-run (powered, debug only, no leg command)
Level 2  suspended standing (gantry; operator e-stop ready)
Level 3  suspended stepping in place
Level 4  foam obstacle 2 → 5 → 8 cm
Level 5  single wooden step 5 → 8 → 10–12 cm
Level 6  real stairs — ONLY after documented safety review + all lower levels pass
```

---

## 4. Go / No-Go checklist (roadmap §9.5)

**No-go** if any of the following is true:

- joint order not verified
- safety supervisor disabled
- emergency stop not tested
- policy output not filtered
- robot not suspended for first test
- logs not recording
- operator not present
- battery / power unstable
- perception confidence low
- latency too high

---

## 5. When in doubt

If a request would publish real leg-joint commands, climb real stairs, or disable a safety
check while the approval flag is false — **stop and ask.** Do not proceed on assumed
authorization. If a topic, joint, or limit named in the roadmap can't be verified on the
actual robot/SDK, flag it rather than guessing.
