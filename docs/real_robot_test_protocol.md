# Real-Robot Test Protocol

Procedures for any test that touches real X2 hardware. Derived from roadmap §6.5 (Phase 2)
and §9.3–§9.5 (Phase 5). **Read [../SAFETY.md](../SAFETY.md) first.** Every real-robot test
produces saved logs — logging is part of the test, not optional.

## Preconditions (all tests)

- Operator present with a tested emergency stop within reach.
- Clear area; bystanders aware.
- Battery / power stable.
- Logging confirmed recording before motion starts.
- Topics verified for this session (`tools/check_topics.sh`, `tools/check_qos.sh`).

## Phase 2 — safe velocity locomotion (no climbing)

Test order (roadmap §6.5):

1. **Dry-run** — publish to debug topic only, no real velocity.
2. **Standing** — zero velocity only; confirm source registration + watchdog.
3. **Flat walk** — 0.05 m/s in open flat area.
4. **Static obstacle / box ahead** — must stop.
5. **Mock stair / curb ahead** — must stop.
6. **Real stairs ahead** — must stop *before* reaching the first step.

Pass criteria: stops are triggered by terrain perception (provable from logs), commands are
smooth, watchdog zeros velocity when perception stops, manual e-stop overrides everything.

## Phase 5 — staged joint-policy bring-up

`REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED` must be `true` (set only after the documented
safety review) before Level 2+. Escalate **in order**; pass the Go/No-Go gate before each
step up.

| Level | Setup | Allowed motion |
|:-----:|-------|----------------|
| 0 | offline replay, no robot command topic | none (shape/obs check) |
| 1 | powered, debug output only | none (compare expected joint targets) |
| 2 | suspended / gantry, operator e-stop ready | standing balance / tiny offsets |
| 3 | suspended | stepping in place (verify joint order, sign, delay) |
| 4 | foam obstacle | 2 → 5 → 8 cm |
| 5 | single wooden step | 5 → 8 → 10–12 cm |
| 6 | real stairs | **only** after documented safety review + all lower levels pass |

## Go / No-Go checklist (roadmap §9.5)

Confirm **none** of these is true before proceeding:

- [ ] joint order not verified
- [ ] safety supervisor disabled
- [ ] emergency stop not tested
- [ ] policy output not filtered
- [ ] robot not suspended (first tests)
- [ ] logs not recording
- [ ] operator not present
- [ ] battery / power unstable
- [ ] perception confidence low
- [ ] latency too high

## Required log fields (roadmap §9.4)

```
timestamp · robot mode · terrain status · heightmap/sampled heights ·
policy observation · policy raw action · filtered action · joint state ·
IMU state · safety supervisor state · operator command · stop reason
```

Logs are saved per test and analysed offline (`tools/analyze_policy_log.py`,
`tools/compare_sim_real.py`).
