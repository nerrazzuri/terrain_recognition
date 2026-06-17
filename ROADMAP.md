# Roadmap

The **source-of-truth** plan for this project lives in:

➡️ [x2_terrain_stair_climbing_roadmap.md](x2_terrain_stair_climbing_roadmap.md)

That document defines the six-phase plan, the repository structure, the message/data
contracts, the training method, the safety rules, and the task backlog.

The **living checklist** of every module and task (with stable IDs and status boxes) is in:

➡️ [TASKS.md](TASKS.md)

## Rule of precedence

> The roadmap is the plan; TASKS.md is the checklist. **If the two disagree, the roadmap
> wins** — fix TASKS.md to match (or raise it). See [AGENTS.md](AGENTS.md) §2.

## ⚠️ Strategy revision (2026-06-17)

See **[roadmap §0.1](x2_terrain_stair_climbing_roadmap.md)**. After inspecting the robot runtime + SDK: the factory MC already ships a **production blind RL walker (`cpgwalk`)**, so we **use it for flat standing/walking instead of training our own** (our from-scratch RL flat-walk is kept only as pipeline validation — it walked but with a limping gait; the factory gait is natural). The real work is a **perception-aware stair-climbing policy**, run externally via **JOINT mode**, **warm-started by distilling the factory `cpgwalk.onnx`** + a height-map input. Phases reframed accordingly.

## Phase summary

| Phase | Name | Risk | Training | Primary output |
|------:|------|:----:|:--------:|----------------|
| 0 | Foundation & Repo Setup | — | No | repo skeleton + shared lib + configs |
| 1 | Terrain Awareness | Low | No | height map + terrain classification |
| 2 | Safe SDK Locomotion | Low/Med | No | safe **factory-`cpgwalk`** velocity adapter + stop-before-stairs |
| 3 | X2 Simulation Model | None | Validated | Isaac Lab / MuJoCo X2 env (pipeline proven by v3) |
| 4 | RL Stair Policy | Sim only | Yes | **distill factory `cpgwalk` → terrain-aware stair policy** |
| 5 | Sim-to-Real Deployment | **High** | Trained | stair policy via **JOINT-mode switch** (gated) |
| 6 | CReF Raw-Depth Policy | **High** | Yes | raw-depth perceptive upgrade |
