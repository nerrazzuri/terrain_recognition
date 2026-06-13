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

## Phase summary

| Phase | Name | Risk | Training | Primary output |
|------:|------|:----:|:--------:|----------------|
| 0 | Foundation & Repo Setup | — | No | repo skeleton + shared lib + configs |
| 1 | Terrain Awareness | Low | No | height map + terrain classification |
| 2 | Safe SDK Locomotion | Low/Med | No | safe velocity adapter |
| 3 | X2 Simulation Model | None | Not yet | Isaac Lab / MuJoCo X2 environment |
| 4 | RL Locomotion Training | Sim only | Yes | PPO height-map policy |
| 5 | Sim-to-Real Deployment | **High** | Trained | suspended / low-step trials |
| 6 | CReF Raw-Depth Policy | **High** | Yes | raw-depth perceptive policy |
