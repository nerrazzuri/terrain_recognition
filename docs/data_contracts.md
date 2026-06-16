# Data Contracts

Field-level contracts for the custom messages in `x2_terrain_msgs`. These define the
interface between perception (Phase 1) and locomotion (Phase 2+). Fields follow roadmap
§5.3. Keep this doc in sync with the `.msg` / `.srv` files — they are the authority once
created.

## Conventions

- Units are SI: metres (`_m`), radians unless suffixed `_deg`, seconds.
- All grid/array messages carry a `std_msgs/Header` with the frame the data is expressed in
  (robot base frame for terrain data).
- Confidence is `0.0`–`1.0`. **Low confidence ⇒ treat as unsafe.**
- `terrain_type` is one of the fixed class strings (see below); unknown ⇒ `unknown_unsafe`.

## Terrain classes

```
flat_ground · rough_ground · slope_up · slope_down · curb_or_single_step ·
stairs_up · stairs_down · gap_or_hole · platform · unknown_unsafe
```

## `TerrainCell.msg`

A single elevation-map cell (used inside `TerrainGrid` or standalone debug).

```
float32 height_m          # terrain height relative to grid origin
float32 confidence        # 0..1
uint8   traversability    # 0 = unknown, 1..254 score, 255 = blocked  (see note)
```

## `TerrainGrid.msg` — `/x2/terrain/heightmap`

Robot-centred elevation map. First version: 2.0 m × 1.6 m, 0.04 m resolution ⇒ 50 × 40.

```
std_msgs/Header header
float32 resolution_m
uint32  width             # cells in x (forward)
uint32  height            # cells in y (lateral)
float32 origin_x_m        # grid origin in base frame
float32 origin_y_m
float32[] height_m        # row-major, length width*height
float32[] confidence      # parallel to height_m, 0..1; unknown cells low/zero
uint8[]   traversability  # parallel to height_m
```

Row-major indexing: `idx = iy * width + ix`. Coordinate conversion and indexing are unit
tested (`tests/unit`). Unknown cells must be marked unknown, **not** flat.

## `TerrainStatus.msg` — `/x2/terrain/status`

Fused single-decision summary of the terrain ahead.

```
std_msgs/Header header
string  terrain_type            # one of the fixed classes
float32 confidence              # 0..1
float32 slope_angle_deg
float32 max_obstacle_height_m
float32 estimated_step_height_m
float32 estimated_step_depth_m
float32 gap_width_m
bool    safe_to_continue        # never true under uncertainty
string  reason                  # human-readable decision reason (always populated)
```

## `StairEstimate.msg` — `/x2/terrain/stair_estimate`

```
std_msgs/Header header
bool    stairs_detected
string  direction                   # "up" | "down" | "none"
float32 confidence
float32 rise_m
float32 tread_m
uint32  visible_step_count
float32 first_step_distance_m
float32 recommended_stop_distance_m
```

## `SafetyDecision.msg` — `/x2/terrain/safety_decision`

Emitted by the safety supervisor; the authoritative stop/go record for logs.

```
std_msgs/Header header
bool    stop                    # true = motion must be zero
string  reason                  # which condition triggered (always populated)
float32 max_forward_velocity    # adapter's allowed forward velocity given terrain
bool    perception_fresh
bool    imu_fresh
bool    operator_estop
```

## `PolicyDebug.msg` — `/x2/policy/debug`

Dry-run / debug channel for the joint policy runtime (no real command while approval flag
is false).

```
std_msgs/Header header
float32[] observation
float32[] raw_action
float32[] filtered_action
bool      would_command          # what the node WOULD send if approved
string    safety_state
```

## `srv/ResetTerrainMap.srv`

```
# request
bool clear_history
---
# response
bool success
string message
```

## Decision policy (roadmap §5.4)

```
if confidence_low:            terrain_type = unknown_unsafe
elif gap_detected:            terrain_type = gap_or_hole
elif stairs_detected and rise > thr: terrain_type = stairs_up | stairs_down
elif single_step_detected:    terrain_type = curb_or_single_step
elif abs(slope_angle) > thr:  terrain_type = slope_up | slope_down
elif roughness > thr:         terrain_type = rough_ground
else:                         terrain_type = flat_ground
```

`safe_to_continue` is never `true` when confidence is low or the class is curb / stairs /
gap / unknown.
