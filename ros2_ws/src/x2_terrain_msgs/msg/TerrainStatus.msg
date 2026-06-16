# Fused single-decision summary of the terrain ahead.
# terrain_type is one of: flat_ground rough_ground slope_up slope_down
#   curb_or_single_step stairs_up stairs_down gap_or_hole platform unknown_unsafe
std_msgs/Header header
string  terrain_type
float32 confidence              # 0..1
float32 slope_angle_deg
float32 max_obstacle_height_m
float32 estimated_step_height_m
float32 estimated_step_depth_m
float32 gap_width_m
bool    safe_to_continue        # never true under uncertainty
string  reason                  # human-readable decision reason (always populated)
