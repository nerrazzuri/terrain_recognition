# Stair detection result.
std_msgs/Header header
bool    stairs_detected
string  direction                   # "up" | "down" | "none"
float32 confidence
float32 rise_m
float32 tread_m
uint32  visible_step_count
float32 first_step_distance_m
float32 recommended_stop_distance_m
