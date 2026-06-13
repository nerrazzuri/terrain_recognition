# Dry-run / debug channel for the joint policy runtime.
# No real command is sent while REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED is false.
std_msgs/Header header
float32[] observation
float32[] raw_action
float32[] filtered_action
bool      would_command          # what the node WOULD send if approved
string    safety_state
