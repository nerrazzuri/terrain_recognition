# Authoritative stop/go record emitted by the safety supervisor (for logs).
std_msgs/Header header
bool    stop                    # true = motion must be zero
string  reason                  # which condition triggered (always populated)
float32 max_forward_velocity    # adapter's allowed forward velocity given terrain
bool    perception_fresh
bool    imu_fresh
bool    operator_estop
