# Robot-centred elevation map. Row-major: idx = iy * width + ix.
# Unknown cells must be marked unknown (low/zero confidence), NOT flat.
std_msgs/Header header
float32 resolution_m
uint32  width             # cells in x (forward)
uint32  height            # cells in y (lateral)
float32 origin_x_m        # grid origin in base frame
float32 origin_y_m
float32[] height_m        # length width*height
float32[] confidence      # parallel to height_m, 0..1
uint8[]   traversability  # parallel to height_m
