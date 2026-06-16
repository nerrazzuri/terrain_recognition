# A single elevation-map cell. See docs/data_contracts.md.
float32 height_m          # terrain height relative to grid origin
float32 confidence        # 0..1
uint8   traversability    # 0 = unknown, 1..254 score, 255 = blocked
