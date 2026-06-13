# X2 Model Assets — provenance

These assets are the official Agibot X2 model, placed here for Phase 3 (sim model).

| File | Source | Notes |
|------|--------|-------|
| `x2_ultra.urdf` | `X2_URDF-v1.3.0` | 29-DoF (12 leg + 3 waist + 12 arm + 2 head); full-collision |
| `x2_ultra_simple_collision.urdf` | `X2_URDF-v1.3.0` | simplified collision — prefer for sim |
| `meshes/*.STL` | `X2_URDF-v1.3.0` | **gitignored** (~112 MB). Repopulate with `tools/fetch_x2_assets.sh` |
| `../../mujoco/model/x2_ultra.xml` | `X2_URDF-v1.3.0` | MuJoCo MJCF (secondary sim); `meshes/` symlinks here |
| `../../mujoco/model/scene.xml` | `X2_URDF-v1.3.0` | MuJoCo scene wrapper |

Joint limits were extracted from `x2_ultra.urdf` into
[`configs/joint_limits_x2_ultra.yaml`](../../../configs/joint_limits_x2_ultra.yaml)
(`verified: true`). Topic names + the AimDK velocity/registration API were verified against
SDK `lx2501_3-v0.9.0.4` and recorded in
[`configs/robot_topics.yaml`](../../../configs/robot_topics.yaml).

## USD conversion (P3-M1-T2)

Isaac Lab can import the URDF directly via its URDF importer, or convert to USD once:

```
# inside an Isaac Lab environment
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    training/isaac_lab/assets/x2_ultra_simple_collision.urdf \
    training/isaac_lab/assets/x2.usd --merge-joints
```

Set `X2_USD_PATH` to the produced `x2.usd`, or point `x2_robot_cfg` at the URDF importer.
The mass/inertia/contact validation (spawn-and-stand AC) still requires Isaac Lab running.
