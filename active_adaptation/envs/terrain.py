from isaaclab.terrains import (
    TerrainImporterCfg,
)

import isaaclab.sim as sim_utils

PLANE_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0
    ),
)

PLANE_LOCO_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
)

TERRAINS = {
    "plane": PLANE_TERRAIN_CFG,
    "plane_loco": PLANE_LOCO_TERRAIN_CFG,
}

