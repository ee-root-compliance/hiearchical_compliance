"""
Object definitions for manipulation environments.

This module provides pre-defined rigid objects that can be added to scenes
via configuration. Objects are defined using IsaacLab's RigidObjectCfg.

Usage in config:
    task:
      objects:
        - type: box_dynamic
          pos: [0.5, 0.0, 0.5]
          size: [0.1, 0.1, 0.1]
        - type: wall
          pos: [1.0, 0.0, 0.5]
          
Usage in code:
    from active_adaptation.envs.objects import create_object_cfg
    obj_cfg = create_object_cfg("box_dynamic", pos=[0.5, 0, 0.5], size=[0.1, 0.1, 0.1])
"""

import torch
from typing import Optional, Tuple, List, Dict, Any

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.assets import AssetBaseCfg
    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False


# ============================================================================
# Object Factory Functions
# ============================================================================

def create_box_cfg(
    name: str,
    pos: Tuple[float, float, float] = (0.5, 0.0, 0.5),
    rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    size: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    color: Tuple[float, float, float] = (0.8, 0.4, 0.2),
    mass: float = 1.0,
    static: bool = False,
    collision: bool = True,
    friction: float = None
) -> "RigidObjectCfg":
    """
    Create a box rigid object configuration.
    
    Args:
        name: Unique name for the object
        pos: Initial position (x, y, z) relative to env origin
        rot: Initial rotation as quaternion (w, x, y, z)
        size: Box dimensions (x, y, z)
        color: RGB color tuple (0-1)
        mass: Mass in kg (ignored if static)
        static: If True, object is fixed in place
        collision: If True, enables collision
        
    Returns:
        RigidObjectCfg for the box
    """
    if not ISAACLAB_AVAILABLE:
        raise RuntimeError("IsaacLab not available")
    
    # For static objects, use very large mass instead of kinematic
    # Kinematic objects don't collide properly with articulations
    if static:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=True,
            linear_damping=1000.0,  # High damping to keep it still
            angular_damping=1000.0,
            max_linear_velocity=0.0,
            max_angular_velocity=0.0,
        )
        mass_props = sim_utils.MassPropertiesCfg(mass=10000.0)  # Very heavy
    else:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
        )
        mass_props = sim_utils.MassPropertiesCfg(mass=mass)
    
    # Prepare collision properties
    collision_props_kwargs = {"collision_enabled": collision}
    if friction is not None:
        collision_props_kwargs["friction"] = friction
    # Build collision_cfg before constructing the RigidObjectCfg so we
    # can use normal statements (some IsaacLab versions expect different
    # kwarg names for friction).
    try:
        if "friction" in collision_props_kwargs and collision_props_kwargs["friction"] is not None:
            friction_val = collision_props_kwargs.pop("friction")
            tried = False
            candidates = ["friction", "static_friction", "friction_coefficient", "mu", "static_friction_coefficient"]
            for cand in candidates:
                try_kw = dict(collision_props_kwargs)
                try_kw[cand] = friction_val
                try:
                    collision_cfg = sim_utils.CollisionPropertiesCfg(**try_kw)
                    tried = True
                    break
                except TypeError:
                    continue
            if not tried:
                collision_cfg = sim_utils.CollisionPropertiesCfg(**collision_props_kwargs)
        else:
            collision_cfg = sim_utils.CollisionPropertiesCfg(**collision_props_kwargs)
    except TypeError:
        try:
            collision_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=collision)
        except Exception:
            collision_cfg = sim_utils.CollisionPropertiesCfg(**{k: v for k, v in collision_props_kwargs.items() if k in ("collision_enabled",)})

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.CuboidCfg(
            size=size,
            visible=True,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
            ),
            rigid_props=rigid_props,
            mass_props=mass_props,
            collision_props=collision_cfg,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
        ),
    )


def create_sphere_cfg(
    name: str,
    pos: Tuple[float, float, float] = (0.5, 0.0, 0.5),
    rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    radius: float = 0.05,
    color: Tuple[float, float, float] = (0.2, 0.6, 0.8),
    mass: float = 0.5,
    static: bool = False,
    collision: bool = True,
) -> "RigidObjectCfg":
    """Create a sphere rigid object configuration."""
    if not ISAACLAB_AVAILABLE:
        raise RuntimeError("IsaacLab not available")
    
    if static:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=True,
            linear_damping=1000.0,
            angular_damping=1000.0,
            max_linear_velocity=0.0,
            max_angular_velocity=0.0,
        )
        mass_props = sim_utils.MassPropertiesCfg(mass=10000.0)
    else:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
        )
        mass_props = sim_utils.MassPropertiesCfg(mass=mass)
    
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.SphereCfg(
            radius=radius,
            visible=True,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
            ),
            rigid_props=rigid_props,
            mass_props=mass_props,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=collision,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
        ),
    )


def create_cylinder_cfg(
    name: str,
    pos: Tuple[float, float, float] = (0.5, 0.0, 0.5),
    rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    radius: float = 0.05,
    height: float = 0.2,
    color: Tuple[float, float, float] = (0.6, 0.6, 0.6),
    mass: float = 1.0,
    static: bool = False,
    collision: bool = True,
) -> "RigidObjectCfg":
    """Create a cylinder rigid object configuration."""
    if not ISAACLAB_AVAILABLE:
        raise RuntimeError("IsaacLab not available")
    
    if static:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=True,
            linear_damping=1000.0,
            angular_damping=1000.0,
            max_linear_velocity=0.0,
            max_angular_velocity=0.0,
        )
        mass_props = sim_utils.MassPropertiesCfg(mass=10000.0)
    else:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
        )
        mass_props = sim_utils.MassPropertiesCfg(mass=mass)
    
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            visible=True,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
            ),
            rigid_props=rigid_props,
            mass_props=mass_props,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=collision,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
        ),
    )


def create_wall_cfg(
    name: str,
    pos: Tuple[float, float, float] = (1.0, 0.0, 0.5),
    rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    size: Tuple[float, float, float] = (0.05, 2.0, 1.0),  # thin, wide, tall
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7),
) -> "RigidObjectCfg":
    """Create a static wall configuration."""
    return create_box_cfg(
        name=name,
        pos=pos,
        rot=rot,
        size=size,
        color=color,
        static=True,
        collision=True,
    )


def create_table_cfg(
    name: str,
    pos: Tuple[float, float, float] = (0.6, 0.0, 0.4),
    rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    size: Tuple[float, float, float] = (0.6, 0.8, 0.02),  # table top
    height: float = 0.4,
    color: Tuple[float, float, float] = (0.55, 0.35, 0.2),
) -> "RigidObjectCfg":
    """Create a static table (just the top surface for simplicity)."""
    return create_box_cfg(
        name=name,
        pos=(pos[0], pos[1], height),
        rot=rot,
        size=size,
        color=color,
        static=True,
        collision=True,
    )


# ============================================================================
# Object Type Registry
# ============================================================================

OBJECT_TYPES = {
    "box_dynamic": lambda **kw: create_box_cfg(static=False, **kw),
    "box_static": lambda **kw: create_box_cfg(static=True, **kw),
    "box": lambda **kw: create_box_cfg(**kw),
    "sphere_dynamic": lambda **kw: create_sphere_cfg(static=False, **kw),
    "sphere_static": lambda **kw: create_sphere_cfg(static=True, **kw),
    "sphere": lambda **kw: create_sphere_cfg(**kw),
    "cylinder_dynamic": lambda **kw: create_cylinder_cfg(static=False, **kw),
    "cylinder_static": lambda **kw: create_cylinder_cfg(static=True, **kw),
    "cylinder": lambda **kw: create_cylinder_cfg(**kw),
    "wall": create_wall_cfg,
    "table": create_table_cfg,
}


def create_object_cfg(obj_type: str, **kwargs) -> "RigidObjectCfg":
    """
    Create an object configuration by type name.
    
    Args:
        obj_type: Type of object (box_dynamic, box_static, wall, table, etc.)
        **kwargs: Object-specific parameters (name, pos, size, etc.)
        
    Returns:
        RigidObjectCfg for the object
        
    Example:
        cfg = create_object_cfg("box_dynamic", name="box1", pos=[0.5, 0, 0.5], size=[0.1, 0.1, 0.1])
    """
    if obj_type not in OBJECT_TYPES:
        raise ValueError(f"Unknown object type: {obj_type}. Available: {list(OBJECT_TYPES.keys())}")
    
    return OBJECT_TYPES[obj_type](**kwargs)


def add_objects_to_scene(scene_cfg, objects_config: List[Dict[str, Any]]):
    """
    Add multiple objects to a scene configuration.
    
    Args:
        scene_cfg: IsaacLab InteractiveSceneCfg
        objects_config: List of object configurations, each with:
            - type: Object type (box_dynamic, wall, etc.)
            - name: Unique name (optional, auto-generated if not provided)
            - pos: Position [x, y, z]
            - ... other type-specific params
            
    Example:
        objects_config = [
            {"type": "box_dynamic", "pos": [0.5, 0, 0.5], "size": [0.1, 0.1, 0.1]},
            {"type": "wall", "pos": [1.0, 0, 0.5]},
        ]
        add_objects_to_scene(scene_cfg, objects_config)
    """
    from omegaconf import OmegaConf, ListConfig, DictConfig
    
    for i, obj_cfg in enumerate(objects_config):
        # Convert OmegaConf to plain dict
        if isinstance(obj_cfg, DictConfig):
            obj_cfg = OmegaConf.to_container(obj_cfg, resolve=True)
        else:
            obj_cfg = dict(obj_cfg)  # Make a copy to avoid modifying original
        
        obj_type = obj_cfg.pop("type")
        name = obj_cfg.pop("name", f"object_{i}")
        
        # Convert lists to tuples for pos, rot, size, color
        for key in ["pos", "rot", "size", "color"]:
            if key in obj_cfg:
                val = obj_cfg[key]
                if isinstance(val, (list, ListConfig)):
                    obj_cfg[key] = tuple(val)
        
        rigid_cfg = create_object_cfg(obj_type, name=name, **obj_cfg)
        setattr(scene_cfg, name, rigid_cfg)
        print(f"[Objects] Added {obj_type}: {name} at {obj_cfg.get('pos', 'default')}")
        print(f"          prim_path: {rigid_cfg.prim_path}")
        print(f"          collision_enabled: {rigid_cfg.spawn.collision_props.collision_enabled if rigid_cfg.spawn.collision_props else 'None'}")
    
    return scene_cfg
