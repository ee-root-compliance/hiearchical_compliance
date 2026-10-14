"""
Utility functions for manipulation task evaluation.

Provides helper functions specific to manipulation tasks.
"""

import torch
from typing import Dict, Tuple, List, Optional
from pathlib import Path


def load_manipulation_checkpoint(checkpoint_path: str):
    """
    Load a manipulation task checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        dict: Checkpoint data
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def get_grasping_action(action: torch.Tensor, gripper_open_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split action into arm control and gripper control.
    
    Args:
        action: Full action tensor
        gripper_open_threshold: Threshold for gripper open/close
        
    Returns:
        arm_action: Action for arm joints
        gripper_action: Action for gripper (open/close)
    """
    # This is a placeholder - adjust based on your actual action space
    # Example: if last action dimension is gripper
    arm_action = action[..., :-1]
    gripper_action = (action[..., -1] > gripper_open_threshold).float()
    
    return arm_action, gripper_action


def compute_object_distance(robot_ee_pos: torch.Tensor, object_pos: torch.Tensor) -> torch.Tensor:
    """
    Compute distance between robot end-effector and object.
    
    Args:
        robot_ee_pos: End-effector position [batch, 3]
        object_pos: Object position [batch, 3]
        
    Returns:
        torch.Tensor: Distance [batch]
    """
    return torch.norm(robot_ee_pos - object_pos, dim=-1)


def check_object_grasped(gripper_contact_force: torch.Tensor, force_threshold: float = 1.0) -> torch.Tensor:
    """
    Check if object is grasped based on contact forces.
    
    Args:
        gripper_contact_force: Contact force [batch, 2] (left, right fingers)
        force_threshold: Minimum force to consider grasped
        
    Returns:
        torch.Tensor: Boolean tensor indicating grasp [batch]
    """
    # Both fingers must have contact
    grasped = (gripper_contact_force > force_threshold).all(dim=-1)
    return grasped


def log_manipulation_metrics(
    episode_num: int,
    step: int,
    reward: torch.Tensor,
    object_dist: torch.Tensor = None,
    is_grasped: torch.Tensor = None,
):
    """
    Log manipulation task metrics.
    
    Args:
        episode_num: Episode number
        step: Step number
        reward: Current reward
        object_dist: Distance to object (optional)
        is_grasped: Whether object is grasped (optional)
    """
    log_str = f"[Ep {episode_num:3d}, Step {step:4d}] Reward: {reward.mean():.4f}"
    
    if object_dist is not None:
        log_str += f" | Dist: {object_dist.mean():.3f}"
    
    if is_grasped is not None:
        grasp_rate = is_grasped.float().mean().item()
        log_str += f" | Grasp: {grasp_rate:.1%}"
    
    print(log_str)


def get_contact_forces(env, body_names: Optional[List[str]] = None, mean_over_history: bool = True):
    """
    Retrieve net contact forces from the environment's ContactSensor.

    Args:
        env: The environment instance (expected to have `scene` with a `contact_forces` sensor).
        body_names: Optional list or pattern of body names to filter (passed to sensor.find_bodies).
        mean_over_history: If True, use the history-averaged forces (net_forces_w_history.mean(1)),
                           otherwise use current net_forces_w.

    Returns:
        Dict mapping body_name -> torch.Tensor of shape [num_envs, 3]. If the sensor or bodies are
        not available, returns an empty dict.
    """
    forces_dict: Dict[str, torch.Tensor] = {}

    # Print target info for user clarity
    if body_names is None:
        print("Visualization targets: all bodies (body_names not specified)")
    else:
        print(f"Visualization targets: {body_names}")

    try:
        contact_sensor = env.scene["contact_forces"]
    except Exception:
        # No contact sensor available
        print("Contact sensor not found (env.scene['contact_forces'] is unavailable)")
        return forces_dict

    # choose data source
    data = None
    try:
        if mean_over_history and hasattr(contact_sensor.data, "net_forces_w_history"):
            data = contact_sensor.data.net_forces_w_history.mean(1)
            print("Using net_forces_w_history.mean(1) as the contact-force source")
        elif hasattr(contact_sensor.data, "net_forces_w"):
            data = contact_sensor.data.net_forces_w
            print("Using net_forces_w as the contact-force source")
        else:
            print("No contact data (contact_sensor.data lacks net_forces_w and net_forces_w_history)")
            return forces_dict
    except Exception:
        print("Error reading contact_sensor.data")
        return forces_dict

    # data expected shape: [num_envs, num_bodies, 3]
    try:
        body_ids, found_names = contact_sensor.find_bodies(body_names if body_names is not None else ".*")
    except Exception:
        print("Error calling contact_sensor.find_bodies")
        return forces_dict

    if len(found_names) == 0:
        print(f"No matching body_names found: {body_names}")
        return forces_dict

    print(f"Found bodies: {found_names}")

    # Quick check: any contact at all?
    try:
        max_norm = data.norm(dim=-1).max().item()
    except Exception:
        max_norm = 0.0

    if max_norm <= 1e-6:
        print("No contact (all body forces are zero)")
        return forces_dict

    for bid, bname in zip(body_ids, found_names):
        try:
            forces_dict[bname] = data[:, bid, :].clone()
        except Exception:
            # skip if indexing fails
            continue

    return forces_dict


def debug_draw_contact_forces(env, body_names: Optional[List[str]] = None, scale: float = 0.01, color=(0.2, 0.8, 0.2, 1.0), env_idx: int = 0, print_forces: bool = True):
    """
    Print and (optionally) draw contact forces for the requested bodies.

    Args:
        env: The environment instance (must expose `debug_draw` and `scene`).
        body_names: Optional list or pattern of body names to visualize (passed to sensor.find_bodies).
        scale: Multiplier applied to force vectors for visualization.
        color: RGBA tuple for drawing arrows/points.
        env_idx: Which environment index to display when printing/drawing single env values.
        print_forces: Whether to print force magnitudes to stdout.

    Notes:
        This function is defensive: it will early-return if ContactSensor or debug drawing
        is not available in the provided `env`.
    """
    if not hasattr(env, "scene"):
        print("env has no scene attribute; cannot read contact information")
        return

    # Print requested target
    if body_names is None:
        print("debug_draw_contact_forces: targets = all bodies")
    else:
        print(f"debug_draw_contact_forces: targets = {body_names}")

    try:
        contact_sensor = env.scene["contact_forces"]
    except Exception:
        print("Contact sensor not found; cannot visualize contact forces")
        return

    forces = get_contact_forces(env, body_names=body_names, mean_over_history=True)
    if not forces:
        print("No matching bodies found, or no contact (get_contact_forces returned empty)")
        return

    print(f"Bodies to visualize: {list(forces.keys())}")

    # Try to obtain body positions for drawing. We'll try several common places and be defensive.
    body_positions = {}
    # 1) contact_sensor.data may contain body positions
    try:
        if hasattr(contact_sensor.data, "body_pos_w"):
            bp = contact_sensor.data.body_pos_w  # [num_envs, num_bodies, 3]
            for name in forces.keys():
                ids, names = contact_sensor.find_bodies(name)
                if len(ids) > 0:
                    body_positions[name] = bp[env_idx, ids[0], :].clone()
    except Exception:
        pass

    # 2) try to find object/articulation in the scene by name
    for name in forces.keys():
        if name in body_positions:
            continue
        try:
            obj = None
            if hasattr(env.scene, "get"):
                obj = env.scene.get(name, None)
            if obj is None and name in env.scene:
                obj = env.scene[name]
            if obj is not None and hasattr(obj, "data"):
                if hasattr(obj.data, "root_pos_w"):
                    body_positions[name] = obj.data.root_pos_w[env_idx].clone()
                elif hasattr(obj.data, "body_pos_w"):
                    body_positions[name] = obj.data.body_pos_w[env_idx, 0].clone()
        except Exception:
            pass

    # If debug draw not available, only print
    has_draw = hasattr(env, "debug_draw") and (getattr(env, "sim", None) is None or env.sim.has_gui())

    any_nonzero = False
    for name, vec in forces.items():
        # vec is [num_envs, 3]
        try:
            v = vec[env_idx]
        except Exception:
            # fallback to first env
            v = vec[0]

        try:
            mag = v.norm().item()
        except Exception:
            mag = float("nan")

        if mag and mag > 1e-6:
            any_nonzero = True

        if print_forces:
            if mag == float("nan"):
                print(f"Contact force [{name}]: unable to compute magnitude")
            else:
                print(f"Contact force [{name}] (env {env_idx}): {mag:.3f} N, vec=({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})")

        if has_draw and name in body_positions and mag > 1e-6:
            try:
                pos = body_positions[name]
                # draw an arrow (vector) and a point at the body position
                env.debug_draw.vector(pos, v * scale, color=color)
                env.debug_draw.point(pos, color=color)
            except Exception:
                # don't abort drawing others
                continue

    if not any_nonzero:
        print("No contact (all queryable body forces are very small)")


def list_contact_sensor_bodies(env):
    """
    Print all body names known to the contact sensor (for debugging name mismatches).
    """
    try:
        contact_sensor = env.scene["contact_forces"]
    except Exception:
        print("Contact sensor not found (env.scene['contact_forces'] is unavailable)")
        return

    try:
        ids, names = contact_sensor.find_bodies(".*")
        if len(names) == 0:
            print("Contact sensor exists, but no bodies are registered")
        else:
            print(f"Contact sensor registered bodies ({len(names)}): {names}")
    except Exception as e:
        print(f"Error calling contact_sensor.find_bodies: {e}")
