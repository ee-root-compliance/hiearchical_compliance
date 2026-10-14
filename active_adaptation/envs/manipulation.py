"""
Manipulation environment for grasping and object manipulation tasks.
Extends the base locomotion environment with object interaction capabilities.
"""

import torch
from isaaclab.assets import Articulation
from omegaconf import OmegaConf

from .locomotion import SimpleEnv
from .mdp.commands.teleoperation import TeleopCommand


class ManipulationEnv(SimpleEnv):
    """
    Manipulation task environment.
    
    Inherits from SimpleEnv (training environment) and adds:
    - Teleoperation command system (VR input via UDP)
    - Static/dynamic objects (boxes, etc.)
    - Object interaction sensors
    - Manipulation-specific observations and rewards
    
    Policy input/output remains compatible with trained policy.
    """

    def __init__(self, cfg):
        """
        Initialize the manipulation environment.
        
        Args:
            cfg: Hydra configuration object
        """
        # Before calling super().__init__, replace command system
        if not hasattr(cfg, 'command'):
            cfg.command = OmegaConf.create()
        
        # Store original command config
        original_command_cfg = OmegaConf.to_container(cfg.command, resolve=True)
        
        # Create modified config with teleoperation
        cfg.command = OmegaConf.create({
            '_target_': 'active_adaptation.envs.mdp.commands.teleoperation.TeleopCommand',
            'bind_port': 15000,
        })
        
        # Initialize parent environment with teleoperation command
        super().__init__(cfg)
        
        # Store original config for reference
        self.original_command_cfg = original_command_cfg
        
        # Setup manipulation-specific components
        self._setup_manipulation_objects()
        self._setup_object_sensors()
    
    def _setup_manipulation_objects(self):
        """
        Load and configure objects for manipulation (boxes, tables, etc.).
        
        This method is called during environment initialization.
        Currently a placeholder - implement based on your needs.
        """
        # TODO: Load objects using IsaacLab API
        # Example:
        # self.box = RigidObject(...)
        # self.table = RigidObject(...)
        pass
    
    def _setup_object_sensors(self):
        """
        Setup sensors for detecting object state and interactions.
        
        This could include:
        - Contact sensors on gripper
        - Proximity sensors
        - Object pose trackers
        """
        # TODO: Setup object interaction sensors
        pass
    
    def reset(self, env_ids: torch.Tensor = None):
        """
        Reset environment and objects.
        
        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        super().reset(env_ids)
        
        # TODO: Reset object positions/velocities if needed
        pass
    
    def step(self, actions: torch.Tensor):
        """
        Step the environment with actions.
        
        Args:
            actions: Policy actions, same shape as training environment
            
        Returns:
            obs: Observations (compatible with trained policy)
            reward: Rewards
            done: Done flags
            info: Additional info
        """
        # Use parent class step - it handles policy-compatible actions
        return super().step(actions)
    
    def get_object_state(self):
        """
        Get state of manipulation objects.
        
        Returns:
            dict: Object states (positions, velocities, etc.)
        """
        # TODO: Return object states for logging/visualization
        return {}
    
    def get_command_manager(self):
        """
        Get the teleoperation command manager.
        
        Returns:
            TeleopCommand: The teleoperation command system
        """
        return self.command_manager
