import torch
import carb
import omni
import weakref

from active_adaptation.utils.math import quat_mul
from typing import Sequence, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.locomotion import SimpleEnv

class Command:
    def __init__(self, env, ) -> None:
        self.env: SimpleEnv = env
        self.asset: Articulation = env.scene["robot"]
        self.init_root_state = self.asset.data.default_root_state.clone()
        self.init_root_state[:, 3:7] = self.asset.data.root_state_w[:, 3:7]
        self.init_joint_pos = self.asset.data.default_joint_pos.clone()
        self.init_joint_vel = self.asset.data.default_joint_vel.clone()

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def step(self, substep: int):
        pass
    
    def before_update(self):
        pass

    def update(self):
        pass

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass

    def sample_init(self, env_ids: torch.Tensor):
        raise NotImplementedError