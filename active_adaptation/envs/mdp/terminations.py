import torch
import abc

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor


class Termination:
    def __init__(self, env):
        self.env = env
    
    def update(self):
        pass

    def reset(self, env_ids):
        pass
    
    @abc.abstractmethod
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError
    
    @property
    def num_envs(self) -> int:
        return self.env.num_envs


def termination_func(func):
    class TermFunc(Termination):
        def __call__(self):
            return func(self.env)
    return TermFunc


def termination_wrapper(func):
    class TerminationWrapper(Termination):
        def __call__(self):
            return func()
    return TerminationWrapper


class crash(Termination):
    def __init__(
        self, 
        env, 
        body_names_expr: str,
        t_thres: float = 0.,
        min_time: float = 0.,
        **kwargs
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_indices, self.body_names = self.contact_sensor.find_bodies(body_names_expr)
        self.t_thres = t_thres
        self._decay = 0.98
        self._thres = (self.t_thres / self.env.physics_dt) * 0.9
        self.count = torch.zeros(self.num_envs, len(self.body_indices), device=self.env.device)
        self.min_steps = int(min_time / self.env.step_dt)
        print(f"Terminate upon contact on {self.body_names}")
    
    def reset(self, env_ids):
        self.count[env_ids] = 0.
    
    def update(self):
        in_contact = self.contact_sensor.data.net_forces_w[:, self.body_indices].norm(dim=-1) > 1.0
        self.count.add_(in_contact.float()).mul_(self._decay)
        
    def __call__(self):
        valid = (self.env.episode_length_buf > self.min_steps)
        undesired_contact = (self.count > self._thres).any(-1)
        return (undesired_contact & valid).reshape(self.num_envs, 1)
    
class fall_over(Termination):
    def __init__(
        self, 
        env, 
        xy_thres: float=0.8,
        z_thres: float=0.5
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.xy_thres = xy_thres
        self.z_thres = z_thres
    
    def __call__(self):
        fall_over = (self.asset.data.projected_gravity_b[:, :2].norm(dim=1, keepdim=True) >= self.xy_thres) | (-self.asset.data.projected_gravity_b[:, 2:] < self.z_thres)
        return fall_over


class root_height_below(Termination):
    def __init__(self, env, min_height: float = 0.55):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.min_height = min_height

    def __call__(self):
        return self.asset.data.root_pos_w[:, 2:3] < self.min_height


class reference_body_height_error(Termination):
    """Terminate when a body's height deviates too far from the motion reference.

    This is the reference-relative height check used by BeyondMimic.  Unlike
    ``root_height_below``, it does not impose one absolute standing height, so
    clips with crouching or other legitimate height changes remain valid.
    """

    def __init__(
        self,
        env,
        body_name: str = "torso_link",
        threshold: float = 0.25,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager = self.env.command_manager
        body_ids, body_names = self.asset.find_bodies(body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"reference_body_height_error expected exactly one body for "
                f"{body_name!r}, found {body_names}."
            )
        self.asset_body_id = body_ids[0]
        self.body_name = body_names[0]
        try:
            self.motion_body_id = self.command_manager.dataset.body_names.index(
                self.body_name
            )
        except ValueError as exc:
            raise ValueError(
                f"Reference motion has no body named {self.body_name!r}."
            ) from exc
        self.threshold = float(threshold)

    def __call__(self):
        current_height = self.asset.data.body_pos_w[:, self.asset_body_id, 2:3]
        reference_height = (
            self.command_manager._motion.body_pos_w[:, 0, self.motion_body_id, 2:3]
            + self.env.scene.env_origins[:, 2:3]
        )
        return (current_height - reference_height).abs() > self.threshold


class cum_error(Termination):
    def __init__(self, env, thres: float = 0.85, min_steps: int = 50):
        super().__init__(env)
        self.thres = torch.tensor(thres, device=self.env.device)
        self.min_steps = min_steps # tolerate the first few steps
        self.error_exceeded_count = torch.zeros(self.env.num_envs, 1, device=self.env.device, dtype=torch.int32)
        self.command_manager = self.env.command_manager
    
    def reset(self, env_ids):
        self.error_exceeded_count[env_ids] = 0

    def update(self):
        error_exceeded = (self.command_manager._cum_error > self.thres).any(-1, True)
        self.error_exceeded_count[error_exceeded] += 1
        self.error_exceeded_count[~error_exceeded] = 0
    
    def __call__(self) -> torch.Tensor:
        return (self.error_exceeded_count > self.min_steps).reshape(-1, 1)
