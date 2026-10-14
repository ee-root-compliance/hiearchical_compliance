import torch
import numpy as np
import abc
import einops
from collections.abc import Sequence as SequenceABC
from typing import Tuple, TYPE_CHECKING, Callable
from omegaconf import ListConfig

from isaaclab.utils.string import resolve_matching_names
import active_adaptation
from active_adaptation.utils.math import clamp_norm, quat_apply, quat_apply_inverse, yaw_quat, quat_mul, quat_conjugate
import active_adaptation.utils.symmetry as sym_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, RayCaster, Imu
    from isaaclab.sensors import Camera, TiledCamera
    from active_adaptation.envs.base import _Env


if active_adaptation.get_backend() == "isaac":
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import UsdGeom, UsdPhysics


def _parse_xyz_or_scalar(value, *, name: str, device: torch.device) -> tuple[torch.Tensor | float, bool]:
    if isinstance(value, (SequenceABC, ListConfig)) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(f"{name} must be a scalar or 3 xyz values, got {value}.")
        tensor = torch.tensor(value, dtype=torch.float32, device=device).reshape(1, 1, 3)
        if torch.any(tensor <= 0.0):
            raise ValueError(f"{name} must be positive, got {value}.")
        return tensor, True
    scalar = float(value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return scalar, False


class Observation:
    """
    Base class for all observations.
    """

    def __init__(self, env):
        self.env: _Env = env
        self.command_manager = env.command_manager

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError
    
    def __call__(self) ->  Tuple[torch.Tensor, torch.Tensor]:
        tensor = self.compute()
        return tensor
    
    def startup(self):
        """Called once upon initialization of the environment"""
        pass
    
    def post_step(self, substep: int):
        """Called after each physics substep"""
        pass

    def update(self):
        """Called after all physics substeps are completed"""
        pass

    def reset(self, env_ids: torch.Tensor):
        """Called after episode termination"""

    def debug_draw(self):
        """Called at each step **after** simulation, if GUI is enabled"""
        pass

    def symmetry_transforms(self):
        raise NotImplementedError(
            "This observation does not support symmetry transforms. "
            "Please implement the symmetry_transforms method if needed."
        )


def observation_func(func):

    class ObsFunc(Observation):
        def __init__(self, env, **params):
            super().__init__(env)
            self.params = params

        def compute(self):
            return func(self.env, **self.params)
    
    return ObsFunc

def observation_wrapper(func: Callable[[], torch.Tensor], func_sym: Callable):
    class ObservationWrapper(Observation):
        def __init__(self, env, **params):
            super().__init__(env)
            self.params = params

        def compute(self):
            return func(**self.params)
        def symmetry_transforms(self):
            return func_sym(**self.params)
    return ObservationWrapper

class root_ang_vel_history(Observation):
    def __init__(self, env, noise_std: float=0., history_steps: list[int]=[1]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.update()
    
    def reset(self, env_ids):
        root_ang_vel_b = self.asset.data.root_ang_vel_b[env_ids]
        root_ang_vel_b = root_ang_vel_b.unsqueeze(1).expand(-1, self.buffer.shape[1], -1)
        if self.noise_std > 0:
            root_ang_vel_b = random_noise(root_ang_vel_b, self.noise_std)
        self.buffer[env_ids] = root_ang_vel_b

    def update(self):
        root_ang_vel_b = self.asset.data.root_ang_vel_b
        if self.noise_std > 0:
            root_ang_vel_b = random_noise(root_ang_vel_b, self.noise_std)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = root_ang_vel_b

    def compute(self) -> torch.Tensor:
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[-1., 1., -1.])
        return transform.repeat(len(self.history_steps))

class projected_gravity_history(Observation):
    def __init__(self, env, noise_std: float=0., history_steps: list[int]=[1]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.update()
    
    def reset(self, env_ids):
        projected_gravity_b = self.asset.data.projected_gravity_b[env_ids]
        projected_gravity_b = projected_gravity_b.unsqueeze(1).expand(-1, self.buffer.shape[1], -1)
        if self.noise_std > 0:
            projected_gravity_b = random_noise(projected_gravity_b, self.noise_std)
            projected_gravity_b = projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True)
        self.buffer[env_ids] = self.asset.data.projected_gravity_b[env_ids].unsqueeze(1)
    
    def update(self):
        projected_gravity_b = self.asset.data.projected_gravity_b
        if self.noise_std > 0:
            projected_gravity_b = random_noise(projected_gravity_b, self.noise_std)
            projected_gravity_b = projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = projected_gravity_b
    
    def compute(self):
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform.repeat(len(self.history_steps))

class root_linvel_b(Observation):
    def __init__(self, env, yaw_only: bool=False, ema: float=1.0):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only
        self.linvel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._ema = ema

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        self.linvel_w[env_ids] = 0.0

    def post_step(self, substep: int):
        self.linvel_w.mul_(1 - self._ema).add_(self.asset.data.root_lin_vel_w * self._ema)
    
    def compute(self) -> torch.Tensor:
        if self.yaw_only:
            quat = yaw_quat(self.asset.data.root_quat_w)
        else:
            quat = self.asset.data.root_quat_w
        linvel = quat_apply_inverse(
            quat,
            self.linvel_w
        )
        return linvel

    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform

    def debug_draw(self):
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            linvel = self.linvel_w
            self.env.debug_draw.vector(
                self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
                linvel,
                color=(0.8, 0.1, 0.1, 1.)
            )
    
class JointObs(Observation):
    def __init__(
        self, 
        env,
        joint_names: str=".*",
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)

class joint_params(Observation):
    def __init__(
        self,
        env,
        joint_names: str=".*",
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)

    def compute(self) -> torch.Tensor:
        return torch.cat([
            self.asset.data.joint_armature[:, self.joint_ids],
            self.asset.data.joint_friction_coeff[:, self.joint_ids],
            self.asset.data.joint_stiffness[:, self.joint_ids],
            self.asset.data.joint_damping[:, self.joint_ids]
        ], dim=-1)
    
    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names).repeat(4)
        return transform

class joint_pos_history(Observation):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        history_steps: list[int]=[1], 
        noise_std: float=0.,
    ):
        super().__init__(env)
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.noise_std = max(noise_std, 0.)
        self.asset: Articulation = self.env.scene["robot"]
        from active_adaptation.envs.mdp.action import JointPosition
        action_manager: JointPosition = self.env.action_manager
        self.joint_pos_offset = action_manager.offset
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.num_joints = len(self.joint_ids)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

        shape = (self.num_envs, self.buffer_size, self.num_joints)
        self.joint_pos = torch.zeros(self.num_envs, 2, self.num_joints, device=self.device)
        self.buffer = torch.zeros(shape, device=self.device)
    
    def post_step(self, substep):
        self.joint_pos[:, substep % 2] = self.asset.data.joint_pos[:, self.joint_ids]
    
    def reset(self, env_ids):
        self.buffer[env_ids] = self.asset.data.joint_pos[env_ids.unsqueeze(1), self.joint_ids.unsqueeze(0)].unsqueeze(1)
    
    def update(self):
        self.buffer = self.buffer.roll(1, 1)
        joint_pos = self.joint_pos.mean(1)
        if self.noise_std > 0:
            joint_pos = random_noise(joint_pos, self.noise_std)
        self.buffer[:, 0] = joint_pos
    
    def compute(self):
        joint_pos = self.buffer - self.joint_pos_offset[:, self.joint_ids].unsqueeze(1)
        joint_pos_selected = joint_pos[:, self.history_steps]
        return joint_pos_selected.reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform.repeat(len(self.history_steps))

class applied_torque(Observation):
    def __init__(self, env, joint_names: str=".*", noise_std: float=0.):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_indices, self.joint_names = self.asset.find_joints(joint_names)
        self.noise_std = max(noise_std, 0.)
    
    def compute(self) -> torch.Tensor:
        applied_efforts = self.asset.data.applied_torque
        return random_noise(applied_efforts[:, self.joint_indices], self.noise_std)

    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform

class contact_forces(Observation):
    def __init__(self, env, body_names, divide_by_mass: bool=True):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.denom = self.default_mass_total if divide_by_mass else 1.
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

    def compute(self) -> torch.Tensor:
        contact_forces = self.contact_sensor.data.net_forces_w_history.mean(1)
        force = (contact_forces[:, self.body_ids] / self.denom).clamp(-10., 10.)
        return force.view(self.num_envs, -1)
    
    def symmetry_transforms(self):
        transform = sym_utils.cartesian_space_symmetry(self.asset, self.body_names, sign=[1, -1, 1])
        return transform

class body_height(Observation):
    def __init__(self, env, body_names=".*_foot"):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
    
    def compute(self) -> torch.Tensor:
        return self.asset.data.body_pos_w[:, self.body_ids, 2].reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        return sym_utils.cartesian_space_symmetry(self.asset, self.body_names, sign=(1,))

class prev_actions(Observation):
    def __init__(self, env, steps: int=1, flatten: bool=True):
        super().__init__(env)
        self.steps = steps
        self.flatten = flatten
        self.action_manager = self.env.action_manager
    
    def compute(self):
        action_buf = self.action_manager.action_buf[:, :self.steps, :]
        if self.flatten:
            return action_buf.reshape(self.num_envs, -1)
        else:
            return action_buf

    def symmetry_transforms(self):
        transform = self.action_manager.symmetry_transforms().repeat(self.steps)
        return transform

class prev_high_actions(Observation):
    """History of high-level commands before they are decoded for the low-level policy."""

    def __init__(self, env, steps: int=1, flatten: bool=True):
        super().__init__(env)
        self.steps = steps
        self.flatten = flatten
        self.action_manager = self.env.action_manager

    def compute(self):
        action_buf = getattr(self.action_manager, "high_action_buf", None)
        if action_buf is None:
            raise RuntimeError("prev_high_actions requires an action manager with high_action_buf.")
        action_buf = action_buf[:, :self.steps, :]
        if self.flatten:
            return action_buf.reshape(self.num_envs, -1)
        return action_buf

    def symmetry_transforms(self):
        dim = self.action_manager.action_dim * self.steps
        return sym_utils.SymmetryTransform(perm=torch.arange(dim), signs=torch.ones(dim))


class prev_ee_commands(Observation):
    """History of decoded EE commands sent from the high-level policy to the low-level policy."""

    def __init__(self, env, steps: int=3, flatten: bool=True):
        super().__init__(env)
        self.steps = steps
        self.flatten = flatten
        self.action_manager = self.env.action_manager

    def compute(self):
        command_buf = getattr(self.action_manager, "ee_command_buf", None)
        if command_buf is None:
            raise RuntimeError("prev_ee_commands requires an action manager with ee_command_buf.")
        if self.steps > command_buf.shape[1]:
            raise RuntimeError(
                f"prev_ee_commands requested {self.steps} steps, but ee_command_buf only stores "
                f"{command_buf.shape[1]} steps."
            )
        command_buf = command_buf[:, :self.steps, :]
        if self.flatten:
            return command_buf.reshape(self.num_envs, -1)
        return command_buf

    def symmetry_transforms(self):
        if hasattr(self.env.command_manager, "root_and_wrist_6d_sym"):
            return self.env.command_manager.root_and_wrist_6d_sym().repeat(self.steps)
        dim = self.action_manager.ee_storage_dim * self.steps
        return sym_utils.SymmetryTransform(perm=torch.arange(dim), signs=torch.ones(dim))


class prev_root_commands(Observation):
    """History of decoded root commands sent from the high-level policy to the low-level policy."""

    def __init__(self, env, steps: int=3, flatten: bool=True):
        super().__init__(env)
        self.steps = steps
        self.flatten = flatten
        self.action_manager = self.env.action_manager

    def compute(self):
        command_buf = getattr(self.action_manager, "root_command_buf", None)
        if command_buf is None:
            raise RuntimeError("prev_root_commands requires an action manager with root_command_buf.")
        if self.steps > command_buf.shape[1]:
            raise RuntimeError(
                f"prev_root_commands requested {self.steps} steps, but root_command_buf only stores "
                f"{command_buf.shape[1]} steps."
            )
        command_buf = command_buf[:, :self.steps, :]
        if self.flatten:
            return command_buf.reshape(self.num_envs, -1)
        return command_buf

    def symmetry_transforms(self):
        dim = self.action_manager.root_storage_dim * self.steps
        return sym_utils.SymmetryTransform(perm=torch.arange(dim), signs=torch.ones(dim))


class ee_compliance_state(Observation):
    """Privileged EE state using the same force-over-stiffness target as the reward."""

    def __init__(
        self,
        env,
        body_names=("left_hand_mimic", "right_hand_mimic"),
        stiffness: float=60.0,
        max_offset: float=0.25,
        force_deadband: float=2.0,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_names = list(body_names)
        self.stiffness, stiffness_directional = _parse_xyz_or_scalar(
            stiffness,
            name="ee_compliance_state.stiffness",
            device=self.device,
        )
        self.max_offset, offset_directional = _parse_xyz_or_scalar(
            max_offset,
            name="ee_compliance_state.max_offset",
            device=self.device,
        )
        self.directional = stiffness_directional or offset_directional
        self.force_deadband = float(force_deadband)

        try:
            self.motion_ids = torch.tensor(
                [self.command_manager.dataset.body_names.index(name) for name in self.body_names],
                device=self.device,
                dtype=torch.long,
            )
            self.asset_ids = torch.tensor(
                [self.asset.body_names.index(name) for name in self.body_names],
                device=self.device,
                dtype=torch.long,
            )
        except ValueError as exc:
            raise ValueError(f"Could not resolve EE bodies {self.body_names}.") from exc

    def _nominal_target_b(self):
        motion = self.command_manager._motion
        if hasattr(motion, "local_body_pos"):
            return motion.local_body_pos[:, 0, self.motion_ids, :]
        if hasattr(motion, "body_pos_b"):
            return motion.body_pos_b[:, 0, self.motion_ids, :]
        raise RuntimeError("ee_compliance_state needs local_body_pos or body_pos_b in the motion data.")

    def _force_b(self):
        if (
            getattr(self.command_manager, "external_force_mode", "legacy") == "net_pull"
            and hasattr(self.command_manager, "get_net_pull_ee_force_b")
        ):
            return self.command_manager.get_net_pull_ee_force_b()

        force_w = torch.zeros(self.num_envs, len(self.asset_ids), 3, device=self.device)
        force_apply_idx = getattr(self.command_manager, "force_apply_idx_asset", None)
        force_applied_w = getattr(self.command_manager, "force_applied_w", None)
        if force_apply_idx is None or force_applied_w is None:
            return force_w

        force_apply_list = force_apply_idx.tolist()
        for ee_i, body_i in enumerate(self.asset_ids.tolist()):
            if body_i in force_apply_list:
                force_w[:, ee_i] = force_applied_w[:, force_apply_list.index(body_i)]
        root_quat = self.asset.data.root_quat_w.unsqueeze(1).expand(-1, len(self.asset_ids), -1)
        return quat_apply_inverse(root_quat, force_w)

    def _compliance_target_b(self, nominal_target_b, force_b):
        if (
            getattr(self.command_manager, "external_force_mode", "legacy") == "net_pull"
            and hasattr(self.command_manager, "get_net_pull_ee_compliance_target_b")
        ):
            return self.command_manager.get_net_pull_ee_compliance_target_b()

        active = force_b.norm(dim=-1, keepdim=True) > self.force_deadband
        active_force_b = torch.where(active, force_b, torch.zeros_like(force_b))
        if self.directional:
            target_offset_b = torch.clamp(
                active_force_b / self.stiffness,
                min=-self.max_offset,
                max=self.max_offset,
            )
        else:
            target_offset_b = clamp_norm(active_force_b / self.stiffness, max=self.max_offset)
        return nominal_target_b + target_offset_b

    def compute(self):
        root_pos = self.asset.data.root_pos_w.unsqueeze(1)
        root_quat = self.asset.data.root_quat_w.unsqueeze(1).expand(-1, len(self.asset_ids), -1)
        actual_pos_b = quat_apply_inverse(
            root_quat,
            self.asset.data.body_pos_w[:, self.asset_ids, :] - root_pos,
        )
        actual_vel_b = quat_apply_inverse(
            root_quat,
            self.asset.data.body_lin_vel_w[:, self.asset_ids, :]
            - self.asset.data.root_lin_vel_w.unsqueeze(1),
        )
        nominal_target_b = self._nominal_target_b()
        force_b = self._force_b()
        active = force_b.norm(dim=-1, keepdim=True) > self.force_deadband
        compliance_target_b = self._compliance_target_b(nominal_target_b, force_b)
        error_b = compliance_target_b - actual_pos_b

        return torch.cat(
            [
                actual_pos_b.flatten(1),
                actual_vel_b.flatten(1),
                nominal_target_b.flatten(1),
                compliance_target_b.flatten(1),
                error_b.flatten(1),
                force_b.flatten(1),
                active.float().flatten(1),
            ],
            dim=-1,
        )

    def symmetry_transforms(self):
        # RootPPO currently does not apply symmetry augmentation. Keep a valid
        # transform here so observation-spec/debug utilities can still inspect it.
        dim = self.compute().shape[-1]
        return sym_utils.SymmetryTransform(perm=torch.arange(dim), signs=torch.ones(dim))


class applied_action(JointObs):
    def __init__(self, env):
        super().__init__(env)
        self.action_manager = self.env.action_manager

    def compute(self) -> torch.Tensor:
        return self.asset.data.joint_pos_target

    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.asset.joint_names)
        return transform

class cum_error(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        return self.command_manager._cum_error

    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(
            perm=torch.arange(self.command_manager._cum_error.shape[-1]),
            signs=[1.] * self.command_manager._cum_error.shape[-1]
        )
        return transform
    
def symlog(x: torch.Tensor, a: float=1.):
    return x.sign() * torch.log(x.abs() * a + 1.) / a

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std
