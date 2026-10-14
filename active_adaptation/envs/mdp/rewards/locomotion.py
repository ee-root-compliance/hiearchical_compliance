from math import inf
from collections.abc import Sequence as SequenceABC
import torch
import abc
from typing import TYPE_CHECKING, Callable, List, Sequence, Tuple
from omegaconf import ListConfig

import isaaclab.utils.string as string_utils
from isaaclab.utils.string import resolve_matching_names
from active_adaptation.utils.math import (
    clamp_norm,
    normalize,
    axis_angle_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    yaw_quat,
)
from active_adaptation.utils.ee_compliance import force_stiffness_consistency_reward
from ..commands import *

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from isaaclab.assets import Articulation
    from active_adaptation.envs.base import _Env


class Reward:
    def __init__(
        self,
        env,
        weight: float,
        enabled: bool = True,
    ):
        self.env: _Env = env
        self.weight = weight
        self.enabled = enabled

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def step(self, substep: int):
        pass

    def post_step(self, substep: int):
        pass

    def update(self):
        pass

    def reset(self, env_ids: torch.Tensor):
        pass

    def __call__(self) -> torch.Tensor:
        result = self.compute()
        if isinstance(result, torch.Tensor):
            rew, count = result, result.numel()
        elif isinstance(result, tuple):
            rew, is_active = result
            rew = rew * is_active.float()
            count = is_active.sum().item()
        return self.weight * rew, count 

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def debug_draw(self):
        pass


def reward_func(func):
    class RewFunc(Reward):
        def compute(self):
            return func(self.env)

    return RewFunc


def reward_wrapper(func: Callable[[], torch.Tensor]):
    class RewardWrapper(Reward):
        def compute(self):
            return func()
    return RewardWrapper


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


@reward_func
def survival(self):
    return torch.ones(self.num_envs, 1, device=self.device)


class joint_torques_l2(Reward):
    def __init__(
        self, env, weight: float, enabled: bool = True, joint_names: str = ".*"
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

    def compute(self) -> torch.Tensor:
        return (
            -self.asset.data.applied_torque[:, self.joint_ids]
            .square()
            .sum(1, keepdim=True)
        )


class impact_force_l2(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_mass_total = (
            self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        )
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

        print(f"Penalizing impact forces on {self.body_names}.")

    def compute(self) -> torch.Tensor:
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ]
        contact_forces = self.contact_sensor.data.net_forces_w_history.norm(
            dim=-1
        ).mean(1)
        force = contact_forces[:, self.body_ids] / self.default_mass_total
        return -(force.square() * first_contact).sum(1, True).clamp_max(20.0)

class feet_slip(Reward):
    def __init__(
        self, env: "LocomotionEnv", body_names: str, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)

    def compute(self) -> torch.Tensor:
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0
        feet_vel = self.asset.data.body_lin_vel_w[:, self.articulation_body_ids, :2]
        slip = (in_contact * feet_vel.norm(dim=-1).square()).sum(dim=1, keepdim=True)
        return -slip

class feet_upright(Reward):
    def __init__(
        self, env, body_names: str, xy_sigma: float, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        
        self.body_ids_asset, _ = self.asset.find_bodies(body_names)

        down = torch.tensor([0.0, 0.0, -1.0], device=self.env.device)
        self.down = down.expand(self.num_envs, len(self.body_ids_asset), -1)
        self.xy_sigma = xy_sigma
        
    def compute(self):
        feet_quat_w = self.asset.data.body_quat_w[:, self.body_ids_asset]
        feet_projected_down = quat_apply(feet_quat_w, self.down)
        feet_projected_down_xy = feet_projected_down[:, :, :2].norm(dim=-1)
        rew = (torch.exp(-feet_projected_down_xy / self.xy_sigma) - 1.0)
        return rew.float().mean(dim=1, keepdim=True)

class feet_air_time_ref(Reward):
    def __init__(
        self,
        env: "LocomotionEnv",
        body_names: str,
        thres: float,
        weight: float,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)

        self.reward_time = torch.zeros(self.num_envs, len(self.body_ids), device=self.env.device)
        self.last_contact = torch.zeros(self.num_envs, len(self.body_ids), device=self.env.device, dtype=bool)
        self.h_low, self.h_high = 0.035, 0.12
        self.c_low, self.c_high = 0.5, 2.0
        self.exp_log_c_ratio = torch.log(torch.tensor(self.c_high / self.c_low, device=self.device))

    def reset(self, env_ids):
        self.reward_time[env_ids] = 0.0
        self.last_contact[env_ids] = False

    def compute(self):
        current_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0
        first_contact = (~self.last_contact) & current_contact
        self.last_contact[:] = current_contact

        feet_height = self.asset.data.body_pos_w[:, self.articulation_body_ids, 2]

        t = (feet_height - self.h_low) / (self.h_high - self.h_low)
        t = torch.clamp(t, 0.0, 1.0)
        feet_height_coef = self.c_low * torch.exp(self.exp_log_c_ratio * t)

        if hasattr(self.env.command_manager, "skip_ref") and self.env.command_manager.skip_ref:
            self.reward_time = self.reward_time + self.env.step_dt * feet_height_coef
        else:
            contact_diff = self.env.command_manager.feet_standing ^ current_contact
            self.reward_time = self.reward_time + torch.where(contact_diff, -self.env.step_dt, self.env.step_dt * feet_height_coef)
        
        self.reward = torch.sum(
            (self.reward_time - self.thres).clamp_max(0.0) * first_contact, dim=1, keepdim=True
        )
        
        self.reward_time = self.reward_time * (~current_contact)
        return self.reward

class feet_contact_count(Reward):
    def __init__(
        self, env: "LocomotionEnv", body_names: str, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.first_contact = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.env.device
        )

    def compute(self):
        self.first_contact[:] = self.contact_sensor.compute_first_contact(
            self.env.step_dt
        )[:, self.body_ids]
        return self.first_contact.sum(1, keepdim=True)


class joint_vel_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _ = self.asset.find_joints(joint_names)
        self.joint_vel = torch.zeros(
            self.num_envs, 2, len(self.joint_ids), device=self.device
        )

    def post_step(self, substep):
        self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        joint_vel = self.joint_vel.mean(1)
        return -joint_vel.square().sum(1, True)

class joint_acc_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
    
    def compute(self) -> torch.Tensor:
        # print(self.asset.data.joint_acc[:, self.joint_ids].max())
        r = - self.asset.data.joint_acc[:, self.joint_ids].clamp_max(100.0).square().sum(1, True)
        return r

class joint_pos_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str | List[str] =".*", soft_factor: float=0.9, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)
        jpos_limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
        jpos_mean = (jpos_limits[..., 0] + jpos_limits[..., 1]) / 2
        jpos_range = jpos_limits[..., 1] - jpos_limits[..., 0]
        self.soft_factor = soft_factor
        self.soft_limits = torch.zeros_like(jpos_limits)
        self.soft_limits[..., 0] = jpos_mean - 0.5 * jpos_range * soft_factor
        self.soft_limits[..., 1] = jpos_mean + 0.5 * jpos_range * soft_factor

    def compute(self) -> torch.Tensor:
        jpos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (self.soft_limits[..., 0] - jpos).clamp_min(0.0)
        violation_max = (jpos - self.soft_limits[..., 1]).clamp_min(0.0)
        return -(violation_min + violation_max).sum(1, keepdim=True) / (1-self.soft_factor)


class root_position_hold(Reward):
    def __init__(
        self,
        env,
        weight: float,
        sigma: float = 0.25,
        mode: str = "xy",
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        if mode not in ("xy", "xyz"):
            raise ValueError(f"Unsupported root_position_hold mode: {mode}")
        self.asset: Articulation = self.env.scene["robot"]
        self.sigma = sigma
        self.mode = mode
        self.root_pos_ref = torch.zeros(self.num_envs, 3, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self.root_pos_ref[env_ids] = self.asset.data.root_pos_w[env_ids]

    def compute(self) -> torch.Tensor:
        diff = self.asset.data.root_pos_w - self.root_pos_ref
        if self.mode == "xy":
            error = diff[:, :2].norm(dim=-1, keepdim=True)
        else:
            error = diff.norm(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)


class root_force_spring_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        stiffness: float = 400.0,
        sigma: float = 0.25,
        max_offset: float = 0.5,
        force_deadband: float = 5.0,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.stiffness = stiffness
        self.sigma = sigma
        self.max_offset = max_offset
        self.force_deadband = force_deadband
        self.root_pos_ref = torch.zeros(self.num_envs, 2, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self.root_pos_ref[env_ids] = self.asset.data.root_pos_w[env_ids, :2]

    def compute(self) -> torch.Tensor:
        force_buffer = getattr(self.env.command_manager, "force_apply_buffer", None)
        if force_buffer is None:
            target_offset_xy = torch.zeros(self.num_envs, 2, device=self.device)
        else:
            force_xy = force_buffer.sum(dim=1)[:, :2]
            force_norm = force_xy.norm(dim=-1, keepdim=True)
            force_xy = torch.where(force_norm > self.force_deadband, force_xy, torch.zeros_like(force_xy))
            target_offset_xy = clamp_norm(force_xy / self.stiffness, max=self.max_offset)

        target_xy = self.root_pos_ref + target_offset_xy
        error = (self.asset.data.root_pos_w[:, :2] - target_xy).norm(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)


class ee_force_compliance_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        stiffness: float = 60.0,
        sigma: float = 0.08,
        max_offset: float = 0.25,
        force_deadband: float = 2.0,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.stiffness, stiffness_directional = _parse_xyz_or_scalar(
            stiffness,
            name="ee_force_compliance_tracking.stiffness",
            device=self.device,
        )
        self.sigma = sigma
        self.max_offset, offset_directional = _parse_xyz_or_scalar(
            max_offset,
            name="ee_force_compliance_tracking.max_offset",
            device=self.device,
        )
        self.directional = stiffness_directional or offset_directional
        self.force_deadband = force_deadband
        self.ee_names = ["left_hand_mimic", "right_hand_mimic"]

    def _ee_indices(self):
        command = self.env.command_manager
        try:
            motion_idx = torch.tensor(
                [command.dataset.body_names.index(name) for name in self.ee_names],
                device=self.device,
                dtype=torch.long,
            )
            asset_idx = torch.tensor(
                [self.asset.body_names.index(name) for name in self.ee_names],
                device=self.device,
                dtype=torch.long,
            )
        except ValueError:
            return None, None
        return motion_idx, asset_idx

    def compute(self) -> torch.Tensor:
        command = self.env.command_manager
        motion_idx, asset_idx = self._ee_indices()
        if motion_idx is None:
            return torch.zeros(self.num_envs, 1, device=self.device)

        motion = command._motion
        if hasattr(motion, "local_body_pos"):
            target_ee_b = motion.local_body_pos[:, 0, motion_idx, :]
        elif hasattr(motion, "body_pos_b"):
            target_ee_b = motion.body_pos_b[:, 0, motion_idx, :]
        else:
            return torch.zeros(self.num_envs, 1, device=self.device)

        root_quat = self.asset.data.root_quat_w.unsqueeze(1)
        if (
            getattr(command, "external_force_mode", "legacy") == "net_pull"
            and hasattr(command, "get_net_pull_ee_compliance_target_b")
        ):
            compliance_target_b = command.get_net_pull_ee_compliance_target_b()
        else:
            force_w = torch.zeros(self.num_envs, 2, 3, device=self.device)
            force_apply_idx = getattr(command, "force_apply_idx_asset", torch.empty(0, device=self.device, dtype=torch.long))
            force_applied_w = getattr(command, "force_applied_w", None)
            if force_applied_w is not None:
                force_apply_list = force_apply_idx.tolist()
                for ee_i, body_i in enumerate(asset_idx.tolist()):
                    if body_i in force_apply_list:
                        force_i = force_apply_list.index(body_i)
                        force_w[:, ee_i] = force_applied_w[:, force_i]

            force_b = quat_apply_inverse(root_quat.expand(-1, 2, -1), force_w)
            force_norm = force_b.norm(dim=-1, keepdim=True)
            active_force = torch.where(force_norm > self.force_deadband, force_b, torch.zeros_like(force_b))
            if self.directional:
                target_offset_b = torch.clamp(
                    active_force / self.stiffness,
                    min=-self.max_offset,
                    max=self.max_offset,
                )
            else:
                target_offset_b = clamp_norm(active_force / self.stiffness, max=self.max_offset)
            compliance_target_b = target_ee_b + target_offset_b

        actual_ee_w = self.asset.data.body_pos_w[:, asset_idx, :]
        actual_ee_b = quat_apply_inverse(
            root_quat.expand(-1, 2, -1),
            actual_ee_w - self.asset.data.root_pos_w.unsqueeze(1),
        )
        error = (actual_ee_b - compliance_target_b).norm(dim=-1).mean(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)


class ee_force_stiffness_consistency(Reward):
    """Reward per-axis agreement between commanded stiffness and EE deflection.

    The residual ``K_cmd * (x_actual - x_nominal) - F_external`` avoids the
    singular ``F / displacement`` stiffness estimate used for reporting. Only
    environments with an active hand pull contribute this reward; the existing
    compliance tracking reward continues to handle no-force nominal tracking.
    """

    def __init__(
        self,
        env,
        weight: float,
        force_scale: float = 15.0,
        sigma: float = 1.0,
        force_deadband: float = 0.5,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        if force_scale <= 0.0:
            raise ValueError(f"force_scale must be positive, got {force_scale}.")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        if force_deadband < 0.0:
            raise ValueError(
                f"force_deadband must be non-negative, got {force_deadband}."
            )
        self.asset: Articulation = self.env.scene["robot"]
        self.force_scale = float(force_scale)
        self.sigma = float(sigma)
        self.force_deadband = float(force_deadband)
        self.ee_names = ["left_hand_mimic", "right_hand_mimic"]
        self.ee_asset_ids = torch.tensor(
            [self.asset.body_names.index(name) for name in self.ee_names],
            device=self.device,
            dtype=torch.long,
        )

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        command = self.env.command_manager
        required = (
            "get_net_pull_ee_nominal_target_b",
            "get_net_pull_ee_force_b",
            "get_net_pull_ee_compliance_stiffness",
        )
        if any(not hasattr(command, name) for name in required):
            inactive = torch.zeros(
                self.num_envs, 1, dtype=torch.bool, device=self.device
            )
            return torch.zeros(self.num_envs, 1, device=self.device), inactive

        nominal_b = command.get_net_pull_ee_nominal_target_b()
        force_b = command.get_net_pull_ee_force_b()
        stiffness = command.get_net_pull_ee_compliance_stiffness()
        root_quat = self.asset.data.root_quat_w.unsqueeze(1)
        actual_b = quat_apply_inverse(
            root_quat.expand(-1, len(self.ee_names), -1),
            self.asset.data.body_pos_w[:, self.ee_asset_ids]
            - self.asset.data.root_pos_w.unsqueeze(1),
        )

        stiffness = stiffness.expand(-1, len(self.ee_names), -1)
        return force_stiffness_consistency_reward(
            actual_b,
            nominal_b,
            force_b,
            stiffness,
            force_scale=self.force_scale,
            sigma=self.sigma,
            force_deadband=self.force_deadband,
        )


class ee_force_limit_penalty(Reward):
    def __init__(
        self,
        env,
        weight: float,
        stiffness: float | Sequence[float] = 200.0,
        force_limit: float | Sequence[float] = 30.0,
        force_deadband: float = 0.0,
        power: float = 1.0,
        use_command_stiffness: bool = False,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.stiffness, _ = _parse_xyz_or_scalar(
            stiffness,
            name="ee_force_limit_penalty.stiffness",
            device=self.device,
        )
        self.force_limit, _ = _parse_xyz_or_scalar(
            force_limit,
            name="ee_force_limit_penalty.force_limit",
            device=self.device,
        )
        self.force_deadband = float(force_deadband)
        self.power = float(power)
        self.use_command_stiffness = bool(use_command_stiffness)
        if self.power <= 0.0:
            raise ValueError(f"ee_force_limit_penalty.power must be positive, got {power}.")
        self.ee_names = ["left_hand_mimic", "right_hand_mimic"]
        self.contact_sensor = None
        try:
            self.contact_sensor = self.env.scene["ee_contact_forces"]
            self.contact_body_ids, _ = self.contact_sensor.find_bodies(".*hand_mimic")
        except Exception:
            self.contact_body_ids = []

    def _ee_asset_ids(self):
        try:
            return torch.tensor(
                [self.asset.body_names.index(name) for name in self.ee_names],
                device=self.device,
                dtype=torch.long,
            )
        except ValueError:
            return None

    def _force_b(self, asset_idx: torch.Tensor) -> torch.Tensor:
        command = self.env.command_manager
        if (
            getattr(command, "external_force_mode", "legacy") == "net_pull"
            and getattr(command, "external_force_enabled", False)
            and hasattr(command, "get_net_pull_ee_force_b")
        ):
            return command.get_net_pull_ee_force_b()

        force_w = torch.zeros(self.num_envs, 2, 3, device=self.device)
        if self.contact_sensor is not None and len(self.contact_body_ids) >= 2:
            if hasattr(self.contact_sensor.data, "net_forces_w_history"):
                data = self.contact_sensor.data.net_forces_w_history.mean(1)
            else:
                data = self.contact_sensor.data.net_forces_w
            force_w[:] = data[:, self.contact_body_ids[:2], :]

        root_quat = self.asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
        return quat_apply_inverse(root_quat, force_w)

    def compute(self) -> torch.Tensor:
        action_manager = getattr(self.env, "action_manager", None)
        command = self.env.command_manager
        if action_manager is None or not getattr(action_manager, "ee_command_enabled", False):
            return torch.zeros(self.num_envs, 1, device=self.device)
        if not hasattr(command, "get_root_and_wrist_6d_reference"):
            return torch.zeros(self.num_envs, 1, device=self.device)

        ee_command = getattr(action_manager, "ee_command", None)
        if ee_command is None or ee_command.shape[-1] < 6:
            return torch.zeros(self.num_envs, 1, device=self.device)

        asset_idx = self._ee_asset_ids()
        if asset_idx is None:
            return torch.zeros(self.num_envs, 1, device=self.device)

        reference = command.get_root_and_wrist_6d_reference()
        if reference.shape[-1] < 6:
            return torch.zeros(self.num_envs, 1, device=self.device)

        target_offset_b = (ee_command[:, :6] - reference[:, :6]).reshape(self.num_envs, 2, 3)
        force_b = self._force_b(asset_idx)
        force_abs = force_b.abs()
        force_sign = torch.sign(force_b)
        stiffness = self.stiffness
        if self.use_command_stiffness and hasattr(command, "get_net_pull_ee_compliance_stiffness"):
            # The command manager owns the sampled K for range-conditioned
            # tasks. [env, 1, xyz] broadcasts over the two hands.
            stiffness = command.get_net_pull_ee_compliance_stiffness()[:, 0:1, :]
        relief = (stiffness * target_offset_b * force_sign).clamp_min(0.0)
        remaining_force = (force_abs - relief).clamp_min(0.0)
        remaining_force = torch.where(
            force_abs > self.force_deadband,
            remaining_force,
            torch.zeros_like(remaining_force),
        )
        excess = (remaining_force - self.force_limit).clamp_min(0.0)
        normalized = excess / self.force_limit
        if self.power != 1.0:
            normalized = normalized.pow(self.power)
        return -normalized.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)


class ee_orientation_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        sigma: Sequence[float] = (1.0, 0.5),
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.sigma = tuple(float(value) for value in sigma)
        self.ee_names = ["left_hand_mimic", "right_hand_mimic"]

    def compute(self) -> torch.Tensor:
        command = self.env.command_manager
        if not hasattr(command, "get_root_and_wrist_6d_reference"):
            return torch.zeros(self.num_envs, 1, device=self.device)

        try:
            asset_idx = torch.tensor(
                [self.asset.body_names.index(name) for name in self.ee_names],
                device=self.device,
                dtype=torch.long,
            )
        except ValueError:
            return torch.zeros(self.num_envs, 1, device=self.device)

        reference = command.get_root_and_wrist_6d_reference()
        target_axis_angle = reference[:, 6:12].reshape(self.num_envs, 2, 3)
        target_angle = target_axis_angle.norm(dim=-1)
        half_angle = 0.5 * target_angle
        scale = torch.where(
            target_angle > 1e-6,
            torch.sin(half_angle) / target_angle,
            0.5 - target_angle.square() / 48.0,
        )
        target_quat_b = torch.cat(
            [torch.cos(half_angle).unsqueeze(-1), target_axis_angle * scale.unsqueeze(-1)],
            dim=-1,
        )

        root_quat_w = self.asset.data.root_quat_w.unsqueeze(1)
        actual_quat_w = self.asset.data.body_quat_w[:, asset_idx, :]
        actual_quat_b = quat_mul(
            quat_conjugate(root_quat_w).expand(-1, 2, -1),
            actual_quat_w,
        )

        orientation_diff = axis_angle_from_quat(
            quat_mul(target_quat_b, quat_conjugate(actual_quat_b))
        )
        error = orientation_diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        rewards = [torch.exp(-error / sigma) for sigma in self.sigma]
        return sum(rewards) / len(rewards)


class root_force_velocity_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        damping: float = 300.0,
        max_speed: float | None = 0.6,
        sigma: float = 0.25,
        force_deadband: float = 20.0,
        add_root_command_reference: bool = False,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.damping = damping
        self.max_speed = None if max_speed is None else float(max_speed)
        self.sigma = sigma
        self.force_deadband = force_deadband
        self.add_root_command_reference = add_root_command_reference

    def _net_force_xy(self) -> torch.Tensor:
        command = self.env.command_manager
        if getattr(command, "external_force_mode", "legacy") == "net_pull" and hasattr(command, "net_pull_force_w"):
            return command.net_pull_force_w[:, :2]
        force_buffer = getattr(command, "force_apply_buffer", None)
        if force_buffer is None:
            return torch.zeros(self.num_envs, 2, device=self.device)
        return force_buffer.sum(dim=1)[:, :2]

    def compute(self) -> torch.Tensor:
        force_xy = self._net_force_xy()
        force_norm = force_xy.norm(dim=-1, keepdim=True)
        active_force_xy = torch.where(force_norm > self.force_deadband, force_xy, torch.zeros_like(force_xy))
        desired_vel_xy = active_force_xy / self.damping
        if self.max_speed is not None:
            desired_vel_xy = clamp_norm(desired_vel_xy, max=self.max_speed)
        if self.add_root_command_reference and hasattr(self.env.command_manager, "get_root_command_reference"):
            root_ref = self.env.command_manager.get_root_command_reference()
            ref_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
            ref_vel_b[:, :2] = root_ref[:, 1:3]
            ref_vel_w = quat_apply(yaw_quat(self.asset.data.root_quat_w), ref_vel_b)
            desired_vel_xy = desired_vel_xy + ref_vel_w[:, :2]
        vel_error = (self.asset.data.root_lin_vel_w[:, :2] - desired_vel_xy).norm(dim=-1, keepdim=True)
        return torch.exp(-vel_error / self.sigma)


class root_force_direction_progress(Reward):
    def __init__(
        self,
        env,
        weight: float,
        target_speed: float = 0.5,
        force_deadband: float = 20.0,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_speed = target_speed
        self.force_deadband = force_deadband

    def _net_force_xy(self) -> torch.Tensor:
        command = self.env.command_manager
        if getattr(command, "external_force_mode", "legacy") == "net_pull" and hasattr(command, "net_pull_force_w"):
            return command.net_pull_force_w[:, :2]
        force_buffer = getattr(command, "force_apply_buffer", None)
        if force_buffer is None:
            return torch.zeros(self.num_envs, 2, device=self.device)
        return force_buffer.sum(dim=1)[:, :2]

    def compute(self) -> torch.Tensor:
        force_xy = self._net_force_xy()
        force_norm = force_xy.norm(dim=-1, keepdim=True)
        is_active = force_norm > self.force_deadband
        force_dir = force_xy / force_norm.clamp_min(1e-6)
        speed_along_force = (self.asset.data.root_lin_vel_w[:, :2] * force_dir).sum(dim=-1, keepdim=True)
        reward = (speed_along_force / max(self.target_speed, 1e-6)).clamp(0.0, 1.0)
        return reward, is_active


class root_height_hold(Reward):
    def __init__(
        self,
        env,
        weight: float,
        sigma: float = 0.08,
        target_height: float | None = None,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.sigma = sigma
        self.target_height = target_height
        self.root_height_ref = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        if self.target_height is None:
            self.root_height_ref[env_ids] = self.asset.data.root_pos_w[env_ids, 2:3]
        else:
            self.root_height_ref[env_ids] = self.target_height

    def compute(self) -> torch.Tensor:
        error = (self.asset.data.root_pos_w[:, 2:3] - self.root_height_ref).abs()
        return torch.exp(-error / self.sigma)


class root_velocity_hold(Reward):
    def __init__(
        self,
        env,
        weight: float,
        sigma: float = 0.25,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.sigma = sigma

    def compute(self) -> torch.Tensor:
        speed_xy = self.asset.data.root_lin_vel_w[:, :2].norm(dim=-1, keepdim=True)
        return torch.exp(-speed_xy / self.sigma)


class root_command_follow_force_l2(Reward):
    def __init__(
        self,
        env,
        weight: float,
        force_deadband: float = 20.0,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.force_deadband = force_deadband

    def compute(self) -> torch.Tensor:
        action_manager = self.env.action_manager
        command = getattr(action_manager, "root_command", None)
        if command is None:
            return torch.zeros(self.num_envs, 1, device=self.device)

        command_vel_b = command[:, 1:3]
        force_b = getattr(self.env.command_manager, "net_pull_force_b", None)
        if force_b is None:
            force_w = getattr(self.env.command_manager, "net_pull_force_w", None)
            if force_w is None:
                return torch.zeros(self.num_envs, 1, device=self.device)
            force_b = quat_apply_inverse(self.env.scene["robot"].data.root_quat_w, force_w)

        force_xy_b = force_b[:, :2]
        force_norm = force_xy_b.norm(dim=-1, keepdim=True)
        force_dir_b = force_xy_b / force_norm.clamp_min(1e-6)
        follow_speed = (command_vel_b * force_dir_b).sum(dim=-1, keepdim=True).clamp_min(0.0)
        is_active = force_norm > self.force_deadband
        return -follow_speed.square(), is_active


@reward_func
def high_action_l2(self):
    action_buf = getattr(self.action_manager, "high_action_buf", None)
    if action_buf is None:
        return torch.zeros(self.num_envs, 1, device=self.device)
    return -action_buf[:, 0].square().sum(dim=-1, keepdim=True)


@reward_func
def high_action_rate_l2(self):
    action_buf = getattr(self.action_manager, "high_action_buf", None)
    if action_buf is None or action_buf.shape[1] < 2:
        return torch.zeros(self.num_envs, 1, device=self.device)
    action_diff = action_buf[:, 0] - action_buf[:, 1]
    return -action_diff.square().sum(dim=-1, keepdim=True)


@reward_func
def high_action_rate2_l2(self):
    action_buf = getattr(self.action_manager, "high_action_buf", None)
    if action_buf is None or action_buf.shape[1] < 3:
        return torch.zeros(self.num_envs, 1, device=self.device)
    action_diff = action_buf[:, 0] - 2 * action_buf[:, 1] + action_buf[:, 2]
    return -action_diff.square().sum(dim=-1, keepdim=True)


@reward_func
def action_rate_l2(self):
    action_diff = self.action_manager.action_buf[:, :, 0] - self.action_manager.action_buf[:, :, 1]
    return - action_diff.square().sum(dim=-1, keepdim=True)


@reward_func
def action_rate2_l2(self):
    action_diff = (
        self.action_manager.action_buf[:, :, 0] - 2 * self.action_manager.action_buf[:, :, 1] + self.action_manager.action_buf[:, :, 2]
    )
    return - action_diff.square().sum(dim=-1, keepdim=True)
