import torch
import numpy as np
import logging
from typing import Union, TYPE_CHECKING, Dict, Tuple

import active_adaptation

import isaaclab.utils.string as string_utils


if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.base import _Env


class Randomization:
    def __init__(self, env):
        self.env: _Env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device
    
    def startup(self):
        pass
    
    def reset(self, env_ids: torch.Tensor):
        pass
    
    def step(self, substep):
        pass

    def update(self):
        pass

    def debug_draw(self):
        pass


RangeType = Tuple[float, float]
NestedRangeType = Union[RangeType, Dict[str, RangeType]]

class motor_params_implicit(Randomization):
    def __init__(
        self,
        env,
        stiffness_range,
        damping_range,
        armature_range,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

        # Store the range dictionaries
        self.stiffness_range = dict(stiffness_range)
        self.damping_range   = dict(damping_range)
        self.armature_range  = dict(armature_range)
        # ------- stiffness -------
        ids, _, val = string_utils.resolve_matching_names_values(
            self.stiffness_range, self.asset.joint_names
        )
        self.stiffness_id  = torch.tensor(ids, device=self.device, dtype=torch.long)
        self.stiffness_def = self.asset.data.joint_stiffness[0, self.stiffness_id]
        low, high          = (
            torch.tensor(val, device=self.device) * self.stiffness_def.unsqueeze(1)
        ).unbind(1)
        self.stiffness_low   = low
        self.stiffness_scale = high - low

        # ------- damping -------
        ids, _, val = string_utils.resolve_matching_names_values(
            self.damping_range, self.asset.joint_names
        )
        self.damping_id  = torch.tensor(ids, device=self.device, dtype=torch.long)
        self.damping_def = self.asset.data.joint_damping[0, self.damping_id]
        low, high        = (
            torch.tensor(val, device=self.device) * self.damping_def.unsqueeze(1)
        ).unbind(1)
        self.damping_low   = low
        self.damping_scale = high - low

        # ------- armature (expressed as a relative value) -------
        ids, _, val = string_utils.resolve_matching_names_values(
            self.armature_range, self.asset.joint_names
        )
        self.armature_id  = torch.tensor(ids, device=self.device, dtype=torch.long)
        self.armature_def = self.asset.data.joint_armature[0, self.armature_id]
        low, high         = (
            torch.tensor(val, device=self.device) * self.armature_def.unsqueeze(1)
        ).unbind(1)
        self.armature_low   = low
        self.armature_scale = high - low

    def _rand_u(self, n_env: int, k: int):
        return torch.rand(n_env, k, device=self.device)

    # ----------------------------------------------------------
    def reset(self, env_ids):
        n_env = len(env_ids)

        # stiffness
        stiff = self._rand_u(n_env, len(self.stiffness_id))
        stiff = stiff * self.stiffness_scale + self.stiffness_low
        self.asset.write_joint_stiffness_to_sim(stiff, self.stiffness_id, env_ids)

        # damping
        damp = self._rand_u(n_env, len(self.damping_id))
        damp = damp * self.damping_scale + self.damping_low
        self.asset.write_joint_damping_to_sim(damp, self.damping_id, env_ids)

        # armature
        arma = self._rand_u(n_env, len(self.armature_id))
        arma = arma * self.armature_scale + self.armature_low
        self.asset.write_joint_armature_to_sim(arma, self.armature_id, env_ids)


class perturb_body_materials(Randomization):
    def __init__(
        self,
        env,
        body_names,
        static_friction_range=(0.6, 1.0),
        dynamic_friction_frac_range=(0.6, 1.0),
        dynamic_friction_range=None,
        restitution_range=(0.0, 0.2),
        homogeneous: bool = False,
        num_buckets: int = 16,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)

        self.static_friction_range = static_friction_range
        self.dynamic_friction_frac_range = dynamic_friction_frac_range
        self.dynamic_friction_range = dynamic_friction_range
        self.restitution_range = restitution_range
        self.homogeneous = homogeneous
        self.num_buckets = num_buckets

        self.default_materials = (
            self.asset.root_physx_view.get_material_properties()
        )
        num_shapes_per_body = []
        for link_path in self.asset.root_physx_view.link_paths[0]:
            link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)
        cumsum = np.cumsum([0] + num_shapes_per_body)
        self.shape_ids = torch.cat(
            [torch.arange(cumsum[i], cumsum[i + 1]) for i in self.body_ids]
        )

    def startup(self):
        logging.info(f"Randomize body materials of {self.body_names} upon startup.")

        materials = self.default_materials.clone()
        shape_ids = self.shape_ids.to(device=materials.device)
        if self.homogeneous:
            shape = (self.num_envs, 1)
        else:
            shape = (self.num_envs, len(self.shape_ids))

        sf  = sample_uniform(shape, *self.static_friction_range, device=materials.device)  # static friction
        if self.dynamic_friction_range is None:
            dff = sample_uniform(shape, *self.dynamic_friction_frac_range, device=materials.device)  # dynamic-fraction
        else:
            df = sample_uniform(shape, *self.dynamic_friction_range, device=materials.device)  # dynamic friction
        res = sample_uniform(shape, *self.restitution_range, device=materials.device)  # restitution

        def _bucketize(x, lo, hi, n):
            step = (hi - lo) / (n - 1)
            idx  = ((x - lo) / step).round().clamp(0, n - 1)
            return lo + idx * step

        N = self.num_buckets                       # 16 => up to 4096 combinations
        sf  = _bucketize(sf,  *self.static_friction_range,           N)
        if self.dynamic_friction_range is None:
            dff = _bucketize(dff, *self.dynamic_friction_frac_range, N)
            df = dff * sf
        else:
            df = _bucketize(df, *self.dynamic_friction_range,        N)
        res = _bucketize(res, *self.restitution_range,               N)
        # -------------------------------------------------------------

        materials[:, shape_ids, 0] = sf
        materials[:, shape_ids, 1] = df
        materials[:, shape_ids, 2] = res

        indices = torch.arange(self.asset.num_instances)

        self.asset.root_physx_view.set_material_properties(
            materials.flatten(), indices
        )
        self.asset.data.body_materials = materials.to(self.device)

class perturb_body_mass(Randomization):
    def __init__(
        self,
        env,
        body_names=None,
        scale_range=None,
        **perturb_ranges: Tuple[float, float],
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        if body_names is not None:
            if scale_range is None:
                raise ValueError("perturb_body_mass scale_range is required when body_names is set.")
            perturb_ranges = {body_names: scale_range}

        self.body_ids, self.body_names, values = string_utils.resolve_matching_names_values(
            perturb_ranges, self.asset.body_names
        )
        self.mass_ranges = torch.tensor(values, device=self.device)
        print(self.body_names)

    def startup(self):
        logging.info(f"Randomize body masses of {self.body_names} upon startup.")
        masses = self.asset.data.default_mass.clone()
        inertias = self.asset.data.default_inertia.clone()
        print(f"Default masses: {masses[0]}")
        mass_ranges = self.mass_ranges.to(device=masses.device)
        scale = uniform(
            mass_ranges[:, 0].expand_as(masses[:, self.body_ids]),
            mass_ranges[:, 1].expand_as(masses[:, self.body_ids])
        )
        masses[:, self.body_ids] *= scale
        inertias[:, self.body_ids] *= scale.unsqueeze(-1)
        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_masses(masses, indices)
        self.asset.root_physx_view.set_inertias(inertias, indices)
        assert torch.allclose(self.asset.root_physx_view.get_masses(), masses)

class perturb_body_com(Randomization):
    def __init__(self, env, body_names = ".*", com_range=(-0.05, 0.05), x=None, y=None, z=None):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        if x is not None or y is not None or z is not None:
            x = x if x is not None else com_range
            y = y if y is not None else com_range
            z = z if z is not None else com_range
            self.com_low = torch.tensor([x[0], y[0], z[0]], device=self.device)
            self.com_high = torch.tensor([x[1], y[1], z[1]], device=self.device)
        else:
            self.com_low = torch.full((3,), float(com_range[0]), device=self.device)
            self.com_high = torch.full((3,), float(com_range[1]), device=self.device)
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.ALL_INDICES = torch.arange(self.asset.num_instances)
    
    def startup(self):
        coms = self.asset.root_physx_view.get_coms()
        com_low = self.com_low.to(device=coms.device)
        com_high = self.com_high.to(device=coms.device)
        rand_offset = uniform(
            com_low.expand(self.asset.num_instances, len(self.body_ids), 3),
            com_high.expand(self.asset.num_instances, len(self.body_ids), 3),
        )
        coms[:, self.body_ids, :3] += rand_offset
        self.asset.root_physx_view.set_coms(coms, indices=self.ALL_INDICES)

class push_robot(Randomization):
    def __init__(
        self,
        env,
        enabled: bool = True,
        interval_range_s=(3.0, 6.0),
        lin_vel_xy_range=(-0.3, 0.3),
        lin_vel_z_range=None,
        roll_vel_range=None,
        pitch_vel_range=None,
        yaw_vel_range=(-0.5, 0.5),
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.enabled = enabled
        self.interval_range_s = interval_range_s
        self.lin_vel_xy_range = lin_vel_xy_range
        self.lin_vel_z_range = lin_vel_z_range
        self.roll_vel_range = roll_vel_range
        self.pitch_vel_range = pitch_vel_range
        self.yaw_vel_range = yaw_vel_range
        self.next_push_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _sample_interval_steps(self, n: int):
        interval = sample_uniform((n,), *self.interval_range_s, device=self.device)
        return torch.clamp((interval / self.env.step_dt).long(), min=1)

    def reset(self, env_ids: torch.Tensor):
        if not self.enabled:
            return
        self.next_push_step[env_ids] = self.env.episode_length_buf[env_ids] + self._sample_interval_steps(len(env_ids))

    def step(self, substep):
        if not self.enabled or substep != 0:
            return
        env_ids = torch.nonzero(self.env.episode_length_buf >= self.next_push_step, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return

        root_vel = self.asset.data.root_vel_w[env_ids].clone()
        root_vel[:, 0:2] = sample_uniform((env_ids.numel(), 2), *self.lin_vel_xy_range, device=self.device)
        if self.lin_vel_z_range is not None:
            root_vel[:, 2] = sample_uniform(
                (env_ids.numel(),), *self.lin_vel_z_range, device=self.device
            )
        if self.roll_vel_range is not None:
            root_vel[:, 3] = sample_uniform(
                (env_ids.numel(),), *self.roll_vel_range, device=self.device
            )
        if self.pitch_vel_range is not None:
            root_vel[:, 4] = sample_uniform(
                (env_ids.numel(),), *self.pitch_vel_range, device=self.device
            )
        root_vel[:, 5] = sample_uniform((env_ids.numel(),), *self.yaw_vel_range, device=self.device)
        self.asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        self.next_push_step[env_ids] = self.env.episode_length_buf[env_ids] + self._sample_interval_steps(env_ids.numel())

class random_joint_offset(Randomization):
    def __init__(self, env, **offset_range: Tuple[float, float]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _, self.offset_range = string_utils.resolve_matching_names_values(dict(offset_range), self.asset.joint_names)
        
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.offset_range = torch.tensor(self.offset_range, device=self.device)

        self.action_manager = self.env.action_manager

    def reset(self, env_ids: torch.Tensor):
        offset = uniform(self.offset_range[:, 0], self.offset_range[:, 1])
        self.action_manager.offset[env_ids.unsqueeze(1), self.joint_ids] = offset

def clamp_norm(x: torch.Tensor, min: float = 0.0, max: float = torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x


def random_scale(x: torch.Tensor, low: float, high: float, homogeneous: bool=False):
    if homogeneous:
        u = torch.rand(*x.shape[:1], 1, device=x.device)
    else:
        u = torch.rand_like(x)
    return x * (u * (high - low) + low), u

def random_shift(x: torch.Tensor, low: float, high: float):
    return x + x * (torch.rand_like(x) * (high - low) + low)

def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def uniform(low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(low)
    return low + r * (high - low)

def uniform_like(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(x)
    return low + r * (high - low)

def log_uniform(low: torch.Tensor, high: torch.Tensor):
    return uniform(low.log(), high.log()).exp()

def angle_mix(a: torch.Tensor, b: torch.Tensor, weight: float=0.1):
    d = a - b
    d[d > torch.pi] -= 2 * torch.pi
    d[d < -torch.pi] += 2 * torch.pi
    return a - d * weight
