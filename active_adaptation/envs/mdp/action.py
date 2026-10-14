from collections.abc import Sequence as SequenceABC

import torch
import einops
from typing import Dict, Literal, Tuple, Union, TYPE_CHECKING
from tensordict import TensorDictBase
import isaaclab.utils.string as string_utils
import hydra
import active_adaptation.utils.symmetry as symmetry_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.base import _Env


class ActionManager:

    action_dim: int

    def __init__(self, env):
        self.env: _Env = env
        self.asset: Articulation = self.env.scene["robot"]

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device
    
    def symmetry_transforms(self):
        raise NotImplementedError(
            "ActionManager subclasses must implement symmetry_transforms method."
            "This method should return a SymmetryTransform object that applies to the action space."
        )


class JointPosition(ActionManager):
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] | float = 0.5,
        max_delay: int | None = None,
        alpha: Tuple[float, float] = (0.9, 0.9),
        alpha_wide: Tuple[float, float] = (0.8, 1.0),
        boot_protect: bool = False,
        alpha_jit_scale: float | None = None,
        **kwargs,
    ):
        super().__init__(env)

        # ------------------------------------------------------------------ cfg
        self.joint_ids, self.joint_names, self.action_scaling = (
            string_utils.resolve_matching_names_values(
                dict(action_scaling), self.asset.joint_names
            )
        )
        # print(self.joint_ids, self.joint_names, self.action_scaling)
        # breakpoint()
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.action_dim = len(self.joint_ids)

        if isinstance(max_delay, SequenceABC) and not isinstance(max_delay, (str, bytes)):
            if len(max_delay) != 2:
                raise ValueError(f"max_delay range must have exactly 2 values, got {max_delay}.")
            self.min_delay = int(max_delay[0])
            self.max_delay = int(max_delay[1])
        else:
            self.min_delay = 0
            self.max_delay = int(max_delay or 0)  # physics steps
        if self.min_delay < 0 or self.max_delay < self.min_delay:
            raise ValueError(
                f"Invalid max_delay range: min_delay={self.min_delay}, max_delay={self.max_delay}."
            )

        self.alpha_range = alpha
        self.alpha_wide_range = alpha_wide

        # Boot‑protection ----------------------------------------------------
        self.boot_protect_enabled = boot_protect
        if self.boot_protect_enabled:
            self.boot_delay = torch.zeros(self.num_envs, 1, dtype=int, device=self.device)

        # α‑jitter -----------------------------------------------------------
        self.alpha_jit_scale = alpha_jit_scale
        if self.alpha_jit_scale is not None:
            self.alpha_jit = torch.zeros(self.num_envs, 1, device=self.device)

        # Persistent tensors -------------------------------------------------
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)

        with torch.device(self.device):
            hist = max((self.max_delay - 1) // self.env.decimation + 1, 3)
            self.action_buf = torch.zeros(self.num_envs, hist, self.action_dim)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)

    # --------------------------------------------------------------------- util
    def resolve(self, spec):
        """Convenience helper for user APIs."""
        return string_utils.resolve_matching_names_values(dict(spec), self.asset.joint_names)
    
    def symmetry_transforms(self):
        transform = symmetry_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform
    # ------------------------------------------------------------------- reset
    def reset(self, env_ids: torch.Tensor):
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        # Delay selection ---------------------------------------------------
        if self.boot_protect_enabled:
            delay = torch.randint(self.min_delay, self.max_delay + 1, (len(env_ids), 1), device=self.device)
            self.boot_delay[env_ids] = delay
            self.delay[env_ids] = delay
        else:
            self.delay[env_ids] = torch.randint(self.min_delay, self.max_delay + 1, (len(env_ids), 1), device=self.device)

        # α per environment --------------------------------------------------
        alpha = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.alpha_range)
        self.alpha[env_ids] = alpha

    # ---------------------------------------------------------------- forward
    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            raw_action = tensordict["action"].clamp(-10, 10)

            ### debug symmetry
            # raw_action = self.symmetry_transforms().to(raw_action.device).forward(raw_action)

            # α with optional jitter -------------------------------------------
            if self.alpha_jit_scale is not None:
                self.alpha_jit.uniform_(-self.alpha_jit_scale, self.alpha_jit_scale)
                self.alpha.add_(self.alpha_jit).clamp_(*self.alpha_wide_range)

            self.action_buf = torch.roll(self.action_buf, shifts=1, dims=1)
            self.action_buf[:, 0, :] = raw_action

        # Communication delay ----------------------------------------------
        idx = (self.delay - substep + self.env.decimation - 1) // self.env.decimation
        delayed_action = self.action_buf.take_along_dim(idx.unsqueeze(1), dim=1).squeeze(1)
        self.applied_action.lerp_(delayed_action, self.alpha)

        # Joint targets -----------------------------------------------------
        pos_tgt = self.default_joint_pos + self.offset
        pos_tgt[:, self.joint_ids] += self.applied_action * self.action_scaling

        # Optional boot‑protection -----------------------------------------
        if self.boot_protect_enabled:
            pos_tgt = torch.where(
                self.boot_delay > 0,
                self.env.command_manager.joint_pos_boot_protect,
                pos_tgt,
            )
            self.boot_delay.sub_(1).clamp_min_(0)

        # Write to simulator -----------------------------------------------
        self.asset.set_joint_position_target(pos_tgt)
        self.asset.write_data_to_sim()


class HierarchicalRootCommand(ActionManager):
    """High-level root-command action manager with a frozen low-level policy."""

    def __init__(
        self,
        env,
        low_action: Dict,
        low_policy: Dict,
        command_dim: int = 5,
        nominal_root_height: float = 0.79,
        command_scale: Tuple[float, float, float, float, float] = (0.25, 0.8, 0.8, 1.0, 1.0),
        root_command: Dict | None = None,
        ee_command: Dict | None = None,
        feet_command: Dict | None = None,
        joint_residual: Dict | None = None,
        low_policy_command_slice: Tuple[int, int] | None = (1, 7),
        low_policy_obs_key: str | None = "policy",
        override_root_command: bool = False,
        **kwargs,
    ):
        super().__init__(env)
        self.root_command_cfg = dict(root_command or {})
        self.root_command_enabled = self.root_command_cfg.get("enabled", True)
        self.root_command_passthrough_reference = self.root_command_cfg.get("passthrough_reference", False)
        self.root_command_mode = self.root_command_cfg.get("mode", "absolute")
        if not self.root_command_enabled:
            self.root_command_dim = 0
        elif self.root_command_mode == "absolute":
            self.root_command_dim = 5
        elif self.root_command_mode == "velocity_delta":
            self.root_command_dim = 2
        elif self.root_command_mode == "velocity_delta_yaw":
            # Two planar velocity residuals plus a heading/yaw residual.
            # The low-level policy still receives the canonical 5D root
            # command: [height, vx, vy, heading_x, heading_y].
            self.root_command_dim = 3
        else:
            raise ValueError(
                "HierarchicalRootCommand root_command.mode must be one of "
                f"['absolute', 'velocity_delta', 'velocity_delta_yaw'], got {self.root_command_mode!r}."
            )
        self.root_storage_dim = 5
        self.ee_command_cfg = dict(ee_command or {})
        self.ee_command_enabled = self.ee_command_cfg.get("enabled", False)
        self.ee_command_mode = self.ee_command_cfg.get("mode", "pose_delta")
        self.ee_storage_dim = 12 if self.ee_command_enabled else 0
        if not self.ee_command_enabled:
            self.ee_command_dim = 0
        elif self.ee_command_mode == "pose_delta":
            self.ee_command_dim = 12
        elif self.ee_command_mode == "pos_delta":
            self.ee_command_dim = 6
        else:
            raise ValueError(
                "HierarchicalRootCommand ee_command.mode must be one of "
                f"['pose_delta', 'pos_delta'], got {self.ee_command_mode!r}."
            )
        self.feet_command_cfg = dict(feet_command or {})
        self.feet_command_enabled = self.feet_command_cfg.get("enabled", False)
        self.feet_command_dim = 6 if self.feet_command_enabled else 0

        # Instantiate the frozen low-level action manager before resolving the
        # optional residual dimension. Its action order is the canonical G1
        # joint order used by the low-level checkpoint.
        self.low_action_manager: JointPosition = hydra.utils.instantiate(low_action, env=env)
        self.joint_residual_cfg = dict(joint_residual or {})
        self.joint_residual_enabled = self.joint_residual_cfg.get("enabled", False)
        self.joint_residual_dim = (
            self.low_action_manager.action_dim if self.joint_residual_enabled else 0
        )

        expected_command_dim = (
            self.root_command_dim
            + self.ee_command_dim
            + self.feet_command_dim
            + self.joint_residual_dim
        )
        legacy_root_only_dim = 5
        valid_dims = {expected_command_dim}
        if self.root_command_enabled and not self.ee_command_enabled and not self.feet_command_enabled:
            valid_dims.add(legacy_root_only_dim)
        if command_dim not in valid_dims:
            raise ValueError(
                "HierarchicalRootCommand got command_dim="
                f"{command_dim}, expected one of {sorted(valid_dims)} for the configured components."
            )
        self.action_dim = expected_command_dim
        self.nominal_root_height = nominal_root_height
        self.command_scale = torch.tensor(command_scale, device=self.device).reshape(1, self.root_storage_dim)
        self.ee_pos_scale = torch.tensor(
            self.ee_command_cfg.get("pos_scale", [0.15, 0.15, 0.15]),
            device=self.device,
        ).reshape(1, 1, 3)
        self.ee_rot_scale = torch.tensor(
            self.ee_command_cfg.get("rot_scale", [0.5, 0.5, 0.5]),
            device=self.device,
        ).reshape(1, 1, 3)
        self.feet_pos_scale = torch.tensor(
            self.feet_command_cfg.get("pos_scale", [0.15, 0.15, 0.08]),
            device=self.device,
        ).reshape(1, 1, 3)
        residual_scale = self.joint_residual_cfg.get("scale", 0.05)
        if isinstance(residual_scale, SequenceABC) and not isinstance(residual_scale, (str, bytes)):
            if len(residual_scale) != self.joint_residual_dim:
                raise ValueError(
                    "HierarchicalRootCommand joint_residual.scale must have one value per low-level action "
                    f"({self.joint_residual_dim}), got {len(residual_scale)}."
                )
            self.joint_residual_scale = torch.tensor(residual_scale, device=self.device).reshape(1, -1)
        else:
            self.joint_residual_scale = torch.full(
                (1, self.joint_residual_dim), float(residual_scale), device=self.device
            )
        self.low_policy_command_slice = tuple(low_policy_command_slice) if low_policy_command_slice is not None else None
        self.low_policy_obs_key = low_policy_obs_key
        self.override_root_command = override_root_command

        from active_adaptation.learning.hierarchical.frozen_low_level import FrozenLowLevelPolicy
        self.low_policy = FrozenLowLevelPolicy(
            env=env,
            action_dim=self.low_action_manager.action_dim,
            **low_policy,
        )

        self.high_action_buf = torch.zeros(self.num_envs, 3, self.action_dim, device=self.device)
        self.root_command = torch.zeros(self.num_envs, self.root_storage_dim, device=self.device)
        self.root_command_buf = torch.zeros(self.num_envs, 3, self.root_storage_dim, device=self.device)
        self.ee_command = torch.zeros(self.num_envs, self.ee_storage_dim, device=self.device)
        self.ee_command_buf = torch.zeros(self.num_envs, 3, self.ee_storage_dim, device=self.device)
        self.feet_command = torch.zeros(self.num_envs, self.feet_command_dim, device=self.device)
        self.joint_residual = torch.zeros(
            self.num_envs, self.joint_residual_dim, device=self.device
        )
        self.low_action = torch.zeros(self.num_envs, self.low_action_manager.action_dim, device=self.device)
        self.ee_force_stiffness_ablation_enabled = False
        self.ee_force_stiffness_ablation_command = torch.zeros(self.num_envs, self.ee_storage_dim, device=self.device)
        self._reset_root_command(torch.arange(self.num_envs, device=self.device))
        self._reset_ee_command(torch.arange(self.num_envs, device=self.device))
        self._reset_feet_command(torch.arange(self.num_envs, device=self.device))

    def preload_low_policy(self):
        """Load the frozen low-level policy after the env specs are ready."""
        if self.low_policy.preload:
            self.low_policy.load()

    @property
    def joint_ids(self):
        return self.low_action_manager.joint_ids

    @property
    def joint_names(self):
        return self.low_action_manager.joint_names

    @property
    def offset(self):
        return self.low_action_manager.offset

    @property
    def action_buf(self):
        return self.low_action_manager.action_buf

    @property
    def applied_action(self):
        return self.low_action_manager.applied_action

    def symmetry_transforms(self):
        return self.low_action_manager.symmetry_transforms()

    def _reset_root_command(self, env_ids: torch.Tensor):
        if (
            self.root_command_enabled
            and self.root_command_mode in {"velocity_delta", "velocity_delta_yaw"}
            and hasattr(self.env.command_manager, "get_root_command_reference")
        ):
            self.root_command[env_ids] = self._get_root_command_reference()[env_ids]
            return
        self.root_command[env_ids] = 0.0
        self.root_command[env_ids, 0] = self.nominal_root_height
        self.root_command[env_ids, 3] = 1.0

    def _set_default_root_command(self):
        if (
            self.root_command_enabled
            and self.root_command_mode in {"velocity_delta", "velocity_delta_yaw"}
            and hasattr(self.env.command_manager, "get_root_command_reference")
        ):
            self.root_command[:] = self._get_root_command_reference()
            return
        self.root_command[:] = 0.0
        self.root_command[:, 0] = self.nominal_root_height
        self.root_command[:, 3] = 1.0

    def _set_root_command_reference(self):
        if hasattr(self.env.command_manager, "get_root_command_reference"):
            self.root_command[:] = self._get_root_command_reference()
            return
        self._set_default_root_command()

    def _default_root_command(self) -> torch.Tensor:
        command = torch.zeros(self.num_envs, self.root_storage_dim, device=self.device)
        command[:, 0] = self.nominal_root_height
        command[:, 3] = 1.0
        return command

    def _get_root_command_reference(self) -> torch.Tensor:
        if not hasattr(self.env.command_manager, "get_root_command_reference"):
            return self._default_root_command()
        try:
            reference = self.env.command_manager.get_root_command_reference()
        except Exception:
            return self._default_root_command()
        if reference.shape[-1] != self.root_storage_dim:
            return self._default_root_command()
        return reference.clone()

    def _get_ee_command_reference(self) -> torch.Tensor:
        if not self.ee_command_enabled:
            return self.ee_command
        if hasattr(self.env.command_manager, "get_root_and_wrist_6d_reference"):
            return self.env.command_manager.get_root_and_wrist_6d_reference()
        if hasattr(self.env.command_manager, "root_and_wrist_6d"):
            return self.env.command_manager.root_and_wrist_6d()
        return torch.zeros(self.num_envs, self.ee_storage_dim, device=self.device)

    def _reset_ee_command(self, env_ids: torch.Tensor):
        if not self.ee_command_enabled:
            return
        self.ee_command[env_ids] = self._get_ee_command_reference()[env_ids]

    def _get_feet_command_reference(self) -> torch.Tensor:
        if not self.feet_command_enabled:
            return self.feet_command
        if hasattr(self.env.command_manager, "get_feet_pos_b_reference"):
            return self.env.command_manager.get_feet_pos_b_reference()
        if hasattr(self.env.command_manager, "feet_pos_b"):
            return self.env.command_manager.feet_pos_b()
        return torch.zeros(self.num_envs, self.feet_command_dim, device=self.device)

    def _reset_feet_command(self, env_ids: torch.Tensor):
        if not self.feet_command_enabled:
            return
        self.feet_command[env_ids] = self._get_feet_command_reference()[env_ids]

    def reset(self, env_ids: torch.Tensor):
        self.low_action_manager.reset(env_ids)
        self.high_action_buf[env_ids] = 0.0
        self.ee_command_buf[env_ids] = 0.0
        self.low_action[env_ids] = 0.0
        self.joint_residual[env_ids] = 0.0
        self._reset_root_command(env_ids)
        self._reset_ee_command(env_ids)
        self._reset_feet_command(env_ids)
        self.root_command_buf[env_ids] = self.root_command[env_ids].unsqueeze(1).expand(
            -1, self.root_command_buf.shape[1], -1
        )
        if self.ee_command_enabled:
            self.ee_command_buf[env_ids] = self.ee_command[env_ids].unsqueeze(1).expand(
                -1, self.ee_command_buf.shape[1], -1
            )
        if hasattr(self.env.command_manager, "set_root_command"):
            self.env.command_manager.set_root_command(self.root_command)
        if self.ee_command_enabled and hasattr(self.env.command_manager, "set_root_and_wrist_6d_command"):
            self.env.command_manager.set_root_and_wrist_6d_command(self.ee_command)
        if self.feet_command_enabled and hasattr(self.env.command_manager, "set_feet_pos_b_command"):
            self.env.command_manager.set_feet_pos_b_command(self.feet_command)

    def debug_draw(self):
        self.low_action_manager.debug_draw()

    def _decode_root_command(self, raw_action: torch.Tensor) -> torch.Tensor:
        if not self.root_command_enabled:
            command = torch.zeros(self.num_envs, self.root_storage_dim, device=self.device)
            command[:, 0] = self.nominal_root_height
            command[:, 3] = 1.0
            return command
        if self.root_command_mode == "velocity_delta":
            action = torch.tanh(raw_action) * self.command_scale[:, 1:3]
            command = self._get_root_command_reference()
            command[:, 1:3] = command[:, 1:3] + action[:, :2]
            return command

        if self.root_command_mode == "velocity_delta_yaw":
            action = torch.tanh(raw_action)
            command = self._get_root_command_reference()
            command[:, 1:3] = command[:, 1:3] + action[:, :2] * self.command_scale[:, 1:3]

            # Rotate the reference heading in the body frame by a bounded
            # scalar residual. This preserves the low-level command format
            # while giving the high-level policy direct yaw authority.
            yaw_scale = float(self.root_command_cfg.get("yaw_scale", 0.5))
            delta_yaw = action[:, 2] * yaw_scale
            heading = command[:, 3:5]
            cos_yaw = torch.cos(delta_yaw)
            sin_yaw = torch.sin(delta_yaw)
            heading_x = cos_yaw * heading[:, 0] - sin_yaw * heading[:, 1]
            heading_y = sin_yaw * heading[:, 0] + cos_yaw * heading[:, 1]
            command[:, 3:5] = torch.stack([heading_x, heading_y], dim=-1)
            return command

        action = torch.tanh(raw_action) * self.command_scale
        command = torch.zeros(self.num_envs, self.root_storage_dim, device=self.device)
        command[:, 0] = self.nominal_root_height + action[:, 0]
        command[:, 1:3] = action[:, 1:3]

        heading_xy = torch.zeros(self.num_envs, 2, device=self.device)
        heading_xy[:, 0] = 1.0
        heading_xy = heading_xy + action[:, 3:5]
        heading_xy = heading_xy / heading_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        command[:, 3:5] = heading_xy
        return command

    def _decode_ee_command(self, raw_action: torch.Tensor) -> torch.Tensor:
        reference = self._get_ee_command_reference()
        action = torch.tanh(raw_action)

        command = reference.clone()
        if self.ee_command_mode == "pose_delta":
            pos_delta = action[:, :6].reshape(self.num_envs, 2, 3) * self.ee_pos_scale
            rot_delta = action[:, 6:12].reshape(self.num_envs, 2, 3) * self.ee_rot_scale
            command[:, :6] = reference[:, :6] + pos_delta.reshape(self.num_envs, 6)
            command[:, 6:12] = reference[:, 6:12] + rot_delta.reshape(self.num_envs, 6)
        elif self.ee_command_mode == "pos_delta":
            pos_delta = action.reshape(self.num_envs, 2, 3) * self.ee_pos_scale
            command[:, :6] = reference[:, :6] + pos_delta.reshape(self.num_envs, 6)
        return command

    def set_ee_force_stiffness_ablation_command(self, command: torch.Tensor):
        if not self.ee_command_enabled:
            raise RuntimeError("EE force/stiffness ablation requires ee_command.enabled=True.")
        command = torch.as_tensor(command, device=self.device, dtype=torch.float32)
        if command.shape != self.ee_command.shape:
            raise ValueError(f"Expected ablation EE command shape {tuple(self.ee_command.shape)}, got {tuple(command.shape)}.")
        self.ee_force_stiffness_ablation_enabled = True
        self.ee_force_stiffness_ablation_command[:] = command

    def clear_ee_force_stiffness_ablation_command(self):
        self.ee_force_stiffness_ablation_enabled = False
        self.ee_force_stiffness_ablation_command.zero_()

    def _decode_feet_command(self, raw_action: torch.Tensor) -> torch.Tensor:
        reference = self._get_feet_command_reference()
        action = torch.tanh(raw_action)
        pos_delta = action.reshape(self.num_envs, 2, 3) * self.feet_pos_scale
        return reference + pos_delta.reshape(self.num_envs, 6)

    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            raw_action = tensordict["action"].clamp(-10, 10)
            self.high_action_buf = torch.roll(self.high_action_buf, shifts=1, dims=1)
            self.high_action_buf[:, 0, :] = raw_action

            cursor = 0
            if self.root_command_enabled:
                self.root_command[:] = self._decode_root_command(raw_action[:, cursor:cursor + self.root_command_dim])
                self.root_command_buf = torch.roll(self.root_command_buf, shifts=1, dims=1)
                self.root_command_buf[:, 0, :] = self.root_command
                cursor += self.root_command_dim
            elif self.root_command_passthrough_reference:
                self._set_root_command_reference()
            else:
                self._set_default_root_command()
            if self.ee_command_enabled:
                self.ee_command[:] = self._decode_ee_command(raw_action[:, cursor:cursor + self.ee_command_dim])
                if self.ee_force_stiffness_ablation_enabled:
                    self.ee_command[:] = self.ee_force_stiffness_ablation_command
                self.ee_command_buf = torch.roll(self.ee_command_buf, shifts=1, dims=1)
                self.ee_command_buf[:, 0, :] = self.ee_command
                cursor += self.ee_command_dim
            if self.feet_command_enabled:
                self.feet_command[:] = self._decode_feet_command(raw_action[:, cursor:cursor + self.feet_command_dim])
                cursor += self.feet_command_dim
            if self.joint_residual_enabled:
                self.joint_residual[:] = (
                    torch.tanh(raw_action[:, cursor:cursor + self.joint_residual_dim])
                    * self.joint_residual_scale
                )
                cursor += self.joint_residual_dim
            if self.override_root_command:
                self._set_default_root_command()
            if not hasattr(self.env.command_manager, "set_root_command"):
                raise RuntimeError(
                    "HierarchicalRootCommand requires a command manager with set_root_command()."
                )
            self.env.command_manager.set_root_command(self.root_command)
            if self.ee_command_enabled:
                if not hasattr(self.env.command_manager, "set_root_and_wrist_6d_command"):
                    raise RuntimeError(
                        "HierarchicalRootCommand with ee_command.enabled=True requires a command manager "
                        "with set_root_and_wrist_6d_command()."
                    )
                self.env.command_manager.set_root_and_wrist_6d_command(self.ee_command)
            if self.feet_command_enabled:
                if not hasattr(self.env.command_manager, "set_feet_pos_b_command"):
                    raise RuntimeError(
                        "HierarchicalRootCommand with feet_command.enabled=True requires a command manager "
                        "with set_feet_pos_b_command()."
                    )
                self.env.command_manager.set_feet_pos_b_command(self.feet_command)
            low_td = tensordict.clone()
            if self.low_policy_obs_key is not None and self.low_policy_obs_key in low_td.keys():
                if self.low_policy_obs_key in self.env.observation_funcs:
                    low_td["policy"] = self.env.observation_funcs[self.low_policy_obs_key]._compute()
                else:
                    low_td["policy"] = low_td[self.low_policy_obs_key].clone()
            if self.low_policy_command_slice is not None:
                command = self.env.command_manager.command()
                start, stop = self.low_policy_command_slice
                low_td["policy"][:, start:stop] = command
                # low_td["policy"][:, stop:stop + 12] = torch.tensor([0.15, 0.1, 0.0, 0.15, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
            self.low_action[:] = self.low_policy.act(low_td)
            if self.joint_residual_enabled:
                self.low_action.add_(self.joint_residual)

        low_td = tensordict.clone()
        low_td["action"] = self.low_action
        self.low_action_manager(low_td, substep)
