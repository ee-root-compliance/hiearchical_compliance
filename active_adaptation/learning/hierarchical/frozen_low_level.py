from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import hydra
import torch
import wandb
import os
import torch.distributed as dist
from omegaconf import OmegaConf
from tensordict import TensorDictBase
from torchrl.data import Composite, UnboundedContinuous
from torchrl.envs.transforms import VecNorm
from torchrl.envs.utils import ExplorationType, set_exploration_type

from active_adaptation.utils.wandb import parse_checkpoint_path
import active_adaptation as aa


class FrozenLowLevelPolicy:
    """Lazy loader for a frozen low-level policy checkpoint."""

    def __init__(
        self,
        env,
        checkpoint_path: str | None,
        action_dim: int,
        run_path: str | None = None,
        checkpoint_iteration: int | None = None,
        phase: str | None = "adapt",
        vecnorm: str | None = "eval",
        in_keys: Sequence[str] | None = None,
        drop_policy_slices: Sequence[Sequence[int]] | None = None,
        preload: bool = False,
        device: str | torch.device | None = None,
    ):
        self.env = env
        self.checkpoint_path = checkpoint_path
        self.run_path = run_path
        self.checkpoint_iteration = checkpoint_iteration
        self.action_dim = action_dim
        self.phase = phase
        self.vecnorm_mode = vecnorm
        self.in_keys = list(in_keys) if in_keys is not None else None
        self.drop_policy_slices = [tuple(s) for s in drop_policy_slices or []]
        self.preload = bool(preload)
        self.device = torch.device(device or env.device)

        self.policy = None
        self.rollout_policy = None
        self.obs_norm = None
        self._loaded = False

    def _checkpoint_obs_dims(self, state_dict) -> dict[str, int]:
        if "vecnorm" not in state_dict:
            return {}
        extra_state = state_dict["vecnorm"].get("_extra_state", {})
        dims = {}
        for key, value in extra_state.items():
            if key.endswith("_sum") and hasattr(value, "shape"):
                dims[key[:-4]] = value.shape[-1]
        return dims

    def _make_low_observation_spec(self, state_dict):
        ckpt_dims = self._checkpoint_obs_dims(state_dict)
        spec = {}
        source_keys = ckpt_dims.keys() if ckpt_dims else self.env.observation_spec.keys(True, True)
        for key in source_keys:
            value = self.env.observation_spec[key]
            if value.dtype == bool or key.endswith("_"):
                continue
            dim = ckpt_dims.get(key, value.shape[-1])
            spec[key] = UnboundedContinuous((self.env.num_envs, dim), dtype=value.dtype)
        return Composite(spec, shape=[self.env.num_envs]).to(self.device)

    def _resolve_run_checkpoint_path(self) -> str:
        api = wandb.Api()
        run = api.run(self.run_path)
        root = os.path.join(os.path.dirname(__file__), "..", "wandb", run.name)
        root = os.path.abspath(root)
        os.makedirs(root, exist_ok=True)

        checkpoints = []
        for file in run.files():
            if "checkpoint" in file.name and file.name.endswith(".pt"):
                checkpoints.append(file)
            elif file.name in ("cfg.yaml", "files/cfg.yaml", "config.yaml"):
                file.download(root, replace=True)

        if not checkpoints:
            raise ValueError(f"No checkpoint .pt files found in run {self.run_path}.")

        if self.checkpoint_iteration is None:
            def sort_by_iter(file):
                number_str = os.path.basename(file.name)[:-3].split("_")[-1]
                return 100000 if number_str == "final" else int(number_str)
            checkpoint = sorted(checkpoints, key=sort_by_iter)[-1]
        else:
            name = f"checkpoint_{self.checkpoint_iteration}.pt"
            matches = [file for file in checkpoints if os.path.basename(file.name) == name]
            if not matches:
                raise ValueError(f"No {name} found in run {self.run_path}.")
            checkpoint = matches[0]

        local_path = os.path.join(root, checkpoint.name)
        should_download = (not aa.is_distributed()) or aa.is_main_process()
        if should_download:
            checkpoint.download(root, replace=True)
        if aa.is_distributed():
            dist.barrier()
        return local_path

    def _resolve_checkpoint_path(self) -> str:
        if self.checkpoint_path is not None:
            checkpoint_path = str(self.checkpoint_path)
            # hydra.job.chdir=true moves the cwd into outputs/, so resolve
            # relative local paths against the original launch directory.
            if not checkpoint_path.startswith("run:") and not os.path.isabs(checkpoint_path):
                checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path)
            return parse_checkpoint_path(checkpoint_path)
        if self.run_path is not None:
            return self._resolve_run_checkpoint_path()
        raise ValueError("Either low-level checkpoint_path or run_path is required.")

    def _make_low_action_spec(self):
        return SimpleNamespace(shape=torch.Size([self.action_dim]))

    def _rollout_obs_keys(self) -> list[str]:
        phase = getattr(self, "_active_phase", self.phase)
        if phase == "train":
            return ["policy", "priv", "joint_target"]
        return ["policy"]

    def _build_obs_norm(self, state_dict, observation_spec):
        if self.vecnorm_mode is None or "vecnorm" not in state_dict:
            return None

        checkpoint_extra_state = state_dict["vecnorm"].get("_extra_state", {})
        obs_keys = [
            key for key in self._rollout_obs_keys()
            if key in observation_spec.keys(True, True) and f"{key}_sum" in checkpoint_extra_state
        ]
        if not obs_keys:
            return None

        vecnorm = VecNorm(obs_keys, decay=0.9999)
        vecnorm(observation_spec.zero())
        filtered_extra_state = {
            stat_key: value
            for stat_key, value in checkpoint_extra_state.items()
            if any(stat_key.startswith(f"{obs_key}_") for obs_key in obs_keys)
        }
        vecnorm.load_state_dict({"_extra_state": filtered_extra_state})
        return vecnorm.to_observation_norm().to(self.device)

    def _adapt_low_tensordict(self, tensordict: TensorDictBase) -> TensorDictBase:
        low_td = tensordict.clone()
        if self.drop_policy_slices and "policy" in low_td.keys():
            policy = low_td["policy"]
            chunks = []
            cursor = 0
            for start, stop in sorted(self.drop_policy_slices):
                if start > cursor:
                    chunks.append(policy[..., cursor:start])
                cursor = stop
            if cursor < policy.shape[-1]:
                chunks.append(policy[..., cursor:])
            low_td["policy"] = torch.cat(chunks, dim=-1)
        return low_td

    def _check_checkpoint_compatible(self, ckpt_policy_state, checkpoint_path):
        """Fail loudly if the checkpoint cannot populate the rollout modules.

        The low-level PPO policy saves one nested state dict per child module
        (``encoder_priv``, ``adapt_module``, ``actor_student``, ``critic``, ...)
        and its ``load_state_dict`` only warns on per-module failures. Only the
        children used by the frozen rollout policy must match exactly; the
        others (critic, teacher branch in adapt/finetune phase, value_norm) are
        built from the current env spec and may legitimately differ.
        """
        rollout_params = {id(p) for p in self.rollout_policy.parameters()}
        model_state = self.policy.state_dict()
        errors = []
        for name, module in self.policy.named_children():
            if not any(id(p) in rollout_params for p in module.parameters()):
                continue
            expected = model_state.get(name, {})
            loaded = ckpt_policy_state.get(name)
            if loaded is None:
                errors.append(f"missing module '{name}'")
                continue
            for key, value in expected.items():
                if key not in loaded:
                    errors.append(f"missing key '{name}.{key}'")
                elif hasattr(value, "shape") and tuple(loaded[key].shape) != tuple(value.shape):
                    errors.append(
                        f"shape mismatch '{name}.{key}': checkpoint "
                        f"{tuple(loaded[key].shape)} vs expected {tuple(value.shape)}"
                    )
            unexpected = sorted(set(loaded.keys()) - set(expected.keys()))
            if unexpected:
                aa.print(f"[LowPolicy] WARNING: unexpected keys in '{name}': {unexpected}")
        if errors:
            raise RuntimeError(
                f"Low-level checkpoint {checkpoint_path} is incompatible with the frozen "
                f"low-level policy (phase={self._active_phase!r}):\n  " + "\n  ".join(errors)
            )

    def load(self):
        if self._loaded:
            return

        aa.print(
            "[LowPolicy] Loading frozen low-level checkpoint "
            f"(run_path={self.run_path!r}, checkpoint_path={self.checkpoint_path!r})"
        )
        checkpoint_path = self._resolve_checkpoint_path()
        aa.print(f"[LowPolicy] Resolved checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ckpt_cfg = state_dict.get("cfg")
        if ckpt_cfg is None:
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain cfg.")

        algo_cfg = ckpt_cfg.algo
        OmegaConf.set_struct(algo_cfg, False)
        if self.phase is not None:
            algo_cfg.phase = self.phase
        if self.in_keys is not None:
            algo_cfg.in_keys = self.in_keys
        self._active_phase = algo_cfg.phase

        observation_spec = self._make_low_observation_spec(state_dict)
        policy_cls = hydra.utils.get_class(algo_cfg._target_)
        self.policy = policy_cls(
            algo_cfg,
            observation_spec,
            self._make_low_action_spec(),
            self.env.reward_spec,
            device=self.device,
            env=self.env,
        )
        self.rollout_policy = self.policy.get_rollout_policy("eval")
        self._check_checkpoint_compatible(state_dict["policy"], checkpoint_path)
        self.policy.load_state_dict(state_dict["policy"], strict=False)
        self.policy.requires_grad_(False)
        self.policy.eval()
        self.obs_norm = self._build_obs_norm(state_dict, observation_spec)
        self._loaded = True
        aa.print("[LowPolicy] Frozen low-level policy ready")

    @torch.no_grad()
    def act(self, tensordict: TensorDictBase) -> torch.Tensor:
        self.load()
        low_td = self._adapt_low_tensordict(tensordict)
        if self.obs_norm is not None:
            self.obs_norm(low_td)
        with set_exploration_type(ExplorationType.MODE):
            self.rollout_policy(low_td)
        return low_td["action"]
