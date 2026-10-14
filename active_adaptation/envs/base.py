import torch
import numpy as np
import hydra
import inspect

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite, 
    Binary,
    UnboundedContinuous,
)
from collections import OrderedDict

from abc import abstractmethod
from typing import NamedTuple, Dict
import time

import active_adaptation
import active_adaptation.envs.mdp as mdp
import active_adaptation.utils.symmetry as symmetry_utils

class ObsGroup:
    
    def __init__(
        self,
        env,
        name: str,
        funcs: Dict[str, mdp.Observation],
        max_delay: int = 0,
    ):
        self.env = env
        self.name = name
        self.funcs = funcs
        self.max_delay = max_delay
        self.timestamp = -1

    @property
    def keys(self):
        return self.funcs.keys()

    @property
    def spec(self):
        if not hasattr(self, "_spec"):
            foo = self.compute({}, 0)
            spec = {}
            spec[self.name] = UnboundedContinuous(foo[self.name].shape, dtype=foo[self.name].dtype)
            self._spec = Composite(spec, shape=[foo[self.name].shape[0]]).to(foo[self.name].device)
        return self._spec

    def compute(self, tensordict: TensorDictBase, timestamp: int) -> torch.Tensor:
        output = self._compute()

        ### debug symmetry
        # output = self.symmetry_transforms().to(output.device).forward(output)

        tensordict[self.name] = output
        return tensordict
    
    def _compute(self) -> torch.Tensor:
        tensors = []
        for obs_key, func in self.funcs.items():
            tensor = func()
            tensors.append(tensor)
        return torch.cat(tensors, dim=-1)

    def symmetry_transforms(self):
        if not hasattr(self, "_symmetry_transforms"):
            transforms = []
            for obs_key, func in self.funcs.items():
                transform = func.symmetry_transforms()
                tensor = func()
                if tensor.shape[-1] != transform.perm.shape[-1]:
                    raise ValueError(
                        f"Observation '{obs_key}' has dim {tensor.shape[-1]} but its symmetry "
                        f"transform has dim {transform.perm.shape[-1]}"
                    )
                transforms.append(transform)
            transform = symmetry_utils.SymmetryTransform.cat(transforms)
            self._symmetry_transforms = transform
        else:
            transform = self._symmetry_transforms
        return transform

class RewardGroup:
    def __init__(self, env, name: str, funcs: OrderedDict[str, mdp.Reward], scale: list[float] | None = None, student_train: bool = False):
        self.env = env
        self.name = name
        self.funcs = funcs
        self.scale = scale
        self.student_train = student_train
        self.current_factor = 1.0
        self.enabled_rewards = sum([func.enabled for func in funcs.values()])
        self.rew_buf = torch.zeros(env.num_envs, self.enabled_rewards, device=env.device)

        if self.scale is not None:
            if len(self.scale) != 4:
                raise ValueError(f"Scale must be a list of 4 elements, got {self.scale}")
            if self.scale[0] >= self.scale[1]:
                raise ValueError(f"Scale[0] must be less than Scale[1], got {self.scale[0]} >= {self.scale[1]}")
            if self.scale[2] >= self.scale[3]:
                raise ValueError(f"Scale[2] must be less than Scale[3], got {self.scale[2]} >= {self.scale[3]}")
            print(f"Reward group '{self.name}' is scaled with progress from {self.scale[0]} to {self.scale[1]} with factor from {self.scale[2]} to {self.scale[3]}")
            self.step_schedule(0.0)  # Initialize current_factor based on scale
    
    def compute(self) -> torch.Tensor:
        rewards = []
        for key, func in self.funcs.items():
            if not func.enabled:
                continue
            reward, count = func()
            self.env.stats[self.name, key].add_(reward)
            sum, cnt = self.env._stats_ema[self.name][key]
            sum.mul_(self.env._stats_ema_decay).add_(reward.sum())
            cnt.mul_(self.env._stats_ema_decay).add_(count)
            rewards.append(reward)
        if len(rewards):
            self.rew_buf[:] = torch.cat(rewards, 1)
        return self.rew_buf.sum(1, True) * self.current_factor
    
    def step_schedule(self, progress: float):
        if self.student_train:
            progress = 1.0
        if self.scale is not None:
            scaled = (progress - self.scale[0]) / (self.scale[1] - self.scale[0])
            scaled = min(max(scaled, 0.0), 1.0)
            self.current_factor = scaled * (self.scale[3] - self.scale[2]) + self.scale[2]

from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene

class _Env(EnvBase):
    scene: InteractiveScene
    sim: SimulationContext
    @torch.no_grad()
    def __init__(self, cfg):
        self.cfg = cfg
        active_adaptation._ENVS = cfg.num_envs
        self.backend = active_adaptation.get_backend()
        self.setup_scene()
        
        self.max_episode_length = self.cfg.max_episode_length
        self.step_dt = self.cfg.sim.step_dt
        self.physics_dt = self.sim.get_physics_dt()
        self.decimation = int(self.step_dt / self.physics_dt)
        
        print(f"Step dt: {self.step_dt}, physics dt: {self.physics_dt}, decimation: {self.decimation}")

        super().__init__(
            device=self.sim.device,
            batch_size=[self.num_envs],
            run_type_checks=False,
        )
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=int, device=self.device)

        # parse obs and reward functions
        self.done_spec = Composite(
            done=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            terminated=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            truncated=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            shape=[self.num_envs, 1],
            device=self.device
        )

        self.reward_spec = Composite(
            {
                "stats": {
                    "episode_len": UnboundedContinuous([self.num_envs, 1]),
                    "success": UnboundedContinuous([self.num_envs, 1]),
                },
            },
            shape=[self.num_envs]
        ).to(self.device)

        members = dict(inspect.getmembers(self.__class__, inspect.isclass))
        self.command_manager: mdp.Command = hydra.utils.instantiate(self.cfg.command, env=self)
        self.command_manager.before_update()
        self.command_manager.update()

        RAND_FUNCS = mdp.RAND_FUNCS
        RAND_FUNCS.update(mdp.get_obj_by_class(members, mdp.Randomization))
        OBS_FUNCS = mdp.OBS_FUNCS
        OBS_FUNCS.update(mdp.get_obj_by_class(members, mdp.Observation))
        REW_FUNCS = mdp.REW_FUNCS
        REW_FUNCS.update(mdp.get_obj_by_class(members, mdp.Reward))
        TERM_FUNCS = mdp.TERM_FUNCS

        for k, v in inspect.getmembers(self.command_manager):
            if getattr(v, "is_reward", False):
                REW_FUNCS[k] = mdp.reward_wrapper(v)
            elif getattr(v, "is_observation", False):
                name = v.__func__.__name__
                name_sym = f"{name}_sym"
                sym_func = getattr(self.command_manager, name_sym)
                OBS_FUNCS[k] = mdp.observation_wrapper(v, sym_func)
            elif getattr(v, "is_termination", False):
                TERM_FUNCS[k] = mdp.termination_wrapper(v)

        self.randomizations = OrderedDict()
        self.observation_funcs: Dict[str, ObsGroup] = OrderedDict()
        self.reward_funcs = OrderedDict()
        self._startup_callbacks = []
        self._update_callbacks = []
        self._reset_callbacks = []
        self._debug_draw_callbacks = []
        self._pre_step_callbacks = []
        self._post_step_callbacks = []

        self._pre_step_callbacks.append(self.command_manager.step)
        # self._update_callbacks.append(self.command_manager.update)
        self._reset_callbacks.append(self.command_manager.reset)
        self._debug_draw_callbacks.append(self.command_manager.debug_draw)
        
        self.action_manager: mdp.ActionManager = hydra.utils.instantiate(self.cfg.action, env=self)
        self._reset_callbacks.append(self.action_manager.reset)
        self._debug_draw_callbacks.append(self.action_manager.debug_draw)
        
        self.action_spec = Composite(
            {
                "action": UnboundedContinuous((self.num_envs, self.action_dim))
            },
            shape=[self.num_envs]
        ).to(self.device)


        for key, params in self.cfg.randomization.items():
            rand = RAND_FUNCS[key](self, **params if params is not None else {})
            self.randomizations[key] = rand
            self._startup_callbacks.append(rand.startup)
            self._reset_callbacks.append(rand.reset)
            self._debug_draw_callbacks.append(rand.debug_draw)
            self._pre_step_callbacks.append(rand.step)
            self._update_callbacks.append(rand.update)

        for group_key, params in self.cfg.observation.items():
            max_delay = params.pop("_max_delay_", 0)
            if max_delay > 1e-6:
                raise NotImplementedError
            funcs = OrderedDict()            
            for key, kwargs in params.items():
                kwargs = dict(kwargs) if kwargs is not None else {}
                enabled = kwargs.pop("enabled", True)
                if enabled is False:
                    continue
                obs = OBS_FUNCS[key](self, **kwargs)
                funcs[key] = obs

                self._startup_callbacks.append(obs.startup)
                self._update_callbacks.append(obs.update)
                self._reset_callbacks.append(obs.reset)
                self._debug_draw_callbacks.append(obs.debug_draw)
                self._post_step_callbacks.append(obs.post_step)
            
            self.observation_funcs[group_key] = ObsGroup(self, group_key, funcs)
        
        for callback in self._startup_callbacks:
            callback()        
       
        reward_spec = Composite({})

        # parse rewards
        self.clip_rewards = self.cfg.reward.pop("_clip_", True)
        self.mult_dt = self.cfg.reward.pop("_mult_dt_", True)
        self.reward_student_train = self.cfg.reward.pop("_student_train_", False)

        self._stats_ema = {}
        self._stats_ema_decay = 0.99

        self.reward_groups = OrderedDict()
        for group_name, func_specs in self.cfg.reward.items():
            print(f"Reward group: {group_name}")
            funcs = OrderedDict()
            scale_factor = func_specs.pop("_scale_factor_", None)
            self._stats_ema[group_name] = {}

            for key, params in func_specs.items():
                reward: mdp.Reward = REW_FUNCS[key](self, **params)
                funcs[key] = reward
                reward_spec["stats", group_name, key] = UnboundedContinuous(1, device=self.device)
                self._update_callbacks.append(reward.update)
                self._reset_callbacks.append(reward.reset)
                self._debug_draw_callbacks.append(reward.debug_draw)
                self._pre_step_callbacks.append(reward.step)
                self._post_step_callbacks.append(reward.post_step)
                print(f"\t{key}: \t{reward.weight:.2f}, \t{reward.enabled}")
                self._stats_ema[group_name][key] = (torch.tensor(0., device=self.device), torch.tensor(0., device=self.device))

            self.reward_groups[group_name] = RewardGroup(self, group_name, funcs, scale=scale_factor, student_train=self.reward_student_train)
            reward_spec["stats", group_name, "return"] = UnboundedContinuous(1, device=self.device)

        reward_spec["reward"] = UnboundedContinuous(max(1, len(self.reward_groups)), device=self.device)
        reward_spec["discount"] = UnboundedContinuous(1, device=self.device)
        self.reward_spec.update(reward_spec.expand(self.num_envs).to(self.device))
        self.discount = torch.ones((self.num_envs, 1), device=self.device)

        observation_spec = {}
        for group_key, group in self.observation_funcs.items():
            observation_spec.update(group.spec)

        self.observation_spec = Composite(
            observation_spec, 
            shape=[self.num_envs],
            device=self.device
        )

        self.termination_funcs = OrderedDict()
        for key, params in self.cfg.termination.items():
            params = dict(params) if params is not None else {}
            enabled = params.pop("enabled", True)
            if enabled is False:
                continue
            term_func = TERM_FUNCS[key](self, **params)
            self.termination_funcs[key] = term_func
            self._update_callbacks.append(term_func.update)
            self._reset_callbacks.append(term_func.reset)
            self.reward_spec["stats", "termination", key] = UnboundedContinuous((self.num_envs, 1), device=self.device)
        
        self.timestamp = 0

        self.stats = self.reward_spec["stats"].zero()
    
        self.input_tensordict = None
        self.lookat_env_i = 0

        self.extra = {}

    @property
    def action_dim(self) -> int:
        return self.action_manager.action_dim

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return self.scene.num_envs

    @property
    def stats_ema(self):
        result = {}
        for group_key, group in self._stats_ema.items():
            result[group_key] = {}
            for rew_key, (sum, cnt) in group.items():
                result[group_key][rew_key] = (sum / cnt.clamp_min(1e-8)).item()
        return result
    
    def setup_scene(self):
        raise NotImplementedError
    
    @torch.no_grad()
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:

        # get envids
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
        else:
            env_mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        env_ids = env_mask.nonzero().squeeze(-1)

        # reset things in simulation
        if len(env_ids):
            self._reset_idx(env_ids)

        # reset episode length buffer
        if env_ids.numel() < self.num_envs * 0.2:
            self.episode_length_buf[env_ids] = torch.randint(0, self.max_episode_length // 5, (env_ids.numel(),), device=self.device)
        else:
            self.episode_length_buf[env_ids] = torch.randint(0, self.max_episode_length, (env_ids.numel(),), device=self.device)

        # reset mdp
        for callback in self._reset_callbacks:
            callback(env_ids)

        # clean up obs
        tensordict = TensorDict({}, self.num_envs, device=self.device)
        tensordict.update(self.observation_spec.zero())

        return tensordict

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        raise NotImplementedError
    
    def apply_action(self, tensordict: TensorDictBase, substep: int):
        self.input_tensordict = tensordict
        self.action_manager(tensordict, substep)

    def _compute_observation(self, tensordict: TensorDictBase):
        try:
            for group_key, obs_group in self.observation_funcs.items():
                obs_group.compute(tensordict, self.timestamp)
        except Exception as e:
            print(f"Error in computing observation for {group_key}: {e}")
            raise e
    
    def _compute_reward(self) -> TensorDictBase:
        if not self.reward_groups:
            return {"reward": torch.ones((self.num_envs, 1), device=self.device)}
        
        rewards = []
        for group, reward_group in self.reward_groups.items():
            reward = reward_group.compute()
            if self.mult_dt:
                reward *= self.step_dt
            rewards.append(reward)
            self.stats[group, "return"].add_(reward)

        rewards = torch.cat(rewards, 1)

        self.stats["episode_len"][:] = self.episode_length_buf.unsqueeze(1)
        self.stats["success"][:] = (self.episode_length_buf >= self.max_episode_length * 0.9).unsqueeze(1).float()
        return {"reward": rewards}
    
    def _compute_termination(self) -> TensorDictBase:
        if not self.termination_funcs:
            return torch.zeros((self.num_envs, 1), dtype=bool, device=self.device)
        flags = []
        for key, func in self.termination_funcs.items():
            flag = func()
            self.stats["termination", key][:] = flag.float()
            flags.append(flag)
        flags = torch.cat(flags, dim=-1)
        return flags.any(dim=-1, keepdim=True)

    def _update(self):
        for callback in self._update_callbacks:
            callback()
        if self.sim.has_gui():
            self.sim.render()
        self.episode_length_buf.add_(1)
        self.timestamp += 1

    def _update_sim(self, tensordict: TensorDictBase):
        for substep in range(self.decimation):
            for asset in self.scene.articulations.values():
                if asset.has_external_wrench:
                    asset._external_force_b.zero_()
                    asset._external_torque_b.zero_()
                    asset.has_external_wrench = False

            # take actions
            self.apply_action(tensordict, substep)

            # do step for obs/action/rand
            for callback in self._pre_step_callbacks:
                callback(substep)

            # Deal with custom world-frame force apply logic from command managers.
            if getattr(self.command_manager, "force_apply_world", False):
                asset = self.command_manager.asset
                force = self.command_manager.force_apply_buffer.view(-1, 3)
                torque = getattr(self.command_manager, "torque_apply_buffer", None)
                position = getattr(self.command_manager, "position_apply_buffer", None)
                if torque is not None:
                    torque = torque.view(-1, 3)
                if position is not None:
                    position = position.view(-1, 3)
                asset.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=force,
                    torque_data=torque,
                    position_data=position,
                    indices=asset._ALL_INDICES,
                    is_global=True,
                )

            self.scene.write_data_to_sim()

            # run simulation
            self.sim.step(render=False)

            # update buffer
            self.scene.update(self.physics_dt)

            # do post step for obs
            for callback in self._post_step_callbacks:
                callback(substep)

    @torch.no_grad()
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        ## update simulation
        self._update_sim(tensordict)

        ## do command update before reward computation
        self.command_manager.before_update()

        self.discount.fill_(1.0)
        
        self._update()
        
        tensordict = TensorDict({}, self.num_envs, device=self.device)
        
        ## compute reward
        tensordict.update(self._compute_reward())

        ## update command
        self.command_manager.update()

        # update observation
        self._compute_observation(tensordict)

        # update termination
        terminated = self._compute_termination()
        terminated = terminated & (self.episode_length_buf > 5).unsqueeze(1) # do not terminate in the first 5 steps
        truncated = (self.episode_length_buf >= self.max_episode_length).unsqueeze(1)
        if hasattr(self.command_manager, "finished"):
            truncated = truncated | self.command_manager.finished.unsqueeze(1)
        tensordict.set("terminated", terminated)
        tensordict.set("truncated", truncated)
        tensordict.set("done", terminated | truncated)
        tensordict.set("discount", self.discount.clone())
        tensordict["stats"] = self.stats.clone()

        if self.sim.has_gui():
            if hasattr(self, "debug_draw"): # isaac only
                self.debug_draw.clear()
            for callback in self._debug_draw_callbacks:
                callback()
            self.debug_vis()
            
        return tensordict
    
    def _set_seed(self, seed: int = -1):
        # import omni.replicator.core as rep
        # rep.set_global_seed(seed)
        torch.manual_seed(seed)

    def render(self, mode: str = "human"):
        self.sim.render()
        if mode == "human":
            return None
        elif mode == "rgb_array":
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            return rgb_data[:, :, :3]
        else:
            raise NotImplementedError
    
    def debug_vis(self):
        pass
    

    def state_dict(self):
        sd = super().state_dict()
        sd["observation_spec"] = self.observation_spec
        sd["action_spec"] = self.action_spec
        sd["reward_spec"] = self.reward_spec
        return sd

    def get_extra_state(self) -> dict:
        return dict(self.extra)

    def close(self):
        if not self.is_closed:
            if self.backend == "isaac":
                # destructor is order-sensitive
                del self.scene
                # clear callbacks and instance
                self.sim.clear_all_callbacks()
                self.sim.clear_instance()
                # update closing status
            super().close()
            
    def step_schedule(self, progress: float, iter: int):
        if hasattr(self.command_manager, "step_schedule"):
            if inspect.signature(self.command_manager.step_schedule).parameters.get("iter"):
                self.command_manager.step_schedule(progress, iter)
            else:
                self.command_manager.step_schedule(progress)
        for rew in self.reward_groups.values():
            if hasattr(rew, "step_schedule"):
                rew.step_schedule(progress)
