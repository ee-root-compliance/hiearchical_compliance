import warnings
import copy
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.transforms import TensorDictPrimer, ExcludeTransform
from torchrl.modules import ProbabilisticActor
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
    CudaGraphModule
)

from hydra.core.config_store import ConfigStore

# ---- utils ------------------------------------------------------------------------------------ #
from ..modules.distributions import IndependentNormal
from ..utils.valuenorm import ValueNorm1, ValueNormFake
from .common import *
import active_adaptation as aa
import functools


def _compile_training_function(function):
    """Compile PPO-only helpers unless evaluation explicitly disables it."""
    if os.environ.get("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE") == "1":
        return function
    return torch.compile(function)

__all__ = ["PPOPolicy", "PPOConfig"]

# ------------------------------------------------------------------------------------------------ #
# 1. Config
# ------------------------------------------------------------------------------------------------ #


@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo.PPOPolicy"
    name: str = "ppo"

    # PPO hyper‑params
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8

    lr: float = 5e-4
    desired_kl: float = 0.01 # kl schedule

    clip_param: float = 0.2

    entropy_coef_start: float = 0.005
    entropy_coef_end: float = 0.002

    init_noise_scale: float = 1.0  # initial std for actor
    load_noise_scale: float | None = None  # initial std for student actor

    latent_dim: int = 256
    # joint prediction weight for adapt-phase estimator
    joint_pred_weight: float = 1.0

    # distillation
    reg_lambda: float = 0.2  # weight of priv-feature alignment
    # misc
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    # phase switch
    phase: str = "train"  # train | finetune | adapt
    vecnorm: Union[str, None] = None

    # I/O keys
    in_keys: List[str] = field(
        default_factory=lambda: [
            OBS_KEY,
            OBS_PRIV_KEY,
            OBS_JOINT_KEY,
            CRITIC_PRIV_KEY
        ]
    )

    command_modes: Union[List[int], None] = None
    checkpoint_path: Union[str, None] = None


cs = ConfigStore.instance()
cs.store("ppo_train", node=PPOConfig(phase="train", vecnorm="train"), group="algo")
cs.store("ppo_adapt", node=PPOConfig(phase="adapt", vecnorm="eval", train_every=16), group="algo")
cs.store("ppo_finetune", node=PPOConfig(phase="finetune", vecnorm="eval", lr=1e-4, entropy_coef_start=0.002, entropy_coef_end=0.0005), group="algo")

class PPOPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device: str = "cuda:0",
        env = None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        assert cfg.phase in {"train", "finetune", "adapt"}

        self.entropy_coef = cfg.entropy_coef_start
        self.clip_param = cfg.clip_param
        self.action_dim = action_spec.shape[-1]
        self.joint_names = env.action_manager.joint_names
        self.gae = GAE(0.99, 0.95)
        self.reg_lambda = 0.0  # will be annealed
        self.num_minibatches = cfg.num_minibatches
        self.progress = 0.0
        self.current_lr = cfg.lr

        self.reward_groups = list(env.cfg.reward.keys())

        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_td = observation_spec.zero().to(device)

        # ---------------------------------------------------------------------------- private encoder
        self.encoder_priv = Seq(
            Mod(nn.Sequential(make_mlp([256]), nn.LazyLinear(self.cfg.latent_dim)), [OBS_PRIV_KEY], ["priv_feature"]),
        ).to(device)

        # ---------------------------------------------------------------------------- state estimator (student)
        self.adapt_module = Mod(
            nn.Sequential(
                make_mlp([512, 256]),
                nn.LazyLinear(self.cfg.latent_dim),
            ),
            [OBS_KEY],
            ["priv_pred"],
        ).to(device)
        # ---------------------------------------------------------------------------- joint predictor (student)
        # predict a privileged joint observation (priv_joint) from OBS_KEY so that
        # actor_student can consume it during adapt/finetune
        # determine joint dim from fake_td
        joint_dim = fake_td[OBS_JOINT_KEY].shape[-1]
        self.adapt_joint_module = Mod(
            nn.Sequential(
                make_mlp([512, 256]),
                nn.LazyLinear(joint_dim),
            ),
            [OBS_KEY],
            ["priv_joint"],
        ).to(device)
        # ---------------------------------------------------------------------------- actor(s)
        # Teacher: uses policy + joint_target (direct) + priv_feature (encoded)
        # Student: uses policy + priv_pred (predicted latent, no joint_target)
        # actor_in_keys_train = [OBS_KEY, "priv_feature"]
        actor_in_keys_train = [OBS_KEY, "priv_feature", OBS_JOINT_KEY]
        actor_in_keys_adapt = [OBS_KEY, "priv_pred", "priv_joint"]

        def build_actor(in_keys):
            return ProbabilisticActor(
                module=Seq(
                    CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),
                    Mod(make_mlp([512, 512, 256]), ["_actor_inp"], ["_actor_feature"]),
                    Mod(Actor(self.action_dim, init_noise_scale=self.cfg.init_noise_scale, load_noise_scale=self.cfg.load_noise_scale), ["_actor_feature"], ["loc", "scale"]),
                ),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True,
            ).to(device)

        self.actor_teacher = build_actor(actor_in_keys_train)
        self.actor_student = build_actor(actor_in_keys_adapt)

        # ---------------------------------------------------------------------------- critic (shared)
        self.critic = Seq(
            CatTensors([OBS_KEY, OBS_PRIV_KEY, CRITIC_PRIV_KEY], "_critic_inp", del_keys=False),
            Mod(nn.Sequential(make_mlp([512, 512, 256]), nn.LazyLinear(1)), ["_critic_inp"], ["state_value"]),
        ).to(device)

        # ---------------------------------------------------------------------------- lazy init pass
        with torch.device(device):
            fake_td["is_init"] = torch.ones(fake_td.shape[0], 1, dtype=torch.bool)
        self.encoder_priv(fake_td)
        self.adapt_module(fake_td)
        self.adapt_joint_module(fake_td)
        self.actor_teacher(fake_td)
        self.actor_student(fake_td)
        self.critic(fake_td)

        # init weights (orthogonal for MLPS/linear)
        def ortho_(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

        self.apply(ortho_)

        self.world_size = 1
        self.num_updates = 0
        if aa.is_distributed():
            self.world_size = aa.get_world_size()
            self._wrap_ddp(local_rank=aa.get_local_rank())

        # ---------------------------------------------------------------------------- optimisers
        self.opt_teacher = torch.optim.Adam(
            [
                {"params": self.actor_teacher.parameters()},
                {"params": self.encoder_priv.parameters()},
            ],
            lr=cfg.lr,
        )
        self.opt_student = torch.optim.Adam(
            [
                {"params": self.actor_student.parameters()},
                {"params": self.adapt_module.parameters()},
            ],
            lr=cfg.lr,
        )
        self.opt_joint = torch.optim.Adam(self.adapt_joint_module.parameters(), lr=cfg.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.opt_estimator = torch.optim.Adam(self.adapt_module.parameters(), lr=cfg.lr)

        self.update_teacher = functools.partial(self._update, actor=self.actor_teacher, encoder=self.encoder_priv, critic=self.critic, opt_actor=self.opt_teacher, opt_critic=self.opt_critic)
        self.update_student = functools.partial(self._update, actor=self.actor_student, encoder=self.adapt_module, critic=self.critic, opt_actor=self.opt_student, opt_critic=self.opt_critic)
        self.update2 = functools.partial(
            self._update2,
            adapt_module=self.adapt_module,
            adapt_joint_module=self.adapt_joint_module,
            opt_estimator=self.opt_estimator,
            opt_joint=self.opt_joint,
        )

        self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transforms()
        self.obs_priv_transform = env.observation_funcs[OBS_PRIV_KEY].symmetry_transforms()
        self.obs_joint_transform = env.observation_funcs[OBS_JOINT_KEY].symmetry_transforms()
        self.critic_priv_transform = env.observation_funcs[CRITIC_PRIV_KEY].symmetry_transforms()
        self.act_transform = env.action_manager.symmetry_transforms()

        self.obs_transform = self.obs_transform.to(self.device)
        self.obs_priv_transform = self.obs_priv_transform.to(self.device)
        self.obs_joint_transform = self.obs_joint_transform.to(self.device)
        self.critic_priv_transform = self.critic_priv_transform.to(self.device)
        self.act_transform = self.act_transform.to(self.device)

    def _wrap_ddp(self, local_rank: int):
        ddp_kwargs = dict(device_ids=[local_rank], output_device=local_rank,
                        broadcast_buffers=True, find_unused_parameters=False)

        self.actor_teacher = DDP(self.actor_teacher, **ddp_kwargs)
        self.actor_student = DDP(self.actor_student, **ddp_kwargs)
        self.encoder_priv  = DDP(self.encoder_priv,  **ddp_kwargs)
        self.critic        = DDP(self.critic,        **ddp_kwargs)
        self.adapt_module  = DDP(self.adapt_module,  **ddp_kwargs)
        self.adapt_joint_module = DDP(self.adapt_joint_module, **ddp_kwargs)

    def broadcast_parameters(self, extra_modules=[]):
        if self.num_updates % 32 == 0:
            update_list = [self.value_norm] + extra_modules
            if aa.is_distributed():
                for m in update_list:
                    for p in m.parameters():
                        dist.broadcast(p, src=0)
                    for p in m.buffers():
                        dist.broadcast(p, src=0)

    def do_lr_schedule(self, kl):
        if not hasattr(self, "current_lr"):
            self.current_lr = self.cfg.lr
        
        if self.progress < 0.1:
            return

        if aa.is_distributed():
            kl_tensor = torch.tensor(kl, device=self.device)
            dist.all_reduce(kl_tensor, op=dist.ReduceOp.SUM)
            kl = (kl_tensor / self.world_size).item()

        new_lr = self.current_lr
        if kl > self.cfg.desired_kl * 2.0:
            new_lr = max(1e-5, new_lr / 1.1)
        elif 0.0 < kl < self.cfg.desired_kl / 2.0:
            new_lr = min(5e-3, new_lr * 1.1)

        self.current_lr = new_lr

        for opt in (self.opt_teacher, self.opt_student):
            for param_group in opt.param_groups:
                param_group["lr"] = self.current_lr

    def make_tensordict_primer(self):
        return None

    def get_rollout_policy(self, mode: str = "train"):
        modules = []
        if self.cfg.phase == "train":
            modules += [self.encoder_priv, self.actor_teacher]
        elif self.cfg.phase == "finetune":
            modules += [self.adapt_module]
            modules += [self.adapt_joint_module]
            modules += [self.actor_student]
        elif self.cfg.phase == "adapt":
            modules += [self.adapt_module]
            modules += [self.adapt_joint_module]
            modules += [self.actor_student]

        policy = Seq(*modules)
        return policy

    def step_schedule(self, progress: float, iter: int):
        self.reg_lambda = progress * self.cfg.reg_lambda
        start = self.cfg.entropy_coef_start
        end = self.cfg.entropy_coef_end
        # exponential decay from start to end based on progress in [0,1]
        self.entropy_coef = start * (end / start) ** progress
        self.progress = progress

    def train_op(self, td: TensorDict, vecnorm):
        """One optimisation step on a batched rollout tensor-dict."""
        if self.cfg.phase == "train":
            info = {}
            info.update(self._ppo_update(td, self.update_teacher))
            info.update(self.train_estimator(td))
        elif self.cfg.phase == "finetune":
            info = {}
            if self.progress > 0.025:
                info.update(self._ppo_update(td, self.update_student))
        else:  # adapt
            info = self.train_estimator(td)
        self.num_updates += 1
        self.broadcast_parameters(extra_modules=[vecnorm])
        return info

    def _ppo_update(self, td, update_func: callable = None):
        infos = []
        self._compute_advantage(td, self.critic, self.gae, self.value_norm, 
                               REWARD_KEY=REWARD_KEY, TERM_KEY=TERM_KEY, DONE_KEY=DONE_KEY)
        self._modewise_adv_norm(td)

        for _ in range(self.cfg.ppo_epochs):
            for mb in make_batch(td, self.num_minibatches):
                infos.append(TensorDict(update_func(mb), []))
        info = {k: v.mean().item() for k, v in torch.stack(infos).items()}

        with torch.no_grad():
            actor = self.actor_teacher if self.cfg.phase == "train" else self.actor_student
            base = actor.module if isinstance(actor, DDP) else actor
            action_std = base.module[0][2].module.actor_std.detach()
            for joint_name, std in zip(self.joint_names, action_std):
                info[f"actor_std/{joint_name}"] = std
            info["actor_std/mean"] = action_std.mean()

        kl = info["actor/kl"]
        self.do_lr_schedule(kl)
        info["lr"] = self.current_lr

        neg_reward_ratio = (td[REWARD_KEY] <= 0.0).float().mean().item()
        info["critic/neg_reward_ratio"] = neg_reward_ratio

        return info

    def _update(self, mb, actor = None, encoder = None, critic = None, opt_actor = None, opt_critic = None):
        bsize = mb.shape[0]
        loc_old, scale_old = mb["loc"].clone(), mb["scale"].clone()
        action_old = mb["action"].clone()
        logp_old = mb["sample_log_prob"].clone()

        mb_sym = mb.clone()
        mb_sym[OBS_KEY] = self.obs_transform(mb_sym[OBS_KEY])
        mb_sym[OBS_PRIV_KEY] = self.obs_priv_transform(mb_sym[OBS_PRIV_KEY])
        if OBS_JOINT_KEY in mb_sym.keys():
            mb_sym[OBS_JOINT_KEY] = self.obs_joint_transform(mb_sym[OBS_JOINT_KEY])
        mb_sym[CRITIC_PRIV_KEY] = self.critic_priv_transform(mb_sym[CRITIC_PRIV_KEY])
        mb_sym["adv"] = mb["adv"]
        mb_sym["ret"] = mb["ret"]
        mb_sym["is_init"] = mb["is_init"]

        mb_sym = mb_sym.exclude("next")
        mb = mb.exclude("next")
        mb = torch.cat([mb, mb_sym], dim=0)
        valid = ~mb["is_init"]
        mb = mb.exclude("sample_log_prob", "action")

        if encoder is not None:
            encoder(mb)
        actor(mb)

        dist = IndependentNormal(mb["loc"][:bsize], mb["scale"][:bsize])
        logp = dist.log_prob(action_old)
        entropy = dist.entropy().mean()

        ratio = torch.exp(logp - logp_old).unsqueeze(-1)
        surr1 = mb["adv"][:bsize] * ratio
        surr2 = mb["adv"][:bsize] * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2) * valid[:bsize])
        entropy_loss = - self.entropy_coef * entropy

        values = critic(mb)["state_value"]
        value_loss = F.mse_loss(mb["ret"], values, reduction="none")
        value_loss = (value_loss * valid).mean(dim=0)

        if self.cfg.phase == "train":
            if "priv_pred" not in mb.keys():
                with torch.no_grad():
                    self.adapt_module(mb)
            reg_loss = F.mse_loss(mb["priv_pred"], mb["priv_feature"], reduction="none")
            reg_loss = self.reg_lambda * torch.mean(reg_loss * valid)
        else:
            reg_loss = 0.0
        
        symmetry_loss_loc = F.mse_loss(mb["loc"][:bsize], self.act_transform(mb["loc"][bsize:])) * 0.2
        symmetry_loss_std = F.mse_loss(mb["scale"][:bsize], self.act_transform(mb["scale"][bsize:], sign=False)) * 10

        loss = policy_loss + entropy_loss + value_loss.mean() + reg_loss + symmetry_loss_loc + symmetry_loss_std

        # do optimisation step
        opt_actor.zero_grad()
        opt_critic.zero_grad()

        loss.backward()

        actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        critic_grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        encoder_grad_norm = nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)

        opt_actor.step()
        opt_critic.step()

        with torch.no_grad():
            explained_var = 1 - value_loss / (mb["ret"] * valid).var(dim=0)
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            loc, scale = mb["loc"][:bsize], mb["scale"][:bsize]
            kl = torch.sum(
                torch.log(scale) - torch.log(scale_old)
                + (torch.square(scale_old) + torch.square(loc_old - loc)) / (2.0 * torch.square(scale))
                - 0.5,
                axis=-1,
            ).mean()

        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "adapt/reg_loss": reg_loss if isinstance(reg_loss, torch.Tensor) else torch.tensor(0.0),
            "actor/actor_grad_norm": actor_grad_norm,
            "action/encoder_grad_norm": encoder_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "critic/critic_grad_norm": critic_grad_norm,
            "actor/kl": kl.detach(),
            "actor/symmetry_loss_loc": symmetry_loss_loc.detach(),
            "actor/symmetry_loss_std": symmetry_loss_std.detach(),
        }

        info["critic/explained_var"] = explained_var.mean().detach()
        info["critic/value_loss"] = value_loss.mean().detach()
    
        return info

    def train_estimator(self, td):
        infos = []
        
        for _ in range(2):
            for mb in make_batch(td, self.num_minibatches, self.cfg.train_every):
                infos.append(TensorDict(self.update2(mb), []))

        return {k: v.mean().item() for k, v in torch.stack(infos).items()}

    def _update2(self, mb, adapt_module, adapt_joint_module, opt_estimator, opt_joint):
        mb_sym = mb.clone()
        mb_sym[OBS_KEY] = self.obs_transform(mb_sym[OBS_KEY])
        mb_sym[OBS_PRIV_KEY] = self.obs_priv_transform(mb_sym[OBS_PRIV_KEY])
        if OBS_JOINT_KEY in mb_sym.keys():
            mb_sym[OBS_JOINT_KEY] = self.obs_joint_transform(mb_sym[OBS_JOINT_KEY])
        mb_sym[CRITIC_PRIV_KEY] = self.critic_priv_transform(mb_sym[CRITIC_PRIV_KEY])
        mb_sym["is_init"] = mb["is_init"]

        mb_sym = mb_sym.exclude("next")
        mb = mb.exclude("next")
        mb = torch.cat([mb, mb_sym], dim=0)

        with torch.no_grad():
            self.encoder_priv(mb)
        # compute student predictions (with grad)
        adapt_module(mb)
        adapt_joint_module(mb)

        valid = ~mb["is_init"]
        loss_pred = torch.mean(F.mse_loss(mb["priv_pred"], mb["priv_feature"], reduction="none") * (valid))

        # joint prediction loss: only compute if joint key exists in batch
        if OBS_JOINT_KEY in mb.keys():
            joint_target = mb[OBS_JOINT_KEY]
            loss_joint = torch.mean(F.mse_loss(mb["priv_joint"], joint_target, reduction="none") * (valid))
        else:
            loss_joint = torch.tensor(0.0, device=loss_pred.device)

        loss = loss_pred + self.cfg.joint_pred_weight * loss_joint

        opt_estimator.zero_grad()
        opt_joint.zero_grad()
        loss.backward()
        opt_estimator.step()
        opt_joint.step()

        return {"adapt/estimator_loss": loss_pred.detach(), "adapt/joint_loss": loss_joint.detach()}

    @staticmethod
    @_compile_training_function
    @torch.no_grad()
    def _compute_advantage(td, critic, gae, value_norm, REWARD_KEY="reward", TERM_KEY="term", DONE_KEY="done"):
        keys = td.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with td.view(-1) as flat:
                critic(flat)
                critic(flat["next"])

        v = td["state_value"]
        v_next = td["next", "state_value"]

        rewards = td[REWARD_KEY].sum(dim=-1, keepdim=True)#.clamp_min(0.)

        adv, ret = gae(
            rewards,
            td[TERM_KEY],
            td[DONE_KEY],
            value_norm.denormalize(v),
            value_norm.denormalize(v_next),
        )

        value_norm.update(ret)
        td["adv"], td["ret"] = adv, value_norm.normalize(ret)

    @staticmethod
    @_compile_training_function
    def get_global_mean_std(x: torch.Tensor, mask: torch.Tensor):
        if aa.is_distributed():
            local_count = mask.sum()

            local_sum = (x * mask).sum()
            local_sum_sq = (x * x * mask).sum()

            stats = torch.stack([local_sum, local_sum_sq, local_count.float()])

            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            global_sum, global_sum_sq, global_count = stats
            global_count.clamp_min_(1)

            global_mean = global_sum / global_count
            global_var = (global_sum_sq / global_count) - (global_mean * global_mean)
            global_std = torch.sqrt(global_var.clamp(min=0.0)).clamp(min=1e-5)
        else:
            count = mask.sum().clamp_min_(1)
            sum = (x * mask).sum()
            sum_sq = (x * x * mask).sum()

            global_mean = sum / count
            global_var = (sum_sq / count) - (global_mean * global_mean)
            global_std = torch.sqrt(global_var.clamp(min=0.0)).clamp(min=1e-5)
        return global_mean, global_std

    def _modewise_adv_norm(self, td):
        adv = td["adv"]
        is_init = td["is_init"]
        
        mask = ~is_init
        mean_mode, std_mode = self.get_global_mean_std(adv, mask)
        adv[mask] = (adv[mask] - mean_mode) / std_mode

    def state_dict(self):
        state = OrderedDict()
        for n, m in self.named_children():
            if isinstance(m, DDP):
                state[n] = m.module.state_dict()
            else:
                state[n] = m.state_dict()

        state["last_phase"] = self.cfg.phase

        state["_meta"] = {
            "current_lr": getattr(self, "current_lr", self.cfg.lr),
            "entropy_coef": getattr(self, "entropy_coef", self.cfg.entropy_coef_start),
            "reg_lambda": getattr(self, "reg_lambda", 0.0),
            "progress": getattr(self, "progress", 0.0),
            "num_updates": getattr(self, "num_updates", 0),
            "world_size": getattr(self, "world_size", 1),
        }

        return state

    def load_state_dict(self, state_dict, strict=True):
        for n, m in self.named_children():
            try:
                if isinstance(m, DDP):
                    m.module.load_state_dict(state_dict.get(n, {}), strict=strict)
                else:
                    m.load_state_dict(state_dict.get(n, {}), strict=strict)
            except Exception as e:
                warnings.warn(f"Failed to load {n}: {e}")

        last_phase = state_dict.get("last_phase", "train")

        # Initialize student actor from teacher if starting from a 'train' phase checkpoint
        if last_phase == "train":
            warnings.warn("Last phase was 'train'. Performing a hard copy from `actor_teacher` to `actor_student`.")
            self.hard_copy_(self.actor_teacher, self.actor_student)

        meta = state_dict.get("_meta", {})
        if last_phase == self.cfg.phase:
            self.current_lr   = meta.get("current_lr", getattr(self, "current_lr", self.cfg.lr))
            self.entropy_coef = meta.get("entropy_coef", self.entropy_coef)
            self.reg_lambda   = meta.get("reg_lambda", self.reg_lambda)
            self.progress     = meta.get("progress", self.progress)
            self.num_updates  = meta.get("num_updates", self.num_updates)

    @staticmethod
    def soft_copy_(src_module: nn.Module, dst_module: nn.Module, tau: float):
        src = src_module.module if isinstance(src_module, DDP) else src_module
        dst = dst_module.module if isinstance(dst_module, DDP) else dst_module

        """Copy parameters from src -> dst.

        Behavior:
        - If parameter shapes match, perform blended copy: dst = tau*src + (1-tau)*dst
        - If shapes differ, copy overlapping slices and leave the remainder of dst unchanged
        - Also attempt to copy buffers (e.g. running stats) with the same rules

        This makes it possible to transfer most weights even when teacher and student
        receive different concatenated inputs (e.g. teacher includes joint obs).
        """
        with torch.no_grad():
            src_params = dict(src.named_parameters())
            dst_params = dict(dst.named_parameters())

            for name, dst_param in dst_params.items():
                if name not in src_params:
                    continue
                src_param = src_params[name]
                s = src_param.data
                d = dst_param.data

                if s.shape == d.shape:
                    d.copy_(tau * s + (1.0 - tau) * d)
                else:
                    # Partial copy for overlapping dimensions
                    try:
                        # Determine common shape along each axis
                        common = tuple(min(a, b) for a, b in zip(s.shape, d.shape))
                        src_slices = tuple(slice(0, c) for c in common)
                        dst_slices = tuple(slice(0, c) for c in common)
                        d[dst_slices].copy_(tau * s[src_slices] + (1.0 - tau) * d[dst_slices])
                        warnings.warn(f"Partial param copy for '{name}': src {s.shape} -> dst {d.shape}")
                    except Exception as e:
                        warnings.warn(f"Failed partial copy for param '{name}': {e}")

            # Copy buffers as well (e.g., running_mean/var). Handle shape mismatches similarly.
            src_bufs = dict(src.named_buffers())
            dst_bufs = dict(dst.named_buffers())
            for name, dst_buf in dst_bufs.items():
                if name not in src_bufs:
                    continue
                s = src_bufs[name].data
                d = dst_buf.data
                if s.shape == d.shape:
                    d.copy_(s)
                else:
                    try:
                        common = tuple(min(a, b) for a, b in zip(s.shape, d.shape))
                        src_slices = tuple(slice(0, c) for c in common)
                        dst_slices = tuple(slice(0, c) for c in common)
                        d[dst_slices].copy_(s[src_slices])
                        warnings.warn(f"Partial buffer copy for '{name}': src {s.shape} -> dst {d.shape}")
                    except Exception as e:
                        warnings.warn(f"Failed partial copy for buffer '{name}': {e}")
    
    @staticmethod
    def hard_copy_(src_module: nn.Module, dst_module: nn.Module):
        PPOPolicy.soft_copy_(src_module, dst_module, 1.0)
