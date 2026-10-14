from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictModuleBase, TensorDictSequential as Seq
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor

import active_adaptation as aa
from active_adaptation.learning.modules.distributions import IndependentNormal
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    DONE_KEY,
    REWARD_KEY,
    TERM_KEY,
    Actor,
    CatTensors,
    GAE,
    make_batch,
    make_mlp,
)


@dataclass
class RootPPOConfig:
    _target_: str = "active_adaptation.learning.hierarchical.root_ppo.RootPPOPolicy"
    name: str = "root_ppo"

    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8

    lr: float = 3e-4
    desired_kl: float = 0.01
    clip_param: float = 0.2
    entropy_coef_start: float = 0.005
    entropy_coef_end: float = 0.001
    init_noise_scale: float = 0.8
    load_noise_scale: float | None = None
    layer_norm: str | None = "before"
    latent_dim: int = 256
    vecnorm: str | None = None

    in_keys: List[str] = field(default_factory=lambda: ["policy"])
    critic_in_keys: List[str] = field(default_factory=lambda: ["policy"])


cs = ConfigStore.instance()
cs.store("root_ppo", node=RootPPOConfig(), group="algo")


class RootPPOPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: RootPPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device: str = "cuda:0",
        env=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.action_dim = action_spec.shape[-1]
        self.entropy_coef = cfg.entropy_coef_start
        self.clip_param = cfg.clip_param
        self.gae = GAE(0.99, 0.95)
        self.num_minibatches = cfg.num_minibatches
        self.progress = 0.0
        self.current_lr = cfg.lr
        self.num_updates = 0
        actor_in_keys = list(cfg.in_keys)
        critic_in_keys = list(cfg.critic_in_keys)

        self.actor = ProbabilisticActor(
            module=Seq(
                CatTensors(actor_in_keys, "_actor_inp", del_keys=False, sort=False),
                Mod(make_mlp([512, 256], norm=cfg.layer_norm), ["_actor_inp"], ["_actor_feature"]),
                Mod(
                    Actor(
                        self.action_dim,
                        init_noise_scale=cfg.init_noise_scale,
                        load_noise_scale=cfg.load_noise_scale,
                    ),
                    ["_actor_feature"],
                    ["loc", "scale"],
                ),
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        self.critic = Seq(
            CatTensors(critic_in_keys, "_critic_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(make_mlp([512, 256], norm=cfg.layer_norm), nn.LazyLinear(1)),
                ["_critic_inp"],
                ["state_value"],
            ),
        ).to(self.device)

        fake_td = observation_spec.zero().to(self.device)
        fake_td["is_init"] = torch.ones(fake_td.shape[0], 1, dtype=torch.bool, device=self.device)
        self.actor(fake_td)
        self.critic(fake_td)

        self.world_size = 1
        if aa.is_distributed():
            self.world_size = aa.get_world_size()
            ddp_kwargs = dict(
                device_ids=[aa.get_local_rank()],
                output_device=aa.get_local_rank(),
                broadcast_buffers=True,
                find_unused_parameters=False,
            )
            self.actor = DDP(self.actor, **ddp_kwargs)
            self.critic = DDP(self.critic, **ddp_kwargs)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    def make_tensordict_primer(self):
        return None

    def get_rollout_policy(self, mode: str = "train"):
        return Seq(self.actor)

    def broadcast_parameters(self, extra_modules=[]):
        return None

    def step_schedule(self, progress: float, iter: int):
        start = self.cfg.entropy_coef_start
        end = self.cfg.entropy_coef_end
        self.entropy_coef = start * (end / start) ** progress
        self.progress = progress

    def _do_lr_schedule(self, kl: float):
        if self.progress < 0.1:
            return

        # Average KL across DDP ranks so every rank applies the same LR.
        if aa.is_distributed():
            kl_tensor = torch.tensor(kl, device=self.device)
            dist.all_reduce(kl_tensor, op=dist.ReduceOp.SUM)
            kl = (kl_tensor / aa.get_world_size()).item()

        new_lr = self.current_lr
        if kl > self.cfg.desired_kl * 2.0:
            new_lr = max(1e-5, new_lr / 1.1)
        elif 0.0 < kl < self.cfg.desired_kl / 2.0:
            new_lr = min(5e-3, new_lr * 1.1)
        self.current_lr = new_lr
        for opt in (self.opt_actor, self.opt_critic):
            for group in opt.param_groups:
                group["lr"] = self.current_lr

    def train_op(self, td: TensorDict, vecnorm):
        info = self._ppo_update(td)
        self.num_updates += 1
        return info

    @torch.no_grad()
    def _compute_advantage(self, td: TensorDict):
        if "state_value" not in td.keys(True, True):
            self.critic(td.view(-1))
        if ("next", "state_value") not in td.keys(True, True):
            self.critic(td["next"].view(-1))

        rewards = td[REWARD_KEY].sum(dim=-1, keepdim=True)
        adv, ret = self.gae(
            rewards,
            td[TERM_KEY],
            td[DONE_KEY],
            td["state_value"],
            td["next", "state_value"],
        )
        td["adv"] = adv
        td["ret"] = ret

        valid = ~td["is_init"]
        mean = td["adv"][valid].mean()
        std = td["adv"][valid].std().clamp_min(1e-5)
        td["adv"][valid] = (td["adv"][valid] - mean) / std

    def _ppo_update(self, td: TensorDict):
        infos = []
        self._compute_advantage(td)

        for _ in range(self.cfg.ppo_epochs):
            for mb in make_batch(td, self.num_minibatches):
                infos.append(TensorDict(self._update(mb), []))

        info = {k: v.mean().item() for k, v in torch.stack(infos).items()}
        self._do_lr_schedule(info["actor/kl"])
        info["lr"] = self.current_lr
        return info

    def _update(self, mb: TensorDict):
        loc_old = mb["loc"].clone()
        scale_old = mb["scale"].clone()
        action_old = mb["action"].clone()
        logp_old = mb["sample_log_prob"].clone()
        valid = ~mb["is_init"]

        mb = mb.exclude("next", "sample_log_prob", "action")
        self.actor(mb)
        values = self.critic(mb)["state_value"]

        dist = IndependentNormal(mb["loc"], mb["scale"])
        logp = dist.log_prob(action_old)
        entropy = dist.entropy().mean()
        ratio = torch.exp(logp - logp_old).unsqueeze(-1)
        surr1 = mb["adv"] * ratio
        surr2 = mb["adv"] * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
        policy_loss = -torch.mean(torch.min(surr1, surr2) * valid)
        entropy_loss = -self.entropy_coef * entropy
        value_loss = F.mse_loss(values, mb["ret"], reduction="none")
        value_loss = (value_loss * valid).mean()
        loss = policy_loss + entropy_loss + value_loss

        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_actor.step()
        self.opt_critic.step()

        with torch.no_grad():
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            kl = torch.sum(
                torch.log(mb["scale"]) - torch.log(scale_old)
                + (scale_old.square() + (loc_old - mb["loc"]).square()) / (2.0 * mb["scale"].square())
                - 0.5,
                dim=-1,
            ).mean()

        return {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/actor_grad_norm": actor_grad_norm.detach(),
            "actor/clamp_ratio": clipfrac.detach(),
            "actor/kl": kl.detach(),
            "critic/value_loss": value_loss.detach(),
            "critic/critic_grad_norm": critic_grad_norm.detach(),
        }

    def state_dict(self):
        actor = self.actor.module if isinstance(self.actor, DDP) else self.actor
        critic = self.critic.module if isinstance(self.critic, DDP) else self.critic
        return OrderedDict(
            actor=actor.state_dict(),
            critic=critic.state_dict(),
            _meta={
                "current_lr": self.current_lr,
                "entropy_coef": self.entropy_coef,
                "progress": self.progress,
                "num_updates": self.num_updates,
            },
        )

    def load_state_dict(self, state_dict, strict=True):
        actor = self.actor.module if isinstance(self.actor, DDP) else self.actor
        critic = self.critic.module if isinstance(self.critic, DDP) else self.critic
        actor.load_state_dict(state_dict.get("actor", {}), strict=strict)
        critic.load_state_dict(state_dict.get("critic", {}), strict=strict)
        meta = state_dict.get("_meta", {})
        self.current_lr = meta.get("current_lr", self.current_lr)
        self.entropy_coef = meta.get("entropy_coef", self.entropy_coef)
        self.progress = meta.get("progress", self.progress)
        self.num_updates = meta.get("num_updates", self.num_updates)
