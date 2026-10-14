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
class RootStudentPPOConfig:
    _target_: str = "active_adaptation.learning.hierarchical.root_student_ppo.RootStudentPPOPolicy"
    name: str = "root_student_ppo"

    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    estimator_epochs: int = 2

    lr: float = 3e-4
    desired_kl: float = 0.01
    clip_param: float = 0.2
    entropy_coef_start: float = 0.005
    entropy_coef_end: float = 0.001
    init_noise_scale: float = 0.8
    load_noise_scale: float | None = None
    layer_norm: str | None = "before"
    latent_dim: int = 128
    reg_lambda: float = 0.2
    vecnorm: str | None = None
    phase: str = "train"  # train | adapt | finetune
    direct_pred_weight: float = 1.0

    in_keys: List[str] = field(default_factory=lambda: ["hl_policy"])
    priv_in_keys: List[str] = field(default_factory=lambda: ["hl_priv"])
    direct_priv_keys: List[str] = field(default_factory=list)
    critic_in_keys: List[str] = field(default_factory=lambda: ["hl_policy", "hl_priv"])
    # Optional structured loss for direct privileged targets whose first
    # dimensions are continuous and remaining dimensions are one-hot classes.
    # Kept disabled by default so legacy direct estimators are unchanged.
    direct_priv_classification: bool = False
    direct_priv_continuous_dim: int | None = None
    direct_priv_classification_weight: float = 1.0


cs = ConfigStore.instance()
cs.store("root_student_ppo", node=RootStudentPPOConfig(), group="algo")


class DirectPrivPredictionDecoder(TensorDictModuleBase):
    """Turn direct-estimator output into the representation consumed by actor."""

    def __init__(self, raw_key: str, output_key: str, continuous_dim: int):
        super().__init__()
        self.in_keys = [raw_key]
        self.out_keys = [output_key]
        self.continuous_dim = int(continuous_dim)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        raw = tensordict[self.in_keys[0]]
        continuous = raw[..., : self.continuous_dim]
        logits = raw[..., self.continuous_dim :]
        # The teacher target masks body identity during no-force phases. Apply
        # the same convention to the student representation.
        body_prob = logits.softmax(dim=-1)
        active = (continuous.norm(dim=-1, keepdim=True) > 1e-4).to(raw.dtype)
        tensordict[self.out_keys[0]] = torch.cat(
            [continuous, body_prob * active], dim=-1
        )
        return tensordict


class RootStudentPPOPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: RootStudentPPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device: str = "cuda:0",
        env=None,
    ) -> None:
        super().__init__()
        if cfg.phase not in {"train", "adapt", "finetune"}:
            raise ValueError(f"Unsupported root_student_ppo phase: {cfg.phase}")

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
        self.reg_lambda = 0.0

        actor_in_keys = list(cfg.in_keys)
        priv_in_keys = list(cfg.priv_in_keys)
        direct_priv_keys = list(cfg.get("direct_priv_keys", []))
        critic_in_keys = list(cfg.critic_in_keys)
        self.direct_priv_keys = direct_priv_keys
        self.direct_pred_weight = float(cfg.get("direct_pred_weight", 1.0))
        self.direct_pred_key = "direct_priv_pred"
        self.direct_priv_classification = bool(cfg.get("direct_priv_classification", False))
        self.direct_priv_continuous_dim = cfg.get("direct_priv_continuous_dim", None)
        self.direct_priv_classification_weight = float(
            cfg.get("direct_priv_classification_weight", 1.0)
        )
        self.direct_priv_raw_key = "direct_priv_raw"
        self.direct_priv_decoder = None

        self.encoder_priv = Seq(
            CatTensors(priv_in_keys, "_priv_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(make_mlp([256], norm=cfg.layer_norm), nn.LazyLinear(cfg.latent_dim)),
                ["_priv_inp"],
                ["priv_feature"],
            ),
        ).to(self.device)

        self.adapt_module = Seq(
            CatTensors(actor_in_keys, "_adapt_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(make_mlp([512, 256], norm=cfg.layer_norm), nn.LazyLinear(cfg.latent_dim)),
                ["_adapt_inp"],
                ["priv_pred"],
            ),
        ).to(self.device)

        self.adapt_direct_module = None
        if self.direct_priv_keys:
            fake_td = observation_spec.zero().to(self.device)
            direct_dim = self._cat_direct_priv(fake_td).shape[-1]
            if self.direct_priv_classification:
                if self.direct_priv_continuous_dim is None:
                    raise ValueError(
                        "direct_priv_continuous_dim is required when "
                        "direct_priv_classification is enabled."
                    )
                continuous_dim = int(self.direct_priv_continuous_dim)
                if not 0 < continuous_dim < direct_dim:
                    raise ValueError(
                        f"direct_priv_continuous_dim must be in (0, {direct_dim}), "
                        f"got {continuous_dim}."
                    )
                direct_out_key = self.direct_priv_raw_key
                self.direct_priv_decoder = DirectPrivPredictionDecoder(
                    self.direct_priv_raw_key,
                    self.direct_pred_key,
                    continuous_dim,
                ).to(self.device)
            else:
                direct_out_key = self.direct_pred_key
            self.adapt_direct_module = Seq(
                CatTensors(actor_in_keys, "_direct_inp", del_keys=False, sort=False),
                Mod(
                    nn.Sequential(make_mlp([512, 256], norm=cfg.layer_norm), nn.LazyLinear(direct_dim)),
                    ["_direct_inp"],
                    [direct_out_key],
                ),
            ).to(self.device)

        teacher_extra_keys = ["priv_feature"] + self.direct_priv_keys
        student_extra_keys = ["priv_pred"]
        if self.direct_priv_keys:
            student_extra_keys.append(self.direct_pred_key)
        self.actor_teacher = self._build_actor(actor_in_keys + teacher_extra_keys)
        self.actor_student = self._build_actor(actor_in_keys + student_extra_keys)

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
        self.encoder_priv(fake_td)
        self.adapt_module(fake_td)
        if self.adapt_direct_module is not None:
            self.adapt_direct_module(fake_td)
            if self.direct_priv_decoder is not None:
                self.direct_priv_decoder(fake_td)
        self.actor_teacher(fake_td)
        self.actor_student(fake_td)
        self.critic(fake_td)

        self.world_size = 1
        if aa.is_distributed() and not bool(cfg.get("disable_ddp", False)):
            self.world_size = aa.get_world_size()
            ddp_kwargs = dict(
                device_ids=[aa.get_local_rank()],
                output_device=aa.get_local_rank(),
                broadcast_buffers=True,
                find_unused_parameters=False,
            )
            self.encoder_priv = DDP(self.encoder_priv, **ddp_kwargs)
            self.adapt_module = DDP(self.adapt_module, **ddp_kwargs)
            if self.adapt_direct_module is not None:
                self.adapt_direct_module = DDP(self.adapt_direct_module, **ddp_kwargs)
            self.actor_teacher = DDP(self.actor_teacher, **ddp_kwargs)
            self.actor_student = DDP(self.actor_student, **ddp_kwargs)
            self.critic = DDP(self.critic, **ddp_kwargs)

        self.opt_teacher = torch.optim.Adam(
            list(self.encoder_priv.parameters()) + list(self.actor_teacher.parameters()),
            lr=cfg.lr,
        )
        student_params = list(self.adapt_module.parameters()) + list(self.actor_student.parameters())
        if self.adapt_direct_module is not None:
            student_params += list(self.adapt_direct_module.parameters())
        self.opt_student = torch.optim.Adam(student_params, lr=cfg.lr)
        self.opt_estimator = torch.optim.Adam(self.adapt_module.parameters(), lr=cfg.lr)
        self.opt_direct_estimator = None
        if self.adapt_direct_module is not None:
            self.opt_direct_estimator = torch.optim.Adam(self.adapt_direct_module.parameters(), lr=cfg.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    def _cat_direct_priv(self, tensordict: TensorDict) -> torch.Tensor:
        if not self.direct_priv_keys:
            raise RuntimeError("_cat_direct_priv called without direct_priv_keys.")
        return torch.cat([tensordict[key] for key in self.direct_priv_keys], dim=-1)

    def _student_estimators(self):
        modules = [self.adapt_module]
        if self.adapt_direct_module is not None:
            modules.append(self.adapt_direct_module)
        if self.direct_priv_decoder is not None:
            modules.append(self.direct_priv_decoder)
        return modules

    @staticmethod
    def _run_encoder(encoder, tensordict: TensorDict):
        if isinstance(encoder, (list, tuple)):
            for module in encoder:
                module(tensordict)
            return tensordict
        return encoder(tensordict)

    @staticmethod
    def _encoder_parameters(encoder):
        if isinstance(encoder, (list, tuple)):
            params = []
            for module in encoder:
                params.extend(list(module.parameters()))
            return params
        return encoder.parameters()

    def _build_actor(self, in_keys: list[str]):
        return ProbabilisticActor(
            module=Seq(
                CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),
                Mod(make_mlp([512, 256], norm=self.cfg.layer_norm), ["_actor_inp"], ["_actor_feature"]),
                Mod(
                    Actor(
                        self.action_dim,
                        init_noise_scale=self.cfg.init_noise_scale,
                        load_noise_scale=self.cfg.load_noise_scale,
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

    def make_tensordict_primer(self):
        return None

    def get_rollout_policy(self, mode: str = "train"):
        if mode in {"eval", "deploy"}:
            return Seq(*self._student_estimators(), self.actor_student)
        if self.cfg.phase == "train":
            return Seq(self.encoder_priv, self.actor_teacher)
        return Seq(*self._student_estimators(), self.actor_student)

    def broadcast_parameters(self, extra_modules=[]):
        return None

    def step_schedule(self, progress: float, iter: int):
        start = self.cfg.entropy_coef_start
        end = self.cfg.entropy_coef_end
        self.entropy_coef = start * (end / start) ** progress
        self.progress = progress
        self.reg_lambda = progress * self.cfg.reg_lambda

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
        opts = [self.opt_teacher, self.opt_student, self.opt_estimator, self.opt_critic]
        if self.opt_direct_estimator is not None:
            opts.append(self.opt_direct_estimator)
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = self.current_lr

    def train_op(self, td: TensorDict, vecnorm):
        if self.cfg.phase == "train":
            info = self._ppo_update(td, actor=self.actor_teacher, encoder=self.encoder_priv, opt_actor=self.opt_teacher)
            info.update(self._train_estimator(td))
            self._copy_teacher_to_student()
        elif self.cfg.phase == "finetune":
            info = self._ppo_update(td, actor=self.actor_student, encoder=self._student_estimators(), opt_actor=self.opt_student)
        else:
            info = self._train_estimator(td)
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

    def _ppo_update(self, td: TensorDict, actor, encoder, opt_actor):
        infos = []
        self._compute_advantage(td)

        for _ in range(self.cfg.ppo_epochs):
            for mb in make_batch(td, self.num_minibatches):
                infos.append(TensorDict(self._update(mb, actor, encoder, opt_actor), []))

        info = {k: v.mean().item() for k, v in torch.stack(infos).items()}
        self._do_lr_schedule(info["actor/kl"])
        info["lr"] = self.current_lr
        return info

    def _update(self, mb: TensorDict, actor, encoder, opt_actor):
        loc_old = mb["loc"].clone()
        scale_old = mb["scale"].clone()
        action_old = mb["action"].clone()
        logp_old = mb["sample_log_prob"].clone()
        valid = ~mb["is_init"]

        mb = mb.exclude("next", "sample_log_prob", "action")
        self._run_encoder(encoder, mb)
        actor(mb)
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

        opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        encoder_grad_norm = nn.utils.clip_grad_norm_(self._encoder_parameters(encoder), 1.0)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        opt_actor.step()
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
            "actor/encoder_grad_norm": encoder_grad_norm.detach(),
            "actor/clamp_ratio": clipfrac.detach(),
            "actor/kl": kl.detach(),
            "critic/value_loss": value_loss.detach(),
            "critic/critic_grad_norm": critic_grad_norm.detach(),
        }

    def _train_estimator(self, td: TensorDict):
        infos = []
        for _ in range(self.cfg.estimator_epochs):
            for mb in make_batch(td, self.num_minibatches):
                infos.append(TensorDict(self._update_estimator(mb), []))
        return {k: v.mean().item() for k, v in torch.stack(infos).items()}

    def _update_estimator(self, mb: TensorDict):
        mb = mb.exclude("next")
        valid = ~mb["is_init"]

        with torch.no_grad():
            self.encoder_priv(mb)
        self.adapt_module(mb)

        estimator_loss = F.mse_loss(mb["priv_pred"], mb["priv_feature"], reduction="none")
        estimator_loss = torch.mean(estimator_loss * valid)
        direct_loss = torch.zeros((), device=estimator_loss.device)
        direct_force_loss = torch.zeros((), device=estimator_loss.device)
        direct_body_loss = torch.zeros((), device=estimator_loss.device)
        if self.adapt_direct_module is not None:
            self.adapt_direct_module(mb)
            direct_target = self._cat_direct_priv(mb)
            if self.direct_priv_decoder is None:
                direct_loss = F.mse_loss(
                    mb[self.direct_pred_key], direct_target, reduction="none"
                )
                direct_loss = torch.mean(direct_loss * valid)
            else:
                raw_pred = mb[self.direct_priv_raw_key]
                continuous_dim = int(self.direct_priv_continuous_dim)
                force_pred = raw_pred[..., :continuous_dim]
                body_logits = raw_pred[..., continuous_dim:]
                force_target = direct_target[..., :continuous_dim]
                body_target = direct_target[..., continuous_dim:]

                direct_force_loss = F.mse_loss(
                    force_pred, force_target, reduction="none"
                ).mean(dim=-1)
                direct_force_loss = torch.mean(
                    direct_force_loss * valid.squeeze(-1)
                )

                # net_pull_force_b_priv masks the body one-hot during rest. Do
                # not force the estimator to classify an arbitrary body then.
                body_active = body_target.sum(dim=-1) > 0.5
                body_valid = valid.squeeze(-1) & body_active
                if body_valid.any():
                    body_index = body_target.argmax(dim=-1)
                    body_loss = F.cross_entropy(
                        body_logits[body_valid], body_index[body_valid], reduction="none"
                    )
                    direct_body_loss = body_loss.mean()
                direct_loss = direct_force_loss + self.direct_priv_classification_weight * direct_body_loss
                self.direct_priv_decoder(mb)

        loss = estimator_loss + self.direct_pred_weight * direct_loss
        self.opt_estimator.zero_grad()
        if self.opt_direct_estimator is not None:
            self.opt_direct_estimator.zero_grad()
        loss.backward()
        estimator_grad_norm = nn.utils.clip_grad_norm_(self.adapt_module.parameters(), 1.0)
        direct_grad_norm = torch.zeros((), device=estimator_loss.device)
        if self.adapt_direct_module is not None:
            direct_grad_norm = nn.utils.clip_grad_norm_(self.adapt_direct_module.parameters(), 1.0)
        self.opt_estimator.step()
        if self.opt_direct_estimator is not None:
            self.opt_direct_estimator.step()

        return {
            "adapt/estimator_loss": estimator_loss.detach(),
            "adapt/direct_loss": direct_loss.detach(),
            "adapt/direct_force_loss": direct_force_loss.detach(),
            "adapt/direct_body_loss": direct_body_loss.detach(),
            "adapt/estimator_grad_norm": estimator_grad_norm.detach(),
            "adapt/direct_grad_norm": direct_grad_norm.detach(),
        }

    def state_dict(self):
        actor_teacher = self.actor_teacher.module if isinstance(self.actor_teacher, DDP) else self.actor_teacher
        actor_student = self.actor_student.module if isinstance(self.actor_student, DDP) else self.actor_student
        encoder_priv = self.encoder_priv.module if isinstance(self.encoder_priv, DDP) else self.encoder_priv
        adapt_module = self.adapt_module.module if isinstance(self.adapt_module, DDP) else self.adapt_module
        adapt_direct_module = None
        if self.adapt_direct_module is not None:
            adapt_direct_module = self.adapt_direct_module.module if isinstance(self.adapt_direct_module, DDP) else self.adapt_direct_module
        critic = self.critic.module if isinstance(self.critic, DDP) else self.critic
        state = OrderedDict(
            actor_teacher=actor_teacher.state_dict(),
            actor_student=actor_student.state_dict(),
            encoder_priv=encoder_priv.state_dict(),
            adapt_module=adapt_module.state_dict(),
            critic=critic.state_dict(),
            last_phase=self.cfg.phase,
            _meta={
                "current_lr": self.current_lr,
                "entropy_coef": self.entropy_coef,
                "reg_lambda": self.reg_lambda,
                "progress": self.progress,
                "num_updates": self.num_updates,
            },
        )
        if adapt_direct_module is not None:
            state["adapt_direct_module"] = adapt_direct_module.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True):
        actor_teacher = self.actor_teacher.module if isinstance(self.actor_teacher, DDP) else self.actor_teacher
        actor_student = self.actor_student.module if isinstance(self.actor_student, DDP) else self.actor_student
        encoder_priv = self.encoder_priv.module if isinstance(self.encoder_priv, DDP) else self.encoder_priv
        adapt_module = self.adapt_module.module if isinstance(self.adapt_module, DDP) else self.adapt_module
        adapt_direct_module = None
        if self.adapt_direct_module is not None:
            adapt_direct_module = self.adapt_direct_module.module if isinstance(self.adapt_direct_module, DDP) else self.adapt_direct_module
        critic = self.critic.module if isinstance(self.critic, DDP) else self.critic

        actor_teacher.load_state_dict(state_dict.get("actor_teacher", {}), strict=strict)
        actor_student.load_state_dict(state_dict.get("actor_student", {}), strict=strict)
        encoder_priv.load_state_dict(state_dict.get("encoder_priv", {}), strict=strict)
        adapt_module.load_state_dict(state_dict.get("adapt_module", {}), strict=strict)
        if adapt_direct_module is not None and "adapt_direct_module" in state_dict:
            adapt_direct_module.load_state_dict(state_dict["adapt_direct_module"], strict=strict)
        critic.load_state_dict(state_dict.get("critic", {}), strict=strict)

        last_phase = state_dict.get("last_phase", "train")
        if last_phase == "train":
            self._copy_teacher_to_student()

        meta = state_dict.get("_meta", {})
        if last_phase == self.cfg.phase:
            self.current_lr = meta.get("current_lr", self.current_lr)
            self.entropy_coef = meta.get("entropy_coef", self.entropy_coef)
            self.reg_lambda = meta.get("reg_lambda", self.reg_lambda)
            self.progress = meta.get("progress", self.progress)
            self.num_updates = meta.get("num_updates", self.num_updates)

    @torch.no_grad()
    def _copy_teacher_to_student(self):
        actor_teacher = self.actor_teacher.module if isinstance(self.actor_teacher, DDP) else self.actor_teacher
        actor_student = self.actor_student.module if isinstance(self.actor_student, DDP) else self.actor_student
        actor_student.load_state_dict(actor_teacher.state_dict())
