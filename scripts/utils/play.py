import torch
import hydra
import numpy as np
import einops
import itertools
import os
import datetime
from omegaconf import OmegaConf

from isaaclab.app import AppLauncher

from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictSequential

from active_adaptation.learning import ALGOS
from active_adaptation.utils.export import export_onnx
from scripts.utils.helpers import EpisodeStats, make_env_policy, ObsNorm


def _print_ee_contact_forces(env, env_idx: int = 0, min_force: float = 1e-3):
    base_env = env.base_env if hasattr(env, "base_env") else env
    try:
        contact_sensor = base_env.scene["ee_contact_forces"]
    except Exception:
        return

    try:
        if hasattr(contact_sensor.data, "net_forces_w_history"):
            forces = contact_sensor.data.net_forces_w_history.mean(1)
        else:
            forces = contact_sensor.data.net_forces_w
        body_ids, body_names = contact_sensor.find_bodies(".*")
    except Exception as exc:
        print(f"[EE contact] failed to read contact force: {exc}", flush=True)
        return

    if len(body_names) == 0:
        return

    env_idx = min(env_idx, forces.shape[0] - 1)
    chunks = []
    for body_id, body_name in zip(body_ids, body_names):
        force = forces[env_idx, body_id]
        mag = force.norm().item()
        if mag > min_force:
            chunks.append(
                f"{body_name}: {mag:.2f} N "
                f"({force[0].item():+.1f}, {force[1].item():+.1f}, {force[2].item():+.1f})"
            )

    print("[EE contact] " + (" | ".join(chunks) if chunks else "no contact"), flush=True)


def _print_hl_direct_priv_pred(tensordict, env, env_idx: int = 0):
    if "direct_priv_pred" not in tensordict.keys():
        return

    pred = tensordict["direct_priv_pred"].detach()
    if pred.ndim < 2 or pred.shape[-1] < 6:
        return

    base_env = env.base_env if hasattr(env, "base_env") else env
    command_manager = getattr(base_env, "command_manager", None)
    force_scale = 1.0
    if command_manager is not None and hasattr(command_manager, "net_pull_force_range"):
        try:
            force_scale = float(command_manager.net_pull_force_range[1])
        except Exception:
            force_scale = 1.0

    env_idx = min(env_idx, pred.shape[0] - 1)
    if pred.shape[-1] == 6:
        force_slots_b = pred[env_idx].reshape(2, 3) * force_scale
        left_force_b, right_force_b = force_slots_b
        print(
            "[HL direct_priv_pred] "
            f"left_force_b_est: {left_force_b.norm().item():.2f} N "
            f"({left_force_b[0].item():+.1f}, {left_force_b[1].item():+.1f}, {left_force_b[2].item():+.1f})"
            " | "
            f"right_force_b_est: {right_force_b.norm().item():.2f} N "
            f"({right_force_b[0].item():+.1f}, {right_force_b[1].item():+.1f}, {right_force_b[2].item():+.1f})",
            flush=True,
        )
        return

    force_b = pred[env_idx, 3:6] * force_scale
    msg = (
        f"[HL direct_priv_pred] force_b_est: {force_b.norm().item():.2f} N "
        f"({force_b[0].item():+.1f}, {force_b[1].item():+.1f}, {force_b[2].item():+.1f})"
    )
    if pred.shape[-1] >= 9:
        force_w = pred[env_idx, 6:9] * force_scale
        msg += (
            f" | force_w_est: {force_w.norm().item():.2f} N "
            f"({force_w[0].item():+.1f}, {force_w[1].item():+.1f}, {force_w[2].item():+.1f})"
        )
    print(msg, flush=True)


def _print_hl_ee_command_offset(env, env_idx: int = 0):
    base_env = env.base_env if hasattr(env, "base_env") else env
    action_manager = getattr(base_env, "action_manager", None)
    command_manager = getattr(base_env, "command_manager", None)
    if action_manager is None or command_manager is None:
        return
    if not getattr(action_manager, "ee_command_enabled", False):
        return
    ee_command = getattr(action_manager, "ee_command", None)
    if ee_command is None or ee_command.shape[-1] < 6:
        return
    if not hasattr(command_manager, "get_root_and_wrist_6d_reference"):
        return

    try:
        reference = command_manager.get_root_and_wrist_6d_reference()
    except Exception as exc:
        print(f"[HL ee offset] failed to read reference: {exc}", flush=True)
        return
    if reference.shape[-1] < 6:
        return

    env_idx = min(env_idx, ee_command.shape[0] - 1, reference.shape[0] - 1)
    offset = (ee_command[env_idx, :6] - reference[env_idx, :6]).detach().reshape(2, 3)
    left = offset[0]
    right = offset[1]
    print(
        "[HL ee offset] "
        f"left: {left.norm().item():.3f} m ({left[0].item():+.3f}, {left[1].item():+.3f}, {left[2].item():+.3f}) | "
        f"right: {right.norm().item():.3f} m ({right[0].item():+.3f}, {right[1].item():+.3f}, {right[2].item():+.3f})",
        flush=True,
    )


def play(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app

    env, policy, vecnorm, _ = make_env_policy(cfg)

    if cfg.export_policy:
        import time
        import copy
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        fake_input = env.observation_spec[0].rand().cpu()
        fake_input["is_init"] = torch.tensor(1, dtype=bool)
        fake_input["context_adapt_hx"] = torch.zeros(128)
        fake_input = fake_input.unsqueeze(0)

        def test(m, x):
            start = time.perf_counter()
            for _ in range(1000):
                m(x)
            return (time.perf_counter() - start) / 1000
        
        FILE_PATH = os.path.dirname(__file__)
        
        deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy"))
        obs_norm = ObsNorm.from_vecnorm(vecnorm, deploy_policy.in_keys)
        _policy = TensorDictSequential(obs_norm, deploy_policy).cpu()
        
        print(f"Inference time of policy: {test(_policy, fake_input)}")

        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        os.makedirs(os.path.join(FILE_PATH, "..", "exports", f"{cfg.task.name}-{time_str}"), exist_ok=True)
        path = os.path.join(FILE_PATH, "..", "exports", f"{cfg.task.name}-{time_str}", "policy.pt")
        torch.save(_policy, path)

        meta = {}
        meta["action_scaling"] = dict(cfg.task.action.get("action_scaling"))
        # meta["stiffness"] = dict(cfg.task.robot.stiffness)
        # meta["damping"] = dict(cfg.task.robot.damping)
        # meta["effort_limit"] = dict(cfg.task.robot.effort_limit)
        export_onnx(_policy, fake_input, path.replace(".pt", ".onnx"), meta)

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    policy = policy.get_rollout_policy("eval")
    print_ee_contact_forces = bool(cfg.get("print_ee_contact_forces", False))
    ee_contact_print_interval = max(int(cfg.get("ee_contact_print_interval", 10)), 1)

    td_ = env.reset()
    
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in itertools.count():
            td_ = policy(td_)
            policy_td = td_
            td, td_ = env.step_and_maybe_reset(td_)
            episode_stats.add(td)
            if print_ee_contact_forces and i % ee_contact_print_interval == 0:
                _print_ee_contact_forces(env)
                _print_hl_direct_priv_pred(policy_td, env)
                _print_hl_ee_command_offset(env)

            if len(episode_stats) >= env.num_envs:
                print("Step", i)
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    print(k, torch.mean(v).item())
    
    env.close()
    simulation_app.close()
