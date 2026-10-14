import torch
# import warp
import hydra
import numpy as np

import einops
import wandb
import logging
import os
import sys
import time
import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle
import torch.distributed as dist

import active_adaptation as aa
from isaaclab.app import AppLauncher
from active_adaptation.utils.torchrl import SyncDataCollector, TDTimeBuffer

# local import
from scripts.utils.helpers import make_env_policy, EpisodeStats, evaluate
from torchrl.envs.utils import set_exploration_type, ExplorationType

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')

import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")

@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    print(f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}")
    app_launcher = AppLauncher(
        OmegaConf.to_container(cfg.app),
        distributed=aa.is_distributed(),
        device=f"cuda:{aa.get_local_rank()}"
    )
    simulation_app = app_launcher.app
    if aa.is_distributed():
        dist.init_process_group(
            backend="nccl",
            world_size=aa.get_world_size(),
            rank=aa.get_local_rank(),
        )
        cfg.seed = cfg.seed + aa.get_local_rank() * 10000

    # hydra.job.chdir=true moves the cwd into outputs/, so resolve relative
    # local checkpoint paths against the original launch directory.
    if cfg.checkpoint_path is not None:
        ckpt = str(cfg.checkpoint_path)
        if not ckpt.startswith("run:") and not os.path.isabs(ckpt):
            cfg.checkpoint_path = hydra.utils.to_absolute_path(ckpt)

    env, policy, vecnorm, primer = make_env_policy(cfg)

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
    total_frames = total_frames // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    save_interval = cfg.get("save_interval", -1)
    start_iter = cfg.get("start_iter", 0)
    start_frame = start_iter * frames_per_batch

    need_logging = aa.is_main_process()
    if need_logging:
        run = wandb.init(
            job_type=cfg.wandb.job_type,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
            id=cfg.wandb.id,
            notes=cfg.wandb.notes,
        )
        run.config.update(OmegaConf.to_container(cfg))

        default_run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        run_idx = (run.name or run.id or "debug").split("-")[-1]
        run.name = f"{run_idx}-{default_run_name}"
        setproctitle(run.name)

        os.makedirs(run.dir, exist_ok=True)
        cfg_save_path = os.path.join(run.dir, "cfg.yaml")
        OmegaConf.save(cfg, cfg_save_path)
        run.save(cfg_save_path, policy="now")
        run.save(os.path.join(run.dir, "config.yaml"), policy="now")

        import inspect
        import shutil
        source_path = inspect.getfile(policy.__class__)
        target_path = os.path.join(run.dir, source_path.split("/")[-1])
        shutil.copy(source_path, target_path)
        wandb.save(target_path, policy="now")

        log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
        logging.info(f"Log interval: {log_interval} steps")

        stats_keys = [
            k for k in env.reward_spec.keys(True, True) 
            if isinstance(k, tuple) and k[0] == "stats"
        ]
        episode_stats = EpisodeStats(stats_keys, device=env.device)

        def save(policy, checkpoint_name: str):
            ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
            state_dict = OrderedDict()
            state_dict["wandb"] = {"name": run.name, "id": run.id}
            state_dict["policy"] = policy.state_dict()
            state_dict["env"] = env.state_dict()
            state_dict["cfg"] = cfg
            if "vecnorm" in locals():
                state_dict["vecnorm"] = vecnorm.state_dict()
            torch.save(state_dict, ckpt_path)
            run.save(ckpt_path, policy="now", base_path=run.dir)
            logging.info(f"Saved checkpoint to {str(ckpt_path)}")

        def should_save(i):
            if not aa.is_main_process():
                return False
            return i > 0 and i % save_interval == 0
    
    rollout_policy = policy.get_rollout_policy("train")
    env_frames = 0
    carry = env.reset()

    assert env.training
    if aa.is_main_process():
        progress = tqdm(range(total_iters))
    else:
        progress = range(total_iters)

    N = env.num_envs
    T = cfg.algo.train_every
    data_buf = TDTimeBuffer(N, T, device=policy.device)

    for i in progress:
        start = time.perf_counter()

        with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
            torch.compiler.cudagraph_mark_step_begin() # for compiled policy
            for t in range(cfg.algo.train_every):
                carry = rollout_policy(carry)

                td, carry = env.step_and_maybe_reset(carry)

                # deal with value
                policy.critic(td)
                policy.critic(td["next"])
                td.get(("next", "state_value"))[:] = torch.where(
                    td["next", "done"], 
                    td["state_value"],
                    td["next", "state_value"]
                )

                # clean up tensordict
                td["next"] = td["next"].exclude(*rollout_policy.in_keys)
                private_keys = [key for key in td.keys(True, True) if isinstance(key, str) and key.startswith('_')]
                td = td.exclude(*private_keys, "priv_pred", "priv_feature")

                data_buf.write_step(t, td)

            data = data_buf.td

        rollout_time = time.perf_counter() - start
        training_start = time.perf_counter()

        if hasattr(policy, "step_schedule"):
            policy.step_schedule(i / total_iters, i)
        if hasattr(env, "step_schedule"):
            env.step_schedule(i / total_iters, i)
        
        train_carry = policy.train_op(data, vecnorm)

        if need_logging:
            info = {}
            env_frames += data.numel()
            episode_stats.add(data)

            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()
            
            info.update(train_carry)
            info.update(env.extra)
            info.update(env.stats_ema)

            info["env_frames"] = env_frames * aa.get_world_size()
            info["rollout_fps"] = data.numel() / rollout_time * aa.get_world_size()
            info["training_time"] = time.perf_counter() - training_start
        
            if save_interval and save_interval > 0 and should_save(i):
                save(policy, f"checkpoint_{i}")

            run.log(info, step=i)
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, (float, int))}))
    
    if aa.is_main_process():
        save(policy, "checkpoint_final")

        policy_eval = policy.get_rollout_policy("eval")
        info, trajs, stats = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
        run.log(info, step = total_iters)

        wandb.finish()
    exit(0)


if __name__ == "__main__":
    main()
