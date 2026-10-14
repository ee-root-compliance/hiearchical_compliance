import torch
import hydra
import numpy as np
import einops
import time
import sys
from tqdm import tqdm
from omegaconf import OmegaConf

from isaaclab.app import AppLauncher

from active_adaptation.learning import ALGOS

import wandb
import logging
from tqdm import tqdm
from scripts.utils.helpers import make_env_policy, evaluate

import os
import datetime
import termcolor

def eval(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    env, agent, vecnorm, _ = make_env_policy(cfg)
    
    keys = [
        ("next", "stats"),
        ("next", "done"), 
        ("next", "reward"),
        "value_obs",
        "value_priv",
        "value_adapt",
        "context_expert",
        "context_scale",
        "context_adapt",
        "context_adapt_scale",
        "action_kl",
    ]
    
    policy_eval = agent.get_rollout_policy("eval")
    info, trajs, stats = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed, keys=keys)
    
    print(termcolor.colored(trajs, "light_yellow"))
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")

    info["task"] = cfg.task.name
    info["algo"] = cfg.algo.name
    info["checkpoint_path"] = cfg.checkpoint_path
    info["argv"] = sys.argv
    print(OmegaConf.to_yaml(info))

    exit(0)
    env.close()
    simulation_app.close()