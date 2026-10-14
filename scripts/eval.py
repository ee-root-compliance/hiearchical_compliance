import torch
import wandb
import os
import sys
import hydra
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omegaconf import OmegaConf
from isaaclab.app import AppLauncher
from scripts.utils.play import play
from scripts.utils.eval import eval

FILE_PATH = os.path.dirname(__file__)

LEGACY_FORCE_APPLY_PATTERN = [".*shoulder_yaw_link", ".*wrist_roll_link", ".*hand_mimic"]


def patch_legacy_force_apply_pattern(cfg):
    command_cfg = cfg["task"]["command"]
    command_target = command_cfg.get("_target_", "")
    if not (
        command_target.startswith("active_adaptation.envs.mdp.commands.motion_tracking.")
        and "impedance" in command_target
    ):
        return
    if "force_apply_pattern" not in command_cfg:
        return

    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is None:
        return
    checkpoint_path = os.path.expanduser(checkpoint_path)
    if checkpoint_path.startswith("run:"):
        return
    if not os.path.exists(checkpoint_path):
        return

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vecnorm = state_dict.get("vecnorm", {})
    extra_state = vecnorm.get("_extra_state", {})
    priv_sum = extra_state.get("priv_sum", None)
    if priv_sum is None:
        return

    if priv_sum.shape[0] == 732:
        active_pattern = list(command_cfg["force_apply_pattern"])
        if active_pattern != LEGACY_FORCE_APPLY_PATTERN:
            command_cfg["force_active_pattern"] = active_pattern
            command_cfg["force_apply_pattern"] = LEGACY_FORCE_APPLY_PATTERN
            print(
                "[Compat] Checkpoint vecnorm expects legacy 6-body force_priv. "
                f"Using force_apply_pattern={LEGACY_FORCE_APPLY_PATTERN} and "
                f"force_active_pattern={active_pattern}."
            )


def patch_missing_reward_sigma(cfg):
    command_cfg = cfg["task"]["command"]
    reward_sigma = command_cfg.get("reward_sigma", None)
    if reward_sigma is None:
        return
    if "feet" not in reward_sigma:
        reward_sigma["feet"] = [0.2]
        print("[Compat] Added missing reward_sigma.feet=[0.2] for feet_tracking reward.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_path", type=str)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("-p", "--play", action="store_true", default=False)
    # whether to override terrain and command
    parser.add_argument("-t", "--terrain", action="store_true", default=False)
    parser.add_argument("-c", "--command", action="store_true", default=False)
    parser.add_argument("-o", "--teleop", action="store_true", default=False)
    
    parser.add_argument("-e", "--export", action="store_true", default=False)
    parser.add_argument("-v", "--video", action="store_true", default=False)
    parser.add_argument("-i", "--iterations", dest="iterations", type=int, default=None)
    parser.add_argument("-s", "--success", action="store_true", default=False)  # test success rate
    parser.add_argument("--baseline_root_command", action="store_true", default=False)
    args = parser.parse_args()

    api = wandb.Api()
    
    run = api.run(args.run_path)
    print(f"Loading run {run.name}")

    root = os.path.join(os.path.dirname(__file__), "wandb", run.name)
    os.makedirs(root, exist_ok=True)

    checkpoints = []
    for file in run.files():
        print(file.name)
        if "checkpoint" in file.name:
            checkpoints.append(file)
        elif file.name == "cfg.yaml":
            file.download(root, replace=True)
        elif file.name == "files/cfg.yaml":
            file.download(root, replace=True)
        elif file.name == "config.yaml":
            file.download(root, replace=True)
    
    if args.iterations is None:
        def sort_by_time(file):
            number_str = file.name[:-3].split("_")[-1]
            if number_str == "final":
                return 100000
            else:
                return int(number_str)

        checkpoints.sort(key=sort_by_time)
        checkpoint = checkpoints[-1]
    else:
        for file in checkpoints:
            if file.name == f"checkpoint_{args.iterations}.pt":
                checkpoint = file
                break
        else:
            available = sorted(
                (file.name[:-3].split("_")[-1] for file in checkpoints),
                key=lambda it: (not it.isdigit(), int(it) if it.isdigit() else 0, it),
            )
            print(
                f"Error: checkpoint_{args.iterations}.pt not found in run {args.run_path}. "
                f"Available iterations: {', '.join(available) or 'none'}"
            )
            sys.exit(1)
    print(f"Downloading {checkpoint.name}")
    checkpoint.download(root, replace=True)

    try:
        cfg = OmegaConf.load(os.path.join(root, "files", "cfg.yaml"))
    except FileNotFoundError:
        cfg = OmegaConf.load(os.path.join(root, "cfg.yaml"))
    OmegaConf.set_struct(cfg, False)

    cfg["checkpoint_path"] = os.path.join(root, checkpoint.name)
    if cfg.get("vecnorm", None) is not None:
        cfg["vecnorm"] = "eval"

    if args.teleop:
        cfg["task"]["command"]["teleop"] = True

    if args.task is not None:
        with hydra.initialize(config_path="../cfg", job_name="eval", version_base=None):
            _cfg = hydra.compose(config_name="eval", overrides=[f"task={args.task}"])
        # cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["reward"] = _cfg.task.reward
        cfg["task"]["termination"] = _cfg.task.termination
        cfg["task"]["observation"] = _cfg.task.observation
        cfg["task"]["action"] = _cfg.task.action
        cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["robot"] = _cfg.task.robot
        if args.terrain:
            cfg["task"]["terrain"] = _cfg.task.terrain
        if args.command:
            cfg["task"]["command"] = _cfg.task.command
        cfg["task"]["flags"] = _cfg.task.flags

    if args.baseline_root_command:
        cfg["task"]["action"]["override_root_command"] = True

    cfg["task"]["randomization"] = {}

    patch_legacy_force_apply_pattern(cfg)
    patch_missing_reward_sigma(cfg)
    
    if args.play:
        if not args.success:
            cfg["app"]["headless"] = False
            cfg["task"]["num_envs"] = 16
        cfg["export_policy"] = args.export
        cfg["perf_test"] = False
        play(cfg)
    else:
        if args.video:
            cfg["task"]["num_envs"] = 16
            cfg["eval_render"] = True
            cfg["app"]["enable_cameras"] = True
            cfg["app"]["headless"] = True
        eval(cfg)
    exit(0)

if __name__ == "__main__":
    main()
