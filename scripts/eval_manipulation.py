"""
Evaluation script for manipulation tasks with teleoperation.

This script loads a trained policy and runs inference in the manipulation environment.
The policy and environment remain compatible - observation/action spaces are unchanged.

Usage (wandb):
    python scripts/eval_manipulation.py --run_path your-entity/your-project/run_name -p

Usage (EE tracking eval):
    python scripts/eval_manipulation.py --run_path your-entity/your-project/run_name --objects cfg/objects/room_scene.yaml -p --full_collision --ee_tracking_eval --external_force off
    
Usage (local checkpoint):
    python scripts/eval_manipulation.py --checkpoint outputs/xxx/model.pt
"""

import torch
import wandb
import hydra
import argparse
import os
import sys
import json
import datetime
import re

# ``active_adaptation.learning`` is imported below by the evaluation helpers.
# It contains training-time ``@torch.compile`` decorators, which are
# materialized during import.  Evaluation never executes PPO optimisation, so
# these helpers must remain eager for *all* deployment policies, including a
# plain PPOPolicy checkpoint such as the ranged low-level policy.  Previously
# this was enabled only for the analytical-MoE command, which left ordinary
# ranged low-level evaluation spawning a large TorchInductor compile pool on
# its first rollout.
_EVAL_EAGER_CONFIGURED = False
os.environ["ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE"] = "1"
_set_compiler_stance = getattr(getattr(torch, "compiler", None), "set_stance", None)
if _set_compiler_stance is not None:
    _set_compiler_stance("force_eager")
    _EVAL_EAGER_CONFIGURED = True

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omegaconf import OmegaConf, DictConfig
from isaaclab.app import AppLauncher
from isaaclab.utils.math import matrix_from_quat
from torchrl.envs.utils import set_exploration_type, ExplorationType
from scripts.utils.play import play
from scripts.utils.helpers import make_env_policy
from active_adaptation.utils.math import (
    clamp_norm,
    quat_apply,
    quat_apply_inverse,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat,
    yaw_quat,
    normalize,
)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")
DEFAULT_EXTERNAL_FORCE_CFG = os.path.join(CONFIG_PATH, "eval", "external_force", "default_ee_net_pull.yaml")

EE_TRACKING_SEED = 0
EE_TRACKING_NUM_POINTS = 20
EE_TRACKING_RADIUS = 0.15
EE_TRACKING_HOLD_STEPS = 100
EE_TRACKING_WARMUP_STEPS = 50
EE_TRACKING_MAX_EPISODE_LENGTH = 1000000
EE_TRACKING_BODY_NAMES = "left_hand_mimic,right_hand_mimic"
EE_TRACKING_FEET_BODY_NAMES = "left_ankle_roll_link,right_ankle_roll_link"
EE_TRACKING_MIN_EE_CENTER_Z = 0.0
EE_TRACKING_DEFAULT_EE_CENTER_B = [
    [0.170, 0.250, 0.080],
    [0.170, -0.250, 0.080],
]
EE_EVAL_MEAN_WINDOW_SEC = 0.5
EE_COMPLIANCE_FORCE_MAGNITUDES = [5.0, 10.0, 15.0, 20.0, 30.0]


def _safe_report_name(name: str) -> str:
    name = name.strip().rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "policy"


def _default_ee_report_path(args, prefix: str) -> str:
    if args.run_path:
        policy_name = _safe_report_name(args.run_path)
    elif args.checkpoint:
        policy_name = _safe_report_name(os.path.splitext(os.path.basename(args.checkpoint))[0])
    elif getattr(args, "moe_experts_config", None):
        policy_name = _safe_report_name(
            os.path.splitext(os.path.basename(args.moe_experts_config))[0]
        )
    else:
        policy_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stiffness_values = getattr(args, "ee_compliance_stiffness", None)
    stiffness_suffix = ""
    if (
        (prefix.startswith("ee_compliance") or prefix.startswith("ee_bimanual_compliance"))
        and stiffness_values is not None
    ):
        stiffness_token = "_".join(f"{value:g}" for value in stiffness_values)
        stiffness_suffix = f"_k{_safe_report_name(stiffness_token)}"
    output_dir = (
        os.path.join("outputs", "root_compliance_eval")
        if prefix.startswith("root_compliance")
        else "outputs"
    )
    return os.path.join(output_dir, f"{prefix}_{policy_name}{stiffness_suffix}.json")


def _policy_source_label(args) -> str:
    return args.run_path or args.checkpoint or getattr(args, "moe_experts_config", None) or "unknown"


def _configure_evaluation_compiler(args, *, force: bool = False) -> None:
    """Keep deploy/evaluation inference out of TorchInductor.

    The analytical MoE, ranged high-level policy, and ranged low-level
    ``PPOPolicy`` can all import PPO helpers containing training-only compiled
    functions.  A compiled function can trigger a very large first-call
    Inductor compilation, leaving the EE sweep with no visible progress.
    Evaluation does not benefit from training-time compilation, so force eager
    execution for every deployment policy.

    ``set_stance`` was added after the first PyTorch 2.x releases; retain a
    no-op fallback so this script remains usable with older environments.
    """
    set_stance = getattr(getattr(torch, "compiler", None), "set_stance", None)
    if set_stance is None:
        print(
            "[Info] Evaluation: torch.compiler.set_stance is unavailable; "
            "training-time torch.compile is disabled through the environment.",
            flush=True,
        )
        return

    if not _EVAL_EAGER_CONFIGURED:
        set_stance("force_eager")
    print(
        "[Info] Evaluation: TorchInductor disabled; using eager inference.",
        flush=True,
    )


EE_COMPLIANCE_FORCE_DIRECTIONS = [
    ("+x", [1.0, 0.0, 0.0]),
    ("-x", [-1.0, 0.0, 0.0]),
    ("+y", [0.0, 1.0, 0.0]),
    ("-y", [0.0, -1.0, 0.0]),
    ("+z", [0.0, 0.0, 1.0]),
    ("-z", [0.0, 0.0, -1.0]),
]
EE_COMPLIANCE_RAMP_STEPS = 25
EE_COMPLIANCE_HOLD_STEPS = 100
EE_COMPLIANCE_RECOVERY_STEPS = 50
EE_COMPLIANCE_BASELINE_STEPS = 50
ROOT_COMPLIANCE_RAMP_STEPS = 50
ROOT_COMPLIANCE_HOLD_STEPS = 250
ROOT_COMPLIANCE_RECOVERY_STEPS = 100
ROOT_COMPLIANCE_BASELINE_STEPS = 100
ROOT_COMPLIANCE_WARMUP_STEPS = 100
ROOT_COMPLIANCE_MEAN_WINDOW_SEC = 1.0
ROOT_COMPLIANCE_BODY_NAMES = [
    "torso_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_hand_mimic",
    "right_hand_mimic",
]
ROOT_COMPLIANCE_FORCE_DIRECTIONS = [
    ("+x", [1.0, 0.0, 0.0]),
    ("-x", [-1.0, 0.0, 0.0]),
    ("+y", [0.0, 1.0, 0.0]),
    ("-y", [0.0, -1.0, 0.0]),
]
ROOT_COMPLIANCE_FORCE_MAGNITUDES = [5.0, 10.0, 20.0, 30.0]
BIMANUAL_PER_EE_REQUIRED_MSG = (
    "--ee_bimanual_compliance_eval requires a per-EE stiffness task "
    "(command.net_pull_ee_compliance_stiffness_per_ee: true), e.g. "
    "--task G1/hl/ee/G1_hl_ee_bimanual_analytical_moe_200_600_force_b_student."
)


def _sample_uniform_ball(num_points: int, radius: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = torch.randn(num_points, 2, 3, generator=generator)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    magnitude = radius * torch.rand(num_points, 2, 1, generator=generator).pow(1.0 / 3.0)
    return direction * magnitude


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


def _quat_to_rpy_wxyz(q: torch.Tensor) -> torch.Tensor:
    q = normalize(q)
    w, x, y, z = q.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack([roll, pitch, yaw], dim=-1)


def _body_pose_in_root_frame(asset, body_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1)
    body_pos_w = asset.data.body_pos_w[:, body_ids]
    body_quat_w = asset.data.body_quat_w[:, body_ids]
    root_quat_expanded = root_quat_w.expand(-1, len(body_ids), -1)
    pos_b = quat_apply_inverse(root_quat_expanded, body_pos_w - root_pos_w)
    quat_b = quat_mul(quat_conjugate(root_quat_expanded), body_quat_w)
    return pos_b, normalize(quat_b)


def _quat_angle_error_deg(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    actual = normalize(actual)
    target = normalize(target)
    dot = (actual * target).sum(dim=-1).abs().clamp(-1.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(dot))


def _summary(values: torch.Tensor) -> dict:
    flat = values.detach().float().reshape(-1).cpu()
    return {
        "mean": float(flat.mean().item()),
        "rmse": float(torch.sqrt((flat * flat).mean()).item()),
        "max": float(flat.max().item()),
        "min": float(flat.min().item()),
        "std": float(flat.std(unbiased=False).item()),
    }


def _cfg_number_or_list(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        return [float(v) for v in value]
    return float(value)


def _param_tensor(value, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _format_scalar_or_xyz(value) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return "[" + ", ".join(_format_scalar_or_xyz(item) for item in value) + "]"
        return "[" + ", ".join(f"{v:.2f}" for v in value) + "]"
    return f"{float(value):.2f}"


def _stiffness_along_direction(value, direction: torch.Tensor, ee_index: int | None = None):
    if value is None:
        return None
    if isinstance(value, list):
        stiffness = torch.as_tensor(value, dtype=torch.float32, device=direction.device)
        if stiffness.ndim >= 2:
            # Per-EE stiffness [[left xyz], [right xyz]]: select this EE's row.
            stiffness = stiffness[0 if ee_index is None else ee_index].reshape(-1)
        if stiffness.numel() == 1:
            return float(stiffness.item())
        return float((stiffness * direction.abs()).sum().item())
    return float(value)


def _mean_tensor_samples(samples: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(samples, dim=0).mean(dim=0)


def _mean_pose_samples(pos_samples: list[torch.Tensor], quat_samples: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    pos = _mean_tensor_samples(pos_samples)
    quat = normalize(_mean_tensor_samples(quat_samples))
    return pos, quat


def _capture_joint_motion_state(asset, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
    """Capture joint and whole-body motion state for compliance probes.

    The snapshot is intentionally raw.  The EE compliance evaluator can later
    derive joint displacement, joint velocity, root drift, or torque-related
    quantities without rerunning the expensive simulator sweep.
    """
    data = asset.data
    fields = {
        "joint_pos": "joint_pos",
        "joint_vel": "joint_vel",
        "joint_pos_target": "joint_pos_target",
        "applied_torque": "applied_torque",
        "root_pos_w": "root_pos_w",
        "root_quat_w": "root_quat_w",
        "root_lin_vel_w": "root_lin_vel_w",
        "root_ang_vel_w": "root_ang_vel_w",
    }
    snapshot = {}
    for key, attribute in fields.items():
        value = getattr(data, attribute, None)
        if value is None:
            snapshot[key] = None
        elif env_ids is None:
            snapshot[key] = value.detach().clone()
        else:
            snapshot[key] = value[env_ids].detach().clone()
    return snapshot


def _capture_ee_jacobian_root(
    asset,
    body_ids: list[int],
    env_ids: torch.Tensor | None = None,
    spatial: bool = False,
) -> torch.Tensor | None:
    """Capture translational EE Jacobians expressed in the robot root frame.

    IsaacLab's PhysX Jacobian is world-frame. For a floating-base articulation,
    body indices are used directly and the first six Jacobian columns are the
    floating-base DoFs. Articulated joint columns are selected as
    ``joint_id + 6``; this is important because the joint order is not the
    same as simply taking the last ``N`` columns. For a fixed-base articulation,
    body and joint indices are used directly.
    The force used by the compliance evaluator is expressed in this same root
    frame, so ``J.T @ F`` and ``pinv(J) @ (F / K)`` are frame-consistent.
    When ``spatial`` is true, both translational and angular rows are returned
    with shape ``[env, ee, 6, joints]``; otherwise only translational rows are
    returned.
    """
    physx_view = getattr(asset, "root_physx_view", None)
    if physx_view is None or not hasattr(physx_view, "get_jacobians"):
        return None
    jacobian_w = physx_view.get_jacobians()
    is_fixed_base = bool(getattr(asset, "is_fixed_base", False))
    jacobian_body_ids = [int(body_id) - 1 if is_fixed_base else int(body_id) for body_id in body_ids]
    if not jacobian_body_ids or min(jacobian_body_ids) < 0 or max(jacobian_body_ids) >= jacobian_w.shape[1]:
        return None
    if env_ids is not None:
        jacobian_w = jacobian_w[env_ids]
        root_quat_w = asset.data.root_quat_w[env_ids]
    else:
        root_quat_w = asset.data.root_quat_w
    jacobian_w = jacobian_w[:, jacobian_body_ids, :, :]
    root_rot_inv = matrix_from_quat(quat_conjugate(root_quat_w))
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, :, :3, :] = torch.einsum("eij,ebjn->ebin", root_rot_inv, jacobian_w[:, :, :3, :])
    jacobian_b[:, :, 3:, :] = torch.einsum("eij,ebjn->ebin", root_rot_inv, jacobian_w[:, :, 3:, :])
    joint_names = list(getattr(asset, "joint_names", []))
    if not joint_names:
        return None
    joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    if is_fixed_base:
        jacobian_joint_ids = [int(joint_id) for joint_id in joint_ids]
    else:
        jacobian_joint_ids = [int(joint_id) + 6 for joint_id in joint_ids]
    if not jacobian_joint_ids or max(jacobian_joint_ids) >= jacobian_b.shape[-1]:
        return None
    row_count = 6 if spatial else 3
    return jacobian_b[:, :, :row_count, jacobian_joint_ids].detach().clone()


def _stiffness_xyz_for_ee(value, ee_index: int, device: torch.device) -> torch.Tensor | None:
    """Normalize scalar, xyz, or per-hand xyz stiffness to a 3-vector."""
    if value is None:
        return None
    stiffness = torch.as_tensor(value, dtype=torch.float32, device=device)
    if stiffness.ndim == 0:
        return stiffness.expand(3)
    if stiffness.ndim == 1:
        if stiffness.numel() == 3:
            return stiffness
        if stiffness.numel() == 2:
            return stiffness[ee_index].expand(3)
        if stiffness.numel() == 1:
            return stiffness.reshape(()).expand(3)
    if stiffness.ndim >= 2:
        hand = stiffness[ee_index].reshape(-1)
        if hand.numel() == 1:
            return hand.expand(3)
        if hand.numel() >= 3:
            return hand[:3]
    return None


def _predict_joint_response_from_ee_compliance(
    jacobian_root: torch.Tensor | None,
    force_root: torch.Tensor,
    stiffness_xyz: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, float | None]:
    """Return ideal EE displacement, minimum-norm IK delta-q, J.T@F, cond(J)."""
    if jacobian_root is None or stiffness_xyz is None:
        return None, None, None, None
    if jacobian_root.ndim != 2 or jacobian_root.shape[0] != 3:
        return None, None, None, None
    ideal_ee_delta = force_root / stiffness_xyz.clamp_min(1e-6)
    joint_delta_ik = torch.linalg.pinv(jacobian_root) @ ideal_ee_delta
    external_torque = jacobian_root.transpose(0, 1) @ force_root
    try:
        condition = float(torch.linalg.cond(jacobian_root).item())
    except RuntimeError:
        condition = None
    return ideal_ee_delta, joint_delta_ik, external_torque, condition


def _stack_motion_state_samples(samples: list[dict], key: str) -> torch.Tensor | None:
    values = [sample[key] for sample in samples if sample.get(key) is not None]
    if not values:
        return None
    return torch.stack(values, dim=0)


def _named_joint_summary(values: torch.Tensor, joint_names: list[str]) -> dict:
    values = values.detach().float()
    return {
        name: _summary(values[..., index])
        for index, name in enumerate(joint_names)
    }


def _mean_window_steps(base_env, window_sec: float = EE_EVAL_MEAN_WINDOW_SEC) -> int:
    step_dt = float(getattr(base_env, "step_dt", 0.02))
    return max(1, int(round(window_sec / step_dt)))


def _get_ee_compliance_params(cfg, force_deadband_override: float | None = None) -> dict:
    reward_cfg = OmegaConf.select(cfg, "task.reward.ee_compliance.ee_force_compliance_tracking")
    found = reward_cfg is not None
    reward_cfg = reward_cfg or {}
    return {
        "found_in_cfg": found,
        "stiffness": _cfg_number_or_list(reward_cfg.get("stiffness", 60.0)),
        "max_offset": _cfg_number_or_list(reward_cfg.get("max_offset", 0.25)),
        "force_deadband": (
            float(force_deadband_override)
            if force_deadband_override is not None
            else float(reward_cfg.get("force_deadband", 2.0))
        ),
    }


def _prompt_low_level_nominal_stiffness(
    compliance_params: dict,
    command_manager,
    action_manager,
    explicit_stiffness: list[float] | None = None,
) -> None:
    if _is_hierarchical_action_manager(action_manager):
        return
    # A command-line stiffness is authoritative for scripted evaluation.  Do
    # not enter the interactive low-level prompt in that case: the old order
    # made ``--ee_compliance_stiffness`` appear ineffective and blocked
    # headless/ranged low-level evaluations waiting on stdin.
    if explicit_stiffness is not None:
        return
    default = compliance_params["stiffness"]
    if hasattr(command_manager, "get_net_pull_ee_compliance_stiffness"):
        active = command_manager.get_net_pull_ee_compliance_stiffness()[0, 0].detach().float().cpu()
        if torch.allclose(active, active[0].expand_as(active)):
            default = float(active[0].item())
        else:
            default = [float(v) for v in active.tolist()]
        compliance_params["stiffness"] = default
    prompt = (
        "Low-level policy detected. Enter EE compliance nominal stiffness xyz "
        f"(default {_format_scalar_or_xyz(default)}), e.g. 600 or 200 600 200: "
    )
    try:
        raw = input(prompt).strip()
    except EOFError:
        raw = ""
    if raw:
        values = raw.replace(",", " ").split()
        if len(values) not in (1, 3):
            raise ValueError(
                "Expected one stiffness value or three xyz values, for example: 600 or 200 600 200"
            )
        compliance_params["stiffness"] = float(values[0]) if len(values) == 1 else [float(v) for v in values]
        compliance_params["found_in_cfg"] = True
    if hasattr(command_manager, "set_net_pull_ee_compliance_stiffness"):
        command_manager.set_net_pull_ee_compliance_stiffness(compliance_params["stiffness"])
    elif hasattr(command_manager, "net_pull_ee_compliance_stiffness"):
        command_manager.net_pull_ee_compliance_stiffness = _param_tensor(
            compliance_params["stiffness"],
            command_manager.device,
        )
    print(
        "Low-level EE compliance nominal stiffness: "
        f"{_format_scalar_or_xyz(compliance_params['stiffness'])}",
        flush=True,
    )


def _restore_low_level_eval_stiffness(compliance_params: dict, command_manager, action_manager) -> None:
    """Restore a user-selected stiffness after a range-task environment reset."""
    if _is_hierarchical_action_manager(action_manager):
        return
    if hasattr(command_manager, "set_net_pull_ee_compliance_stiffness"):
        command_manager.set_net_pull_ee_compliance_stiffness(compliance_params["stiffness"])


def _set_explicit_eval_stiffness(
    stiffness_values: list[float] | None,
    compliance_params: dict,
    command_manager,
) -> None:
    """Set the active EE stiffness used by both the target model and policy obs.

    Range-trained high-level policies read the active stiffness through the
    ``net_pull_ee_compliance_stiffness_command`` observation.  The command
    manager is therefore the single source of truth for both that observation
    and the compliance target calculation.
    """
    if stiffness_values is None:
        return
    per_ee = bool(getattr(command_manager, "net_pull_ee_compliance_stiffness_per_ee", False))
    valid_lengths = (1, 2, 3, 6) if per_ee else (1, 3)
    if len(stiffness_values) not in valid_lengths:
        expected = "one, two, three, or six values" if per_ee else "one isotropic value or three xyz values"
        raise ValueError(f"--ee_compliance_stiffness expects {expected}.")
    stiffness = float(stiffness_values[0]) if len(stiffness_values) == 1 else [float(v) for v in stiffness_values]
    if not hasattr(command_manager, "set_net_pull_ee_compliance_stiffness"):
        raise RuntimeError(
            "--ee_compliance_stiffness requires a command manager with "
            "set_net_pull_ee_compliance_stiffness()."
        )
    command_manager.set_net_pull_ee_compliance_stiffness(stiffness)
    compliance_params["stiffness"] = stiffness
    compliance_params["found_in_cfg"] = True


def _validate_explicit_eval_stiffness(cfg, stiffness_values: list[float] | None) -> None:
    """Reject directional commands for policies trained with scalar stiffness input."""
    if stiffness_values is None:
        return
    command_cfg = cfg.get("task", {}).get("command", {})
    scalar_range = command_cfg.get("net_pull_ee_compliance_stiffness_range", None)
    xyz_range = command_cfg.get("net_pull_ee_compliance_stiffness_xyz_range", None)
    if scalar_range is None or xyz_range is not None:
        return
    values = [float(value) for value in stiffness_values]
    if len(values) == 3 and not all(abs(value - values[0]) < 1e-6 for value in values[1:]):
        raise ValueError(
            "This policy is configured with scalar stiffness range "
            f"[{float(scalar_range[0]):g}, {float(scalar_range[1]):g}], so it only "
            "accepts an isotropic --ee_compliance_stiffness value (for example "
            "--ee_compliance_stiffness 300). Directional xyz commands such as "
            "300 600 600 require a policy trained with "
            "net_pull_ee_compliance_stiffness_xyz_range."
        )


def _restore_explicit_eval_stiffness(
    stiffness_values: list[float] | None,
    command_manager,
) -> None:
    """Reapply an explicitly requested stiffness after an environment reset."""
    if stiffness_values is None:
        return
    stiffness = float(stiffness_values[0]) if len(stiffness_values) == 1 else [float(v) for v in stiffness_values]
    if hasattr(command_manager, "set_net_pull_ee_compliance_stiffness"):
        command_manager.set_net_pull_ee_compliance_stiffness(stiffness)


def _tensor_summary(value: torch.Tensor) -> dict:
    value = value.detach().float().reshape(-1).cpu()
    return {
        "mean": float(value.mean().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
    }


def _component_summary(values: torch.Tensor) -> dict:
    values = values.detach().float().cpu()
    return {
        "x": _summary(values[..., 0]),
        "y": _summary(values[..., 1]),
        "z": _summary(values[..., 2]),
    }


def _direct_force_pred_b_from_tensordict(td, command_manager):
    if "direct_priv_pred" not in td.keys():
        return None
    pred = td["direct_priv_pred"]
    if pred.shape[-1] < 6:
        return None
    force_limit = float(getattr(command_manager, "net_pull_force_range", (1.0, 1.0))[1])

    # EE-only force estimators use [left_force_b, right_force_b]. Select the
    # slot for the currently loaded hand so legacy eval records stay 3-D.
    if pred.shape[-1] == 6:
        force_slots_b = pred.reshape(*pred.shape[:-1], 2, 3)
        if (
            pred.ndim == 2
            and hasattr(command_manager, "net_pull_idx_asset")
            and hasattr(command_manager, "net_pull_body_local_idx")
            and hasattr(command_manager, "net_pull_ee_idx_asset")
        ):
            body_idx = command_manager.net_pull_idx_asset[command_manager.net_pull_body_local_idx]
            ee_match = body_idx.unsqueeze(-1) == command_manager.net_pull_ee_idx_asset.unsqueeze(0)
            if ee_match.any(dim=-1).all():
                ee_idx = ee_match.int().argmax(dim=-1)
                env_idx = torch.arange(pred.shape[0], device=pred.device)
                return force_slots_b[env_idx, ee_idx] * max(force_limit, 1e-6)
        return force_slots_b.sum(dim=-2) * max(force_limit, 1e-6)

    # Legacy net_pull_force_priv layout: point_b[0:3], force_b[3:6], ...
    return pred[..., 3:6] * max(force_limit, 1e-6)


def _force_over_stiffness_offset_b(force_b: torch.Tensor, params: dict) -> torch.Tensor:
    force_norm = force_b.norm(dim=-1, keepdim=True)
    active_force = torch.where(
        force_norm > params["force_deadband"],
        force_b,
        torch.zeros_like(force_b),
    )
    stiffness = _param_tensor(params["stiffness"], active_force.device)
    max_offset = _param_tensor(params["max_offset"], active_force.device)
    offset_b = active_force / stiffness
    if max_offset.ndim > 0:
        return torch.clamp(offset_b, min=-max_offset, max=max_offset)
    return clamp_norm(offset_b, max=float(max_offset.item()))


def _make_force_stiffness_ee_command(
    reference_command: torch.Tensor,
    nominal_target_b: torch.Tensor,
    force_b: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    command = reference_command.clone()
    offset_b = _force_over_stiffness_offset_b(force_b, params)
    command[:, :6] = (nominal_target_b + offset_b).reshape(command.shape[0], 6)
    return command


def _set_oracle_force_stiffness_target(
    command_manager,
    action_manager,
    reference_command: torch.Tensor,
    nominal_target_b: torch.Tensor,
    oracle_force_b: torch.Tensor,
    params: dict,
):
    """Write the ground-truth ``nominal + F/K`` target used by the oracle baseline.

    This deliberately does not read ``direct_priv_pred`` or any other policy
    output.  ``oracle_force_b`` is the force scripted by the evaluator (or the
    force applied by the command manager), so this path is usable with the raw
    stiff low-level policy as well as with a hierarchical action manager.
    """
    command = _make_force_stiffness_ee_command(
        reference_command,
        nominal_target_b,
        oracle_force_b,
        params,
    )
    _set_ee_eval_target(command_manager, action_manager, command)
    return command


def _set_ee_force_stiffness_ablation_from_policy_td(
    action_manager,
    command_manager,
    policy_td,
    reference_command: torch.Tensor,
    nominal_target_b: torch.Tensor,
    params: dict,
):
    if not hasattr(action_manager, "set_ee_force_stiffness_ablation_command"):
        raise RuntimeError(
            "EE force/stiffness ablation requires a hierarchical action manager with "
            "set_ee_force_stiffness_ablation_command()."
        )
    force_pred_b = _direct_force_pred_b_from_tensordict(policy_td, command_manager)
    if force_pred_b is None:
        raise RuntimeError(
            "EE force/stiffness ablation requires policy_td['direct_priv_pred']; "
            "this checkpoint does not expose a direct force estimator."
        )
    force_pred_b = force_pred_b.to(device=nominal_target_b.device, dtype=torch.float32)
    if force_pred_b.shape != (nominal_target_b.shape[0], 3):
        raise RuntimeError(
            f"Expected force estimator output shape {(nominal_target_b.shape[0], 3)}, "
            f"got {tuple(force_pred_b.shape)}."
        )

    force_b = torch.zeros_like(nominal_target_b)
    if getattr(command_manager, "eval_ee_force_enabled", False):
        local_idx = getattr(command_manager, "eval_ee_force_body_local_idx", None)
        if local_idx is not None and hasattr(command_manager, "net_pull_ee_idx_asset") and hasattr(command_manager, "net_pull_idx_asset"):
            for ee_i, body_idx in enumerate(command_manager.net_pull_ee_idx_asset.detach().cpu().tolist()):
                matches = (command_manager.net_pull_idx_asset == int(body_idx)).nonzero(as_tuple=False).flatten()
                if matches.numel() > 0:
                    mask = local_idx.to(device=force_b.device) == matches[0].to(device=force_b.device)
                    force_b[mask, ee_i] = force_pred_b[mask]
    command = _make_force_stiffness_ee_command(reference_command, nominal_target_b, force_b, params)
    action_manager.set_ee_force_stiffness_ablation_command(command)


def _get_ee_compliance_eval_info(command_manager, params: dict, use_command_manager_target: bool = True) -> dict:
    if (
        use_command_manager_target
        and
        getattr(command_manager, "external_force_mode", "legacy") == "net_pull"
        and hasattr(command_manager, "get_net_pull_ee_compliance_target_b")
    ):
        actual_stiffness = params["stiffness"]
        if hasattr(command_manager, "get_net_pull_ee_compliance_stiffness"):
            stiffness_tensor = command_manager.get_net_pull_ee_compliance_stiffness()[0].detach().float().cpu()
            if stiffness_tensor.shape[0] == 2:
                actual_stiffness = [[float(v) for v in hand.tolist()] for hand in stiffness_tensor]
            else:
                actual_stiffness = (
                    float(stiffness_tensor[0, 0].item())
                    if torch.allclose(stiffness_tensor[0], stiffness_tensor[0, 0].expand_as(stiffness_tensor[0]))
                    else [float(v) for v in stiffness_tensor[0].tolist()]
                )
        elif hasattr(command_manager, "net_pull_ee_compliance_stiffness"):
            stiffness_tensor = command_manager.net_pull_ee_compliance_stiffness.detach().float().reshape(-1).cpu()
            actual_stiffness = (
                float(stiffness_tensor.item())
                if stiffness_tensor.numel() == 1
                else [float(v) for v in stiffness_tensor.tolist()]
            )
        return {
            "target_mode": "net_pull_dynamic_ee_compliance_target_b",
            "actual_stiffness": actual_stiffness,
            "force_limit": None,
            "effective_stiffness": None,
        }

    if params["found_in_cfg"]:
        return {
            "target_mode": "explicit_force_over_stiffness",
            "actual_stiffness": params["stiffness"],
            "force_limit": None,
            "effective_stiffness": None,
        }

    if hasattr(command_manager, "force_keypoint_b"):
        force_limit = None
        effective_stiffness = None
        if hasattr(command_manager, "force_safe_limit_tl"):
            force_limit = _tensor_summary(command_manager.force_safe_limit_tl.current)
            effective_stiffness = {
                key: val / 0.05
                for key, val in force_limit.items()
            }
        return {
            "target_mode": "low_level_force_keypoint_b",
            "actual_stiffness": None,
            "force_limit": force_limit,
            "effective_stiffness": effective_stiffness,
        }

    return {
        "target_mode": "fallback_force_over_stiffness",
        "actual_stiffness": params["stiffness"],
        "force_limit": None,
        "effective_stiffness": None,
    }


def _slice_envs(value: torch.Tensor, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return value
    return value[env_ids]


def _compute_ee_compliance_target_b(
    command_manager,
    asset,
    body_ids: list[int],
    nominal_target_b: torch.Tensor,
    params: dict,
    env_ids: torch.Tensor | None = None,
    use_command_manager_target: bool = True,
):
    if (
        use_command_manager_target
        and
        getattr(command_manager, "external_force_mode", "legacy") == "net_pull"
        and hasattr(command_manager, "get_net_pull_ee_compliance_target_b")
    ):
        compliance_target_b = command_manager.get_net_pull_ee_compliance_target_b()
        if hasattr(command_manager, "get_net_pull_ee_force_b"):
            force_b = command_manager.get_net_pull_ee_force_b()
        else:
            force_b = torch.zeros_like(nominal_target_b)
        target_offset_b = compliance_target_b - nominal_target_b
        return (
            _slice_envs(compliance_target_b, env_ids),
            _slice_envs(target_offset_b, env_ids),
            _slice_envs(force_b, env_ids),
        )

    if (
        getattr(command_manager, "external_force_mode", "legacy") == "net_pull"
        and hasattr(command_manager, "get_net_pull_ee_force_b")
    ):
        force_b = command_manager.get_net_pull_ee_force_b()
        if env_ids is not None and force_b.shape[0] != nominal_target_b.shape[0]:
            force_b = _slice_envs(force_b, env_ids)
    else:
        force_b = None

    force_w = torch.zeros_like(nominal_target_b)
    force_apply_idx = getattr(command_manager, "force_apply_idx_asset", None)
    force_applied_w = getattr(command_manager, "force_applied_w", None)

    if force_b is None and force_apply_idx is not None and force_applied_w is not None:
        force_apply_list = force_apply_idx.detach().cpu().tolist()
        for ee_i, body_i in enumerate(body_ids):
            if body_i in force_apply_list:
                force_i = force_apply_list.index(body_i)
                force_w[:, ee_i] = force_applied_w[:, force_i]

        root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, len(body_ids), -1)
        force_b = quat_apply_inverse(root_quat, force_w)
    elif force_b is None:
        force_b = torch.zeros_like(nominal_target_b)
    if not params["found_in_cfg"] and force_apply_idx is not None and hasattr(command_manager, "force_keypoint_b"):
        compliance_target_b = nominal_target_b.clone()
        force_apply_list = force_apply_idx.detach().cpu().tolist()
        active_force = force_b.norm(dim=-1, keepdim=True) > params["force_deadband"]
        for ee_i, body_i in enumerate(body_ids):
            if body_i in force_apply_list:
                force_i = force_apply_list.index(body_i)
                compliance_target_b[:, ee_i] = torch.where(
                    active_force[:, ee_i],
                    command_manager.force_keypoint_b[:, force_i],
                    nominal_target_b[:, ee_i],
                )
        target_offset_b = compliance_target_b - nominal_target_b
        return (
            _slice_envs(compliance_target_b, env_ids),
            _slice_envs(target_offset_b, env_ids),
            _slice_envs(force_b, env_ids),
        )

    force_norm = force_b.norm(dim=-1, keepdim=True)
    active_force = torch.where(
        force_norm > params["force_deadband"],
        force_b,
        torch.zeros_like(force_b),
    )
    stiffness = _param_tensor(params["stiffness"], active_force.device)
    max_offset = _param_tensor(params["max_offset"], active_force.device)
    target_offset_b = active_force / stiffness
    if max_offset.ndim > 0:
        target_offset_b = torch.clamp(target_offset_b, min=-max_offset, max=max_offset)
    else:
        target_offset_b = clamp_norm(target_offset_b, max=float(max_offset.item()))
    compliance_target_b = nominal_target_b + target_offset_b
    return (
        _slice_envs(compliance_target_b, env_ids),
        _slice_envs(target_offset_b, env_ids),
        _slice_envs(force_b, env_ids),
    )


def _set_ee_command(command_manager, command: torch.Tensor):
    if hasattr(command_manager, "set_root_and_wrist_6d_command"):
        command_manager.set_root_and_wrist_6d_command(command)
        return
    raise RuntimeError(
        "EE tracking eval needs a command manager with set_root_and_wrist_6d_command(). "
        "Please use a high-level manipulation/root command task."
    )


def _is_hierarchical_action_manager(action_manager) -> bool:
    return hasattr(action_manager, "low_policy") and hasattr(action_manager, "_decode_ee_command")


def _set_ee_reference(command_manager, command: torch.Tensor):
    if hasattr(command_manager, "set_root_and_wrist_6d_reference_override"):
        command_manager.set_root_and_wrist_6d_reference_override(command)
        return
    _set_ee_command(command_manager, command)


def _set_ee_eval_target(command_manager, action_manager, command: torch.Tensor):
    if _is_hierarchical_action_manager(action_manager):
        _set_ee_reference(command_manager, command)
    else:
        _set_ee_command(command_manager, command)
        # Low-level policies consume the command observation directly, while
        # net-pull compliance targets use the reference path. Keep both paths
        # anchored to the same evaluation target.
        if hasattr(command_manager, "set_root_and_wrist_6d_reference_override"):
            command_manager.set_root_and_wrist_6d_reference_override(command)


def _get_hl_ee_command_pos_b(action_manager, env_ids: torch.Tensor | None = None):
    if not _is_hierarchical_action_manager(action_manager):
        return None
    ee_command = getattr(action_manager, "ee_command", None)
    if ee_command is None or ee_command.shape[-1] < 6:
        return None
    command_pos_b = ee_command[:, :6].reshape(ee_command.shape[0], 2, 3)
    return _slice_envs(command_pos_b, env_ids)


def _refresh_moe_stiffness_observation(td_, command_manager):
    """Refresh the analytical-MoE gate input after an eval stiffness update."""
    if "hl_moe" not in td_.keys():
        return td_
    observation_fn = getattr(command_manager, "net_pull_ee_compliance_stiffness_command", None)
    if observation_fn is not None:
        td_["hl_moe"] = observation_fn()
    return td_


def _set_external_force(command_manager, enabled: bool):
    if hasattr(command_manager, "set_external_force_enabled"):
        command_manager.set_external_force_enabled(enabled)
        return
    if not enabled:
        print(
            "[WARNING] --external_force off requested, but this command manager "
            "does not expose set_external_force_enabled().",
            flush=True,
        )


def _configure_external_force(cfg, enabled: bool):
    command_cfg = cfg["task"]["command"]
    command_target = command_cfg.get("_target_", "")
    if command_target.startswith("active_adaptation.envs.mdp.commands.motion_tracking.") and "impedance" in command_target:
        command_cfg["external_force_enabled"] = enabled
        print(f"  External force: {'on' if enabled else 'off'}")
        return True
    if not enabled:
        print(
            f"  External force: off requested, but command target does not support it: {command_target}"
        )
    return False


def _apply_external_force_mode(cfg, mode: str):
    if mode == "default":
        external_force_cfg = OmegaConf.load(DEFAULT_EXTERNAL_FORCE_CFG)
        command_overrides = external_force_cfg.get("command", {})
        cfg["task"]["command"].update(command_overrides)
        print(f"  External force config: default ({DEFAULT_EXTERNAL_FORCE_CFG})")

    enabled = mode != "off"
    return _configure_external_force(cfg, enabled)


def _uses_hierarchical_action_cfg(cfg) -> bool:
    action_cfg = cfg["task"].get("action", {})
    return action_cfg.get("_target_", "") == "active_adaptation.envs.mdp.action.HierarchicalRootCommand"


def _fix_low_level_force_limit_for_ee_eval(cfg):
    if _uses_hierarchical_action_cfg(cfg):
        return None
    if OmegaConf.select(cfg, "task.reward.ee_compliance.ee_force_compliance_tracking.stiffness") is not None:
        return None

    command_cfg = cfg["task"]["command"]
    if "force_safe_default" not in command_cfg:
        command_cfg["force_safe_default"] = 10.0
    force_limit = float(command_cfg["force_safe_default"])
    command_cfg["force_safe_bounds"] = [force_limit, force_limit]
    print(
        "  Low-level EE eval: fixed force_safe_limit "
        f"to default {force_limit:.2f} because no EE compliance stiffness config was found."
    )
    return force_limit


def _enable_root_passthrough_for_ee_only_hl(cfg):
    action_cfg = cfg["task"]["action"]
    action_target = action_cfg.get("_target_", "")
    if action_target != "active_adaptation.envs.mdp.action.HierarchicalRootCommand":
        return
    root_command_cfg = action_cfg.get("root_command", {})
    root_enabled = root_command_cfg.get("enabled", True)
    if root_enabled:
        return
    if "root_command" not in action_cfg or action_cfg["root_command"] is None:
        action_cfg["root_command"] = {}
    action_cfg["root_command"]["passthrough_reference"] = True
    print("  Root command: passthrough reference enabled for EE-only high-level teleop")


def _set_static_root_command(command_manager, asset):
    if hasattr(command_manager, "set_static_root_command"):
        command_manager.set_static_root_command(
            root_height=asset.data.root_pos_w[:, 2:3],
        )
        return
    if not hasattr(command_manager, "set_command_override"):
        return
    command = torch.zeros(command_manager.num_envs, 6, device=command_manager.device)
    command[:, 0] = asset.data.root_pos_w[:, 2]
    command[:, 3] = 1.0
    if hasattr(command_manager, "force_safe_limit_tl"):
        command[:, 5:6] = command_manager.force_safe_limit_tl.current
    command_manager.set_command_override(command)


def _set_default_feet_command(command_manager, asset):
    if not hasattr(command_manager, "set_feet_pos_b_command"):
        return False

    body_names = [name.strip() for name in EE_TRACKING_FEET_BODY_NAMES.split(",")]
    try:
        body_ids = [asset.body_names.index(name) for name in body_names]
    except ValueError:
        print(
            f"[WARNING] Cannot find feet body names {body_names}; skip feet command override.",
            flush=True,
        )
        return False

    feet_pos_b, _ = _body_pose_in_root_frame(asset, body_ids)
    command_manager.set_feet_pos_b_command(feet_pos_b.reshape(command_manager.num_envs, 6))
    return True


def _get_ee_sample_center(
    default_pos_b: torch.Tensor,
    device: torch.device,
    initial_position: list[float] | None = None,
) -> torch.Tensor:
    if initial_position is None:
        center = torch.tensor(EE_TRACKING_DEFAULT_EE_CENTER_B, dtype=torch.float32, device=device)
    else:
        if len(initial_position) != 6:
            raise ValueError(
                "EE compliance initial position must contain six values: "
                "left_x left_y left_z right_x right_y right_z."
            )
        center = torch.tensor(initial_position, dtype=torch.float32, device=device).reshape(2, 3)
    center = center.unsqueeze(0).expand(default_pos_b.shape[0], -1, -1).clone()
    center[..., 2].clamp_min_(EE_TRACKING_MIN_EE_CENTER_Z)
    return center


def _disable_eval_timer_reset(base_env):
    base_env.episode_length_buf.zero_()
    if hasattr(base_env.command_manager, "finished"):
        base_env.command_manager.finished.zero_()


def _warn_if_done(tensordict, step_label: str):
    if "done" in tensordict.keys() and tensordict["done"].any():
        print(f"[WARNING] Env reset triggered during scripted eval at {step_label}.", flush=True)


def evaluate_ee_tracking(cfg, args):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app
    env = None

    try:
        print("Root compliance: loading environment and checkpoint.", flush=True)
        env, policy, _vecnorm, _ = make_env_policy(cfg)
        print("Root compliance: policy loaded.", flush=True)
        rollout_policy = policy.get_rollout_policy("eval")
        print("Root compliance: rollout policy created.", flush=True)
        base_env = env.base_env if hasattr(env, "base_env") else env
        asset = base_env.scene["robot"]
        command_manager = base_env.command_manager
        action_manager = base_env.action_manager
        _set_external_force(command_manager, args.external_force != "off")

        body_names = [name.strip() for name in EE_TRACKING_BODY_NAMES.split(",")]
        try:
            body_ids = [asset.body_names.index(name) for name in body_names]
        except ValueError as exc:
            raise RuntimeError(
                f"Cannot find EE body name from {body_names}. Available bodies: {asset.body_names}"
            ) from exc
        if len(body_ids) != 2:
            raise RuntimeError(f"Expected exactly two EE body names, got {body_names}.")

        td_ = env.reset()
        _disable_eval_timer_reset(base_env)
        _set_static_root_command(command_manager, asset)
        print("Static root command enabled for EE tracking eval.", flush=True)
        if _set_default_feet_command(command_manager, asset):
            print("Default feet command enabled for EE tracking eval.", flush=True)
        default_pos_b, default_quat_b = _body_pose_in_root_frame(asset, body_ids)
        sample_center_b = _get_ee_sample_center(default_pos_b, base_env.device)
        compliance_params = _get_ee_compliance_params(
            cfg, getattr(args, "ee_compliance_force_deadband", None)
        )
        use_command_manager_compliance_target = _is_hierarchical_action_manager(action_manager)
        compliance_eval_info = _get_ee_compliance_eval_info(
            command_manager,
            compliance_params,
            use_command_manager_target=use_command_manager_compliance_target,
        )
        target_quat_b = torch.zeros_like(default_quat_b)
        target_quat_b[..., 0] = 1.0
        target_axis_angle_b = torch.zeros(base_env.num_envs, 2, 3, device=base_env.device)
        target_rpy_b = torch.zeros(base_env.num_envs, 2, 3, device=base_env.device)

        default_command = torch.cat(
            [sample_center_b.reshape(base_env.num_envs, 6), target_axis_angle_b.reshape(base_env.num_envs, 6)],
            dim=-1,
        )
        _set_ee_eval_target(command_manager, action_manager, default_command)
        if _is_hierarchical_action_manager(action_manager):
            print("EE tracking eval target is written as high-level EE reference override.", flush=True)
        else:
            print("EE tracking eval target is written as low-level EE command override.", flush=True)
        print(
            "EE sample center_b env0 "
            f"default={default_pos_b[0].detach().cpu().tolist()} "
            f"used={sample_center_b[0].detach().cpu().tolist()}",
            flush=True,
        )
        print("EE compliance target mode: " + compliance_eval_info["target_mode"], flush=True)
        if compliance_eval_info["actual_stiffness"] is not None:
            print(
                "EE compliance actual stiffness "
                f"{_format_scalar_or_xyz(compliance_eval_info['actual_stiffness'])}; "
                f"max_offset={_format_scalar_or_xyz(compliance_params['max_offset'])}, "
                f"force_deadband={compliance_params['force_deadband']} "
                f"(from_cfg={compliance_params['found_in_cfg']})",
                flush=True,
            )
        elif compliance_eval_info["effective_stiffness"] is not None:
            force_limit = compliance_eval_info["force_limit"]
            effective_stiffness = compliance_eval_info["effective_stiffness"]
            print(
                "EE compliance effective low-level stiffness "
                f"mean/min/max={effective_stiffness['mean']:.2f}/"
                f"{effective_stiffness['min']:.2f}/"
                f"{effective_stiffness['max']:.2f} N/m "
                f"from force_safe_limit mean/min/max={force_limit['mean']:.2f}/"
                f"{force_limit['min']:.2f}/"
                f"{force_limit['max']:.2f} N; "
                f"force_deadband={compliance_params['force_deadband']}",
                flush=True,
            )
        else:
            print(
                "EE compliance actual stiffness unavailable; "
                f"fallback stiffness={_format_scalar_or_xyz(compliance_params['stiffness'])}, "
                f"force_deadband={compliance_params['force_deadband']}",
                flush=True,
            )

        offsets = _sample_uniform_ball(
            EE_TRACKING_NUM_POINTS,
            EE_TRACKING_RADIUS,
            EE_TRACKING_SEED,
        ).to(base_env.device)
        target_pos_b = sample_center_b.unsqueeze(0) + offsets.unsqueeze(1)

        records = []
        print("Starting EE tracking rollout...", flush=True)
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for _ in range(EE_TRACKING_WARMUP_STEPS):
                td_ = rollout_policy(td_)
                td, td_ = env.step_and_maybe_reset(td_)
                _warn_if_done(td, "warmup")

            for point_idx in range(EE_TRACKING_NUM_POINTS):
                command = torch.cat(
                    [
                        target_pos_b[point_idx].reshape(base_env.num_envs, 6),
                        target_axis_angle_b.reshape(base_env.num_envs, 6),
                    ],
                    dim=-1,
                )
                _set_ee_eval_target(command_manager, action_manager, command)

                for _ in range(EE_TRACKING_HOLD_STEPS):
                    td_ = rollout_policy(td_)
                    td, td_ = env.step_and_maybe_reset(td_)
                    _warn_if_done(td, f"point {point_idx + 1}")

                actual_pos_b, actual_quat_b = _body_pose_in_root_frame(asset, body_ids)
                pos_error = (actual_pos_b - target_pos_b[point_idx]).norm(dim=-1)
                compliance_target_b, compliance_offset_b, force_b = _compute_ee_compliance_target_b(
                    command_manager,
                    asset,
                    body_ids,
                    target_pos_b[point_idx],
                    compliance_params,
                    use_command_manager_target=use_command_manager_compliance_target,
                )
                compliance_pos_error = (actual_pos_b - compliance_target_b).norm(dim=-1)
                rpy_error = torch.rad2deg(_wrap_to_pi(_quat_to_rpy_wxyz(actual_quat_b) - target_rpy_b).abs())
                quat_error = _quat_angle_error_deg(actual_quat_b, target_quat_b)

                records.append({
                    "point_index": point_idx,
                    "target_pos_b": target_pos_b[point_idx].detach().cpu().tolist(),
                    "compliance_target_pos_b": compliance_target_b.detach().cpu().tolist(),
                    "compliance_offset_b": compliance_offset_b.detach().cpu().tolist(),
                    "ee_force_b": force_b.detach().cpu().tolist(),
                    "actual_pos_b": actual_pos_b.detach().cpu().tolist(),
                    "target_quat_b_wxyz": target_quat_b.detach().cpu().tolist(),
                    "target_axis_angle_b": target_axis_angle_b.detach().cpu().tolist(),
                    "target_rpy_b_deg": torch.rad2deg(target_rpy_b).detach().cpu().tolist(),
                    "actual_rpy_b_deg": torch.rad2deg(_quat_to_rpy_wxyz(actual_quat_b)).detach().cpu().tolist(),
                    "pos_error_m": pos_error.detach().cpu().tolist(),
                    "compliance_pos_error_m": compliance_pos_error.detach().cpu().tolist(),
                    "rpy_abs_error_deg": rpy_error.detach().cpu().tolist(),
                    "quat_angle_error_deg": quat_error.detach().cpu().tolist(),
                })

                mean_l = pos_error[:, 0].mean().item()
                mean_r = pos_error[:, 1].mean().item()
                compliance_mean_l = compliance_pos_error[:, 0].mean().item()
                compliance_mean_r = compliance_pos_error[:, 1].mean().item()
                print(
                    f"EE point {point_idx + 1:02d}/{EE_TRACKING_NUM_POINTS}: "
                    f"left_pos_err={mean_l:.4f} m, right_pos_err={mean_r:.4f} m, "
                    f"compliance_left_err={compliance_mean_l:.4f} m, "
                    f"compliance_right_err={compliance_mean_r:.4f} m"
                )

        pos_errors = torch.tensor([r["pos_error_m"] for r in records])
        compliance_pos_errors = torch.tensor([r["compliance_pos_error_m"] for r in records])
        compliance_offsets = torch.tensor([r["compliance_offset_b"] for r in records])
        ee_forces_b = torch.tensor([r["ee_force_b"] for r in records])
        rpy_errors = torch.tensor([r["rpy_abs_error_deg"] for r in records])
        quat_errors = torch.tensor([r["quat_angle_error_deg"] for r in records])

        report = {
            "checkpoint": args.checkpoint,
            "run_path": args.run_path,
            "moe_experts_config": getattr(args, "moe_experts_config", None),
            "task": args.task,
            "seed": EE_TRACKING_SEED,
            "num_envs": args.num_envs,
            "ee_body_names": body_names,
            "feet_body_names": EE_TRACKING_FEET_BODY_NAMES.split(","),
            "default_ee_center_b": default_pos_b.detach().cpu().tolist(),
            "sample_ee_center_b": sample_center_b.detach().cpu().tolist(),
            "configured_ee_center_b": EE_TRACKING_DEFAULT_EE_CENTER_B,
            "min_ee_center_z": EE_TRACKING_MIN_EE_CENTER_Z,
            "ee_radius_m": EE_TRACKING_RADIUS,
            "ee_points": EE_TRACKING_NUM_POINTS,
            "ee_hold_steps": EE_TRACKING_HOLD_STEPS,
            "ee_warmup_steps": EE_TRACKING_WARMUP_STEPS,
            "external_force": args.external_force,
            "external_force_default_cfg": DEFAULT_EXTERNAL_FORCE_CFG if args.external_force == "default" else None,
            "ee_compliance_target": compliance_params,
            "ee_compliance_eval_info": compliance_eval_info,
            "summary": {
                "position_error_m": {
                    "combined": _summary(pos_errors),
                    "left": _summary(pos_errors[:, :, 0]),
                    "right": _summary(pos_errors[:, :, 1]),
                },
                "compliance_position_error_m": {
                    "combined": _summary(compliance_pos_errors),
                    "left": _summary(compliance_pos_errors[:, :, 0]),
                    "right": _summary(compliance_pos_errors[:, :, 1]),
                },
                "compliance_offset_norm_m": {
                    "combined": _summary(compliance_offsets.norm(dim=-1)),
                    "left": _summary(compliance_offsets[:, :, 0].norm(dim=-1)),
                    "right": _summary(compliance_offsets[:, :, 1].norm(dim=-1)),
                },
                "ee_force_b_norm_n": {
                    "combined": _summary(ee_forces_b.norm(dim=-1)),
                    "left": _summary(ee_forces_b[:, :, 0].norm(dim=-1)),
                    "right": _summary(ee_forces_b[:, :, 1].norm(dim=-1)),
                },
                "rpy_abs_error_deg": {
                    "combined": _summary(rpy_errors),
                    "left": _summary(rpy_errors[:, :, 0]),
                    "right": _summary(rpy_errors[:, :, 1]),
                },
                "quat_angle_error_deg": {
                    "combined": _summary(quat_errors),
                    "left": _summary(quat_errors[:, :, 0]),
                    "right": _summary(quat_errors[:, :, 1]),
                },
            },
            "records": records,
        }

        if args.ee_output is None:
            args.ee_output = _default_ee_report_path(args, "ee_tracking_eval")
        os.makedirs(os.path.dirname(args.ee_output) or ".", exist_ok=True)
        with open(args.ee_output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        pos = report["summary"]["position_error_m"]["combined"]
        compliance_pos = report["summary"]["compliance_position_error_m"]["combined"]
        compliance_offset = report["summary"]["compliance_offset_norm_m"]["combined"]
        ee_force = report["summary"]["ee_force_b_norm_n"]["combined"]
        quat = report["summary"]["quat_angle_error_deg"]["combined"]
        rpy = report["summary"]["rpy_abs_error_deg"]["combined"]
        print("\n" + "=" * 60)
        print("EE TRACKING EVAL")
        print("=" * 60)
        print(f"  Nominal position error mean/rmse/max: {pos['mean']:.4f} / {pos['rmse']:.4f} / {pos['max']:.4f} m")
        print(
            "  Compliance position error mean/rmse/max: "
            f"{compliance_pos['mean']:.4f} / {compliance_pos['rmse']:.4f} / {compliance_pos['max']:.4f} m"
        )
        print(
            "  Compliance offset norm mean/max: "
            f"{compliance_offset['mean']:.4f} / {compliance_offset['max']:.4f} m"
        )
        print(f"  EE force_b norm mean/max: {ee_force['mean']:.2f} / {ee_force['max']:.2f} N")
        print(f"  RPY abs error mean/rmse/max: {rpy['mean']:.2f} / {rpy['rmse']:.2f} / {rpy['max']:.2f} deg")
        print(f"  Quat angle error mean/rmse/max: {quat['mean']:.2f} / {quat['rmse']:.2f} / {quat['max']:.2f} deg")
        print(f"  Report: {args.ee_output}")
        print("=" * 60 + "\n")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def _rollout_one_step(env, rollout_policy, td_, step_label: str, return_policy_td: bool = False, after_policy=None):
    td_ = rollout_policy(td_)
    policy_td = td_ if return_policy_td else None
    if after_policy is not None:
        after_policy(td_)
    td, td_ = env.step_and_maybe_reset(td_)
    _warn_if_done(td, step_label)
    if return_policy_td:
        return td, td_, policy_td
    return td, td_


def _get_root_compliance_params(cfg) -> dict:
    velocity_cfg = OmegaConf.select(cfg, "task.reward.root_hold.root_force_velocity_tracking")
    position_cfg = OmegaConf.select(cfg, "task.reward.root_hold.root_position_hold")
    found = velocity_cfg is not None
    velocity_cfg = velocity_cfg or {}
    return {
        "found_in_cfg": found,
        # A low-level baseline has no root_hold reward, but is still a valid
        # zero-drift reference for the root compliance evaluator.
        "mode": "damping" if found else ("position_hold" if position_cfg is not None else "baseline"),
        "position_hold_found_in_cfg": position_cfg is not None,
        "damping": float(velocity_cfg.get("damping", 60.0)),
        "force_deadband": float(velocity_cfg.get("force_deadband", 0.0)),
        "max_speed": velocity_cfg.get("max_speed", None),
        "add_root_command_reference": bool(velocity_cfg.get("add_root_command_reference", True)),
    }


def _velocity_xy_b_to_w(asset, velocity_xy_b: torch.Tensor) -> torch.Tensor:
    velocity_b = torch.zeros(velocity_xy_b.shape[0], 3, dtype=torch.float32, device=velocity_xy_b.device)
    velocity_b[:, :2] = velocity_xy_b
    return quat_apply(yaw_quat(asset.data.root_quat_w), velocity_b)[:, :2]


def _get_root_reference_velocity_w(command_manager, asset, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    if hasattr(command_manager, "get_root_command_reference"):
        root_reference = command_manager.get_root_command_reference()
    elif hasattr(command_manager, "root_command"):
        root_reference = command_manager.root_command
    else:
        root_reference = torch.zeros(asset.data.root_pos_w.shape[0], 5, device=asset.data.root_pos_w.device)
    velocity_w = _velocity_xy_b_to_w(asset, root_reference[:, 1:3])
    return _slice_envs(velocity_w, env_ids)


def _get_root_command_velocity_w(action_manager, command_manager, asset, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    root_command = getattr(action_manager, "root_command", None)
    if root_command is None:
        root_command = getattr(command_manager, "root_command", None)
    if root_command is None:
        return _get_root_reference_velocity_w(command_manager, asset, env_ids)
    velocity_w = _velocity_xy_b_to_w(asset, root_command[:, 1:3])
    return _slice_envs(velocity_w, env_ids)


def _get_root_eval_sample(command_manager, action_manager, asset, env_ids: torch.Tensor | None = None) -> dict:
    root_rpy_w = _quat_to_rpy_wxyz(asset.data.root_quat_w)
    return {
        "reference_velocity_w": _get_root_reference_velocity_w(command_manager, asset, env_ids),
        "command_velocity_w": _get_root_command_velocity_w(action_manager, command_manager, asset, env_ids),
        "actual_velocity_w": _slice_envs(asset.data.root_lin_vel_w[:, :2], env_ids),
        "root_pos_w": _slice_envs(asset.data.root_pos_w[:, :2], env_ids),
        "root_height": _slice_envs(asset.data.root_pos_w[:, 2:3], env_ids),
        "root_yaw": _slice_envs(root_rpy_w[:, 2:3], env_ids),
    }


def _mean_sample_dict(samples: list[dict]) -> dict:
    return {
        key: _mean_tensor_samples([sample[key] for sample in samples])
        for key in samples[0].keys()
    }


def _safe_ratio(force_n: float, velocity_mps: float) -> float | None:
    if abs(velocity_mps) <= 1e-5:
        return None
    return float(force_n / velocity_mps)


def _available_root_eval_bodies(command_manager, asset) -> list[str]:
    available = []
    net_pull_idx_asset = getattr(command_manager, "net_pull_idx_asset", None)
    if net_pull_idx_asset is None:
        return available
    net_pull_ids = set(int(idx) for idx in net_pull_idx_asset.detach().cpu().tolist())
    for body_name in ROOT_COMPLIANCE_BODY_NAMES:
        if body_name not in asset.body_names:
            continue
        if asset.body_names.index(body_name) not in net_pull_ids:
            continue
        available.append(body_name)
    return available


def evaluate_root_compliance(cfg, args):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app
    env = None

    try:
        env, policy, _vecnorm, _ = make_env_policy(cfg)
        rollout_policy = policy.get_rollout_policy("eval")
        base_env = env.base_env if hasattr(env, "base_env") else env
        asset = base_env.scene["robot"]
        command_manager = base_env.command_manager
        action_manager = base_env.action_manager
        _set_external_force(command_manager, True)

        if not hasattr(command_manager, "set_eval_root_force_w") or not hasattr(command_manager, "clear_eval_root_force"):
            raise RuntimeError(
                "Root compliance eval needs MotionTrackingCommand_impedance with set_eval_root_force_w() "
                "and clear_eval_root_force()."
            )

        body_names = _available_root_eval_bodies(command_manager, asset)
        if not body_names:
            raise RuntimeError(
                "No root compliance eval bodies are present in net_pull_apply_pattern. "
                f"Requested candidates: {ROOT_COMPLIANCE_BODY_NAMES}"
            )

        ee_body_names = [name.strip() for name in EE_TRACKING_BODY_NAMES.split(",")]
        try:
            ee_body_ids = [asset.body_names.index(name) for name in ee_body_names]
        except ValueError as exc:
            raise RuntimeError(
                f"Cannot find EE body name from {ee_body_names}. Available bodies: {asset.body_names}"
            ) from exc
        default_ee_pos_b, _ = _body_pose_in_root_frame(asset, ee_body_ids)
        default_ee_center_b = _get_ee_sample_center(default_ee_pos_b, base_env.device)
        default_ee_axis_angle_b = torch.zeros(base_env.num_envs, 2, 3, device=base_env.device)
        default_ee_command = torch.cat(
            [
                default_ee_center_b.reshape(base_env.num_envs, 6),
                default_ee_axis_angle_b.reshape(base_env.num_envs, 6),
            ],
            dim=-1,
        )

        root_params = _get_root_compliance_params(cfg)
        if root_params["mode"] == "unknown":
            raise RuntimeError(
                "Root compliance eval requires either "
                "task.reward.root_hold.root_force_velocity_tracking (damping) or "
                "task.reward.root_hold.root_position_hold (resist) in the task config."
            )
        position_hold_eval = root_params["mode"] in {"position_hold", "baseline"}
        damping = float(root_params["damping"])
        mean_window_steps = min(
            _mean_window_steps(base_env, ROOT_COMPLIANCE_MEAN_WINDOW_SEC),
            ROOT_COMPLIANCE_HOLD_STEPS,
        )
        baseline_window_steps = min(mean_window_steps, ROOT_COMPLIANCE_BASELINE_STEPS)

        print("Root compliance: resetting environment.", flush=True)
        td_ = env.reset()
        print("Root compliance: reset complete.", flush=True)
        _disable_eval_timer_reset(base_env)
        _set_static_root_command(command_manager, asset)
        _set_ee_eval_target(command_manager, action_manager, default_ee_command)
        print("Static root command enabled for root compliance eval: reference velocity = 0.", flush=True)
        print("Default EE target enabled for root compliance eval.", flush=True)
        if position_hold_eval:
            print(
                "Root compliance target: position hold, target root XY drift = 0 m. "
                f"mode={root_params['mode']}.",
                flush=True,
            )
        else:
            print(
                "Root compliance target: "
                f"delta_v_w = force_xy_w / damping, damping={damping:.2f} N/(m/s), "
                f"force_deadband={root_params['force_deadband']:.2f} N, "
                f"max_speed={root_params['max_speed']}",
                flush=True,
            )
        print(
            f"Starting root compliance sweep... bodies={body_names}, "
            f"mean_window_steps={mean_window_steps}, baseline_window_steps={baseline_window_steps}",
            flush=True,
        )

        directions = [
            (name, torch.tensor(vec, dtype=torch.float32, device=base_env.device))
            for name, vec in ROOT_COMPLIANCE_FORCE_DIRECTIONS
        ]
        case_specs = []
        for body_name in body_names:
            for direction_name, direction_w in directions:
                for magnitude in ROOT_COMPLIANCE_FORCE_MAGNITUDES:
                    case_specs.append({
                        "body_name": body_name,
                        "direction_name": direction_name,
                        "direction_w": direction_w,
                        "force_n": magnitude,
                    })

        records = []
        batch_size = min(max(1, int(getattr(args, "root_compliance_num_envs", 1))), base_env.num_envs)

        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for batch_start in range(0, len(case_specs), batch_size):
                batch_cases = case_specs[batch_start: batch_start + batch_size]
                active_envs = len(batch_cases)
                batch_end = batch_start + active_envs
                print(
                    f"Root compliance progress: cases {batch_start + 1}-{batch_end}/"
                    f"{len(case_specs)} ({batch_end / len(case_specs):.0%})",
                    flush=True,
                )
                env_ids = torch.arange(active_envs, device=base_env.device)
                command_manager.clear_eval_root_force()
                td_ = env.reset()
                _disable_eval_timer_reset(base_env)
                _set_static_root_command(command_manager, asset)
                _set_ee_eval_target(command_manager, action_manager, default_ee_command)

                for _ in range(ROOT_COMPLIANCE_WARMUP_STEPS):
                    _, td_ = _rollout_one_step(env, rollout_policy, td_, "root compliance warmup")

                baseline_samples = []
                for step in range(ROOT_COMPLIANCE_BASELINE_STEPS):
                    command_manager.clear_eval_root_force()
                    _, td_ = _rollout_one_step(env, rollout_policy, td_, "root compliance baseline")
                    if step >= ROOT_COMPLIANCE_BASELINE_STEPS - baseline_window_steps:
                        baseline_samples.append(
                            _get_root_eval_sample(command_manager, action_manager, asset, env_ids=env_ids)
                        )
                baseline = _mean_sample_dict(baseline_samples)

                for _ in range(ROOT_COMPLIANCE_RECOVERY_STEPS):
                    command_manager.clear_eval_root_force()
                    _, td_ = _rollout_one_step(env, rollout_policy, td_, "root compliance recovery")

                for ramp_step in range(ROOT_COMPLIANCE_RAMP_STEPS):
                    ramp_ratio = float(ramp_step + 1) / float(ROOT_COMPLIANCE_RAMP_STEPS)
                    for env_i, case in enumerate(batch_cases):
                        force_w = case["direction_w"] * (case["force_n"] * ramp_ratio)
                        command_manager.set_eval_root_force_w(
                            case["body_name"],
                            force_w,
                            env_ids=env_ids[env_i:env_i + 1],
                        )
                    _, td_ = _rollout_one_step(env, rollout_policy, td_, "root compliance ramp")

                hold_samples = [[] for _ in range(active_envs)]
                full_forces = [
                    case["direction_w"] * case["force_n"]
                    for case in batch_cases
                ]
                for hold_step in range(ROOT_COMPLIANCE_HOLD_STEPS):
                    for env_i, case in enumerate(batch_cases):
                        command_manager.set_eval_root_force_w(
                            case["body_name"],
                            full_forces[env_i],
                            env_ids=env_ids[env_i:env_i + 1],
                        )
                    _, td_ = _rollout_one_step(env, rollout_policy, td_, "root compliance hold")
                    if hold_step >= ROOT_COMPLIANCE_HOLD_STEPS - mean_window_steps:
                        sample = _get_root_eval_sample(command_manager, action_manager, asset, env_ids=env_ids)
                        for env_i in range(active_envs):
                            hold_samples[env_i].append({
                                key: value[env_i:env_i + 1]
                                for key, value in sample.items()
                            })

                for env_i, case in enumerate(batch_cases):
                    hold = _mean_sample_dict(hold_samples[env_i])
                    base_slice = {
                        key: value[env_i:env_i + 1]
                        for key, value in baseline.items()
                    }
                    force_w = full_forces[env_i]
                    force_xy_w = force_w[:2]
                    force_dir_xy_w = force_xy_w / force_xy_w.norm().clamp_min(1e-6)
                    target_delta_v_w = (
                        torch.zeros_like(force_xy_w)
                        if position_hold_eval
                        else force_xy_w / damping
                    )

                    command_delta_from_baseline_w = hold["command_velocity_w"] - base_slice["command_velocity_w"]
                    actual_delta_from_baseline_w = hold["actual_velocity_w"] - base_slice["actual_velocity_w"]
                    command_residual_from_reference_w = hold["command_velocity_w"] - hold["reference_velocity_w"]
                    actual_residual_from_reference_w = hold["actual_velocity_w"] - hold["reference_velocity_w"]

                    command_along = (command_delta_from_baseline_w * force_dir_xy_w).sum(dim=-1)
                    actual_along = (actual_delta_from_baseline_w * force_dir_xy_w).sum(dim=-1)
                    command_orth = (
                        command_delta_from_baseline_w - command_along.unsqueeze(-1) * force_dir_xy_w
                    ).norm(dim=-1)
                    actual_orth = (
                        actual_delta_from_baseline_w - actual_along.unsqueeze(-1) * force_dir_xy_w
                    ).norm(dim=-1)
                    command_error = (command_delta_from_baseline_w - target_delta_v_w).norm(dim=-1)
                    actual_error = (actual_delta_from_baseline_w - target_delta_v_w).norm(dim=-1)

                    height_delta = hold["root_height"] - base_slice["root_height"]
                    yaw_delta_deg = torch.rad2deg(_wrap_to_pi(hold["root_yaw"] - base_slice["root_yaw"]))
                    position_delta_xy = hold["root_pos_w"] - base_slice["root_pos_w"]
                    position_along = (position_delta_xy * force_dir_xy_w).sum(dim=-1)
                    position_orth = (
                        position_delta_xy
                        - position_along.unsqueeze(-1) * force_dir_xy_w
                    ).norm(dim=-1)
                    position_drift = position_delta_xy.norm(dim=-1)
                    command_along_mean = float(command_along.mean().item())
                    actual_along_mean = float(actual_along.mean().item())

                    record = {
                        "body": case["body_name"],
                        "direction": case["direction_name"],
                        "direction_w": case["direction_w"].detach().cpu().tolist(),
                        "force_w_n": force_w.detach().cpu().tolist(),
                        "force_n": case["force_n"],
                        "target_delta_velocity_w_mps": target_delta_v_w.detach().cpu().tolist(),
                        "target_delta_velocity_norm_mps": float(target_delta_v_w.norm().item()),
                        "cfg_damping_n_per_mps": damping,
                        "measured_command_damping_signed_n_per_mps": _safe_ratio(case["force_n"], command_along_mean),
                        "measured_command_damping_abs_n_per_mps": _safe_ratio(case["force_n"], abs(command_along_mean)),
                        "measured_actual_damping_signed_n_per_mps": _safe_ratio(case["force_n"], actual_along_mean),
                        "measured_actual_damping_abs_n_per_mps": _safe_ratio(case["force_n"], abs(actual_along_mean)),
                        "baseline_reference_velocity_w_mps": base_slice["reference_velocity_w"].detach().cpu().tolist(),
                        "baseline_command_velocity_w_mps": base_slice["command_velocity_w"].detach().cpu().tolist(),
                        "baseline_actual_velocity_w_mps": base_slice["actual_velocity_w"].detach().cpu().tolist(),
                        "hold_reference_velocity_w_mps": hold["reference_velocity_w"].detach().cpu().tolist(),
                        "hold_command_velocity_w_mps": hold["command_velocity_w"].detach().cpu().tolist(),
                        "hold_actual_velocity_w_mps": hold["actual_velocity_w"].detach().cpu().tolist(),
                        "command_delta_from_baseline_w_mps": command_delta_from_baseline_w.detach().cpu().tolist(),
                        "actual_delta_from_baseline_w_mps": actual_delta_from_baseline_w.detach().cpu().tolist(),
                        "command_residual_from_reference_w_mps": command_residual_from_reference_w.detach().cpu().tolist(),
                        "actual_residual_from_reference_w_mps": actual_residual_from_reference_w.detach().cpu().tolist(),
                        "command_delta_error_mps": command_error.detach().cpu().tolist(),
                        "actual_delta_error_mps": actual_error.detach().cpu().tolist(),
                        "command_delta_along_force_mps": command_along.detach().cpu().tolist(),
                        "actual_delta_along_force_mps": actual_along.detach().cpu().tolist(),
                        "command_delta_orthogonal_mps": command_orth.detach().cpu().tolist(),
                        "actual_delta_orthogonal_mps": actual_orth.detach().cpu().tolist(),
                        "baseline_root_position_xy_w_m": base_slice["root_pos_w"].detach().cpu().tolist(),
                        "hold_root_position_xy_w_m": hold["root_pos_w"].detach().cpu().tolist(),
                        "root_position_delta_xy_w_m": position_delta_xy.detach().cpu().tolist(),
                        "root_position_drift_xy_m": position_drift.detach().cpu().tolist(),
                        "root_position_along_force_m": position_along.detach().cpu().tolist(),
                        "root_position_orthogonal_m": position_orth.detach().cpu().tolist(),
                        "baseline_root_height_m": base_slice["root_height"].detach().cpu().tolist(),
                        "hold_root_height_m": hold["root_height"].detach().cpu().tolist(),
                        "root_height_delta_m": height_delta.detach().cpu().tolist(),
                        "root_yaw_delta_deg": yaw_delta_deg.detach().cpu().tolist(),
                    }
                    records.append(record)

                    print(
                        f"{case['body_name']:>22s} {case['direction_name']:>2s} {case['force_n']:>4.0f}N "
                        f"[env {batch_start + env_i:02d}]: "
                        f"target={target_delta_v_w.norm().item():.3f} m/s, "
                        f"cmd_along={command_along_mean:.3f} m/s, "
                        f"actual_along={actual_along_mean:.3f} m/s, "
                        f"cmd_err={command_error.mean().item():.3f}, "
                        f"actual_err={actual_error.mean().item():.3f}",
                        flush=True,
                    )

            command_manager.clear_eval_root_force()

        command_errors = torch.tensor([r["command_delta_error_mps"] for r in records])
        actual_errors = torch.tensor([r["actual_delta_error_mps"] for r in records])
        command_along = torch.tensor([r["command_delta_along_force_mps"] for r in records])
        actual_along = torch.tensor([r["actual_delta_along_force_mps"] for r in records])
        command_orth = torch.tensor([r["command_delta_orthogonal_mps"] for r in records])
        actual_orth = torch.tensor([r["actual_delta_orthogonal_mps"] for r in records])
        target_norm = torch.tensor([r["target_delta_velocity_norm_mps"] for r in records])
        height_delta = torch.tensor([r["root_height_delta_m"] for r in records])
        yaw_delta = torch.tensor([r["root_yaw_delta_deg"] for r in records])
        position_drift = torch.tensor([r["root_position_drift_xy_m"] for r in records])
        position_along = torch.tensor([r["root_position_along_force_m"] for r in records])
        position_orth = torch.tensor([r["root_position_orthogonal_m"] for r in records])
        command_damping = [
            r["measured_command_damping_abs_n_per_mps"]
            for r in records
            if r["measured_command_damping_abs_n_per_mps"] is not None
        ]
        actual_damping = [
            r["measured_actual_damping_abs_n_per_mps"]
            for r in records
            if r["measured_actual_damping_abs_n_per_mps"] is not None
        ]
        command_damping_tensor = (
            torch.tensor(command_damping, dtype=torch.float32)
            if command_damping
            else torch.empty(0, dtype=torch.float32)
        )
        actual_damping_tensor = (
            torch.tensor(actual_damping, dtype=torch.float32)
            if actual_damping
            else torch.empty(0, dtype=torch.float32)
        )

        report = {
            "checkpoint": args.checkpoint,
            "run_path": args.run_path,
            "moe_experts_config": getattr(args, "moe_experts_config", None),
            "task": args.task,
            "num_envs": args.root_compliance_num_envs if args.root_compliance_eval else args.num_envs,
            "root_reference": "static_zero_velocity",
            "default_ee_center_b": default_ee_center_b.detach().cpu().tolist(),
            "configured_ee_center_b": EE_TRACKING_DEFAULT_EE_CENTER_B,
            "root_force_bodies": body_names,
            "force_directions": ROOT_COMPLIANCE_FORCE_DIRECTIONS,
            "force_magnitudes_n": ROOT_COMPLIANCE_FORCE_MAGNITUDES,
            "warmup_steps": ROOT_COMPLIANCE_WARMUP_STEPS,
            "ramp_steps": ROOT_COMPLIANCE_RAMP_STEPS,
            "hold_steps": ROOT_COMPLIANCE_HOLD_STEPS,
            "recovery_steps": ROOT_COMPLIANCE_RECOVERY_STEPS,
            "baseline_steps": ROOT_COMPLIANCE_BASELINE_STEPS,
            "mean_window_sec": ROOT_COMPLIANCE_MEAN_WINDOW_SEC,
            "mean_window_steps": mean_window_steps,
            "external_force": "manual_root_sweep",
            "root_compliance_target": root_params,
            "summary": {
                "target_delta_velocity_norm_mps": _summary(target_norm),
                "command_delta_error_mps": _summary(command_errors),
                "actual_delta_error_mps": _summary(actual_errors),
                "command_delta_along_force_mps": _summary(command_along),
                "actual_delta_along_force_mps": _summary(actual_along),
                "command_delta_orthogonal_mps": _summary(command_orth),
                "actual_delta_orthogonal_mps": _summary(actual_orth),
                "measured_command_damping_abs_n_per_mps": (
                    _summary(command_damping_tensor)
                    if command_damping_tensor.numel() > 0
                    else None
                ),
                "measured_actual_damping_abs_n_per_mps": (
                    _summary(actual_damping_tensor)
                    if actual_damping_tensor.numel() > 0
                    else None
                ),
                "root_height_delta_m": _summary(height_delta),
                "root_yaw_delta_deg": _summary(yaw_delta),
                "root_position_drift_xy_m": _summary(position_drift),
                "root_position_along_force_m": _summary(position_along),
                "root_position_orthogonal_m": _summary(position_orth),
            },
            "records": records,
        }

        if args.root_output is None:
            args.root_output = _default_ee_report_path(args, "root_compliance_eval")
        os.makedirs(os.path.dirname(args.root_output) or ".", exist_ok=True)
        with open(args.root_output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        cmd_error = report["summary"]["command_delta_error_mps"]
        actual_error = report["summary"]["actual_delta_error_mps"]
        cmd_damping = report["summary"]["measured_command_damping_abs_n_per_mps"]
        actual_damping_summary = report["summary"]["measured_actual_damping_abs_n_per_mps"]
        print("\n" + "=" * 60)
        print("ROOT COMPLIANCE EVAL")
        print("=" * 60)
        if position_hold_eval:
            drift = report["summary"]["root_position_drift_xy_m"]
            along = report["summary"]["root_position_along_force_m"]
            orth = report["summary"]["root_position_orthogonal_m"]
            print(
                "  Root XY drift mean/rmse/max: "
                f"{drift['mean']:.4f} / {drift['rmse']:.4f} / {drift['max']:.4f} m"
            )
            print(
                "  Root drift along-force mean/rmse/max: "
                f"{along['mean']:.4f} / {along['rmse']:.4f} / {along['max']:.4f} m"
            )
            print(
                "  Root drift orthogonal mean/rmse/max: "
                f"{orth['mean']:.4f} / {orth['rmse']:.4f} / {orth['max']:.4f} m"
            )
        else:
            print(
                "  Command delta error mean/rmse/max: "
                f"{cmd_error['mean']:.4f} / {cmd_error['rmse']:.4f} / {cmd_error['max']:.4f} m/s"
            )
            print(
                "  Actual delta error mean/rmse/max: "
                f"{actual_error['mean']:.4f} / {actual_error['rmse']:.4f} / {actual_error['max']:.4f} m/s"
            )
            if cmd_damping is not None:
                print(
                    "  Measured command damping abs mean/min/max: "
                    f"{cmd_damping['mean']:.1f} / {cmd_damping['min']:.1f} / {cmd_damping['max']:.1f} N/(m/s)"
                )
            if actual_damping_summary is not None:
                print(
                    "  Measured actual damping abs mean/min/max: "
                    f"{actual_damping_summary['mean']:.1f} / {actual_damping_summary['min']:.1f} / {actual_damping_summary['max']:.1f} N/(m/s)"
                )
            print(f"  Config damping: {damping:.2f} N/(m/s)")
        print(f"  Evaluation mode: {root_params['mode']}")
        print(f"  Report: {args.root_output}")
        print("=" * 60 + "\n")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def evaluate_ee_bimanual_compliance(cfg, args):
    """Evaluate both EEs under the same force sweep in each environment.

    This is intentionally a separate mode: the historical EE compliance eval
    applies force to one hand at a time and its report schema is preserved.
    The bimanual mode expects a per-EE stiffness command and reports each hand
    independently as well as the combined result.
    """
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app
    env = None
    try:
        env, policy, _vecnorm, _ = make_env_policy(cfg)
        rollout_policy = policy.get_rollout_policy("eval")
        base_env = env.base_env if hasattr(env, "base_env") else env
        asset = base_env.scene["robot"]
        command_manager = base_env.command_manager
        action_manager = base_env.action_manager
        _set_external_force(command_manager, True)
        if not hasattr(command_manager, "set_eval_bimanual_ee_force_b"):
            raise RuntimeError(
                "Bimanual EE compliance eval requires the updated MotionTrackingCommand_impedance."
            )
        if not getattr(command_manager, "net_pull_ee_compliance_stiffness_per_ee", False):
            raise RuntimeError(BIMANUAL_PER_EE_REQUIRED_MSG)

        body_names = [name.strip() for name in EE_TRACKING_BODY_NAMES.split(",")]
        body_ids = [asset.body_names.index(name) for name in body_names]
        td_ = env.reset()
        _disable_eval_timer_reset(base_env)
        _set_static_root_command(command_manager, asset)
        _set_default_feet_command(command_manager, asset)

        default_pos_b, default_quat_b = _body_pose_in_root_frame(asset, body_ids)
        sample_center_b = _get_ee_sample_center(
            default_pos_b,
            base_env.device,
            getattr(args, "ee_compliance_initial_position", None),
        )
        target_quat_b = torch.zeros_like(default_quat_b)
        target_quat_b[..., 0] = 1.0
        target_axis_angle_b = torch.zeros(base_env.num_envs, 2, 3, device=base_env.device)
        target_rpy_b = torch.zeros(base_env.num_envs, 2, 3, device=base_env.device)
        default_command = torch.cat(
            [sample_center_b.reshape(base_env.num_envs, 6), target_axis_angle_b.reshape(base_env.num_envs, 6)],
            dim=-1,
        )
        _set_ee_eval_target(command_manager, action_manager, default_command)

        compliance_params = _get_ee_compliance_params(
            cfg, getattr(args, "ee_compliance_force_deadband", None)
        )
        _set_explicit_eval_stiffness(
            getattr(args, "ee_compliance_stiffness", None), compliance_params, command_manager
        )
        _refresh_moe_stiffness_observation(td_, command_manager)
        use_command_manager_target = (
            getattr(command_manager, "external_force_mode", "legacy") == "net_pull"
            and hasattr(command_manager, "get_net_pull_ee_compliance_target_b")
        )
        compliance_eval_info = _get_ee_compliance_eval_info(
            command_manager, compliance_params, use_command_manager_target=use_command_manager_target
        )
        mean_window_steps = min(_mean_window_steps(base_env), EE_COMPLIANCE_HOLD_STEPS)
        baseline_window_steps = min(mean_window_steps, EE_COMPLIANCE_BASELINE_STEPS)
        directions = [
            (name, torch.tensor(vec, dtype=torch.float32, device=base_env.device))
            for name, vec in EE_COMPLIANCE_FORCE_DIRECTIONS
        ]
        case_specs = [
            {"direction_name": name, "direction_b": direction_b, "force_n": magnitude}
            for name, direction_b in directions
            for magnitude in EE_COMPLIANCE_FORCE_MAGNITUDES
        ]
        records = []
        batch_size = min(max(1, int(getattr(args, "ee_compliance_num_envs", 1))), base_env.num_envs)

        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            command_manager.clear_eval_ee_force()
            for _ in range(EE_TRACKING_WARMUP_STEPS):
                _, td_ = _rollout_one_step(env, rollout_policy, td_, "bimanual compliance warmup")

            for batch_start in range(0, len(case_specs), batch_size):
                batch_cases = case_specs[batch_start: batch_start + batch_size]
                active_envs = len(batch_cases)
                env_ids = torch.arange(active_envs, device=base_env.device)
                td_ = env.reset()
                _disable_eval_timer_reset(base_env)
                _restore_explicit_eval_stiffness(getattr(args, "ee_compliance_stiffness", None), command_manager)
                _refresh_moe_stiffness_observation(td_, command_manager)
                _set_static_root_command(command_manager, asset)
                _set_default_feet_command(command_manager, asset)
                _set_ee_eval_target(command_manager, action_manager, default_command)

                for _ in range(EE_TRACKING_WARMUP_STEPS):
                    _, td_ = _rollout_one_step(env, rollout_policy, td_, "bimanual compliance warmup")

                baseline_pos_samples = []
                for step in range(EE_COMPLIANCE_BASELINE_STEPS):
                    command_manager.clear_eval_ee_force()
                    _, td_ = _rollout_one_step(env, rollout_policy, td_, "bimanual compliance baseline")
                    if step >= EE_COMPLIANCE_BASELINE_STEPS - baseline_window_steps:
                        baseline_pos_samples.append(_body_pose_in_root_frame(asset, body_ids)[0])
                baseline_pos_b = _mean_tensor_samples(baseline_pos_samples)

                for ramp_step in range(EE_COMPLIANCE_RAMP_STEPS):
                    force = torch.stack([
                        case["direction_b"] * case["force_n"] * float(ramp_step + 1) / EE_COMPLIANCE_RAMP_STEPS
                        for case in batch_cases
                    ], dim=0)
                    command_manager.set_eval_bimanual_ee_force_b(
                        force.unsqueeze(1).expand(-1, 2, -1), env_ids=env_ids
                    )
                    _, td_ = _rollout_one_step(env, rollout_policy, td_, "bimanual compliance ramp")

                pos_samples = [[] for _ in batch_cases]
                target_samples = [[] for _ in batch_cases]
                force_samples = [[] for _ in batch_cases]
                for hold_step in range(EE_COMPLIANCE_HOLD_STEPS):
                    force = torch.stack([
                        case["direction_b"] * case["force_n"] for case in batch_cases
                    ], dim=0)
                    command_manager.set_eval_bimanual_ee_force_b(
                        force.unsqueeze(1).expand(-1, 2, -1), env_ids=env_ids
                    )
                    _, td_ = _rollout_one_step(env, rollout_policy, td_, "bimanual compliance hold")
                    if hold_step >= EE_COMPLIANCE_HOLD_STEPS - mean_window_steps:
                        actual_pos_b = _body_pose_in_root_frame(asset, body_ids)[0]
                        for env_i, case in enumerate(batch_cases):
                            target_b, _, force_b = _compute_ee_compliance_target_b(
                                command_manager, asset, body_ids, sample_center_b,
                                compliance_params, env_ids=env_ids[env_i:env_i + 1],
                                use_command_manager_target=use_command_manager_target,
                            )
                            pos_samples[env_i].append(actual_pos_b[env_ids[env_i]:env_ids[env_i] + 1])
                            target_samples[env_i].append(target_b)
                            force_samples[env_i].append(force_b)

                active_stiffness = command_manager.get_net_pull_ee_compliance_stiffness()[0].detach().cpu()
                for env_i, case in enumerate(batch_cases):
                    actual_pos_b = _mean_tensor_samples(pos_samples[env_i])
                    target_b = _mean_tensor_samples(target_samples[env_i])
                    force_b = _mean_tensor_samples(force_samples[env_i])
                    nominal_delta_b = actual_pos_b - sample_center_b[env_ids[env_i]:env_ids[env_i] + 1]
                    compliance_delta_b = actual_pos_b - target_b
                    nominal_error = nominal_delta_b.norm(dim=-1)[0]
                    compliance_error = compliance_delta_b.norm(dim=-1)[0]
                    measured = []
                    for ee_i in range(2):
                        displacement = ((actual_pos_b[0, ee_i] - baseline_pos_b[env_ids[env_i], ee_i]) * case["direction_b"]).sum()
                        measured.append(case["force_n"] / abs(float(displacement)) if abs(float(displacement)) > 1e-5 else float("nan"))
                    records.append({
                        "direction": case["direction_name"],
                        "force_n": case["force_n"],
                        "nominal_position_error_m": nominal_error.tolist(),
                        "compliance_position_error_m": compliance_error.tolist(),
                        "measured_stiffness_abs_n_per_m": measured,
                        "target_stiffness_n_per_m": active_stiffness.tolist(),
                        "ee_force_b": force_b[0].tolist(),
                    })
                    print(
                        f"both {case['direction_name']:>2s} {case['force_n']:>4.0f}N [env {batch_start + env_i:02d}]: "
                        f"left_comp={compliance_error[0].item():.4f} m, right_comp={compliance_error[1].item():.4f} m, "
                        f"k_left={measured[0]:.1f}, k_right={measured[1]:.1f} N/m",
                        flush=True,
                    )

        if args.ee_output is None:
            args.ee_output = _default_ee_report_path(args, "ee_bimanual_compliance_eval")
        os.makedirs(os.path.dirname(args.ee_output) or ".", exist_ok=True)
        # Write the raw records first so a failure in the summary below does
        # not lose the whole sweep; the full report overwrites this file.
        with open(args.ee_output, "w", encoding="utf-8") as f:
            json.dump(
                {"mode": "bimanual_ee_compliance", "policy_source": _policy_source_label(args), "records": records},
                f,
                indent=2,
            )

        nominal = torch.tensor([record["nominal_position_error_m"] for record in records])
        compliance = torch.tensor([record["compliance_position_error_m"] for record in records])
        measured = torch.tensor([record["measured_stiffness_abs_n_per_m"] for record in records])
        target = torch.tensor([record["target_stiffness_n_per_m"] for record in records])
        axis_names = ("x", "y", "z")
        measured_xyz_summary = {"left": {}, "right": {}}
        stiffness_error_xyz_summary = {"left": {}, "right": {}}
        stiffness_mae_xyz = {"left": {}, "right": {}}
        for ee_i, ee_name in enumerate(("left", "right")):
            all_abs_errors = []
            for axis_i, axis_name in enumerate(axis_names):
                axis_mask = torch.tensor(
                    [record["direction"].endswith(axis_name) for record in records],
                    dtype=torch.bool,
                )
                axis_measured = measured[axis_mask, ee_i]
                axis_target = target[axis_mask, ee_i, axis_i]
                axis_error = axis_measured - axis_target
                axis_abs_error = axis_error.abs()
                measured_xyz_summary[ee_name][axis_name] = _summary(axis_measured)
                stiffness_error_xyz_summary[ee_name][axis_name] = _summary(axis_error)
                stiffness_mae_xyz[ee_name][axis_name] = float(axis_abs_error.mean().item())
                all_abs_errors.append(axis_abs_error)
            all_abs_errors = torch.cat(all_abs_errors)
            stiffness_mae_xyz[ee_name]["overall"] = float(all_abs_errors.mean().item())
        report = {
            "mode": "bimanual_ee_compliance",
            "policy_source": _policy_source_label(args),
            "summary": {
                "nominal_position_error_m": {"left": _summary(nominal[:, 0]), "right": _summary(nominal[:, 1]), "combined": _summary(nominal)},
                "compliance_position_error_m": {"left": _summary(compliance[:, 0]), "right": _summary(compliance[:, 1]), "combined": _summary(compliance)},
                "measured_stiffness_abs_n_per_m": {"left": _summary(measured[:, 0]), "right": _summary(measured[:, 1]), "combined": _summary(measured)},
                "measured_stiffness_abs_n_per_m_xyz": measured_xyz_summary,
                "stiffness_error_n_per_m_xyz": stiffness_error_xyz_summary,
                "stiffness_mae_n_per_m_xyz": stiffness_mae_xyz,
                "target_stiffness_n_per_m": {"left": target[0, 0].tolist(), "right": target[0, 1].tolist()},
            },
            "records": records,
        }
        with open(args.ee_output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("\n" + "=" * 60)
        print("BIMANUAL EE COMPLIANCE EVAL")
        print("=" * 60)
        for name, values in (("Nominal", nominal), ("Compliance", compliance), ("Measured stiffness", measured)):
            print(
                f"  {name} left mean±std: {values[:, 0].mean():.4f} ± {values[:, 0].std(unbiased=False):.4f}; "
                f"right mean±std: {values[:, 1].mean():.4f} ± {values[:, 1].std(unbiased=False):.4f}"
            )
        for ee_name in ("left", "right"):
            print(
                f"  Measured stiffness {ee_name} xyz mean±std: "
                + ", ".join(
                    f"{axis}={measured_xyz_summary[ee_name][axis]['mean']:.1f} ± "
                    f"{measured_xyz_summary[ee_name][axis]['std']:.1f}"
                    for axis in axis_names
                )
                + " N/m"
            )
            print(
                f"  Stiffness MAE {ee_name} xyz/overall: "
                + ", ".join(
                    f"{axis}={stiffness_mae_xyz[ee_name][axis]:.1f}"
                    for axis in axis_names
                )
                + f", overall={stiffness_mae_xyz[ee_name]['overall']:.1f} N/m"
            )
        print(f"  Target stiffness left/right: {target[0, 0].tolist()} / {target[0, 1].tolist()} N/m")
        print(f"  Report: {args.ee_output}")
        print("=" * 60 + "\n")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def evaluate_ee_compliance(cfg, args):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app
    env = None

    try:
        env, policy, _vecnorm, _ = make_env_policy(cfg)
        rollout_policy = policy.get_rollout_policy("eval")
        base_env = env.base_env if hasattr(env, "base_env") else env
        asset = base_env.scene["robot"]
        command_manager = base_env.command_manager
        action_manager = base_env.action_manager
        _set_external_force(command_manager, True)

        if not hasattr(command_manager, "set_eval_ee_force_b"):
            raise RuntimeError(
                "EE compliance eval needs MotionTrackingCommand_impedance with set_eval_ee_force_b()."
            )

        body_names = [name.strip() for name in EE_TRACKING_BODY_NAMES.split(",")]
        try:
            body_ids = [asset.body_names.index(name) for name in body_names]
        except ValueError as exc:
            raise RuntimeError(
                f"Cannot find EE body name from {body_names}. Available bodies: {asset.body_names}"
            ) from exc
        joint_names = [str(name) for name in asset.joint_names]

        td_ = env.reset()
        _disable_eval_timer_reset(base_env)
        _set_static_root_command(command_manager, asset)
        print("Static root command enabled for EE compliance eval.", flush=True)
        if _set_default_feet_command(command_manager, asset):
            print("Default feet command enabled for EE compliance eval.", flush=True)

        default_pos_b, default_quat_b = _body_pose_in_root_frame(asset, body_ids)
        sample_center_b = _get_ee_sample_center(
            default_pos_b,
            base_env.device,
            getattr(args, "ee_compliance_initial_position", None),
        )
        target_quat_b = torch.zeros_like(default_quat_b)
        target_quat_b[..., 0] = 1.0
        target_axis_angle_b = torch.zeros(base_env.num_envs, 2, 3, device=base_env.device)
        target_rpy_b = torch.zeros(base_env.num_envs, 2, 3, device=base_env.device)
        default_command = torch.cat(
            [sample_center_b.reshape(base_env.num_envs, 6), target_axis_angle_b.reshape(base_env.num_envs, 6)],
            dim=-1,
        )
        _set_ee_eval_target(command_manager, action_manager, default_command)

        compliance_params = _get_ee_compliance_params(
            cfg, getattr(args, "ee_compliance_force_deadband", None)
        )
        _prompt_low_level_nominal_stiffness(
            compliance_params,
            command_manager,
            action_manager,
            getattr(args, "ee_compliance_stiffness", None),
        )
        _set_explicit_eval_stiffness(
            getattr(args, "ee_compliance_stiffness", None),
            compliance_params,
            command_manager,
        )
        _refresh_moe_stiffness_observation(td_, command_manager)
        force_estimator_ablation = bool(getattr(args, "ee_compliance_force_estimator_ablation", False))
        oracle_force_baseline = bool(getattr(args, "ee_compliance_oracle_force", False))
        if force_estimator_ablation and oracle_force_baseline:
            raise RuntimeError(
                "--ee_compliance_force_estimator_ablation and "
                "--ee_compliance_oracle_force are mutually exclusive."
            )
        if force_estimator_ablation and not _is_hierarchical_action_manager(action_manager):
            raise RuntimeError(
                "--ee_compliance_force_estimator_ablation is only supported for hierarchical policies; "
                "low-level-only EE compliance eval is intentionally not handled."
            )
        if oracle_force_baseline and _is_hierarchical_action_manager(action_manager):
            raise RuntimeError(
                "--ee_compliance_oracle_force is intended for the raw stiff low-level policy. "
                "Use the 3kp stiff checkpoint/run directly; it must not load a high-level policy."
            )
        if force_estimator_ablation:
            action_manager.clear_ee_force_stiffness_ablation_command()
        use_command_manager_compliance_target = (
            not force_estimator_ablation
            and not oracle_force_baseline
            and getattr(command_manager, "external_force_mode", "legacy") == "net_pull"
            and hasattr(command_manager, "get_net_pull_ee_compliance_target_b")
        )
        compliance_eval_info = _get_ee_compliance_eval_info(
            command_manager,
            compliance_params,
            use_command_manager_target=use_command_manager_compliance_target,
        )
        if force_estimator_ablation:
            compliance_eval_info["target_mode"] = "force_estimator_over_cfg_stiffness_ablation"
            compliance_eval_info["actual_stiffness"] = compliance_params["stiffness"]
        elif oracle_force_baseline:
            compliance_eval_info["target_mode"] = "oracle_force_over_cfg_stiffness"
            compliance_eval_info["actual_stiffness"] = compliance_params["stiffness"]
        mean_window_steps = min(_mean_window_steps(base_env), EE_COMPLIANCE_HOLD_STEPS)
        baseline_window_steps = min(mean_window_steps, EE_COMPLIANCE_BASELINE_STEPS)

        def force_estimator_ablation_after_policy(policy_td):
            if not force_estimator_ablation:
                return
            _set_ee_force_stiffness_ablation_from_policy_td(
                action_manager,
                command_manager,
                policy_td,
                default_command,
                sample_center_b,
                compliance_params,
            )

        def set_oracle_target(oracle_force_b=None):
            if not oracle_force_baseline:
                return
            if oracle_force_b is None:
                oracle_force_b = torch.zeros_like(sample_center_b)
            _set_oracle_force_stiffness_target(
                command_manager,
                action_manager,
                default_command,
                sample_center_b,
                oracle_force_b,
                compliance_params,
            )

        print(
            "EE compliance eval target is written as "
            + (
                "force-estimator/stiffness low-level EE command override."
                if force_estimator_ablation
                else (
                    "oracle ground-truth-force/stiffness low-level EE command override."
                    if oracle_force_baseline
                    else ("high-level EE reference override." if _is_hierarchical_action_manager(action_manager) else "low-level EE command override.")
                )
            ),
            flush=True,
        )
        print(
            "EE compliance target mode: "
            f"{compliance_eval_info['target_mode']}; "
            f"actual stiffness={_format_scalar_or_xyz(compliance_eval_info['actual_stiffness']) if compliance_eval_info['actual_stiffness'] is not None else 'unavailable'}; "
            f"max_offset={_format_scalar_or_xyz(compliance_params['max_offset'])}; "
            f"force_deadband={compliance_params['force_deadband']}",
            flush=True,
        )
        if force_estimator_ablation:
            print(
                "EE compliance force-estimator ablation enabled: high-level EE delta is bypassed; "
                "low-level EE command position = nominal + direct_priv_pred_force_b / cfg stiffness.",
                flush=True,
            )
        if oracle_force_baseline:
            print(
                "EE compliance oracle-force baseline enabled: learned force estimation and high-level "
                "EE deltas are bypassed; low-level EE command position = nominal + applied_force_b / cfg stiffness.",
                flush=True,
            )
        print(
            f"Starting EE compliance sweep... mean_window_steps={mean_window_steps}, "
            f"baseline_window_steps={baseline_window_steps}",
            flush=True,
        )

        directions = [
            (name, torch.tensor(vec, dtype=torch.float32, device=base_env.device))
            for name, vec in EE_COMPLIANCE_FORCE_DIRECTIONS
        ]
        case_specs = []
        for ee_i, ee_name in enumerate(["left", "right"]):
            for direction_name, direction_b in directions:
                for magnitude in EE_COMPLIANCE_FORCE_MAGNITUDES:
                    case_specs.append({
                        "ee_i": ee_i,
                        "ee_name": ee_name,
                        "direction_name": direction_name,
                        "direction_b": direction_b,
                        "force_n": magnitude,
                    })
        records = []

        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            command_manager.clear_eval_ee_force()
            set_oracle_target()
            for _ in range(EE_TRACKING_WARMUP_STEPS):
                _, td_ = _rollout_one_step(
                    env,
                    rollout_policy,
                    td_,
                    "compliance warmup",
                    after_policy=force_estimator_ablation_after_policy,
                )

            baseline_pos_samples = []
            baseline_quat_samples = []
            baseline_compliance_target_samples = []
            for step in range(EE_COMPLIANCE_BASELINE_STEPS):
                command_manager.clear_eval_ee_force()
                set_oracle_target()
                _, td_ = _rollout_one_step(
                    env,
                    rollout_policy,
                    td_,
                    "compliance baseline",
                    after_policy=force_estimator_ablation_after_policy,
                )
                if step >= EE_COMPLIANCE_BASELINE_STEPS - baseline_window_steps:
                    sample_pos_b, sample_quat_b = _body_pose_in_root_frame(asset, body_ids)
                    sample_compliance_target_b, _, _ = _compute_ee_compliance_target_b(
                        command_manager,
                        asset,
                        body_ids,
                        sample_center_b,
                        compliance_params,
                        use_command_manager_target=use_command_manager_compliance_target,
                    )
                    baseline_pos_samples.append(sample_pos_b)
                    baseline_quat_samples.append(sample_quat_b)
                    baseline_compliance_target_samples.append(sample_compliance_target_b)

            baseline_pos_b, baseline_quat_b = _mean_pose_samples(baseline_pos_samples, baseline_quat_samples)
            baseline_compliance_target_b = _mean_tensor_samples(baseline_compliance_target_samples)
            baseline_nominal_error = (baseline_pos_b - sample_center_b).norm(dim=-1)
            baseline_compliance_error = (baseline_pos_b - baseline_compliance_target_b).norm(dim=-1)
            print(
                "Baseline no-force EE error "
                f"left={baseline_nominal_error[:, 0].mean().item():.4f} m, "
                f"right={baseline_nominal_error[:, 1].mean().item():.4f} m",
                flush=True,
            )

            batch_size = min(max(1, int(getattr(args, "ee_compliance_num_envs", 1))), base_env.num_envs)
            for batch_start in range(0, len(case_specs), batch_size):
                batch_cases = case_specs[batch_start: batch_start + batch_size]
                active_envs = len(batch_cases)
                env_ids = torch.arange(active_envs, device=base_env.device)
                command_manager.clear_eval_ee_force()
                td_ = env.reset()
                _disable_eval_timer_reset(base_env)
                # Range-trained low-level tasks resample stiffness in
                # sample_init(). Restore the value entered for this eval
                # after every batch reset so policy input and target K agree.
                _restore_low_level_eval_stiffness(compliance_params, command_manager, action_manager)
                # Range-trained high-level tasks also resample stiffness on
                # reset. Reapply an explicit eval value so the policy input,
                # compliance target, and report all use the same K.
                _restore_explicit_eval_stiffness(
                    getattr(args, "ee_compliance_stiffness", None),
                    command_manager,
                )
                _refresh_moe_stiffness_observation(td_, command_manager)
                _set_static_root_command(command_manager, asset)
                if _set_default_feet_command(command_manager, asset):
                    pass
                _set_ee_eval_target(command_manager, action_manager, default_command)
                set_oracle_target()

                for _ in range(EE_TRACKING_WARMUP_STEPS):
                    set_oracle_target()
                    _, td_ = _rollout_one_step(
                        env,
                        rollout_policy,
                        td_,
                        "compliance warmup",
                        after_policy=force_estimator_ablation_after_policy,
                    )

                baseline_pos_samples = []
                baseline_quat_samples = []
                baseline_compliance_target_samples = []
                baseline_joint_state_samples = []
                baseline_ee_jacobian_samples = []
                baseline_ee_spatial_jacobian_samples = []
                for step in range(EE_COMPLIANCE_BASELINE_STEPS):
                    command_manager.clear_eval_ee_force()
                    set_oracle_target()
                    _, td_ = _rollout_one_step(
                        env,
                        rollout_policy,
                        td_,
                        "compliance baseline",
                        after_policy=force_estimator_ablation_after_policy,
                    )
                    if step >= EE_COMPLIANCE_BASELINE_STEPS - baseline_window_steps:
                        sample_pos_b, sample_quat_b = _body_pose_in_root_frame(asset, body_ids)
                        sample_compliance_target_b, _, _ = _compute_ee_compliance_target_b(
                            command_manager,
                            asset,
                            body_ids,
                            sample_center_b,
                            compliance_params,
                            use_command_manager_target=use_command_manager_compliance_target,
                        )
                        baseline_pos_samples.append(sample_pos_b)
                        baseline_quat_samples.append(sample_quat_b)
                        baseline_compliance_target_samples.append(sample_compliance_target_b)
                        baseline_joint_state_samples.append(
                            _capture_joint_motion_state(asset, env_ids)
                        )
                        baseline_ee_jacobian_samples.append(
                            _capture_ee_jacobian_root(asset, body_ids, env_ids)
                        )
                        baseline_ee_spatial_jacobian_samples.append(
                            _capture_ee_jacobian_root(asset, body_ids, env_ids, spatial=True)
                        )

                baseline_pos_b, baseline_quat_b = _mean_pose_samples(baseline_pos_samples, baseline_quat_samples)
                baseline_compliance_target_b = _mean_tensor_samples(baseline_compliance_target_samples)
                baseline_nominal_error = (baseline_pos_b - sample_center_b).norm(dim=-1)
                baseline_compliance_error = (baseline_pos_b - baseline_compliance_target_b).norm(dim=-1)
                baseline_joint_pos_window = _stack_motion_state_samples(
                    baseline_joint_state_samples, "joint_pos"
                )
                baseline_joint_vel_window = _stack_motion_state_samples(
                    baseline_joint_state_samples, "joint_vel"
                )
                baseline_joint_target_window = _stack_motion_state_samples(
                    baseline_joint_state_samples, "joint_pos_target"
                )
                baseline_applied_torque_window = _stack_motion_state_samples(
                    baseline_joint_state_samples, "applied_torque"
                )
                baseline_root_pos_window = _stack_motion_state_samples(
                    baseline_joint_state_samples, "root_pos_w"
                )
                baseline_root_quat_window = _stack_motion_state_samples(
                    baseline_joint_state_samples, "root_quat_w"
                )
                baseline_root_lin_vel_window = _stack_motion_state_samples(
                    baseline_joint_state_samples, "root_lin_vel_w"
                )
                baseline_root_ang_vel_window = _stack_motion_state_samples(
                    baseline_joint_state_samples, "root_ang_vel_w"
                )
                baseline_ee_jacobian_window = (
                    torch.stack(
                        [sample for sample in baseline_ee_jacobian_samples if sample is not None],
                        dim=0,
                    )
                    if any(sample is not None for sample in baseline_ee_jacobian_samples)
                    else None
                )
                baseline_ee_spatial_jacobian_window = (
                    torch.stack(
                        [sample for sample in baseline_ee_spatial_jacobian_samples if sample is not None],
                        dim=0,
                    )
                    if any(sample is not None for sample in baseline_ee_spatial_jacobian_samples)
                    else None
                )

                for _ in range(EE_COMPLIANCE_RECOVERY_STEPS):
                    command_manager.clear_eval_ee_force()
                    set_oracle_target()
                    _, td_ = _rollout_one_step(
                        env,
                        rollout_policy,
                        td_,
                        "compliance recovery",
                        after_policy=force_estimator_ablation_after_policy,
                    )

                ramp_joint_state_samples = []
                for ramp_step in range(EE_COMPLIANCE_RAMP_STEPS):
                    ramp_forces = torch.zeros(active_envs, 3, device=base_env.device)
                    for env_i, case in enumerate(batch_cases):
                        ramp_forces[env_i] = case["direction_b"] * (
                            case["force_n"] * float(ramp_step + 1) / float(EE_COMPLIANCE_RAMP_STEPS)
                        )
                    for ee_i in (0, 1):
                        mask = torch.tensor([case["ee_i"] == ee_i for case in batch_cases], device=base_env.device, dtype=torch.bool)
                        if mask.any():
                            force_b = ramp_forces[mask]
                            command_manager.set_eval_ee_force_b(ee_i, force_b, env_ids=env_ids[mask])
                    if oracle_force_baseline:
                        oracle_force_b = torch.zeros_like(sample_center_b)
                        for env_i, case in enumerate(batch_cases):
                            oracle_force_b[env_ids[env_i], case["ee_i"]] = ramp_forces[env_i]
                        set_oracle_target(oracle_force_b)
                    _, td_ = _rollout_one_step(
                        env,
                        rollout_policy,
                        td_,
                        "compliance ramp",
                        after_policy=force_estimator_ablation_after_policy,
                    )
                    ramp_joint_state_samples.append(
                        _capture_joint_motion_state(asset, env_ids)
                    )

                ramp_joint_pos_window = _stack_motion_state_samples(
                    ramp_joint_state_samples, "joint_pos"
                )
                ramp_joint_vel_window = _stack_motion_state_samples(
                    ramp_joint_state_samples, "joint_vel"
                )
                ramp_root_pos_window = _stack_motion_state_samples(
                    ramp_joint_state_samples, "root_pos_w"
                )

                pos_samples = [[] for _ in range(active_envs)]
                quat_samples = [[] for _ in range(active_envs)]
                compliance_target_samples = [[] for _ in range(active_envs)]
                compliance_offset_samples = [[] for _ in range(active_envs)]
                force_samples = [[] for _ in range(active_envs)]
                force_pred_samples = [[] for _ in range(active_envs)]
                hl_command_samples = [[] for _ in range(active_envs)]
                moe_gate_samples = [[] for _ in range(active_envs)]
                force_joint_state_samples = []
                force_ee_jacobian_samples = []
                force_ee_spatial_jacobian_samples = []
                full_forces = [
                    case["direction_b"] * case["force_n"]
                    for case in batch_cases
                ]
                for hold_step in range(EE_COMPLIANCE_HOLD_STEPS):
                    for ee_i in (0, 1):
                        mask = torch.tensor([case["ee_i"] == ee_i for case in batch_cases], device=base_env.device, dtype=torch.bool)
                        if mask.any():
                            force_b = torch.stack([full_forces[i] for i, m in enumerate(mask.tolist()) if m], dim=0)
                            command_manager.set_eval_ee_force_b(ee_i, force_b, env_ids=env_ids[mask])
                    if oracle_force_baseline:
                        oracle_force_b = torch.zeros_like(sample_center_b)
                        for env_i, case in enumerate(batch_cases):
                            oracle_force_b[env_ids[env_i], case["ee_i"]] = full_forces[env_i]
                        set_oracle_target(oracle_force_b)
                    _, td_, policy_td = _rollout_one_step(
                        env,
                        rollout_policy,
                        td_,
                        "compliance hold",
                        return_policy_td=True,
                        after_policy=force_estimator_ablation_after_policy,
                    )
                    if hold_step >= EE_COMPLIANCE_HOLD_STEPS - mean_window_steps:
                        direct_force_pred_b = _direct_force_pred_b_from_tensordict(policy_td, command_manager)
                        sample_pos_b, sample_quat_b = _body_pose_in_root_frame(asset, body_ids)
                        force_joint_state_samples.append(
                            _capture_joint_motion_state(asset, env_ids)
                        )
                        force_ee_jacobian_samples.append(
                            _capture_ee_jacobian_root(asset, body_ids, env_ids)
                        )
                        force_ee_spatial_jacobian_samples.append(
                            _capture_ee_jacobian_root(asset, body_ids, env_ids, spatial=True)
                        )
                        sample_hl_command_pos_b = _get_hl_ee_command_pos_b(action_manager, env_ids=env_ids)
                        for env_i, case in enumerate(batch_cases):
                            sample_compliance_target_b, sample_compliance_offset_b, sample_force_b = _compute_ee_compliance_target_b(
                                command_manager,
                                asset,
                                body_ids,
                                sample_center_b,
                                compliance_params,
                                env_ids=env_ids[env_i:env_i + 1],
                                use_command_manager_target=use_command_manager_compliance_target,
                            )
                            pos_samples[env_i].append(sample_pos_b[env_ids[env_i]:env_ids[env_i]+1])
                            quat_samples[env_i].append(sample_quat_b[env_ids[env_i]:env_ids[env_i]+1])
                            compliance_target_samples[env_i].append(sample_compliance_target_b)
                            compliance_offset_samples[env_i].append(sample_compliance_offset_b)
                            force_samples[env_i].append(sample_force_b)
                            if direct_force_pred_b is not None:
                                force_pred_samples[env_i].append(
                                    direct_force_pred_b[env_ids[env_i]:env_ids[env_i] + 1]
                                )
                            if "moe_gate_weights" in policy_td.keys():
                                moe_gate_samples[env_i].append(
                                    policy_td["moe_gate_weights"][env_ids[env_i]:env_ids[env_i] + 1]
                                )
                            if sample_hl_command_pos_b is not None:
                                hl_command_samples[env_i].append(sample_hl_command_pos_b[env_i:env_i + 1])

                force_joint_pos_window = _stack_motion_state_samples(
                    force_joint_state_samples, "joint_pos"
                )
                force_joint_vel_window = _stack_motion_state_samples(
                    force_joint_state_samples, "joint_vel"
                )
                force_joint_target_window = _stack_motion_state_samples(
                    force_joint_state_samples, "joint_pos_target"
                )
                force_applied_torque_window = _stack_motion_state_samples(
                    force_joint_state_samples, "applied_torque"
                )
                force_root_pos_window = _stack_motion_state_samples(
                    force_joint_state_samples, "root_pos_w"
                )
                force_root_quat_window = _stack_motion_state_samples(
                    force_joint_state_samples, "root_quat_w"
                )
                force_root_lin_vel_window = _stack_motion_state_samples(
                    force_joint_state_samples, "root_lin_vel_w"
                )
                force_root_ang_vel_window = _stack_motion_state_samples(
                    force_joint_state_samples, "root_ang_vel_w"
                )
                force_ee_jacobian_window = (
                    torch.stack(
                        [sample for sample in force_ee_jacobian_samples if sample is not None],
                        dim=0,
                    )
                    if any(sample is not None for sample in force_ee_jacobian_samples)
                    else None
                )
                force_ee_spatial_jacobian_window = (
                    torch.stack(
                        [sample for sample in force_ee_spatial_jacobian_samples if sample is not None],
                        dim=0,
                    )
                    if any(sample is not None for sample in force_ee_spatial_jacobian_samples)
                    else None
                )

                for env_i, case in enumerate(batch_cases):
                    actual_pos_b, actual_quat_b = _mean_pose_samples(pos_samples[env_i], quat_samples[env_i])
                    compliance_target_b = _mean_tensor_samples(compliance_target_samples[env_i])
                    compliance_offset_b = _mean_tensor_samples(compliance_offset_samples[env_i])
                    force_b = _mean_tensor_samples(force_samples[env_i])
                    force_pred_b = (
                        _mean_tensor_samples(force_pred_samples[env_i])
                        if force_pred_samples[env_i]
                        else None
                    )
                    force_pred_error_b = force_pred_b - force_b[:, case["ee_i"]] if force_pred_b is not None else None
                    force_pred_error_norm = force_pred_error_b.norm(dim=-1) if force_pred_error_b is not None else None
                    force_pred_actual_norm = force_b[:, case["ee_i"]].norm(dim=-1)
                    force_pred_norm = force_pred_b.norm(dim=-1) if force_pred_b is not None else None
                    hl_command_pos_b = (
                        _mean_tensor_samples(hl_command_samples[env_i])
                        if hl_command_samples[env_i]
                        else None
                    )
                    moe_gate_weights = (
                        _mean_tensor_samples(moe_gate_samples[env_i])
                        if moe_gate_samples[env_i]
                        else None
                    )
                    nominal_error = (actual_pos_b - sample_center_b[env_ids[env_i]:env_ids[env_i]+1]).norm(dim=-1)
                    compliance_error = (actual_pos_b - compliance_target_b).norm(dim=-1)
                    nominal_delta_b = actual_pos_b - sample_center_b[env_ids[env_i]:env_ids[env_i]+1]
                    compliance_delta_b = actual_pos_b - compliance_target_b
                    if hl_command_pos_b is not None:
                        hl_command_nominal_delta_b = hl_command_pos_b - sample_center_b[env_ids[env_i]:env_ids[env_i]+1]
                        hl_command_compliance_delta_b = hl_command_pos_b - compliance_target_b
                        actual_to_hl_command_delta_b = actual_pos_b - hl_command_pos_b
                        actual_to_hl_command_error = actual_to_hl_command_delta_b.norm(dim=-1)
                        hl_command_to_compliance_error = hl_command_compliance_delta_b.norm(dim=-1)
                        selected_hl_command_delta_b = hl_command_nominal_delta_b[:, case["ee_i"]]
                        selected_hl_command_along = (selected_hl_command_delta_b * case["direction_b"]).sum(dim=-1)
                        hl_command_along_mean = float(selected_hl_command_along.mean().item())
                    else:
                        hl_command_nominal_delta_b = None
                        hl_command_compliance_delta_b = None
                        actual_to_hl_command_delta_b = None
                        actual_to_hl_command_error = None
                        hl_command_to_compliance_error = None
                        selected_hl_command_along = None
                        hl_command_along_mean = None
                    selected_delta_b = actual_pos_b[:, case["ee_i"]] - baseline_pos_b[env_ids[env_i], case["ee_i"]]
                    selected_along = (selected_delta_b * case["direction_b"]).sum(dim=-1)
                    selected_orth = (selected_delta_b - selected_along.unsqueeze(-1) * case["direction_b"]).norm(dim=-1)
                    along_mean = float(selected_along.mean().item())
                    measured_stiffness_signed = None
                    measured_stiffness_abs = None
                    if abs(along_mean) > 1e-5:
                        measured_stiffness_signed = float(case["force_n"] / along_mean)
                        measured_stiffness_abs = float(case["force_n"] / abs(along_mean))
                    hl_command_measured_stiffness_signed = None
                    hl_command_measured_stiffness_abs = None
                    if hl_command_along_mean is not None and abs(hl_command_along_mean) > 1e-5:
                        hl_command_measured_stiffness_signed = float(case["force_n"] / hl_command_along_mean)
                        hl_command_measured_stiffness_abs = float(case["force_n"] / abs(hl_command_along_mean))

                    rpy_error = torch.rad2deg(_wrap_to_pi(_quat_to_rpy_wxyz(actual_quat_b) - target_rpy_b[env_ids[env_i]:env_ids[env_i]+1]).abs())
                    quat_error = _quat_angle_error_deg(actual_quat_b, target_quat_b[env_ids[env_i]:env_ids[env_i]+1])

                    joint_motion = {}
                    if baseline_ee_jacobian_window is not None:
                        baseline_jacobian_mean = baseline_ee_jacobian_window[:, env_i].mean(dim=0)
                        force_jacobian_mean = (
                            force_ee_jacobian_window[:, env_i].mean(dim=0)
                            if force_ee_jacobian_window is not None
                            else None
                        )
                        selected_baseline_jacobian = baseline_jacobian_mean[case["ee_i"]]
                        selected_force_jacobian = (
                            force_jacobian_mean[case["ee_i"]]
                            if force_jacobian_mean is not None
                            else None
                        )
                        baseline_spatial_jacobian_mean = (
                            baseline_ee_spatial_jacobian_window[:, env_i].mean(dim=0)
                            if baseline_ee_spatial_jacobian_window is not None
                            else None
                        )
                        force_spatial_jacobian_mean = (
                            force_ee_spatial_jacobian_window[:, env_i].mean(dim=0)
                            if force_ee_spatial_jacobian_window is not None
                            else None
                        )
                        selected_baseline_spatial_jacobian = (
                            baseline_spatial_jacobian_mean[case["ee_i"]]
                            if baseline_spatial_jacobian_mean is not None
                            else None
                        )
                        selected_force_spatial_jacobian = (
                            force_spatial_jacobian_mean[case["ee_i"]]
                            if force_spatial_jacobian_mean is not None
                            else None
                        )
                        selected_force_root = force_b[0, case["ee_i"]]
                        selected_stiffness_xyz = _stiffness_xyz_for_ee(
                            compliance_eval_info["actual_stiffness"],
                            case["ee_i"],
                            selected_force_root.device,
                        )
                        ideal_ee_delta, joint_delta_ik, external_torque, jacobian_condition = (
                            _predict_joint_response_from_ee_compliance(
                                selected_baseline_jacobian,
                                selected_force_root,
                                selected_stiffness_xyz,
                            )
                        )
                        joint_motion.update({
                            "ee_jacobian_baseline_root_3xn": selected_baseline_jacobian.detach().cpu().tolist(),
                            "ee_jacobian_force_root_3xn": (
                                selected_force_jacobian.detach().cpu().tolist()
                                if selected_force_jacobian is not None
                                else None
                            ),
                            "ee_spatial_jacobian_baseline_root_6xn": (
                                selected_baseline_spatial_jacobian.detach().cpu().tolist()
                                if selected_baseline_spatial_jacobian is not None
                                else None
                            ),
                            "ee_spatial_jacobian_force_root_6xn": (
                                selected_force_spatial_jacobian.detach().cpu().tolist()
                                if selected_force_spatial_jacobian is not None
                                else None
                            ),
                            "jacobian_joint_count": int(selected_baseline_jacobian.shape[-1]),
                            "jacobian_condition_number": jacobian_condition,
                            "ee_delta_ideal_root_m": (
                                ideal_ee_delta.detach().cpu().tolist()
                                if ideal_ee_delta is not None
                                else None
                            ),
                            "joint_delta_q_ik_rad": (
                                joint_delta_ik.detach().cpu().tolist()
                                if joint_delta_ik is not None
                                else None
                            ),
                            "external_torque_jt_force_nm": (
                                external_torque.detach().cpu().tolist()
                                if external_torque is not None
                                else None
                            ),
                        })
                    if baseline_joint_pos_window is not None and force_joint_pos_window is not None:
                        baseline_joint_pos_samples = baseline_joint_pos_window[:, env_i]
                        force_joint_pos_samples = force_joint_pos_window[:, env_i]
                        baseline_joint_pos_mean = baseline_joint_pos_samples.mean(dim=0)
                        force_joint_pos_mean = force_joint_pos_samples.mean(dim=0)
                        joint_motion.update({
                            "joint_pos_baseline_rad": baseline_joint_pos_mean.detach().cpu().tolist(),
                            "joint_pos_force_rad": force_joint_pos_mean.detach().cpu().tolist(),
                            "joint_delta_pos_rad": (
                                force_joint_pos_mean - baseline_joint_pos_mean
                            ).detach().cpu().tolist(),
                            "joint_pos_baseline_std_rad": (
                                baseline_joint_pos_samples.std(dim=0, unbiased=False)
                            ).detach().cpu().tolist(),
                            "joint_pos_force_std_rad": (
                                force_joint_pos_samples.std(dim=0, unbiased=False)
                            ).detach().cpu().tolist(),
                            "joint_pos_baseline_samples_rad": baseline_joint_pos_samples.detach().cpu().tolist(),
                            "joint_pos_force_samples_rad": force_joint_pos_samples.detach().cpu().tolist(),
                        })
                        trajectory_joint_pos_samples = [
                            samples[:, env_i]
                            for samples in (ramp_joint_pos_window, force_joint_pos_window)
                            if samples is not None
                        ]
                        if trajectory_joint_pos_samples:
                            trajectory_joint_pos = torch.cat(trajectory_joint_pos_samples, dim=0)
                            joint_motion["joint_peak_abs_delta_rad"] = (
                                (trajectory_joint_pos - baseline_joint_pos_mean).abs().max(dim=0).values
                                .detach().cpu().tolist()
                            )
                        if ramp_joint_pos_window is not None:
                            joint_motion["joint_pos_ramp_samples_rad"] = (
                                ramp_joint_pos_window[:, env_i].detach().cpu().tolist()
                            )
                        if ramp_joint_vel_window is not None:
                            joint_motion["joint_vel_ramp_samples_rad_s"] = (
                                ramp_joint_vel_window[:, env_i].detach().cpu().tolist()
                            )
                        if baseline_joint_target_window is not None and force_joint_target_window is not None:
                            joint_motion.update({
                                "joint_pos_target_baseline_rad": (
                                    baseline_joint_target_window[:, env_i].mean(dim=0).detach().cpu().tolist()
                                ),
                                "joint_pos_target_force_rad": (
                                    force_joint_target_window[:, env_i].mean(dim=0).detach().cpu().tolist()
                                ),
                            })
                    if baseline_joint_vel_window is not None and force_joint_vel_window is not None:
                        baseline_joint_vel_samples = baseline_joint_vel_window[:, env_i]
                        force_joint_vel_samples = force_joint_vel_window[:, env_i]
                        joint_motion.update({
                            "joint_vel_baseline_rad_s_mean": baseline_joint_vel_samples.mean(dim=0).detach().cpu().tolist(),
                            "joint_vel_force_rad_s_mean": force_joint_vel_samples.mean(dim=0).detach().cpu().tolist(),
                            "joint_vel_force_rad_s_std": force_joint_vel_samples.std(dim=0, unbiased=False).detach().cpu().tolist(),
                            "joint_vel_force_samples_rad_s": force_joint_vel_samples.detach().cpu().tolist(),
                        })
                    if baseline_root_pos_window is not None and force_root_pos_window is not None:
                        baseline_root_pos_samples = baseline_root_pos_window[:, env_i]
                        force_root_pos_samples = force_root_pos_window[:, env_i]
                        baseline_root_pos_mean = baseline_root_pos_samples.mean(dim=0)
                        force_root_pos_mean = force_root_pos_samples.mean(dim=0)
                        joint_motion.update({
                            "root_pos_baseline_w_m": baseline_root_pos_mean.detach().cpu().tolist(),
                            "root_pos_force_w_m": force_root_pos_mean.detach().cpu().tolist(),
                            "root_delta_pos_w_m": (
                                force_root_pos_mean - baseline_root_pos_mean
                            ).detach().cpu().tolist(),
                            "root_delta_pos_norm_m": float(
                                (force_root_pos_mean - baseline_root_pos_mean).norm().item()
                            ),
                            "root_pos_baseline_samples_w_m": baseline_root_pos_samples.detach().cpu().tolist(),
                            "root_pos_force_samples_w_m": force_root_pos_samples.detach().cpu().tolist(),
                        })
                        if ramp_root_pos_window is not None:
                            ramp_root_pos_samples = ramp_root_pos_window[:, env_i]
                            root_trajectory = torch.cat(
                                [
                                    ramp_root_pos_samples,
                                    force_root_pos_samples,
                                ],
                                dim=0,
                            )
                            joint_motion.update({
                                "root_pos_ramp_samples_w_m": ramp_root_pos_samples.detach().cpu().tolist(),
                                "root_peak_delta_pos_norm_m": float(
                                    (root_trajectory - baseline_root_pos_mean).norm(dim=-1).max().item()
                                ),
                            })
                    if baseline_root_quat_window is not None and force_root_quat_window is not None:
                        baseline_root_quat_mean = normalize(
                            baseline_root_quat_window[:, env_i].mean(dim=0, keepdim=True)
                        )
                        force_root_quat_mean = normalize(
                            force_root_quat_window[:, env_i].mean(dim=0, keepdim=True)
                        )
                        joint_motion.update({
                            "root_quat_baseline_wxyz": baseline_root_quat_mean[0].detach().cpu().tolist(),
                            "root_quat_force_wxyz": force_root_quat_mean[0].detach().cpu().tolist(),
                            "root_orientation_delta_deg": float(
                                _quat_angle_error_deg(
                                    force_root_quat_mean, baseline_root_quat_mean
                                )[0].item()
                            ),
                            "ee_force_w": quat_apply(
                                force_root_quat_mean,
                                force_b[:, case["ee_i"]],
                            )[0].detach().cpu().tolist(),
                        })
                    if force_root_lin_vel_window is not None:
                        root_lin_vel_samples = force_root_lin_vel_window[:, env_i]
                        joint_motion.update({
                            "root_lin_vel_force_w_m_s_mean": root_lin_vel_samples.mean(dim=0).detach().cpu().tolist(),
                            "root_lin_vel_force_w_m_s_std": root_lin_vel_samples.std(dim=0, unbiased=False).detach().cpu().tolist(),
                            "root_lin_vel_force_samples_w_m_s": root_lin_vel_samples.detach().cpu().tolist(),
                        })
                    if force_root_ang_vel_window is not None:
                        root_ang_vel_samples = force_root_ang_vel_window[:, env_i]
                        joint_motion.update({
                            "root_ang_vel_force_w_rad_s_mean": root_ang_vel_samples.mean(dim=0).detach().cpu().tolist(),
                            "root_ang_vel_force_w_rad_s_std": root_ang_vel_samples.std(dim=0, unbiased=False).detach().cpu().tolist(),
                            "root_ang_vel_force_samples_w_rad_s": root_ang_vel_samples.detach().cpu().tolist(),
                        })
                    if baseline_applied_torque_window is not None and force_applied_torque_window is not None:
                        baseline_torque_samples = baseline_applied_torque_window[:, env_i]
                        force_torque_samples = force_applied_torque_window[:, env_i]
                        joint_motion.update({
                            "applied_torque_baseline_nm": baseline_torque_samples.mean(dim=0).detach().cpu().tolist(),
                            "applied_torque_force_nm": force_torque_samples.mean(dim=0).detach().cpu().tolist(),
                            "applied_torque_delta_nm": (
                                force_torque_samples.mean(dim=0) - baseline_torque_samples.mean(dim=0)
                            ).detach().cpu().tolist(),
                        })

                    records.append({
                        "ee": case["ee_name"],
                        "ee_index": case["ee_i"],
                        "direction": case["direction_name"],
                        "direction_b": case["direction_b"].detach().cpu().tolist(),
                        "force_n": case["force_n"],
                        "cfg_stiffness_n_per_m": _stiffness_along_direction(
                            compliance_eval_info["actual_stiffness"],
                            case["direction_b"],
                            case["ee_i"],
                        ),
                        "measured_stiffness_signed_n_per_m": measured_stiffness_signed,
                        "measured_stiffness_abs_n_per_m": measured_stiffness_abs,
                        "hl_command_measured_stiffness_signed_n_per_m": hl_command_measured_stiffness_signed,
                        "hl_command_measured_stiffness_abs_n_per_m": hl_command_measured_stiffness_abs,
                        "deflection_along_m_mean": along_mean,
                        "deflection_along_m": selected_along.detach().cpu().tolist(),
                        "deflection_orth_m": selected_orth.detach().cpu().tolist(),
                        "hl_command_deflection_along_m_mean": hl_command_along_mean,
                        "hl_command_deflection_along_m": (
                            selected_hl_command_along.detach().cpu().tolist()
                            if selected_hl_command_along is not None
                            else None
                        ),
                        "nominal_delta_xyz_m": nominal_delta_b.detach().cpu().tolist(),
                        "compliance_delta_xyz_m": compliance_delta_b.detach().cpu().tolist(),
                        "hl_command_nominal_delta_xyz_m": (
                            hl_command_nominal_delta_b.detach().cpu().tolist()
                            if hl_command_nominal_delta_b is not None
                            else None
                        ),
                        "hl_command_compliance_delta_xyz_m": (
                            hl_command_compliance_delta_b.detach().cpu().tolist()
                            if hl_command_compliance_delta_b is not None
                            else None
                        ),
                        "actual_to_hl_command_delta_xyz_m": (
                            actual_to_hl_command_delta_b.detach().cpu().tolist()
                            if actual_to_hl_command_delta_b is not None
                            else None
                        ),
                        "target_pos_b": sample_center_b[env_ids[env_i]].detach().cpu().tolist(),
                        "baseline_pos_b": baseline_pos_b[env_ids[env_i]].detach().cpu().tolist(),
                        "baseline_quat_b_wxyz": baseline_quat_b[env_ids[env_i], case["ee_i"]].detach().cpu().tolist(),
                        "actual_pos_b": actual_pos_b.detach().cpu().tolist(),
                        "actual_quat_b_wxyz": actual_quat_b[0, case["ee_i"]].detach().cpu().tolist(),
                        "hl_command_pos_b": (
                            hl_command_pos_b.detach().cpu().tolist()
                            if hl_command_pos_b is not None
                            else None
                        ),
                        "moe_gate_weights_600_400_200": (
                            moe_gate_weights.detach().cpu().tolist()
                            if moe_gate_weights is not None
                            else None
                        ),
                        "compliance_target_pos_b": compliance_target_b.detach().cpu().tolist(),
                        "compliance_offset_b": compliance_offset_b.detach().cpu().tolist(),
                        "ee_force_b": force_b.detach().cpu().tolist(),
                        "force_estimator_pred_b": (
                            force_pred_b.detach().cpu().tolist()
                            if force_pred_b is not None
                            else None
                        ),
                        "force_estimator_error_b": (
                            force_pred_error_b.detach().cpu().tolist()
                            if force_pred_error_b is not None
                            else None
                        ),
                        "force_estimator_error_norm_n": (
                            force_pred_error_norm.detach().cpu().tolist()
                            if force_pred_error_norm is not None
                            else None
                        ),
                        "force_estimator_pred_norm_n": (
                            force_pred_norm.detach().cpu().tolist()
                            if force_pred_norm is not None
                            else None
                        ),
                        "force_estimator_actual_norm_n": force_pred_actual_norm.detach().cpu().tolist(),
                        "nominal_pos_error_m": nominal_error.detach().cpu().tolist(),
                        "compliance_pos_error_m": compliance_error.detach().cpu().tolist(),
                        "actual_to_hl_command_error_m": (
                            actual_to_hl_command_error.detach().cpu().tolist()
                            if actual_to_hl_command_error is not None
                            else None
                        ),
                        "hl_command_to_compliance_error_m": (
                            hl_command_to_compliance_error.detach().cpu().tolist()
                            if hl_command_to_compliance_error is not None
                            else None
                        ),
                        "rpy_abs_error_deg": rpy_error.detach().cpu().tolist(),
                        "quat_angle_error_deg": quat_error.detach().cpu().tolist(),
                        **joint_motion,
                    })

                    print(
                        f"{case['ee_name']:>5s} {case['direction_name']:>2s} {case['force_n']:>4.0f}N "
                        f"[env {batch_start + env_i:02d}]: "
                        f"nom_err={nominal_error[:, case['ee_i']].mean().item():.4f} m, "
                        f"comp_err={compliance_error[:, case['ee_i']].mean().item():.4f} m, "
                        f"defl={along_mean:.4f} m, "
                        f"k_meas={measured_stiffness_abs if measured_stiffness_abs is not None else float('nan'):.1f} N/m, "
                        f"k_cfg={records[-1]['cfg_stiffness_n_per_m'] if records[-1]['cfg_stiffness_n_per_m'] is not None else float('nan'):.1f} N/m",
                        flush=True,
                    )

            command_manager.clear_eval_ee_force()

        nominal_errors = torch.tensor([r["nominal_pos_error_m"] for r in records])
        compliance_errors = torch.tensor([r["compliance_pos_error_m"] for r in records])
        deflections = torch.tensor([r["deflection_along_m"] for r in records])
        nominal_delta_xyz = torch.tensor([r["nominal_delta_xyz_m"] for r in records])
        compliance_delta_xyz = torch.tensor([r["compliance_delta_xyz_m"] for r in records])
        hl_command_records = [r for r in records if r["hl_command_pos_b"] is not None]
        force_estimator_records = [r for r in records if r["force_estimator_pred_b"] is not None]
        if hl_command_records:
            hl_command_nominal_delta_xyz = torch.tensor([r["hl_command_nominal_delta_xyz_m"] for r in hl_command_records])
            hl_command_compliance_delta_xyz = torch.tensor([r["hl_command_compliance_delta_xyz_m"] for r in hl_command_records])
            actual_to_hl_command_delta_xyz = torch.tensor([r["actual_to_hl_command_delta_xyz_m"] for r in hl_command_records])
            actual_to_hl_command_errors = torch.tensor([r["actual_to_hl_command_error_m"] for r in hl_command_records])
            hl_command_to_compliance_errors = torch.tensor([r["hl_command_to_compliance_error_m"] for r in hl_command_records])
        else:
            hl_command_nominal_delta_xyz = None
            hl_command_compliance_delta_xyz = None
            actual_to_hl_command_delta_xyz = None
            actual_to_hl_command_errors = None
            hl_command_to_compliance_errors = None
        if force_estimator_records:
            force_estimator_pred_b = torch.tensor([r["force_estimator_pred_b"] for r in force_estimator_records])
            force_estimator_error_b = torch.tensor([r["force_estimator_error_b"] for r in force_estimator_records])
            force_estimator_error_norm = torch.tensor([r["force_estimator_error_norm_n"] for r in force_estimator_records])
            force_estimator_pred_norm = torch.tensor([r["force_estimator_pred_norm_n"] for r in force_estimator_records])
            force_estimator_actual_norm = torch.tensor([r["force_estimator_actual_norm_n"] for r in force_estimator_records])
        else:
            force_estimator_pred_b = None
            force_estimator_error_b = None
            force_estimator_error_norm = None
            force_estimator_pred_norm = None
            force_estimator_actual_norm = None
        measured_stiffness = [
            r["measured_stiffness_abs_n_per_m"]
            for r in records
            if r["measured_stiffness_abs_n_per_m"] is not None
        ]
        hl_command_measured_stiffness = [
            r["hl_command_measured_stiffness_abs_n_per_m"]
            for r in records
            if r["hl_command_measured_stiffness_abs_n_per_m"] is not None
        ]
        measured_stiffness_tensor = (
            torch.tensor(measured_stiffness, dtype=torch.float32)
            if measured_stiffness
            else torch.empty(0, dtype=torch.float32)
        )
        measured_stiffness_error_abs = [
            abs(
                r["measured_stiffness_abs_n_per_m"]
                - r["cfg_stiffness_n_per_m"]
            )
            for r in records
            if r["measured_stiffness_abs_n_per_m"] is not None
            and r["cfg_stiffness_n_per_m"] is not None
        ]
        measured_stiffness_error_abs_tensor = (
            torch.tensor(measured_stiffness_error_abs, dtype=torch.float32)
            if measured_stiffness_error_abs
            else torch.empty(0, dtype=torch.float32)
        )
        hl_command_measured_stiffness_tensor = (
            torch.tensor(hl_command_measured_stiffness, dtype=torch.float32)
            if hl_command_measured_stiffness
            else torch.empty(0, dtype=torch.float32)
        )
        measured_stiffness_xyz = {}
        measured_stiffness_error_xyz = {}
        hl_command_measured_stiffness_xyz = {}
        for axis in ("x", "y", "z"):
            axis_values = [
                r["measured_stiffness_abs_n_per_m"]
                for r in records
                if r["direction"].endswith(axis) and r["measured_stiffness_abs_n_per_m"] is not None
            ]
            axis_tensor = (
                torch.tensor(axis_values, dtype=torch.float32)
                if axis_values
                else torch.empty(0, dtype=torch.float32)
            )
            measured_stiffness_xyz[axis] = _summary(axis_tensor) if axis_tensor.numel() > 0 else None
            axis_error_values = [
                abs(
                    r["measured_stiffness_abs_n_per_m"]
                    - r["cfg_stiffness_n_per_m"]
                )
                for r in records
                if r["direction"].endswith(axis)
                and r["measured_stiffness_abs_n_per_m"] is not None
                and r["cfg_stiffness_n_per_m"] is not None
            ]
            axis_error_tensor = (
                torch.tensor(axis_error_values, dtype=torch.float32)
                if axis_error_values
                else torch.empty(0, dtype=torch.float32)
            )
            measured_stiffness_error_xyz[axis] = (
                _summary(axis_error_tensor) if axis_error_tensor.numel() > 0 else None
            )
            hl_axis_values = [
                r["hl_command_measured_stiffness_abs_n_per_m"]
                for r in records
                if r["direction"].endswith(axis) and r["hl_command_measured_stiffness_abs_n_per_m"] is not None
            ]
            hl_axis_tensor = (
                torch.tensor(hl_axis_values, dtype=torch.float32)
                if hl_axis_values
                else torch.empty(0, dtype=torch.float32)
            )
            hl_command_measured_stiffness_xyz[axis] = _summary(hl_axis_tensor) if hl_axis_tensor.numel() > 0 else None

        joint_motion_summary = None
        if records and records[0].get("joint_delta_pos_rad") is not None:
            joint_delta_pos = torch.tensor(
                [record["joint_delta_pos_rad"] for record in records], dtype=torch.float32
            )
            joint_force_pos_std = torch.tensor(
                [record["joint_pos_force_std_rad"] for record in records], dtype=torch.float32
            )
            joint_force_vel_std = torch.tensor(
                [record["joint_vel_force_rad_s_std"] for record in records], dtype=torch.float32
            )
            joint_peak_abs_delta = torch.tensor(
                [record["joint_peak_abs_delta_rad"] for record in records], dtype=torch.float32
            )
            root_delta_pos = torch.tensor(
                [record["root_delta_pos_norm_m"] for record in records], dtype=torch.float32
            )
            root_peak_delta_pos = torch.tensor(
                [record["root_peak_delta_pos_norm_m"] for record in records], dtype=torch.float32
            )
            root_orientation_delta = torch.tensor(
                [record["root_orientation_delta_deg"] for record in records], dtype=torch.float32
            )
            joint_motion_summary = {
                "joint_delta_abs_rad": _named_joint_summary(joint_delta_pos.abs(), joint_names),
                "joint_pos_force_std_rad": _named_joint_summary(joint_force_pos_std, joint_names),
                "joint_vel_force_std_rad_s": _named_joint_summary(joint_force_vel_std, joint_names),
                "joint_peak_abs_delta_rad": _named_joint_summary(joint_peak_abs_delta, joint_names),
                "root_delta_pos_norm_m": _summary(root_delta_pos),
                "root_peak_delta_pos_norm_m": _summary(root_peak_delta_pos),
                "root_orientation_delta_deg": _summary(root_orientation_delta),
            }

        report = {
            "checkpoint": args.checkpoint,
            "run_path": args.run_path,
            "moe_experts_config": getattr(args, "moe_experts_config", None),
            "task": args.task,
            "num_envs": args.ee_compliance_num_envs if (args.ee_compliance_eval or args.ee_bimanual_compliance_eval) else args.num_envs,
            "ee_body_names": body_names,
            "joint_names": joint_names,
            "joint_state_recording": {
                "enabled": joint_motion_summary is not None,
                "joint_position_unit": "rad",
                "joint_velocity_unit": "rad/s",
                "applied_torque_unit": "N*m",
                "root_position_frame": "world",
                "baseline_window_steps": baseline_window_steps,
                "force_hold_window_steps": mean_window_steps,
                "raw_window_samples": True,
            },
            "jacobian_recording": {
                "enabled": bool(records and records[0].get("ee_jacobian_baseline_root_3xn") is not None),
                "spatial_enabled": bool(records and records[0].get("ee_spatial_jacobian_baseline_root_6xn") is not None),
                "frame": "robot_root",
                "jacobian_type": "translational_geometric_jacobian",
                "shape_per_ee": "3 x n_joints",
                "spatial_jacobian_type": "translational_and_angular_geometric_jacobian",
                "spatial_shape_per_ee": "6 x n_joints",
                "baseline_ee_orientation": "mean quaternion over the no-force baseline window",
                "ideal_ee_displacement": "K_x^{-1} F in the robot-root frame",
                "minimum_norm_joint_prediction": "pinv(J) @ ideal_ee_displacement",
                "external_joint_torque": "J.T @ F",
                "note": "The predicted joint displacement is a local kinematic IK reference, not intrinsic joint compliance.",
            },
            "force_directions": EE_COMPLIANCE_FORCE_DIRECTIONS,
            "force_magnitudes_n": EE_COMPLIANCE_FORCE_MAGNITUDES,
            "ramp_steps": EE_COMPLIANCE_RAMP_STEPS,
            "hold_steps": EE_COMPLIANCE_HOLD_STEPS,
            "recovery_steps": EE_COMPLIANCE_RECOVERY_STEPS,
            "baseline_steps": EE_COMPLIANCE_BASELINE_STEPS,
            "mean_window_sec": EE_EVAL_MEAN_WINDOW_SEC,
            "mean_window_steps": mean_window_steps,
            "external_force": "manual_default",
            "external_force_default_cfg": DEFAULT_EXTERNAL_FORCE_CFG,
            "ee_compliance_force_estimator_ablation": force_estimator_ablation,
            "ee_compliance_oracle_force": oracle_force_baseline,
            "ee_compliance_position_name": getattr(args, "ee_compliance_position_name", None),
            "ee_compliance_initial_position_b": sample_center_b[0].detach().cpu().tolist(),
            "compliance_controller": (
                "oracle_force_over_cfg_stiffness_stiff_low_level"
                if oracle_force_baseline
                else (
                    "force_estimator_over_cfg_stiffness"
                    if force_estimator_ablation
                    else "policy_or_command_manager"
                )
            ),
            "target_pos_b": sample_center_b.detach().cpu().tolist(),
            "baseline_pos_b": baseline_pos_b.detach().cpu().tolist(),
            "baseline_nominal_error_m": baseline_nominal_error.detach().cpu().tolist(),
            "baseline_compliance_error_m": baseline_compliance_error.detach().cpu().tolist(),
            "ee_compliance_target": compliance_params,
            "ee_compliance_eval_info": compliance_eval_info,
            "summary": {
                "nominal_position_error_m": {
                    "combined": _summary(nominal_errors),
                    "left": _summary(nominal_errors[:, :, 0]),
                    "right": _summary(nominal_errors[:, :, 1]),
                },
                "nominal_position_delta_xyz_m": _component_summary(nominal_delta_xyz),
                "compliance_position_error_m": {
                    "combined": _summary(compliance_errors),
                    "left": _summary(compliance_errors[:, :, 0]),
                    "right": _summary(compliance_errors[:, :, 1]),
                },
                "compliance_position_delta_xyz_m": _component_summary(compliance_delta_xyz),
                "hl_command_nominal_delta_xyz_m": (
                    _component_summary(hl_command_nominal_delta_xyz)
                    if hl_command_nominal_delta_xyz is not None
                    else None
                ),
                "hl_command_compliance_delta_xyz_m": (
                    _component_summary(hl_command_compliance_delta_xyz)
                    if hl_command_compliance_delta_xyz is not None
                    else None
                ),
                "actual_to_hl_command_delta_xyz_m": (
                    _component_summary(actual_to_hl_command_delta_xyz)
                    if actual_to_hl_command_delta_xyz is not None
                    else None
                ),
                "actual_to_hl_command_error_m": (
                    {
                        "combined": _summary(actual_to_hl_command_errors),
                        "left": _summary(actual_to_hl_command_errors[:, :, 0]),
                        "right": _summary(actual_to_hl_command_errors[:, :, 1]),
                    }
                    if actual_to_hl_command_errors is not None
                    else None
                ),
                "hl_command_to_compliance_error_m": (
                    {
                        "combined": _summary(hl_command_to_compliance_errors),
                        "left": _summary(hl_command_to_compliance_errors[:, :, 0]),
                        "right": _summary(hl_command_to_compliance_errors[:, :, 1]),
                    }
                    if hl_command_to_compliance_errors is not None
                    else None
                ),
                "deflection_along_m": _summary(deflections),
                "measured_stiffness_abs_n_per_m": (
                    _summary(measured_stiffness_tensor)
                    if measured_stiffness_tensor.numel() > 0
                    else None
                ),
                "measured_stiffness_abs_xyz_n_per_m": measured_stiffness_xyz,
                "measured_stiffness_error_abs_n_per_m": (
                    _summary(measured_stiffness_error_abs_tensor)
                    if measured_stiffness_error_abs_tensor.numel() > 0
                    else None
                ),
                "measured_stiffness_error_abs_xyz_n_per_m": measured_stiffness_error_xyz,
                "hl_command_measured_stiffness_abs_n_per_m": (
                    _summary(hl_command_measured_stiffness_tensor)
                    if hl_command_measured_stiffness_tensor.numel() > 0
                    else None
                ),
                "hl_command_measured_stiffness_abs_xyz_n_per_m": hl_command_measured_stiffness_xyz,
                "force_estimator": (
                    {
                        "pred_force_b_n": _component_summary(force_estimator_pred_b),
                        "error_force_b_n": _component_summary(force_estimator_error_b),
                        "error_norm_n": _summary(force_estimator_error_norm),
                        "pred_norm_n": _summary(force_estimator_pred_norm),
                        "actual_norm_n": _summary(force_estimator_actual_norm),
                    }
                    if force_estimator_records
                    else None
                ),
                "joint_motion": joint_motion_summary,
            },
            "records": records,
        }

        if args.ee_output is None:
            report_prefix = (
                "ee_compliance_eval_oracle_force_stiff_low_level"
                if oracle_force_baseline
                else (
                    "ee_compliance_eval_force_estimator_ablation"
                    if force_estimator_ablation
                    else "ee_compliance_eval"
                )
            )
            args.ee_output = _default_ee_report_path(args, report_prefix)
        os.makedirs(os.path.dirname(args.ee_output) or ".", exist_ok=True)
        with open(args.ee_output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        nominal = report["summary"]["nominal_position_error_m"]["combined"]
        compliance = report["summary"]["compliance_position_error_m"]["combined"]
        stiffness_summary = report["summary"]["measured_stiffness_abs_n_per_m"]
        stiffness_xyz = report["summary"]["measured_stiffness_abs_xyz_n_per_m"]
        stiffness_error_summary = report["summary"]["measured_stiffness_error_abs_n_per_m"]
        stiffness_error_xyz = report["summary"]["measured_stiffness_error_abs_xyz_n_per_m"]
        force_estimator_summary = report["summary"].get("force_estimator")
        print("\n" + "=" * 60)
        print("EE COMPLIANCE EVAL")
        print("=" * 60)
        print(
            "  Nominal position error mean±std/rmse/max: "
            f"{nominal['mean']:.4f} ± {nominal['std']:.4f} / "
            f"{nominal['rmse']:.4f} / {nominal['max']:.4f} m"
        )
        print(
            "  Compliance position error mean±std/rmse/max: "
            f"{compliance['mean']:.4f} ± {compliance['std']:.4f} / "
            f"{compliance['rmse']:.4f} / {compliance['max']:.4f} m"
        )
        if stiffness_summary is not None:
            print(
                "  Measured nominal stiffness abs mean±std/rmse/min/max: "
                f"{stiffness_summary['mean']:.1f} ± {stiffness_summary['std']:.1f} / "
                f"{stiffness_summary['rmse']:.1f} / {stiffness_summary['min']:.1f} / "
                f"{stiffness_summary['max']:.1f} N/m"
            )
            print(
                "  Measured nominal stiffness xyz mean±std/min/max: "
                f"x={stiffness_xyz['x']['mean']:.1f} ± {stiffness_xyz['x']['std']:.1f}/"
                f"{stiffness_xyz['x']['min']:.1f}/{stiffness_xyz['x']['max']:.1f}, "
                f"y={stiffness_xyz['y']['mean']:.1f} ± {stiffness_xyz['y']['std']:.1f}/"
                f"{stiffness_xyz['y']['min']:.1f}/{stiffness_xyz['y']['max']:.1f}, "
                f"z={stiffness_xyz['z']['mean']:.1f} ± {stiffness_xyz['z']['std']:.1f}/"
                f"{stiffness_xyz['z']['min']:.1f}/{stiffness_xyz['z']['max']:.1f} N/m"
            )
            if (
                stiffness_error_summary is not None
                and all(stiffness_error_xyz[axis] is not None for axis in ("x", "y", "z"))
            ):
                print(
                    "  Measured nominal stiffness error MAE±std xyz/overall: "
                    f"x={stiffness_error_xyz['x']['mean']:.1f} ± {stiffness_error_xyz['x']['std']:.1f}, "
                    f"y={stiffness_error_xyz['y']['mean']:.1f} ± {stiffness_error_xyz['y']['std']:.1f}, "
                    f"z={stiffness_error_xyz['z']['mean']:.1f} ± {stiffness_error_xyz['z']['std']:.1f}, "
                    f"overall={stiffness_error_summary['mean']:.1f} ± {stiffness_error_summary['std']:.1f} N/m"
                )
        if force_estimator_summary is not None:
            pred_norm = force_estimator_summary["pred_norm_n"]
            actual_norm = force_estimator_summary["actual_norm_n"]
            err_norm = force_estimator_summary["error_norm_n"]
            err_xyz = force_estimator_summary["error_force_b_n"]
            print(
                "  Force estimator pred/actual norm mean±std: "
                f"{pred_norm['mean']:.1f} ± {pred_norm['std']:.1f} / "
                f"{actual_norm['mean']:.1f} ± {actual_norm['std']:.1f} N"
            )
            print(
                "  Force estimator error norm mean±std/rmse/max: "
                f"{err_norm['mean']:.1f} ± {err_norm['std']:.1f} / "
                f"{err_norm['rmse']:.1f} / {err_norm['max']:.1f} N"
            )
            print(
                "  Force estimator error xyz mean±std: "
                f"x={err_xyz['x']['mean']:.1f} ± {err_xyz['x']['std']:.1f}, "
                f"y={err_xyz['y']['mean']:.1f} ± {err_xyz['y']['std']:.1f}, "
                f"z={err_xyz['z']['mean']:.1f} ± {err_xyz['z']['std']:.1f} N"
            )
        print(f"  Config stiffness: {compliance_eval_info['actual_stiffness']}")
        print(f"  Report: {args.ee_output}")
        print("=" * 60 + "\n")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def main():
    parser = argparse.ArgumentParser(description="Manipulation evaluation with teleoperation")
    parser.add_argument("-r", "--run_path", "--run-path", type=str, help="WandB run path")
    parser.add_argument("--checkpoint", type=str, help="Local checkpoint path (alternative to wandb)")
    parser.add_argument(
        "--config_file",
        "--config-file",
        type=str,
        default=None,
        help=(
            "Local training cfg.yaml to load together with --checkpoint. "
            "This avoids requiring W&B network access and preserves the checkpoint's policy/task config."
        ),
    )
    parser.add_argument(
        "--moe_experts_config",
        "--moe-experts-config",
        type=str,
        default=None,
        help=(
            "YAML manifest for the frozen experts of the analytical EE MoE "
            "(e.g. cfg/moe/G1_ee_analytical_100_200_400_600_xyz600_direct_posscale035.yaml, "
            "ten experts). This is an alternative to --run_path/--checkpoint."
        ),
    )
    parser.add_argument(
        "--low_checkpoint",
        "--low-checkpoint",
        type=str,
        default=None,
        help=(
            "Frozen low-level policy checkpoint for --moe_experts_config: a local .pt path or "
            "run:entity/project/run_id[:iter]. Overrides the manifest's low_policy.checkpoint_path."
        ),
    )
    parser.add_argument(
        "--low_run_path",
        "--low-run-path",
        type=str,
        default=None,
        help=(
            "Frozen low-level policy W&B run (entity/project/run_id) for --moe_experts_config; "
            "the latest checkpoint is used. Overrides the manifest's low_policy.run_path."
        ),
    )
    parser.add_argument("--task", type=str, default=None, help="Override task config")
    parser.add_argument("-p", "--play", action="store_true", default=False, help="Play mode (visualize)")
    parser.add_argument("-i", "--iterations", type=int, default=None, help="Checkpoint iteration to load")
    parser.add_argument("-n", "--num_envs", "--num-envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the evaluation seed (cfg.seed); default keeps the config seed")
    parser.add_argument("-e", "--export", action="store_true", default=False, help="Export policy")
    parser.add_argument("--objects", type=str, default=None, 
                        help="Objects config file (e.g., cfg/objects/room_scene.yaml)")
    parser.add_argument("--full_collision", "--full-collision", action="store_true", default=False,
                        help="Use robot USD with full collision meshes (g1_col_full)")
    parser.add_argument("--obs_source", "--obs-source", choices=["udp", "motion"], default="udp",
                        help="Observation source for command/root_and_wrist_6d in play mode")
    parser.add_argument("--ee_tracking_eval", "--ee-tracking-eval", action="store_true", default=False,
                        help="Evaluate EE tracking accuracy with scripted random EE commands")
    parser.add_argument("--ee_compliance_eval", "--ee-compliance-eval", action="store_true", default=False,
                        help="Evaluate EE compliance with deterministic EE force sweeps")
    parser.add_argument(
        "--ee_bimanual_compliance_eval",
        "--ee-bimanual-compliance-eval",
        action="store_true",
        default=False,
        help="Evaluate both EEs simultaneously with per-EE stiffness targets",
    )
    parser.add_argument("--ee_compliance_num_envs", "--ee-compliance-num-envs", type=int, default=1,
                        help="Number of environments to use for ee_compliance_eval")
    parser.add_argument(
        "--ee_compliance_stiffness",
        "--ee-compliance-stiffness",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional EE nominal stiffness for compliance eval. Use one isotropic "
            "value, three xyz values, or in bimanual mode two isotropic / six "
            "[left xyz, right xyz] values. For range-trained high-level policies "
            "this is both the policy input and compliance reference. A policy "
            "configured with net_pull_ee_compliance_stiffness_range is scalar "
            "range-conditioned and accepts one isotropic value; xyz values "
            "require an xyz-range config. Omitting it keeps cfg/sampler behavior."
        ),
    )
    parser.add_argument(
        "--ee_compliance_force_deadband",
        "--ee-compliance-force-deadband",
        type=float,
        default=None,
        help="Override the EE compliance force deadband for a matched evaluation protocol.",
    )
    parser.add_argument(
        "--ee_compliance_initial_position",
        "--ee-compliance-initial-position",
        type=float,
        nargs=6,
        default=None,
        metavar=("LX", "LY", "LZ", "RX", "RY", "RZ"),
        help=(
            "Override the nominal EE anchor for EE compliance evaluation as six "
            "root-frame values: left_x left_y left_z right_x right_y right_z."
        ),
    )
    parser.add_argument(
        "--ee_compliance_position_name",
        "--ee-compliance-position-name",
        type=str,
        default=None,
        help="Optional label recorded with a custom EE compliance initial position.",
    )
    parser.add_argument("--root_compliance_eval", "--root-compliance-eval", action="store_true", default=False,
                        help="Evaluate root/locomotion compliance with deterministic world-frame force sweeps")
    parser.add_argument("--root_compliance_num_envs", "--root-compliance-num-envs", type=int, default=1,
                        help="Number of environments to use for root_compliance_eval")
    parser.add_argument("--external_force", "--external-force", choices=["on", "off", "default"], default="on",
                        help="External force mode: on uses run cfg, off disables it, default loads the shared eval force cfg")
    parser.add_argument("--ee_output", "--ee-output", type=str, default=None, help="Path to write EE tracking/compliance JSON report")
    parser.add_argument("--root_output", "--root-output", type=str, default=None, help="Path to write root compliance JSON report")
    parser.add_argument(
        "--ee_compliance_force_estimator_ablation",
        "--ee-compliance-force-estimator-ablation",
        action="store_true",
        default=False,
        help=(
            "Only with --ee_compliance_eval: bypass the high-level EE delta output and send "
            "delta_x = force_estimator_b / cfg stiffness to the low-level policy."
        ),
    )
    parser.add_argument(
        "--ee_compliance_oracle_force",
        "--ee-compliance-oracle-force",
        action="store_true",
        default=False,
        help=(
            "Only with --ee_compliance_eval: use the evaluator's ground-truth applied force and "
            "send nominal + force_b / cfg stiffness directly to the raw stiff low-level policy. "
            "This is the oracle-force analytical compliance baseline and is incompatible with "
            "--ee_compliance_force_estimator_ablation."
        ),
    )
    args = parser.parse_args()

    eval_modes = [
        args.ee_tracking_eval,
        args.ee_compliance_eval,
        args.ee_bimanual_compliance_eval,
        args.root_compliance_eval,
    ]
    if sum(bool(mode) for mode in eval_modes) > 1:
        print("Error: --ee_tracking_eval, --ee_compliance_eval, and --root_compliance_eval are mutually exclusive.")
        sys.exit(1)
    if args.ee_compliance_force_estimator_ablation and not args.ee_compliance_eval:
        print("Error: --ee_compliance_force_estimator_ablation can only be used with --ee_compliance_eval.")
        sys.exit(1)
    if args.ee_compliance_oracle_force and not args.ee_compliance_eval:
        print("Error: --ee_compliance_oracle_force can only be used with --ee_compliance_eval.")
        sys.exit(1)
    if args.ee_compliance_force_estimator_ablation and args.ee_compliance_oracle_force:
        print(
            "Error: --ee_compliance_force_estimator_ablation and "
            "--ee_compliance_oracle_force are mutually exclusive."
        )
        sys.exit(1)
    policy_sources = [args.run_path, args.checkpoint, args.moe_experts_config]
    if sum(source is not None for source in policy_sources) != 1:
        print(
            "Error: specify exactly one of --run_path, --checkpoint, or "
            "--moe_experts_config."
        )
        sys.exit(1)
    if args.moe_experts_config and args.export:
        print("Error: analytical MoE export is not supported; evaluate or play it directly.")
        sys.exit(1)
    if (args.low_checkpoint or args.low_run_path) and not args.moe_experts_config:
        print("Error: --low_checkpoint/--low_run_path can only be used with --moe_experts_config.")
        sys.exit(1)
    if args.low_checkpoint and args.low_run_path:
        print("Error: --low_checkpoint and --low_run_path are mutually exclusive.")
        sys.exit(1)

    _configure_evaluation_compiler(args)

    # Determine checkpoint source
    if args.run_path:
        # Load from wandb
        api = wandb.Api()
        run = api.run(args.run_path)
        print(f"Loading run: {run.name}")

        root = os.path.join(os.path.dirname(__file__), "wandb", run.name)
        os.makedirs(root, exist_ok=True)

        # Download config and checkpoints
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
        
        # Select checkpoint
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

        # Load config
        try:
            cfg = OmegaConf.load(os.path.join(root, "files", "cfg.yaml"))
        except FileNotFoundError:
            cfg = OmegaConf.load(os.path.join(root, "cfg.yaml"))
        OmegaConf.set_struct(cfg, False)

        cfg["checkpoint_path"] = os.path.join(root, checkpoint.name)
        if cfg.get("vecnorm", None) is not None:
            cfg["vecnorm"] = "eval"

    elif args.checkpoint:
        # Load from a local training config when supplied; otherwise retain the
        # historical default eval config behavior for backwards compatibility.
        if args.config_file:
            if not os.path.isfile(args.config_file):
                print(f"Error: local config file does not exist: {args.config_file}")
                sys.exit(1)
            cfg = OmegaConf.load(args.config_file)
        else:
            with hydra.initialize(config_path="../cfg", job_name="eval_manipulation", version_base=None):
                cfg = hydra.compose(config_name="eval", overrides=[])
        OmegaConf.set_struct(cfg, False)
        cfg["checkpoint_path"] = args.checkpoint
        cfg["vecnorm"] = "eval"

    elif args.moe_experts_config:
        if not os.path.isfile(args.moe_experts_config):
            print(f"Error: MoE experts config does not exist: {args.moe_experts_config}")
            sys.exit(1)
        manifest = OmegaConf.load(args.moe_experts_config)
        manifest_task = manifest.get("task", None)
        default_task = (
            "G1/hl/ee/G1_hl_ee_bimanual_analytical_moe_200_600_force_b_student"
            if args.ee_bimanual_compliance_eval
            else "G1/hl/ee/G1_hl_ee_xyz_analytical_moe_200_600_force_b_student"
        )
        selected_task = str(manifest_task or default_task)
        with hydra.initialize(config_path="../cfg", job_name="eval_manipulation", version_base=None):
            cfg = hydra.compose(
                config_name="eval",
                overrides=[
                    f"task={selected_task}",
                    "+algo=root_student_force_analytical_moe",
                ],
            )
        OmegaConf.set_struct(cfg, False)
        if "algo" in manifest:
            algo_override = manifest.algo
        else:
            # ``task`` is a manifest-level deployment hint, not an algorithm
            # parameter. Keep it out of cfg.algo when using the flat format.
            algo_override = OmegaConf.create(
                {key: value for key, value in manifest.items() if key not in ("task", "low_policy")}
            )
        cfg["algo"] = OmegaConf.merge(cfg.algo, algo_override)
        cfg["checkpoint_path"] = None
        cfg["vecnorm"] = None

        missing_experts = []
        for expert_name, expert_cfg in cfg.algo.experts.items():
            if not expert_cfg.get("checkpoint_path", None) and not expert_cfg.get("run_path", None):
                missing_experts.append(expert_name)
        if missing_experts:
            print(
                "Error: MoE experts config has no checkpoint_path/run_path for: "
                + ", ".join(missing_experts)
            )
            sys.exit(1)
        print(f"Loading analytical EE MoE experts from: {args.moe_experts_config}")
        manifest_low_policy = manifest.get("low_policy", None) or {}
    else:
        print("Error: no policy source was selected.")
        sys.exit(1)

    # Note: UdpTeleopReceiver is already started by default in MotionTrackingCommand
    # The robot will listen on UDP port 15000 for teleoperation commands

    # Override task config if specified
    if args.task is not None:
        with hydra.initialize(config_path="../cfg", job_name="eval_manipulation", version_base=None):
            _cfg = hydra.compose(config_name="eval", overrides=[f"task={args.task}"])
        cfg["task"]["reward"] = _cfg.task.reward
        cfg["task"]["termination"] = _cfg.task.termination
        cfg["task"]["observation"] = _cfg.task.observation
        cfg["task"]["action"] = _cfg.task.action
        cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["robot"] = _cfg.task.robot
        cfg["task"]["command"] = _cfg.task.command
        cfg["task"]["flags"] = _cfg.task.flags

    if args.seed is not None:
        cfg["seed"] = args.seed
        print(f"Seed override: {args.seed}")

    if args.moe_experts_config:
        # Low-level policy: CLI flags override the manifest's low_policy block.
        if args.low_checkpoint or args.low_run_path:
            low_checkpoint, low_run_path, low_iteration = args.low_checkpoint, args.low_run_path, None
        else:
            low_checkpoint = manifest_low_policy.get("checkpoint_path", None)
            low_run_path = manifest_low_policy.get("run_path", None)
            low_iteration = manifest_low_policy.get("checkpoint_iteration", None)
        low_policy_cfg = cfg["task"].get("action", {}).get("low_policy", None)
        if low_policy_cfg is not None:
            low_policy_cfg["checkpoint_path"] = low_checkpoint
            low_policy_cfg["run_path"] = low_run_path
            low_policy_cfg["checkpoint_iteration"] = low_iteration
            if not low_checkpoint and not low_run_path:
                print(
                    "Error: the analytical MoE needs a frozen low-level policy. Set low_policy.checkpoint_path "
                    f"or low_policy.run_path in {args.moe_experts_config}, or pass "
                    "--low_checkpoint <path or run:entity/project/run_id[:iter]> / "
                    "--low_run_path <entity/project/run_id>."
                )
                sys.exit(1)
            print(f"  Low-level policy: {low_checkpoint or low_run_path}")

    if args.ee_bimanual_compliance_eval and not cfg["task"].get("command", {}).get(
        "net_pull_ee_compliance_stiffness_per_ee", False
    ):
        print(f"Error: {BIMANUAL_PER_EE_REQUIRED_MSG}")
        sys.exit(1)

    external_force_mode = "default" if (args.ee_compliance_eval or args.ee_bimanual_compliance_eval) else args.external_force
    _apply_external_force_mode(cfg, external_force_mode)
    if args.ee_compliance_force_deadband is not None:
        command_cfg = cfg["task"].get("command", {})
        if command_cfg is not None:
            command_cfg["net_pull_ee_compliance_force_deadband"] = float(args.ee_compliance_force_deadband)
        print(
            "EE compliance force deadband override: "
            f"{float(args.ee_compliance_force_deadband):.3f}"
        )

    if args.ee_tracking_eval or args.ee_compliance_eval or args.ee_bimanual_compliance_eval or args.root_compliance_eval:
        cfg["app"]["headless"] = not args.play
        if args.ee_compliance_eval or args.ee_bimanual_compliance_eval:
            cfg["task"]["num_envs"] = args.ee_compliance_num_envs
        elif args.root_compliance_eval:
            cfg["task"]["num_envs"] = args.root_compliance_num_envs
        else:
            cfg["task"]["num_envs"] = args.num_envs
        cfg["task"]["max_episode_length"] = EE_TRACKING_MAX_EPISODE_LENGTH
        cfg["task"]["termination"] = {}
        cfg["task"]["randomization"] = {}
        cfg["export_policy"] = False
        cfg["perf_test"] = False

        command_target = cfg["task"]["command"].get("_target_", "")
        if command_target == "active_adaptation.envs.mdp.commands.teleoperation.TeleopCommand":
            cfg["task"]["command"]["mode"] = "programmatic"
        elif command_target.startswith("active_adaptation.envs.mdp.commands.motion_tracking."):
            cfg["task"]["command"]["disable_motion_finish"] = True
            cfg["task"]["command"]["teleop"] = {"enabled": False, "obs_source": "motion"}

        fixed_low_level_force_limit = None
        if args.ee_tracking_eval or args.ee_compliance_eval or args.ee_bimanual_compliance_eval:
            fixed_low_level_force_limit = _fix_low_level_force_limit_for_ee_eval(cfg)

        if "init_noise" in cfg["task"]["command"]:
            cfg["task"]["command"]["init_noise"] = {
                "root_pos": 0.0,
                "root_ori": 0.0,
                "root_lin_vel": 0.0,
                "root_ang_vel": 0.0,
                "joint_pos": 0.0,
                "joint_vel": 0.0,
            }

        if args.full_collision:
            cfg["task"]["robot"]["name"] = "g1_col_full"
            print("  Robot: Using full collision USD (g1_col_full)")

        if args.objects:
            objects_cfg = OmegaConf.load(args.objects)
            cfg["task"]["objects"] = objects_cfg.get("objects", [])
            print(f"  Objects: Loaded {len(cfg['task']['objects'])} objects from {args.objects}")

        if args.ee_compliance_eval or args.ee_bimanual_compliance_eval:
            cfg["task"]["objects"] = []
            print("  Objects: cleared for empty compliance eval scene")
        if args.root_compliance_eval:
            cfg["task"]["objects"] = []
            print("  Objects: cleared for empty root compliance eval scene")

    if args.ee_compliance_eval or args.ee_bimanual_compliance_eval:
        _validate_explicit_eval_stiffness(
            cfg,
            getattr(args, "ee_compliance_stiffness", None),
        )

    if args.ee_tracking_eval:
        print("\n" + "="*60)
        print("MANIPULATION TASK (EE Tracking Eval)")
        print("="*60)
        print(f"  Run: {_policy_source_label(args)}")
        print(f"  Num envs: {args.num_envs}")
        print(f"  Seed: {EE_TRACKING_SEED}")
        print(f"  EE points: {EE_TRACKING_NUM_POINTS}")
        print(f"  EE radius: {EE_TRACKING_RADIUS} m")
        print(f"  Hold steps: {EE_TRACKING_HOLD_STEPS}")
        print(f"  EE bodies: {EE_TRACKING_BODY_NAMES}")
        print(f"  External force: {args.external_force}")
        if fixed_low_level_force_limit is not None:
            print(f"  Low-level force limit: fixed at {fixed_low_level_force_limit:.2f}")
        print("="*60 + "\n")

        evaluate_ee_tracking(cfg, args)

    elif args.ee_bimanual_compliance_eval:
        print("\n" + "="*60)
        print("MANIPULATION TASK (Bimanual EE Compliance Eval)")
        print("="*60)
        print(f"  Run: {_policy_source_label(args)}")
        print(f"  Num envs: {args.ee_compliance_num_envs}")
        print("  EE bodies: left_hand_mimic,right_hand_mimic")
        print("  Force mode: same direction and magnitude on both hands")
        print("="*60 + "\n")
        evaluate_ee_bimanual_compliance(cfg, args)

    elif args.ee_compliance_eval:
        print("\n" + "="*60)
        print("MANIPULATION TASK (EE Compliance Eval)")
        print("="*60)
        print(f"  Run: {_policy_source_label(args)}")
        print(f"  Num envs: {args.ee_compliance_num_envs}")
        print(f"  EE bodies: {EE_TRACKING_BODY_NAMES}")
        print(f"  Force directions: {[name for name, _ in EE_COMPLIANCE_FORCE_DIRECTIONS]}")
        print(f"  Force magnitudes: {EE_COMPLIANCE_FORCE_MAGNITUDES} N")
        print(f"  Hold steps: {EE_COMPLIANCE_HOLD_STEPS}")
        print(f"  Mean window: {EE_EVAL_MEAN_WINDOW_SEC:.2f} s")
        print("  External force: manual sweep with default eval cfg")
        if fixed_low_level_force_limit is not None:
            print(f"  Low-level force limit: fixed at {fixed_low_level_force_limit:.2f}")
        print("="*60 + "\n")

        evaluate_ee_compliance(cfg, args)

    elif args.root_compliance_eval:
        print("\n" + "="*60)
        print("MANIPULATION TASK (Root Compliance Eval)")
        print("="*60)
        print(f"  Run: {_policy_source_label(args)}")
        print(f"  Num envs: {args.root_compliance_num_envs}")
        print(f"  Root force bodies: {ROOT_COMPLIANCE_BODY_NAMES}")
        print(f"  Force directions: {[name for name, _ in ROOT_COMPLIANCE_FORCE_DIRECTIONS]}")
        print(f"  Force magnitudes: {ROOT_COMPLIANCE_FORCE_MAGNITUDES} N")
        print(f"  Reference velocity: 0 m/s")
        print(f"  Warmup steps: {ROOT_COMPLIANCE_WARMUP_STEPS}")
        print(f"  Ramp steps: {ROOT_COMPLIANCE_RAMP_STEPS}")
        print(f"  Hold steps: {ROOT_COMPLIANCE_HOLD_STEPS}")
        print(f"  Baseline/recovery steps: {ROOT_COMPLIANCE_BASELINE_STEPS}/{ROOT_COMPLIANCE_RECOVERY_STEPS}")
        print(f"  Mean window: {ROOT_COMPLIANCE_MEAN_WINDOW_SEC:.2f} s")
        print("  External force: manual sweep using task net_pull cfg")
        print("="*60 + "\n")

        evaluate_root_compliance(cfg, args)

    # Play mode settings
    elif args.play:
        cfg["app"]["headless"] = False
        cfg["task"]["num_envs"] = args.num_envs
        cfg["task"]["max_episode_length"] = 1000000  # Very long episode for teleop (no auto-reset)
        # Disable all termination conditions for teleop mode
        cfg["task"]["termination"] = {}
        # Disable motion finish triggering reset only for live teleop mode.
        cfg["task"]["command"]["disable_motion_finish"] = args.obs_source == "udp"
        cfg["task"]["command"]["teleop"] = {
            "enabled": args.obs_source == "udp",
            "obs_source": args.obs_source,
        }
        _enable_root_passthrough_for_ee_only_hl(cfg)
        cfg["export_policy"] = args.export
        cfg["perf_test"] = False
        cfg["print_ee_contact_forces"] = True
        cfg["ee_contact_print_interval"] = 10
        
        # Disable all init noise for consistent starting pose in teleop mode
        cfg["task"]["command"]["init_noise"] = {
            "root_pos": 0.0,
            "root_ori": 0.0,
            "root_lin_vel": 0.0,
            "root_ang_vel": 0.0,
            "joint_pos": 0.0,
            "joint_vel": 0.0,
        }
        
        # Use full collision robot USD if requested
        if args.full_collision:
            cfg["task"]["robot"]["name"] = "g1_col_full"
            print("  Robot: Using full collision USD (g1_col_full)")
        
        # Load objects configuration
        if args.objects:
            objects_cfg = OmegaConf.load(args.objects)
            cfg["task"]["objects"] = objects_cfg.get("objects", [])
            print(f"  Objects: Loaded {len(cfg['task']['objects'])} objects from {args.objects}")
        
        print("\n" + "="*60)
        print("MANIPULATION TASK (Teleoperation Mode)")
        print("="*60)
        print(f"  Run: {_policy_source_label(args)}")
        print(f"  Num envs: {args.num_envs}")
        print(f"  Max episode length: 1000000 steps (~5.5 hours at 50Hz)")
        print(f"  External force: {args.external_force}")
        print("  EE contact / HL force estimate / HL EE offset print: every 10 steps")
        if args.obs_source == "udp":
            print(f"  Obs source: UDP teleop (waiting for input on port 15000)")
        else:
            print(f"  Obs source: motion dataset")
        if args.objects:
            print(f"  Objects: {args.objects}")
        print("="*60 + "\n")
        
        play(cfg)
    else:
        print("Error: Currently only play mode (-p) is supported for manipulation")
        sys.exit(1)
    
    exit(0)


if __name__ == "__main__":
    main()
