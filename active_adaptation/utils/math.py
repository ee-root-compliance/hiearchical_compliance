# This file contains additional math utilities
# that are not covered by IsaacLab

import torch
import torch.distributions as D
from isaaclab.utils.math import (
    wrap_to_pi,
    quat_mul,
    quat_conjugate,
    quat_from_angle_axis,
)

def clamp_norm(x: torch.Tensor, min: float=0., max: float=torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x

def clamp_along(x: torch.Tensor, axis: torch.Tensor, min: float, max: float):
    projection = (x * axis).sum(dim=-1, keepdim=True)
    return x - projection * axis + projection.clamp(min, max) * axis

def normalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]    # (..., 3)
    w = quat[..., :1]      # (..., 1)
    t = torch.cross(xyz, vec, dim=-1) * 2   # (..., 3)
    return vec + w * t + torch.cross(xyz, t, dim=-1)  # (..., 3)

@torch.jit.script
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]    # (..., 3)
    w = quat[..., :1]      # (..., 1)
    t = torch.cross(xyz, vec, dim=-1) * 2
    return vec - w * t + torch.cross(xyz, t, dim=-1)

@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(
        angle.abs() > 1.0e-6, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)

def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    qw = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]

    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    half_yaw = yaw * 0.5
    out = torch.zeros_like(quat)
    out[..., 0] = torch.cos(half_yaw)  # w
    out[..., 3] = torch.sin(half_yaw)  # z

    return normalize(out)


__all__ = [
    "yaw_quat", "wrap_to_pi", "quat_mul", "quat_conjugate", "quat_from_angle_axis",
    "quat_apply", "quat_apply_inverse", "axis_angle_from_quat",
    "clamp_norm", "clamp_along", "normalize"
]