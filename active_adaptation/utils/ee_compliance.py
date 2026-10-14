from __future__ import annotations

import torch


def sample_mixed_xyz_stiffness(
    count: int,
    stiffness_min: torch.Tensor,
    stiffness_max: torch.Tensor,
    *,
    levels: torch.Tensor | None = None,
    levels_prob: float = 0.0,
    anchors: torch.Tensor | None = None,
    anchor_prob: float = 0.0,
) -> torch.Tensor:
    """Sample continuous xyz stiffness with optional level/anchor mixtures."""

    device = stiffness_min.device
    sampled = stiffness_min.reshape(1, 3) + torch.rand(
        count, 3, device=device
    ) * (stiffness_max - stiffness_min).reshape(1, 3)
    if count == 0:
        return sampled

    modes = torch.rand(count, device=device)
    anchor_mask = modes < anchor_prob
    level_mask = (modes >= anchor_prob) & (modes < anchor_prob + levels_prob)
    if anchor_mask.any():
        if anchors is None:
            raise ValueError("Anchor sampling requires a non-empty anchors tensor.")
        anchor_ids = torch.randint(
            anchors.shape[0],
            (int(anchor_mask.sum().item()),),
            device=device,
        )
        sampled[anchor_mask] = anchors[anchor_ids]
    if level_mask.any():
        if levels is None:
            raise ValueError("Level sampling requires a non-empty levels tensor.")
        level_ids = torch.randint(
            levels.numel(),
            (int(level_mask.sum().item()), 3),
            device=device,
        )
        sampled[level_mask] = levels[level_ids]
    return sampled


def force_stiffness_consistency_reward(
    actual_b: torch.Tensor,
    nominal_b: torch.Tensor,
    force_b: torch.Tensor,
    stiffness: torch.Tensor,
    *,
    force_scale: float,
    sigma: float,
    force_deadband: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a robust reward for ``K * displacement == external force``."""

    force_residual = stiffness * (actual_b - nominal_b) - force_b
    per_hand_error = force_residual.abs().mean(dim=-1) / force_scale
    active_hands = force_b.norm(dim=-1) > force_deadband
    active_count = active_hands.sum(dim=-1, keepdim=True)
    error = (
        (per_hand_error * active_hands.to(per_hand_error.dtype)).sum(
            dim=-1, keepdim=True
        )
        / active_count.clamp_min(1).to(per_hand_error.dtype)
    )
    reward = torch.exp(-error / sigma)
    return reward, active_count > 0
