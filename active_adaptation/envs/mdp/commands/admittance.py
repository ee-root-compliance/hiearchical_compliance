import torch
from typing import Optional

def _norm(x, eps=1e-6):
    return x.norm(dim=-1, keepdim=True).clamp_min(eps)

def clamp_norm(x: torch.Tensor, max=100.0, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    x_norm = x.norm(dim=dim, keepdim=True).clamp_min(eps)
    max_t = torch.as_tensor(max, device=x.device, dtype=x.dtype)
    scale = (max_t / x_norm).clamp(max=1.0)
    return x * scale

class AdmittanceMassChain:
    def __init__(
        self,
        num_envs: int,
        num_points: int,
        dt: float,
        mixed_loop_steps: int,
        device: torch.device,
        mass: float = 1.0,
        damping: float = 40.0,
        vel_clip: float = 15.0,
        acc_clip: float = 500.0,
    ):
        self.N = num_envs
        self.M = num_points
        self.H = mixed_loop_steps
        self.dt = float(dt)
        self.device = device

        self.mass = mass
        self.damping = damping
        self.vel_clip = vel_clip
        self.acc_clip = acc_clip

        with torch.device(device):
            self.x = torch.zeros(self.H, self.N, self.M, 3)
            self.v = torch.zeros(self.H, self.N, self.M, 3)

    @torch.no_grad()
    def reset(
        self,
        env_ids: torch.Tensor,
        x0_b: torch.Tensor,
        v0_b: Optional[torch.Tensor] = None,
    ):
        self.x[:, env_ids] = x0_b.unsqueeze(0)
        if v0_b is None:
            self.v[:, env_ids] = 0.0
        else:
            self.v[:, env_ids] = v0_b.unsqueeze(0)

    @torch.no_grad()
    def ingest_state(self, x_b: torch.Tensor, v_b: torch.Tensor):
        self.x[:] = torch.roll(self.x, shifts=1, dims=0)
        self.v[:] = torch.roll(self.v, shifts=1, dims=0)
        self.x[0] = x_b
        self.v[0] = v_b

    @torch.no_grad()
    def step(
        self,
        F_drive_b: torch.Tensor,
        F_ext_b: torch.Tensor,
    ):
        dt = self.dt
        F_damp = - self.damping * self.v
        F_total = F_drive_b + F_ext_b + F_damp
        a = clamp_norm(F_total / self.mass, max=self.acc_clip)

        self.v[:] = clamp_norm(self.v + a * dt, max=self.vel_clip)
        self.x[:] = self.x + self.v * dt
