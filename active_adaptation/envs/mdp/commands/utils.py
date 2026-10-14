import torch
from typing import Optional
import math

class TemporalLerp:
    """
    A tiny helper to manage time-varying tensors with start/end/duration.
    Works on arbitrary tensor shapes like (N, M, D). Time is discrete (steps).
    """

    def __init__(
        self,
        shape,
        device: torch.device,
        default: float = 0.0,
        easing: str = "linear",  # "linear" | "smoothstep"
        clamp: tuple[float, float] | None = None,
    ):
        self.device = device
        self.value = torch.full(shape, default, dtype=torch.float32, device=device)  # current value
        self._start = self.value.clone()
        self._end = self.value.clone()
        # time states broadcast to last dim
        tshape = self.value.shape[:1] + (1,) * (len(self.value.shape) - 1)
        self._t = torch.zeros(tshape, dtype=torch.int32, device=device)
        self._T = torch.ones(tshape, dtype=torch.int32, device=device)  # avoid /0
        self._active = torch.zeros(tshape, dtype=torch.bool, device=device)
        self.easing = easing
        self.clamp = clamp

    # ---------- public API ----------
    @torch.no_grad()
    def set(
        self,
        env_ids: Optional[torch.Tensor],
        end: Optional[torch.Tensor | float] = None,
        delta: Optional[torch.Tensor | float] = None,
        total_steps: torch.Tensor | int = 0,
        start: Optional[torch.Tensor] = None,
    ):
        """
        Start (or override) a ramp for a subset of the first-dimension indices (env_ids).
        - If start is None, it uses current value at those indices.
        - total_steps can be int or tensor broadcastable to (..., 1).
        """
        if end is None and delta is None:
            raise ValueError("Either 'end' or 'delta' must be provided.")
        if end is not None and delta is not None:
            raise ValueError("Only one of 'end' or 'delta' can be provided.")

        index = (slice(None),) if env_ids is None else (env_ids,)
        # set start/end/clock
        cur_start = self.value[index] if start is None else start
        self._start[index] = cur_start
        if end is not None:
            self._end[index] = end
        elif delta is not None:
            self._end[index] = cur_start + delta
        self._t[index] = 0
        if isinstance(total_steps, int):
            self._T[index] = total_steps
        elif isinstance(total_steps, torch.Tensor):
            self._T[index] = total_steps.view((-1,) + (1,) * (len(self.value.shape) - 1))  # ensure broadcastable
        self._active[index] = torch.ones_like(self._active[index], dtype=torch.bool)
        self.value[index] = cur_start

    @torch.no_grad()
    def update_time(self, steps: int = 1):
        """Advance time by `steps` for all active elements and update current value."""
        if not self._active.any():
            return
        self._t[self._active] += int(steps)
        # compute current value
        self._update_value()
        # mark done
        done = self._t >= self._T
        self._active = self._active & (~done)

    @torch.no_grad()
    def reset(self, env_ids: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None):
        index = (slice(None),) if env_ids is None else (env_ids,)
        if value is not None:
            self.value[index] = value
        self._start[index] = self.value[index]
        self._end[index]   = self.value[index]
        self._t[index].zero_()
        self._T[index].fill_(1)
        self._active[index].zero_()

    @property
    def current(self) -> torch.Tensor:
        return self.value

    @property
    def mask_active(self) -> torch.Tensor:
        return self._active.view(-1)  # drop last dim for convenience

    @property
    def mask_done(self) -> torch.Tensor:
        return (~self._active).view(-1)

    @property
    def time_left(self) -> torch.Tensor:
        return (self._T - self._t).clamp_min(0).view(-1)

    # ---------- internals ----------
    def _ease(self, a: torch.Tensor) -> torch.Tensor:
        if self.easing == "linear":
            return a
        elif self.easing == "smoothstep":
            # 3a^2 - 2a^3
            return a * a * (3 - 2 * a)
        else:
            raise ValueError(f"Unknown easing function: {self.easing}. Use 'linear' or 'smoothstep'.")

    def _update_value(self):
        self.value = self._ease((self._t / self._T).clamp_(0.0, 1.0)) * (self._end - self._start) + self._start
        if self.clamp is not None:
            self.value.clamp_(min=self.clamp[0], max=self.clamp[1])



def clamp_norm(x: torch.Tensor, max=100.0, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    x_norm = x.norm(dim=dim, keepdim=True).clamp_min(eps)
    max_t = torch.as_tensor(max, device=x.device, dtype=x.dtype)
    scale = (max_t / x_norm).clamp(max=1.0)
    return x * scale

def create_mapping(a: torch.Tensor | list[int], b: torch.Tensor | list[int], device: torch.device = torch.device("cpu")) -> torch.Tensor:
    a = a.tolist() if isinstance(a, torch.Tensor) else a
    b = b.tolist() if isinstance(b, torch.Tensor) else b

    return torch.tensor([a.index(item) for item in b], device=device, dtype=torch.int32)

def random_uniform(shape: int|tuple[int], min = 0.0, max = 0.0, device: torch.device = torch.device("cpu")):
    return torch.rand(shape, device=device, dtype=torch.float32) * (max - min) + min

def rand_points_isotropic(
    N: int,
    M: int,
    r_max: float = 1.0,
    *,
    device=None,
    dtype=torch.float32,
    generator: torch.Generator | None = None,
):
    r = torch.rand((N, M, 1), device=device, dtype=dtype, generator=generator)

    v = torch.rand((N, M, 1), device=device, dtype=dtype, generator=generator)
    w = torch.rand((N, M, 1), device=device, dtype=dtype, generator=generator)

    z = 1 - 2 * v
    phi = 2 * math.pi * w
    xy_norm = torch.sqrt(torch.clamp(1 - z*z, min=0))

    x = xy_norm * torch.cos(phi)
    y = xy_norm * torch.sin(phi)

    direction = torch.cat([x, y, z], dim=-1)
    pts = direction * r * r_max
    return pts

def rand_points_disk(
    N: int,
    M: int,
    r_max: float = 1.0,
    *,
    device=None,
    dtype=torch.float32,
    generator: torch.Generator | None = None,
):
    u = torch.rand((N, M, 1), device=device, dtype=dtype, generator=generator)
    r = torch.sqrt(u) * r_max

    phi = 2 * math.pi * torch.rand((N, M, 1), device=device, dtype=dtype, generator=generator)

    x = r * torch.cos(phi)
    y = r * torch.sin(phi)

    pts = torch.cat([x, y], dim=-1)
    return pts