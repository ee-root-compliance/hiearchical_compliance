"""
Teleoperation command system for manipulation tasks.

Provides command interface compatible with VR input (UDP) for robot control.
Similar to motion_tracking.py but simplified for direct teleoperation.
"""

import torch
import numpy as np
from typing import Optional
import threading
import socket
import struct
import time

from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.envs.mdp import observation
import active_adaptation.utils.symmetry as sym_utils


# UDP packet: b"G6D1" + seq(u32) + 28 float32 (root/head/left/right, pos3 + quat4).
# Extra appended floats are ignored.
MAGIC = b"G6D1"
PACK_HEADER_FMT = "<4sI"
PACK_HEADER_SIZE = struct.calcsize(PACK_HEADER_FMT)
PACK_NUM_FLOATS = 7 * 4
PACK_MIN_SIZE = PACK_HEADER_SIZE + PACK_NUM_FLOATS * 4

# Legacy packet format: magic 0x12345678 + seq + exactly 28 floats.
LEGACY_MAGIC = 0x12345678
LEGACY_PACK_FMT = f"=II{PACK_NUM_FLOATS}f"
LEGACY_PACK_SIZE = struct.calcsize(LEGACY_PACK_FMT)


def parse_teleop_packet(data: bytes):
    """Return (seq, 28 floats) for a G6D1 or legacy packet, else None."""
    if len(data) >= PACK_MIN_SIZE and data[:4] == MAGIC:
        _, seq = struct.unpack(PACK_HEADER_FMT, data[:PACK_HEADER_SIZE])
        floats = struct.unpack(f"<{PACK_NUM_FLOATS}f", data[PACK_HEADER_SIZE:PACK_MIN_SIZE])
        return seq, floats
    if len(data) == LEGACY_PACK_SIZE:
        magic, seq, *floats = struct.unpack(LEGACY_PACK_FMT, data)
        if magic == LEGACY_MAGIC:
            return seq, floats
    return None


class UdpTeleopReceiver:
    """
    Receive UDP packets: root/head/left/right, each (pos3 + quat4) in WORLD frame.
    Thread updates latest sample (CPU tensors).
    """
    def __init__(self, bind_ip="0.0.0.0", bind_port=15000, timeout=0.2):
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.timeout = timeout

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._seq: int = -1
        self._t_recv: float = 0.0

        # store latest as torch CPU tensors
        self._root_pos = torch.zeros(3, dtype=torch.float32)
        self._root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self._head_pos = torch.zeros(3, dtype=torch.float32)
        self._head_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self._l_pos = torch.zeros(3, dtype=torch.float32)
        self._l_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self._r_pos = torch.zeros(3, dtype=torch.float32)
        self._r_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.bind_ip, self.bind_port))
        sock.settimeout(self.timeout)

        while self._running:
            try:
                data, _ = sock.recvfrom(2048)
                packet = parse_teleop_packet(data)
                if packet is None:
                    continue
                seq, floats = packet

                # 4 bodies * 7 floats
                vals = torch.tensor(floats, dtype=torch.float32)  # shape (28,)
                root = vals[0:7]
                head = vals[7:14]
                left = vals[14:21]
                right = vals[21:28]

                with self._lock:
                    self._seq = int(seq)
                    self._t_recv = time.time()

                    self._root_pos = root[0:3].clone()
                    self._root_quat = root[3:7].clone()
                    self._head_pos = head[0:3].clone()
                    self._head_quat = head[3:7].clone()
                    self._l_pos = left[0:3].clone()
                    self._l_quat = left[3:7].clone()
                    self._r_pos = right[0:3].clone()
                    self._r_quat = right[3:7].clone()

            except socket.timeout:
                continue
            except Exception:
                continue

    def get_latest(self):
        """
        Returns:
            (seq, t_recv, root_pos, root_quat, head_pos, head_quat, l_pos, l_quat, r_pos, r_quat)
        """
        with self._lock:
            return (
                self._seq,
                self._t_recv,
                self._root_pos.clone(),
                self._root_quat.clone(),
                self._head_pos.clone(),
                self._head_quat.clone(),
                self._l_pos.clone(),
                self._l_quat.clone(),
                self._r_pos.clone(),
                self._r_quat.clone(),
            )


class TeleopCommand(Command):
    """
    Teleoperation command system for manipulation tasks.
    
    Receives target poses for root, head, and hands from VR input (UDP).
    Converts to observations and rewards compatible with the trained policy.
    """

    def __init__(self, env, bind_port: int = 15000, mode: str = "udp"):
        """
        Initialize teleoperation command system.
        
        Args:
            env: Base environment
            bind_port: UDP port for receiving VR input
            mode: "udp" receives commands from the UDP receiver. "programmatic"
                keeps targets set through set_targets/set_root_and_wrist_6d_command.
        """
        super().__init__(env)
        
        self.bind_port = bind_port
        self.mode = mode
        self._teleop = None
        if self.mode not in ("udp", "programmatic"):
            raise ValueError(f"Unsupported TeleopCommand mode: {self.mode}")
        if self.mode == "udp":
            self._teleop = UdpTeleopReceiver(bind_port=bind_port)
            self._teleop.start()
        
        # Store latest target poses [N, 3] and [N, 4]
        self._target_root_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_root_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_root_quat[:, -1] = 1.0  # identity quaternion
        
        self._target_head_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_head_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_head_quat[:, -1] = 1.0
        
        self._target_l_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_l_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_l_quat[:, -1] = 1.0
        
        self._target_r_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_r_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_r_quat[:, -1] = 1.0

        self.root_command = torch.zeros(self.num_envs, 5, device=self.device)
        self.root_command[:, 0] = self.asset.data.default_root_state[:, 2]
        self.root_command[:, 3] = 1.0
        self.root_and_wrist_6d_command = torch.zeros(self.num_envs, 12, device=self.device)
        self.use_root_and_wrist_6d_command = False
        self.root_and_wrist_6d_reference_override = torch.zeros(self.num_envs, 12, device=self.device)
        self.use_root_and_wrist_6d_reference_override = False

    def update(self):
        """
        Update target poses from UDP input.
        Called once per environment step.
        """
        if self.mode == "programmatic":
            return

        # Get latest UDP data
        seq, t_recv, root_pos, root_quat, head_pos, head_quat, l_pos, l_quat, r_pos, r_quat = self._teleop.get_latest()
        
        # Convert to device and expand to all environments
        root_pos = root_pos.to(self.device)
        root_quat = root_quat.to(self.device)
        head_pos = head_pos.to(self.device)
        head_quat = head_quat.to(self.device)
        l_pos = l_pos.to(self.device)
        l_quat = l_quat.to(self.device)
        r_pos = r_pos.to(self.device)
        r_quat = r_quat.to(self.device)
        
        # All environments receive the same command from VR
        self._target_root_pos[:] = root_pos.unsqueeze(0)
        self._target_root_quat[:] = root_quat.unsqueeze(0)
        self._target_head_pos[:] = head_pos.unsqueeze(0)
        self._target_head_quat[:] = head_quat.unsqueeze(0)
        self._target_l_pos[:] = l_pos.unsqueeze(0)
        self._target_l_quat[:] = l_quat.unsqueeze(0)
        self._target_r_pos[:] = r_pos.unsqueeze(0)
        self._target_r_quat[:] = r_quat.unsqueeze(0)

    def reset(self, env_ids: torch.Tensor):
        """Reset command state."""
        self._target_root_pos[env_ids] = 0.0
        self._target_root_quat[env_ids, :3] = 0.0
        self._target_root_quat[env_ids, 3] = 1.0
        self.root_command[env_ids] = 0.0
        self.root_command[env_ids, 0] = self.asset.data.default_root_state[env_ids, 2]
        self.root_command[env_ids, 3] = 1.0

    def _expand_target(self, value: torch.Tensor, dim: int) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if value.ndim == 1:
            value = value.unsqueeze(0).expand(self.num_envs, -1)
        if value.shape != (self.num_envs, dim):
            raise ValueError(f"Expected target shape {(self.num_envs, dim)} or ({dim},), got {tuple(value.shape)}.")
        return value

    def set_targets(
        self,
        root_pos: Optional[torch.Tensor] = None,
        root_quat: Optional[torch.Tensor] = None,
        head_pos: Optional[torch.Tensor] = None,
        head_quat: Optional[torch.Tensor] = None,
        l_pos: Optional[torch.Tensor] = None,
        l_quat: Optional[torch.Tensor] = None,
        r_pos: Optional[torch.Tensor] = None,
        r_quat: Optional[torch.Tensor] = None,
    ):
        """Set teleoperation pose targets directly from a script."""
        if root_pos is not None:
            self._target_root_pos[:] = self._expand_target(root_pos, 3)
        if root_quat is not None:
            self._target_root_quat[:] = self._expand_target(root_quat, 4)
        if head_pos is not None:
            self._target_head_pos[:] = self._expand_target(head_pos, 3)
        if head_quat is not None:
            self._target_head_quat[:] = self._expand_target(head_quat, 4)
        if l_pos is not None:
            self._target_l_pos[:] = self._expand_target(l_pos, 3)
        if l_quat is not None:
            self._target_l_quat[:] = self._expand_target(l_quat, 4)
        if r_pos is not None:
            self._target_r_pos[:] = self._expand_target(r_pos, 3)
        if r_quat is not None:
            self._target_r_quat[:] = self._expand_target(r_quat, 4)

    def set_root_command(self, command: torch.Tensor):
        if command.shape[-1] != 5:
            raise ValueError(f"Root command must have 5 dims, got {command.shape[-1]}.")
        self.root_command[:] = command

    def get_root_and_wrist_6d_reference(self):
        if self.use_root_and_wrist_6d_reference_override:
            return self.root_and_wrist_6d_reference_override
        return self.root_and_wrist_6d_command

    def set_root_and_wrist_6d_reference_override(self, command: torch.Tensor):
        if command.shape[-1] != 12:
            raise ValueError(f"Root-and-wrist reference override must have 12 dims, got {command.shape[-1]}.")
        self.use_root_and_wrist_6d_reference_override = True
        self.root_and_wrist_6d_reference_override[:] = command

    def clear_root_and_wrist_6d_reference_override(self):
        self.use_root_and_wrist_6d_reference_override = False

    def set_root_and_wrist_6d_command(self, command: torch.Tensor):
        if command.shape[-1] != 12:
            raise ValueError(f"Root-and-wrist command must have 12 dims, got {command.shape[-1]}.")
        self.use_root_and_wrist_6d_command = True
        self.root_and_wrist_6d_command[:] = command

    @observation
    def root_and_wrist_6d(self):
        return self.root_and_wrist_6d_command

    def root_and_wrist_6d_sym(self):
        pos_perm = torch.tensor([3, 4, 5, 0, 1, 2])
        pos_signs = torch.tensor([1., -1., 1., 1., -1., 1.])
        ori_perm = torch.tensor([9, 10, 11, 6, 7, 8])
        ori_signs = torch.tensor([-1., 1., -1., -1., 1., -1.])
        return sym_utils.SymmetryTransform(
            perm=torch.cat([pos_perm, ori_perm]),
            signs=torch.cat([pos_signs, ori_signs]),
        )

    @observation
    def command(self):
        force_limit = torch.full((self.num_envs, 1), 0.3, device=self.device)
        return torch.cat([self.root_command, force_limit], dim=-1)

    def get_root_command_reference(self):
        return self.command()[..., :5]

    def command_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(6),
            signs=[1., 1., -1., 1., -1., 1.],
        )

    def __del__(self):
        """Cleanup UDP receiver thread."""
        if hasattr(self, '_teleop') and self._teleop is not None:
            self._teleop.stop()
