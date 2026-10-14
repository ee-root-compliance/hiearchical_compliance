import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.sensors import ContactSensor

from active_adaptation.envs.mdp import reward, termination, observation
from active_adaptation.utils.motion import MotionDataset
from active_adaptation.utils.multimotion import ProgressiveMultiMotionDataset
from active_adaptation.utils import symmetry as sym_utils
from active_adaptation.utils.ee_compliance import sample_mixed_xyz_stiffness
from active_adaptation.utils.math import (
    quat_apply_inverse,
    quat_apply,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat,
    quat_from_angle_axis,
    yaw_quat,
    normalize,
)
from .base import Command
import re
import math
import gc
from typing import Sequence

from active_adaptation.envs.mdp.observations import random_noise
from active_adaptation.envs.mdp.commands.admittance import AdmittanceMassChain

import socket
import struct
import threading
import time
from typing import Optional, Tuple
import os

MAGIC = b"G6D1"
PACK_HEADER_FMT = "<4sI"
PACK_HEADER_SIZE = struct.calcsize(PACK_HEADER_FMT)
PACK_MIN_FLOATS = 28
PACK_MIN_SIZE = PACK_HEADER_SIZE + PACK_MIN_FLOATS * 4

class UdpTeleopReceiver:
    """
    Receive UDP packets: root/head/left/right, each (pos3 + quat4), plus optional
    appended keypoint data. The first optional extension is feet_pos_b with
    left/right foot xyz appended after the original 28 floats.
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
        self._has_feet_pos_b = False
        self._feet_pos_b = torch.zeros(6, dtype=torch.float32)

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
                if len(data) < PACK_MIN_SIZE:
                    continue
                float_bytes = len(data) - PACK_HEADER_SIZE
                if float_bytes % 4 != 0:
                    continue
                num_floats = float_bytes // 4
                if num_floats < PACK_MIN_FLOATS:
                    continue

                magic, seq = struct.unpack(PACK_HEADER_FMT, data[:PACK_HEADER_SIZE])
                if magic != MAGIC:
                    continue
                floats = struct.unpack("<" + "f" * num_floats, data[PACK_HEADER_SIZE:])

                vals = torch.tensor(floats, dtype=torch.float32)
                root = vals[0:7]
                head = vals[7:14]
                left = vals[14:21]
                right = vals[21:28]
                has_feet_pos_b = num_floats >= 34
                feet_pos_b = vals[28:34] if has_feet_pos_b else self._feet_pos_b

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
                    self._has_feet_pos_b = bool(has_feet_pos_b)
                    if has_feet_pos_b:
                        self._feet_pos_b = feet_pos_b.clone()

            except socket.timeout:
                continue
            except Exception:
                continue

    def get_latest(self) -> Tuple[int, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          seq, t_recv, root_pos(3), root_quat(4), head_pos(3), head_quat(4), l_pos(3), l_quat(4), r_pos(3), r_quat(4)
        """
        with self._lock:
            return (
                self._seq, self._t_recv,
                self._root_pos.clone(), self._root_quat.clone(),
                self._head_pos.clone(), self._head_quat.clone(),
                self._l_pos.clone(), self._l_quat.clone(),
                self._r_pos.clone(), self._r_quat.clone(),
            )

    def get_latest_feet_pos_b(self) -> Tuple[bool, torch.Tensor]:
        with self._lock:
            return self._has_feet_pos_b, self._feet_pos_b.clone()


def _match_indices(motion_names, asset_names, patterns, name_map=None, device=None, debug=False):
    asset_idx, motion_idx = [], []
    for i, a in enumerate(asset_names):
        if any(re.match(p, a) for p in patterns):
            m = name_map.get(a, a) if name_map else a
            if m in motion_names:
                asset_idx.append(i)
                motion_idx.append(motion_names.index(m))
                if debug:
                    print(f"Matched asset '{a}' (idx {i}) to motion '{m}' (idx {motion_names.index(m)})")
    return torch.tensor(motion_idx, device=device), torch.tensor(asset_idx, device=device)

def _calc_exp_sigma(error : torch.Tensor, sigma_list : list[float], reduce_last_dim : bool = False):
    count = len(sigma_list)
    if reduce_last_dim:
        rewards = [torch.exp(- error / sigma).mean(dim=-1, keepdim=True) for sigma in sigma_list]
    else:
        rewards = [torch.exp(- error / sigma) for sigma in sigma_list]
    return sum(rewards) / count

def get_items_by_index(list, indexes):
    if isinstance(indexes, torch.Tensor):
        indexes = indexes.tolist()
    return [list[i] for i in indexes]

def convert_dtype(dtype_str):
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
        'bool': bool,
        'long': torch.long
    }
    if isinstance(dtype_str, str):
        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype string: {dtype_str}")
        return dtype_map[dtype_str]
    return dtype_str

def quat_wxyz_to_rpy(q):
    """
    Convert a quaternion in (w, x, y, z) order to roll-pitch-yaw (r, p, y) in radians.
    Accepts a list/tuple/np-array/torch tensor of length 4 and returns a tuple (r, p, y).
    """
    # avoid heavy deps here; work on Python floats
    try:
        # support torch tensors
        import numpy as _np
    except Exception:
        _np = None

    if hasattr(q, "tolist"):
        q = q.tolist()

    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])

    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return (roll, pitch, yaw)

class MotionTrackingCommand(Command):
    def __init__(self, env, dataset: dict,
                dataset_extra_keys: list[dict] = [],
                keypoint_map: dict = {
                    # "left_wrist_roll_link": "left_wrist_roll_rubber_hand",
                    # "right_wrist_roll_link": "right_wrist_roll_rubber_hand"
                },
                keypoint_patterns: list[str] = ["head_mimic", ".*_hand_mimic", ".*wrist_roll_link.*", ".*shoulder_yaw_link", ".*knee.*", ".*ankle_roll_link"],
                lower_keypoint_patterns: list[str] = [".*knee.*", ".*ankle_roll_link"],
                joint_patterns: list[str] = ["waist_*", ".*_hip_.*", ".*_knee.*",".*shoulder.*", ".*elbow.*", ".*wrist.*"],
                ignore_joint_patterns: list[str] = [".*ankle_roll_joint"],
                feet_patterns: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"],
                init_noise: dict[str, float] = {},
                reward_sigma: dict[str, list[float]] = {},
                student_train: bool = False,
                disable_motion_finish: bool = False,
                teleop: dict | None = None,
                synthetic_ee_target: dict | None = None,
                motion_ee_coverage_resample: dict | None = None,):
        super().__init__(env)
        
        # Lists for logging UDP target and actual EE position/pose
        self.target_position_list = []
        self.target_pose_list = []
        self.actual_position_list = []
        self.actual_pose_list = []
        self._last_csv_save_time = time.time()
        
        self.disable_motion_finish = disable_motion_finish
        teleop = teleop or {}
        self.teleop_enabled = bool(teleop.get("enabled", False))
        self.teleop_obs_source = teleop.get("obs_source", "motion")
        if self.teleop_obs_source not in ("motion", "udp"):
            raise ValueError(f"Unsupported teleop obs_source: {self.teleop_obs_source}")
        self.use_root_and_wrist_6d_command = False
        self.root_and_wrist_6d_command = torch.zeros(self.num_envs, 12, device=self.device)
        self.use_root_and_wrist_6d_reference_override = False
        self.root_and_wrist_6d_reference_override = torch.zeros(self.num_envs, 12, device=self.device)
        synthetic_ee_target = synthetic_ee_target or {}
        self.synthetic_ee_target_enabled = bool(synthetic_ee_target.get("enabled", False))
        self.synthetic_ee_target_prob = float(synthetic_ee_target.get("prob", 0.0))
        self.synthetic_ee_target_train_only = bool(synthetic_ee_target.get("train_only", True))
        self.synthetic_ee_target_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.synthetic_ee_target_pos_b = torch.zeros(self.num_envs, 2, 3, dtype=torch.float32, device=self.device)
        self.synthetic_ee_target_left_range = self._parse_synthetic_ee_range(
            synthetic_ee_target.get("left_range", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            name="synthetic_ee_target.left_range",
        )
        self.synthetic_ee_target_right_range = self._parse_synthetic_ee_range(
            synthetic_ee_target.get("right_range", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            name="synthetic_ee_target.right_range",
        )
        if not 0.0 <= self.synthetic_ee_target_prob <= 1.0:
            raise ValueError(f"synthetic_ee_target.prob must be in [0, 1], got {self.synthetic_ee_target_prob}.")
        motion_ee_coverage_resample = motion_ee_coverage_resample or {}
        self.motion_ee_coverage_resample_enabled = bool(motion_ee_coverage_resample.get("enabled", False))
        self.motion_ee_coverage_train_only = bool(motion_ee_coverage_resample.get("train_only", True))
        self.motion_ee_coverage_candidates = int(motion_ee_coverage_resample.get("candidates", 32))
        self.motion_ee_coverage_bins = int(motion_ee_coverage_resample.get("bins", 16))
        self.motion_ee_coverage_power = float(motion_ee_coverage_resample.get("power", 0.5))
        self.motion_ee_coverage_left_range = self._parse_synthetic_ee_range(
            motion_ee_coverage_resample.get("left_range", self.synthetic_ee_target_left_range),
            name="motion_ee_coverage_resample.left_range",
        )
        self.motion_ee_coverage_right_range = self._parse_synthetic_ee_range(
            motion_ee_coverage_resample.get("right_range", self.synthetic_ee_target_right_range),
            name="motion_ee_coverage_resample.right_range",
        )
        if self.motion_ee_coverage_candidates <= 0:
            raise ValueError("motion_ee_coverage_resample.candidates must be positive.")
        if self.motion_ee_coverage_bins <= 1:
            raise ValueError("motion_ee_coverage_resample.bins must be > 1.")
        if self.motion_ee_coverage_power < 0.0:
            raise ValueError("motion_ee_coverage_resample.power must be non-negative.")
        self.use_feet_pos_b_command = False
        self.feet_pos_b_command = torch.zeros(self.num_envs, 6, device=self.device)
        self.use_command_override = False
        self.command_override = torch.zeros(self.num_envs, 6, device=self.device)
        self.use_static_root_command = False
        self.static_root_height = torch.zeros(self.num_envs, 1, device=self.device)
        self.static_heading_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.static_heading_w[:, 0] = 1.0
        self.future_steps = torch.tensor([0, 2, 4, 8, 16], device=self.device)

        self.zero_init_prob = 1.0

        dataset_extra_keys = [
            {**k, 'dtype': convert_dtype(k['dtype'])} 
            for k in dataset_extra_keys
        ]

        self.student_train = student_train

        self.dataset = ProgressiveMultiMotionDataset(**dataset, env_size=self.num_envs, max_step_size=1000, dataset_extra_keys=dataset_extra_keys, device=self.device, ds_device=torch.device("cpu"), refresh_threshold=1000 * 20)
        self.dataset.set_limit(self.asset.data.soft_joint_pos_limits, self.asset.data.soft_joint_vel_limits, self.asset.joint_names)
        self._configure_motion_ee_coverage_resampling()

        # bodies for full‑body keypoint tracking
        self.keypoint_patterns = keypoint_patterns
        self.lower_keypoint_patterns = lower_keypoint_patterns
        self.keypoint_map = keypoint_map
        self.keypoint_idx_motion, self.keypoint_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.keypoint_patterns,
            name_map=self.keypoint_map,
            device=self.device
        )
        self.lower_keypoint_idx_motion, self.lower_keypoint_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.lower_keypoint_patterns,
            name_map=self.keypoint_map,
            device=self.device
        )

        # joints for full‑body joint tracking
        self.joint_patterns = joint_patterns
        self.joint_idx_motion, self.joint_idx_asset = _match_indices(
            self.dataset.joint_names,
            self.asset.joint_names,
            self.joint_patterns,
            device=self.device
        )
        
        self.feet_patterns = feet_patterns
        self.feet_idx_motion, self.feet_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.feet_patterns,
            device=self.device
        )

        # bodies for teleoperation (head and wrists / hands) used for 6D teleop input
        self.teleop_body_patterns = ["head_mimic", ".*_hand_mimic", ".*wrist_roll_link.*"]
        self.teleop_idx_motion, self.teleop_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.teleop_body_patterns,
            name_map=self.keypoint_map,
            device=self.device,
            debug=False,
        )

        self._teleop = None
        if self.teleop_enabled or self.teleop_obs_source == "udp":
            self._teleop = UdpTeleopReceiver(bind_port=15000)
            self._teleop.start()
        print(
            f"[MotionTrackingCommand] teleop.enabled={self.teleop_enabled}, "
            f"obs_source={self.teleop_obs_source}, "
            f"udp_receiver={'on' if self._teleop is not None else 'off'}",
            flush=True,
        )

        # Optional UDP broadcaster for root state (disabled by default).
        # Enable by setting environment variable MOTION_TRACKING_UDP_BROADCAST to "host:port" (e.g. 127.0.0.1:15001)
        self._udp_broadcast_enabled = False
        self._udp_broadcast_addr = ("127.0.0.1", 15001)
        self._udp_broadcast_sock = None
        
        try:
            b_cfg = os.getenv("MOTION_TRACKING_UDP_BROADCAST", None)
            print(f"[MOTION_TRACKING] UDP broadcaster config: {b_cfg}", flush=True)
            if b_cfg:
                parts = b_cfg.split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 15001
                self._udp_broadcast_addr = (host, port)
                self._udp_broadcast_enabled = True
                # create non-blocking UDP socket
                try:
                    self._udp_broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    self._udp_broadcast_sock.setblocking(False)
                    self._udp_broadcast_seq = 0
                    print(f"[MOTION_TRACKING] UDP broadcaster enabled -> {self._udp_broadcast_addr}", flush=True)
                    # Print debug send every N sends (option B). Configure via env var:
                    # MOTION_TRACKING_UDP_BROADCAST_PRINT_EVERY (integer > 0). Default 10.
                    try:
                        _pe = os.getenv("MOTION_TRACKING_UDP_BROADCAST_PRINT_EVERY", "10")
                        self._udp_broadcast_print_every = max(1, int(_pe))
                    except Exception:
                        self._udp_broadcast_print_every = 10
                    print(f"[MOTION_TRACKING] UDP broadcaster debug: printing every {self._udp_broadcast_print_every} sends", flush=True)
                except Exception:
                    self._udp_broadcast_sock = None
                    self._udp_broadcast_enabled = False
        except Exception:
            self._udp_broadcast_enabled = False

        # all joints except ankles
        self.ignore_joint_patterns = ignore_joint_patterns
        all_j_m, all_j_a = [], []
        for j in self.asset.joint_names:
            if j in self.dataset.joint_names and not any(re.match(p, j) for p in self.ignore_joint_patterns):
                all_j_m.append(self.dataset.joint_names.index(j))
                all_j_a.append(self.asset.joint_names.index(j))
        self.all_joint_idx_dataset, self.all_joint_idx_asset = all_j_m, all_j_a
        self.all_joint_idx_dataset = torch.tensor(self.all_joint_idx_dataset, device=self.device)
        self.all_joint_idx_asset = torch.tensor(self.all_joint_idx_asset, device=self.device)

        self.last_reset_env_ids = None

        self._cum_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._cum_root_pos_scale = 0.3
        self._cum_keypoint_scale = 0.25
        self._cum_orientation_scale = 0.7
        self.feet_standing = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

        self.lengths = torch.full((self.num_envs,), 1, dtype=torch.int32, device=self.device)
        self.t = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.boot_indicator = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.boot_indicator_max = 25

        self.joint_pos_last = torch.zeros(self.num_envs, len(self.joint_idx_asset), device=self.device)
        self.joint_pos_boot_protect = self.asset.data.default_joint_pos.clone()

        ## init noise
        self.init_noise_params = init_noise
        ## reward sigma
        self.reward_sigma = reward_sigma

    def _synthetic_ee_target_active(self) -> bool:
        if not self.synthetic_ee_target_enabled:
            return False
        if self.synthetic_ee_target_train_only and not getattr(self.env, "training", True):
            return False
        return True

    def _motion_ee_coverage_active(self) -> bool:
        if not self.motion_ee_coverage_resample_enabled:
            return False
        if self.motion_ee_coverage_train_only and not getattr(self.env, "training", True):
            return False
        return True

    def _parse_synthetic_ee_range(self, value, *, name: str) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if value.shape != (2, 3):
            raise ValueError(f"{name} must have shape [2, 3] as [min_xyz, max_xyz], got {tuple(value.shape)}.")
        if torch.any(value[1] < value[0]):
            raise ValueError(f"{name} max must be >= min, got {value}.")
        return value

    def _sample_synthetic_ee_target(self, env_ids: torch.Tensor):
        if not self._synthetic_ee_target_active():
            self.synthetic_ee_target_mask[env_ids] = False
            return
        use_synth = torch.rand(env_ids.shape[0], device=self.device) < self.synthetic_ee_target_prob
        self.synthetic_ee_target_mask[env_ids] = use_synth
        if not use_synth.any():
            return
        synth_env_ids = env_ids[use_synth]

        def sample_range(target_range: torch.Tensor, count: int) -> torch.Tensor:
            low = target_range[0].view(1, 3)
            high = target_range[1].view(1, 3)
            return low + torch.rand(count, 3, device=self.device) * (high - low)

        self.synthetic_ee_target_pos_b[synth_env_ids, 0] = sample_range(
            self.synthetic_ee_target_left_range,
            synth_env_ids.numel(),
        )
        self.synthetic_ee_target_pos_b[synth_env_ids, 1] = sample_range(
            self.synthetic_ee_target_right_range,
            synth_env_ids.numel(),
        )

    def _configure_motion_ee_coverage_resampling(self):
        self.motion_ee_coverage_hist = None
        if not self.motion_ee_coverage_resample_enabled:
            return
        ee_names = ["left_hand_mimic", "right_hand_mimic"]
        self.motion_ee_coverage_idx_motion = torch.tensor(
            [self.dataset.body_names.index(name) for name in ee_names],
            dtype=torch.long,
            device=self.device,
        )
        ranges = torch.stack([
            self.motion_ee_coverage_left_range,
            self.motion_ee_coverage_right_range,
        ]).cpu()
        bins = self.motion_ee_coverage_bins
        hist = torch.zeros(2, bins, bins, bins, dtype=torch.float32)
        for ds in self.dataset.datasets:
            body_ids_cpu = [ds.body_names.index(name) for name in ee_names]
            for start in range(0, ds.num_steps, 250_000):
                end = min(start + 250_000, ds.num_steps)
                pos = ds.data.body_pos_b[start:end, body_ids_cpu, :].to(torch.float32).cpu()
                for ee_i in range(2):
                    lo = ranges[ee_i, 0]
                    hi = ranges[ee_i, 1]
                    valid = ((pos[:, ee_i] >= lo) & (pos[:, ee_i] <= hi)).all(dim=-1)
                    if not valid.any():
                        continue
                    idx = ((pos[valid, ee_i] - lo) / (hi - lo).clamp_min(1e-6) * bins).floor().long()
                    idx = idx.clamp(0, bins - 1)
                    flat = idx[:, 0] * bins * bins + idx[:, 1] * bins + idx[:, 2]
                    hist[ee_i].view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        self.motion_ee_coverage_hist = hist.to(self.device)
        occupied = (hist > 0).float().mean(dim=(1, 2, 3))
        print(
            "[MotionTrackingCommand] motion EE coverage resampling enabled: "
            f"candidates={self.motion_ee_coverage_candidates}, bins={bins}, "
            f"power={self.motion_ee_coverage_power}, occupied={occupied.tolist()}",
            flush=True,
        )

    def _motion_ee_coverage_density(self, pos_b: torch.Tensor) -> torch.Tensor:
        ranges = torch.stack([
            self.motion_ee_coverage_left_range,
            self.motion_ee_coverage_right_range,
        ])
        bins = self.motion_ee_coverage_bins
        density = torch.zeros(pos_b.shape[:-2], dtype=torch.float32, device=self.device)
        for ee_i in range(2):
            lo = ranges[ee_i, 0]
            hi = ranges[ee_i, 1]
            valid = ((pos_b[..., ee_i, :] >= lo) & (pos_b[..., ee_i, :] <= hi)).all(dim=-1)
            idx = ((pos_b[..., ee_i, :] - lo) / (hi - lo).clamp_min(1e-6) * bins).floor().long()
            idx = idx.clamp(0, bins - 1)
            counts = self.motion_ee_coverage_hist[
                ee_i,
                idx[..., 0],
                idx[..., 1],
                idx[..., 2],
            ]
            density = density + torch.where(valid, counts, counts.new_full(counts.shape, counts.max() + 1.0))
        return density

    def _sample_coverage_start_offsets(self, env_ids: torch.Tensor, max_start: torch.Tensor) -> torch.Tensor:
        count = env_ids.numel()
        candidates = self.motion_ee_coverage_candidates
        max_start = max_start.clamp_min(1)
        candidate_t = (
            torch.rand(count, candidates, device=self.device)
            * max_start.to(torch.float32).unsqueeze(1)
        ).floor().to(torch.int32)
        env_ids_flat = env_ids.repeat_interleave(candidates)
        candidate_motion = self.dataset.get_slice(
            env_ids_flat,
            candidate_t.reshape(-1),
            steps=1,
        )
        candidate_pos = candidate_motion.body_pos_b[:, 0, self.motion_ee_coverage_idx_motion, :]
        candidate_pos = candidate_pos.reshape(count, candidates, 2, 3)
        density = self._motion_ee_coverage_density(candidate_pos)
        weights = (density + 1.0).pow(-self.motion_ee_coverage_power)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        selected = torch.multinomial(weights, num_samples=1).squeeze(1)
        return candidate_t[torch.arange(count, device=self.device), selected]

    def _apply_synthetic_ee_target(self, reference: torch.Tensor) -> torch.Tensor:
        if not self._synthetic_ee_target_active() or not self.synthetic_ee_target_mask.any():
            return reference
        reference = reference.clone()
        reference[self.synthetic_ee_target_mask, :6] = self.synthetic_ee_target_pos_b[
            self.synthetic_ee_target_mask
        ].reshape(-1, 6)
        return reference

    def sample_init(self, env_ids: torch.Tensor):
        t = self.t[env_ids]
        lengths = self.lengths[env_ids]
        self.last_reset_env_ids = env_ids
        # resample motion
        lengths = self.dataset.reset(env_ids)
        
        max_start = lengths - self.future_steps[-1] - 1
        if self._motion_ee_coverage_active():
            offsets = self._sample_coverage_start_offsets(env_ids, max_start)
            t[:] = offsets
        else:
            rand_float = torch.rand(env_ids.shape[0], device=env_ids.device) * 0.75  # [0,0.75)
            offsets = (rand_float * max_start.to(torch.float32)).floor().int()
            t[:] = offsets * (torch.rand(env_ids.shape[0], device=env_ids.device) > self.zero_init_prob)

        self.lengths[env_ids] = lengths
        self.t[env_ids] = t
        self._sample_synthetic_ee_target(env_ids)

        motion = self.dataset.get_slice(env_ids, self.t[env_ids], 1)

        # set robot state
        self.sample_init_robot(env_ids, motion)
        return None

    def sample_init_robot(self, env_ids: Sequence[int], motion, lift_height: float = 0.04):
        # Get subsets for the current envs
        init_root_state = self.init_root_state[env_ids].clone()
        init_joint_pos = self.init_joint_pos[env_ids].clone()
        init_joint_vel = self.init_joint_vel[env_ids].clone()
        env_origins = self.env.scene.env_origins[env_ids]
        num_envs = len(env_ids)

        # Extract motion data
        motion_root_pos = motion.root_pos_w[:, 0]
        motion_root_quat = motion.root_quat_w[:, 0]
        motion_root_lin_vel = motion.root_lin_vel_w[:, 0]
        motion_root_ang_vel = motion.root_ang_vel_w[:, 0]
        motion_joint_pos = motion.joint_pos[:, 0]
        motion_joint_vel = motion.joint_vel[:, 0]

        # -------- root state ----------------------------------------------------
        init_root_state[:, :3] = env_origins + motion_root_pos
        init_root_state[:, 2] += lift_height
        root_pos_noise = torch.randn_like(init_root_state[:, :3]).clamp(-1, 1) * self.init_noise_params["root_pos"]
        root_pos_noise[:, 2].clamp_min_(0.0)
        init_root_state[:, :3] += root_pos_noise

        init_root_state[:, 3:7] = motion_root_quat
        random_axis = torch.rand(num_envs, 3, device=self.device)
        random_angle = torch.randn(num_envs, device=self.device).clamp(-1, 1) * self.init_noise_params["root_ori"]
        random_quat = quat_from_angle_axis(random_angle, random_axis)
        init_root_state[:, 3:7] = quat_mul(random_quat, init_root_state[:, 3:7])

        init_root_state[:, 7:10] = motion_root_lin_vel
        lin_vel_noise = torch.randn_like(init_root_state[:, 7:10]).clamp(-1, 1) * self.init_noise_params["root_lin_vel"]
        init_root_state[:, 7:10] += lin_vel_noise
        
        init_root_state[:, 10:13] = motion_root_ang_vel
        ang_vel_noise = torch.randn_like(init_root_state[:, 10:13]).clamp(-1, 1) * self.init_noise_params["root_ang_vel"]
        init_root_state[:, 10:13] += ang_vel_noise

        # -------- joint state ----------------------------------------------------
        init_joint_pos[:, self.all_joint_idx_asset] = motion_joint_pos[:, self.all_joint_idx_dataset]
        init_joint_vel[:, self.all_joint_idx_asset] = motion_joint_vel[:, self.all_joint_idx_dataset]
        joint_pos_noise = torch.randn_like(init_joint_pos).clamp(-1, 1) * self.init_noise_params["joint_pos"]
        joint_vel_noise = torch.randn_like(init_joint_vel).clamp(-1, 1) * self.init_noise_params["joint_vel"]
        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        # Apply the calculated states to the simulation
        self.asset.write_root_state_to_sim(init_root_state, env_ids=env_ids)

        self.joint_pos_last[env_ids] = init_joint_pos[:, self.joint_idx_asset]
        self.joint_pos_boot_protect[env_ids] = init_joint_pos

        self.asset.write_joint_position_to_sim(init_joint_pos, env_ids=env_ids)
        self.asset.set_joint_position_target(init_joint_pos, env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim(init_joint_vel, env_ids=env_ids)

        self.asset.write_data_to_sim()
    
    def reset(self, env_ids):
        self.finished[env_ids] = False
        self.boot_indicator[env_ids] = self.boot_indicator_max
        self._cum_error[env_ids] = 0.0

    @observation
    def target_pos_b_obs(self):
        current_pos = self.asset.data.root_pos_w.unsqueeze(1) - self.env.scene.env_origins.unsqueeze(1)
        current_quat = self.asset.data.root_quat_w.unsqueeze(1)
        target_pos_b = quat_apply_inverse(
            current_quat,
            (self._motion.root_pos_w - current_pos)
        )

        # # UDP teleop control for xy position (z stays 0)
        # if hasattr(self, "_teleop") and self._teleop is not None:
        #     seq, t_recv, \
        #         root_pos_cmd, root_quat_cmd, \
        #         head_pos_b, head_quat_b, \
        #         l_pos_b, l_quat_b, \
        #         r_pos_b, r_quat_b = self._teleop.get_latest()
            
        #     if seq >= 0:
        #         # root_pos_cmd is the desired xy offset in world frame from UDP
        #         # Convert to body frame: rotate by inverse of current orientation
        #         # root_pos_cmd: [3] tensor with (x, y, z) - we use only xy
        #         target_xy_w = root_pos_cmd[:2].to(self.device)  # [2]
                
        #         # Expand for all envs and future steps
        #         target_xy_w = target_xy_w.unsqueeze(0).unsqueeze(0).expand(self.num_envs, len(self.future_steps), -1)  # [N, S, 2]
                
        #         # Convert to body frame using current robot orientation
        #         # Create full 3D vector for rotation (z=0)
        #         target_xyz_w = torch.cat([target_xy_w, torch.zeros(self.num_envs, len(self.future_steps), 1, device=self.device)], dim=-1)  # [N, S, 3]
        #         target_pos_b = quat_apply_inverse(current_quat, target_xyz_w)  # [N, S, 3]
                
        #         # Keep only xy in body frame, z stays 0
        #         target_pos_b[:, :, 2] = 0.0
        
        # # no move test
        # target_pos_b = torch.zeros(self.num_envs, len(self.future_steps), 3, device=self.device)
        
        return target_pos_b.reshape(self.num_envs, -1)

    def target_pos_b_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1., -1., 1.]
        ).repeat(len(self.future_steps))
    
    @observation
    def target_linvel_b_obs(self):
        target_linvel_b = quat_apply_inverse(self.asset.data.root_quat_w.unsqueeze(1), self._motion.root_lin_vel_w)
        
        # # UDP teleop control for xy velocity (proportional to position command)
        # if hasattr(self, "_teleop") and self._teleop is not None:
        #     seq, t_recv, \
        #         root_pos_cmd, root_quat_cmd, \
        #         head_pos_b, head_quat_b, \
        #         l_pos_b, l_quat_b, \
        #         r_pos_b, r_quat_b = self._teleop.get_latest()
            
        #     if seq >= 0:
        #         # Use position command as velocity (proportional control)
        #         # Scale factor: how fast to move towards target (m/s per m of offset)
        #         vel_scale = 1.0  # 1.0 means if target is 1m away, velocity is 1 m/s
                
        #         target_vel_xy_w = root_pos_cmd[:2].to(self.device) * vel_scale  # [2]
                
        #         # Expand for all envs and future steps
        #         target_vel_xy_w = target_vel_xy_w.unsqueeze(0).unsqueeze(0).expand(self.num_envs, len(self.future_steps), -1)  # [N, S, 2]
                
        #         # Convert to body frame using current robot orientation
        #         target_vel_xyz_w = torch.cat([target_vel_xy_w, torch.zeros(self.num_envs, len(self.future_steps), 1, device=self.device)], dim=-1)  # [N, S, 3]
        #         target_linvel_b = quat_apply_inverse(self.asset.data.root_quat_w.unsqueeze(1), target_vel_xyz_w)  # [N, S, 3]
                
        #         # Keep only xy velocity in body frame, z stays 0
        #         target_linvel_b[:, :, 2] = 0.0

        # # no move test
        # target_linvel_b = torch.zeros_like(target_linvel_b)
        
        return target_linvel_b.reshape(self.num_envs, -1)
    def target_linvel_b_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1., -1., 1.]
        ).repeat(len(self.future_steps))

    @observation
    def target_projected_gravity_b(self):
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).reshape(1, 1, 3)
        g_b = quat_apply_inverse(self._motion.root_quat_w, gravity)  # [N, S, 3]
        # no move test: use identity quaternion for default upright stance
        # target_root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).reshape(1, 1, 4).repeat(self.num_envs, len(self.future_steps), 1)
        # g_b = quat_apply_inverse(target_root_quat, gravity)
        
        return g_b.reshape(self.num_envs, -1)

    def target_projected_gravity_b_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1., -1., 1.]
        ).repeat(len(self.future_steps))

    @observation
    def target_keypoints_b_obs(self):
        target_keypoints_b = self._motion.body_pos_b[:, :, self.keypoint_idx_motion]
        return target_keypoints_b.reshape(self.num_envs, -1)
    def target_keypoints_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1]).repeat(len(self.future_steps))
    
    @observation
    def target_keypoints_diff_b_obs(self):
        actual_w = self.asset.data.body_pos_w[:, self.keypoint_idx_asset] - self.env.scene.env_origins.unsqueeze(1)
        target_w = self._motion.body_pos_w[:, :, self.keypoint_idx_motion]
        diff_w = target_w - actual_w.unsqueeze(1)
        diff_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1).unsqueeze(1),
            diff_w
        )
        return diff_b.reshape(self.num_envs, -1)
    def target_keypoints_diff_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1]).repeat(len(self.future_steps))

    @observation
    def relative_quat_obs(self):
        relative_quat = axis_angle_from_quat(quat_mul(
            self._motion.root_quat_w,
            quat_conjugate(self.asset.data.root_quat_w.unsqueeze(1).expand_as(self._motion.root_quat_w))
        ))  # [N, S, 3] axis-angle
        # no move test: use zero axis-angle (identity rotation = no relative rotation)
        # relative_quat = torch.zeros(self.num_envs, len(self.future_steps), 3, device=self.device)
        return relative_quat.reshape(self.num_envs, -1)
    def relative_quat_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[-1, 1, -1]
        ).repeat(len(self.future_steps))

    @observation
    def target_joint_pos_obs(self):
        return self._motion.joint_pos.reshape(self.num_envs, -1)
    def target_joint_pos_obs_sym(self):
        return sym_utils.joint_space_symmetry(self.asset, self.dataset.joint_names).repeat(len(self.future_steps))

    def _resolve_future_step_indices(self, steps: Sequence[int] | None):
        if steps is None:
            return list(range(len(self.future_steps)))

        future_steps = [int(step) for step in self.future_steps.detach().cpu().tolist()]
        indices = []
        for step in steps:
            step = int(step)
            if step in future_steps:
                indices.append(future_steps.index(step))
            elif 0 <= step < len(future_steps):
                indices.append(step)
            else:
                raise ValueError(
                    f"Requested future step {step} is not in future_steps={future_steps} "
                    f"and is not a valid future-step index."
                )
        return indices

    @observation
    def target_joint_pos_steps_obs(self, steps: Sequence[int] | None = None):
        step_indices = self._resolve_future_step_indices(steps)
        joint_pos = self._motion.joint_pos[:, step_indices]
        return joint_pos.reshape(self.num_envs, -1)
    def target_joint_pos_steps_obs_sym(self, steps: Sequence[int] | None = None):
        step_indices = self._resolve_future_step_indices(steps)
        return sym_utils.joint_space_symmetry(self.asset, self.dataset.joint_names).repeat(len(step_indices))

    def _root_and_wrist_6d_from_motion(self):
        """
        Teleoperation observation: wrist 6D poses only (each 3 pos + 3 axis-angle) in ROOT frame.
        
        Output: [N, 12]
        Order:
          left_wrist_pos_b(3), right_wrist_pos_b(3),
          left_wrist_axis_angle_b(3), right_wrist_axis_angle_b(3)
        
        Note: wrist pos/ori are in ROOT frame.
        """
        motion = self._motion

        # --- Root pose in world frame (still needed for coordinate transform) ---
        if not hasattr(motion, "root_pos_w") or not hasattr(motion, "root_quat_w"):
            print("[WARNING] root_and_wrist_6d: motion missing root_pos_w or root_quat_w; returning zeros", flush=True)
            return torch.zeros(self.num_envs, 12, device=self.device)
        
        root_pos_w = motion.root_pos_w[:, 0]      # [N, 3] world position
        root_quat_w = motion.root_quat_w[:, 0]    # [N, 4] world quaternion
        root_axis_ang = axis_angle_from_quat(root_quat_w)  # [N, 3]

        # --- Wrist poses in root frame ---
        has_local_pos = hasattr(motion, "local_body_pos")
        has_local_rot = hasattr(motion, "local_body_rot")
        
        if has_local_pos:
            pos_step = motion.local_body_pos[:, 0]   # [N, B, 3] root frame
        elif hasattr(motion, "body_pos_b"):
            pos_step = motion.body_pos_b[:, 0]       # [N, B, 3] root frame
        else:
            print("[WARNING] root_and_wrist_6d: motion missing body position data; returning zeros", flush=True)
            return torch.zeros(self.num_envs, 12, device=self.device)

        if has_local_rot:
            rot_step = motion.local_body_rot[:, 0]   # [N, B, 4] root-relative
        elif hasattr(motion, "body_quat_w"):
            rot_w = motion.body_quat_w[:, 0]
            root_quat_conj = quat_conjugate(root_quat_w).unsqueeze(1)
            rot_step = quat_mul(root_quat_conj.expand(-1, rot_w.shape[1], -1), rot_w)
        else:
            print("[WARNING] root_and_wrist_6d: motion missing body rotation data; returning zeros", flush=True)
            return torch.zeros(self.num_envs, 12, device=self.device)

        # wrist body names (left and right only, no head)
        wrist_names = ["left_hand_mimic", "right_hand_mimic"]
        try:
            wrist_idx = [self.dataset.body_names.index(n) for n in wrist_names]
        except ValueError:
            print("[WARNING] root_and_wrist_6d: expected wrist body names not found; returning zeros", flush=True)
            return torch.zeros(self.num_envs, 12, device=self.device)

        wrist_pos = pos_step[:, wrist_idx, :]   # [N, 2, 3]
        wrist_rot = rot_step[:, wrist_idx, :]   # [N, 2, 4]

        # fixed ee pos and rot test - left and right hands with symmetric y
        # left_wrist_pos = torch.tensor([0.15, 0.1, 0.0], device=self.device, dtype=torch.float32)
        # right_wrist_pos = torch.tensor([0.15, -0.1, 0.0], device=self.device, dtype=torch.float32)
        # wrist_pos = torch.stack([left_wrist_pos, right_wrist_pos]).unsqueeze(0).expand(self.num_envs, -1, -1)
        # wrist_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).reshape(1, 1, 4).expand(self.num_envs, 2, -1)

        wrist_axis_ang = axis_angle_from_quat(wrist_rot)  # [N, 2, 3]

        # Pack: left_pos(3), right_pos(3), left_aa(3), right_aa(3) = 12 dims
        out = torch.cat([
            wrist_pos.reshape(self.num_envs, -1),        # [N, 6]
            wrist_axis_ang.reshape(self.num_envs, -1),   # [N, 6]
        ], dim=-1)  # [N, 12]
        # print(f"root_and_wrist_6d output shape: {out.shape}", flush=True)
        # print(f"root_and_wrist_6d output sample: {out[0]}", flush=True)
        

        return out

    def _root_and_wrist_6d_from_udp(self):
        """
        Wrist 6D poses from UDP teleop input, matching _root_and_wrist_6d_from_motion.

        Output: [N, 12]
        Order: left_pos(3), right_pos(3), left_axis_angle(3), right_axis_angle(3).
        """
        if self._teleop is None:
            return torch.zeros(self.num_envs, 12, device=self.device)

        seq, _t_recv, \
            _root_pos_unused, _root_quat_unused, \
            _head_pos_b, _head_quat_b, \
            l_pos_b, l_quat_b, \
            r_pos_b, r_quat_b = self._teleop.get_latest()

        if seq < 0:
            return torch.zeros(self.num_envs, 12, device=self.device)

        pos_sel_b = torch.stack([l_pos_b, r_pos_b], dim=0).to(self.device)
        pos_sel_b = pos_sel_b.unsqueeze(0).expand(self.num_envs, -1, -1)

        quat_sel_b = torch.stack([l_quat_b, r_quat_b], dim=0).to(self.device)
        quat_sel_b = quat_sel_b / (torch.norm(quat_sel_b, dim=-1, keepdim=True) + 1e-8)
        quat_sel_b = quat_sel_b.unsqueeze(0).expand(self.num_envs, -1, -1)

        axis_ang_b = axis_angle_from_quat(quat_sel_b)
        return torch.cat([
            pos_sel_b.reshape(self.num_envs, -1),
            axis_ang_b.reshape(self.num_envs, -1),
        ], dim=-1)

    @observation
    def root_and_wrist_6d(self):
        if self.use_root_and_wrist_6d_command:
            return self.root_and_wrist_6d_command
        if not hasattr(self, "_printed_root_and_wrist_6d_source"):
            print(
                f"[MotionTrackingCommand] root_and_wrist_6d source={self.teleop_obs_source}",
                flush=True,
            )
            self._printed_root_and_wrist_6d_source = True
        if self.teleop_obs_source == "udp":
            return self._root_and_wrist_6d_from_udp()
        return self._root_and_wrist_6d_from_motion()

    def get_root_and_wrist_6d_reference(self):
        if self.use_root_and_wrist_6d_reference_override:
            return self.root_and_wrist_6d_reference_override
        if self.teleop_obs_source == "udp":
            return self._root_and_wrist_6d_from_udp()
        return self._apply_synthetic_ee_target(self._root_and_wrist_6d_from_motion())

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
    def root_and_wrist_6d_reference(self):
        return self.get_root_and_wrist_6d_reference()

    def root_and_wrist_6d_reference_sym(self):
        return self.root_and_wrist_6d_sym()

    def root_and_wrist_6d_sym(self):
        """
        Symmetry for root_and_wrist_6d (wrist only, root commented out):
        - left/right wrist: swap and flip y sign
        """
        # Now only wrist data: left_wrist(3) + right_wrist(3) + left_aa(3) + right_aa(3) = 12
        # Position: [0,1,2] left, [3,4,5] right -> swap to [3,4,5] right, [0,1,2] left with flipped y
        pos_perm = torch.tensor([3, 4, 5, 0, 1, 2])
        pos_signs = torch.tensor([1., -1., 1., 1., -1., 1.])
        
        # Orientation: same pattern as position [6-8] left, [9-11] right -> swap with flipped signs
        ori_perm = torch.tensor([9, 10, 11, 6, 7, 8])
        ori_signs = torch.tensor([-1., 1., -1., -1., 1., -1.])
        
        perm = torch.cat([pos_perm, ori_perm])
        signs = torch.cat([pos_signs, ori_signs])
        
        return sym_utils.SymmetryTransform(perm=perm, signs=signs)

    @observation
    def feet_pos_b(self):
        if self.use_feet_pos_b_command:
            return self.feet_pos_b_command
        return self.get_feet_pos_b_reference()

    def get_feet_pos_b_reference(self):
        if self.teleop_obs_source == "udp":
            return self._feet_pos_b_from_udp()
        return self._feet_pos_b_from_motion()

    def _feet_pos_b_from_motion(self):
        motion = self._motion
        if hasattr(motion, "local_body_pos"):
            target_feet_b = motion.local_body_pos[:, 0, self.feet_idx_motion, :]
        elif hasattr(motion, "body_pos_b"):
            target_feet_b = motion.body_pos_b[:, 0, self.feet_idx_motion, :]
        else:
            return torch.zeros(self.num_envs, 6, device=self.device)
        return target_feet_b.reshape(self.num_envs, -1)

    def _feet_pos_b_from_udp(self):
        if self._teleop is not None:
            has_feet_pos_b, feet_pos_b = self._teleop.get_latest_feet_pos_b()
            if has_feet_pos_b:
                return feet_pos_b.to(self.device).unsqueeze(0).expand(self.num_envs, -1)
        return self._feet_pos_b_from_motion()

    def set_feet_pos_b_command(self, command: torch.Tensor):
        if command.shape[-1] != 6:
            raise ValueError(f"Feet position command must have 6 dims, got {command.shape[-1]}.")
        self.use_feet_pos_b_command = True
        self.feet_pos_b_command[:] = command

    def feet_pos_b_sym(self):
        return sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.feet_idx_asset),
            sign=[1, -1, 1],
        )


    @observation
    def head_and_wrist_6d(self):
        """
        Teleoperation observation from motion data: for selected bodies (head, hand/wrist)
        return 6D pose per body as position (3) + orientation as axis-angle (3),
        both in ROOT frame.
        
        Uses `local_body_pos` and `local_body_rot` which are already root-relative.
        """
        motion = self._motion

        # Check for required fields (local_body_pos and local_body_rot are root-relative)
        has_local_pos = hasattr(motion, "local_body_pos")
        has_local_rot = hasattr(motion, "local_body_rot")
        
        # Fallback to body_pos_b if local_body_pos not available
        if has_local_pos:
            pos_step = motion.local_body_pos[:, 0]   # [N, B, 3] -- already in root frame
        elif hasattr(motion, "body_pos_b"):
            pos_step = motion.body_pos_b[:, 0]       # [N, B, 3] -- also root frame
        else:
            print("[WARNING] head_and_wrist_6d: motion missing position data; returning zeros", flush=True)
            return torch.zeros(self.num_envs, 18, device=self.device)

        # For rotation: local_body_rot is already root-relative, no conversion needed!
        if has_local_rot:
            rot_step = motion.local_body_rot[:, 0]   # [N, B, 4] -- already root-relative
        elif hasattr(motion, "body_quat_w"):
            # Fallback: convert world quaternions to root-relative
            rot_w = motion.body_quat_w[:, 0]         # [N, B, 4] world-frame
            root_quat = motion.root_quat_w[:, 0]     # [N, 4]
            root_quat_conj = quat_conjugate(root_quat).unsqueeze(1)  # [N, 1, 4]
            rot_step = quat_mul(root_quat_conj.expand(-1, rot_w.shape[1], -1), rot_w)
        else:
            print("[WARNING] head_and_wrist_6d: motion missing rotation data; returning zeros", flush=True)
            return torch.zeros(self.num_envs, 18, device=self.device)

        # explicit names and their order (must match symmetry builder)
        sel_names = ["head_mimic", "left_hand_mimic", "right_hand_mimic"]
        try:
            sel_idx = [self.dataset.body_names.index(n) for n in sel_names]
        except ValueError:
            print("[WARNING] head_and_wrist_6d: expected body names not found in dataset; returning zeros", flush=True)
            return torch.zeros(self.num_envs, 18, device=self.device)

        # select bodies
        pos_sel = pos_step[:, sel_idx, :]   # [N, 3, 3]
        rot_sel = rot_step[:, sel_idx, :]   # [N, 3, 4] root-relative quaternions

        # axis-angle from quaternion (result [N, 3, 3])
        axis_ang = axis_angle_from_quat(rot_sel)

        out = torch.cat([pos_sel.reshape(self.num_envs, -1), axis_ang.reshape(self.num_envs, -1)], dim=-1)
        return out
    
    def head_and_wrist_6d_sym(self):
        # build symmetry for the three selected bodies (head, left hand, right hand)
        # Must match the order used in `head_and_wrist_6d` (head, left_hand, right_hand).
        sel_names = ["head_mimic", "left_hand_mimic", "right_hand_mimic"]
        return sym_utils.cartesian_space_symmetry(self.asset, sel_names, sign=[1, -1, 1]).repeat(2)


    @observation
    def current_keypoint_b(self):
        actual_w = self.asset.data.body_pos_w[:, self.keypoint_idx_asset]
        actual_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            actual_w - self.asset.data.root_pos_w.unsqueeze(1)
        )
        return actual_b.reshape(self.num_envs, -1)
    def current_keypoint_b_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1])

    @observation
    def current_keypoint_vel_b(self):
        actual_vel_w = self.asset.data.body_lin_vel_w[:, self.keypoint_idx_asset]
        actual_vel_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            actual_vel_w
        )
        return actual_vel_b.reshape(self.num_envs, -1)
    def current_keypoint_vel_b_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1])
    
    @observation
    def boot_indicator_state(self):
        return self.boot_indicator / self.boot_indicator_max
    def boot_indicator_state_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1.])

    @reward
    def root_pos_tracking(self):
        current_pos = self.asset.data.root_pos_w
        target_pos = self.reward_root_pos_w
        diff = target_pos - current_pos
        error = diff.norm(dim=-1, keepdim=True)
        self._cum_error[:, 0:1] = error / self._cum_root_pos_scale
        return _calc_exp_sigma(error, self.reward_sigma["root_pos"])

    @reward
    def root_vel_tracking(self):
        current_linvel_w = self.asset.data.root_lin_vel_w
        current_quat = self.asset.data.root_quat_w
        ref_linvel_w = self._motion.root_lin_vel_w[:, 0]
        ref_quat = self._motion.root_quat_w[:, 0, :]

        current_linvel_b = quat_apply_inverse(current_quat, current_linvel_w)
        ref_linvel_b = quat_apply_inverse(ref_quat, ref_linvel_w)
        diff = ref_linvel_b - current_linvel_b

        error = diff.norm(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["root_vel"])

    @reward
    def root_rot_tracking(self):
        current_quat = self.asset.data.root_quat_w
        target_quat = self.reward_root_quat_w
        diff = axis_angle_from_quat(quat_mul(
            target_quat,
            quat_conjugate(current_quat)
        ))
        error = torch.norm(diff, dim=-1, keepdim=True)
        self._cum_error[:, 1:2] = error / self._cum_orientation_scale
        return _calc_exp_sigma(error, self.reward_sigma["root_rot"])
    
    @reward
    def root_ang_vel_tracking(self):
        current_angvel_w = self.asset.data.root_ang_vel_w
        current_quat = self.asset.data.root_quat_w
        ref_angvel_w = self._motion.root_ang_vel_w[:, 0]
        ref_quat = self._motion.root_quat_w[:, 0, :]

        current_angvel_b = quat_apply_inverse(current_quat, current_angvel_w)
        ref_angvel_b = quat_apply_inverse(ref_quat, ref_angvel_w)
        diff = ref_angvel_b - current_angvel_b

        error = diff.norm(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["root_ang_vel"])

    @reward
    def keypoint_tracking(self):
        actual = self.asset.data.body_pos_w[:, self.keypoint_idx_asset]
        target = self.reward_keypoints_w[:, self.keypoint_idx_motion]
        diff = target - actual
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        self._cum_error[:, 2:3] = error / self._cum_keypoint_scale
        return _calc_exp_sigma(error, self.reward_sigma["keypoint"])
    
    @reward
    def lower_keypoint_tracking(self):
        actual = self.asset.data.body_pos_w[:, self.lower_keypoint_idx_asset]
        target = self.reward_keypoints_w[:, self.lower_keypoint_idx_motion]
        diff = target - actual
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["lower_keypoint"])

    @reward
    def ee_tracking(self):
        """
        Reward for tracking end-effector (wrist) positions.
        Target comes from motion data (same source as root_and_wrist_6d observation).
        Wrist positions are in root frame, so we compare in root frame.
        """
        # Get target wrist positions from motion (root frame)
        motion = self._motion
        wrist_names = ["left_hand_mimic", "right_hand_mimic"]
        try:
            wrist_idx_motion = [self.dataset.body_names.index(n) for n in wrist_names]
            wrist_idx_asset = [self.asset.body_names.index(n) for n in wrist_names]
        except ValueError:
            return torch.zeros(self.num_envs, 1, device=self.device)
        
        # Target wrist positions in root frame
        if hasattr(motion, "local_body_pos"):
            target_wrist_b = motion.local_body_pos[:, 0, wrist_idx_motion, :].clone()  # [N, 2, 3]
        elif hasattr(motion, "body_pos_b"):
            target_wrist_b = motion.body_pos_b[:, 0, wrist_idx_motion, :].clone()      # [N, 2, 3]
        else:
            return torch.zeros(self.num_envs, 1, device=self.device)
        
        # Actual wrist positions (convert to root frame)
        actual_wrist_w = self.asset.data.body_pos_w[:, wrist_idx_asset, :]  # [N, 2, 3]
        root_pos = self.asset.data.root_pos_w.unsqueeze(1)                   # [N, 1, 3]
        root_quat = self.asset.data.root_quat_w.unsqueeze(1)                 # [N, 1, 4]
        actual_wrist_b = quat_apply_inverse(root_quat, actual_wrist_w - root_pos)  # [N, 2, 3]

        if (
            getattr(self, "external_force_mode", "legacy") == "net_pull"
            and getattr(self, "use_net_pull_ee_target_for_tracking", False)
            and hasattr(self, "get_net_pull_ee_compliance_target_b")
        ):
            target_wrist_b = self.get_net_pull_ee_compliance_target_b().clone()
        elif getattr(self, "compliance", False) and hasattr(self, "force_keypoint_w"):
            force_apply_idx_asset = self.force_apply_idx_asset.tolist()
            for wrist_i, asset_i in enumerate(wrist_idx_asset):
                if asset_i in force_apply_idx_asset:
                    force_i = force_apply_idx_asset.index(asset_i)
                    target_wrist_b[:, wrist_i, :] = quat_apply_inverse(
                        root_quat[:, 0],
                        self.force_keypoint_w[:, force_i, :] - root_pos[:, 0],
                    )
        
        # Compute error
        diff = target_wrist_b - actual_wrist_b
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)  # [N, 1]
        return _calc_exp_sigma(error, self.reward_sigma["ee"])

    @reward
    def ee_rot_tracking(self):
        """
        Optional reward for tracking wrist orientations in the robot root frame.
        This is separate from ee_tracking, which only tracks wrist positions.
        """
        motion = self._motion
        wrist_names = ["left_hand_mimic", "right_hand_mimic"]
        try:
            wrist_idx_motion = [self.dataset.body_names.index(n) for n in wrist_names]
            wrist_idx_asset = [self.asset.body_names.index(n) for n in wrist_names]
        except ValueError:
            return torch.zeros(self.num_envs, 1, device=self.device)

        if hasattr(motion, "local_body_rot"):
            target_wrist_quat_b = motion.local_body_rot[:, 0, wrist_idx_motion, :]
        elif hasattr(motion, "body_quat_w") and hasattr(motion, "root_quat_w"):
            target_root_quat_w = motion.root_quat_w[:, 0, :].unsqueeze(1)
            target_wrist_quat_w = motion.body_quat_w[:, 0, wrist_idx_motion, :]
            target_wrist_quat_b = quat_mul(
                quat_conjugate(target_root_quat_w).expand(-1, len(wrist_idx_motion), -1),
                target_wrist_quat_w,
            )
        else:
            return torch.zeros(self.num_envs, 1, device=self.device)

        actual_root_quat_w = self.asset.data.root_quat_w.unsqueeze(1)
        actual_wrist_quat_w = self.asset.data.body_quat_w[:, wrist_idx_asset, :]
        actual_wrist_quat_b = quat_mul(
            quat_conjugate(actual_root_quat_w).expand(-1, len(wrist_idx_asset), -1),
            actual_wrist_quat_w,
        )

        diff_quat = quat_mul(target_wrist_quat_b, quat_conjugate(actual_wrist_quat_b))
        error = axis_angle_from_quat(diff_quat).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["ee_rot"])

    @reward
    def feet_tracking(self):
        motion = self._motion
        if hasattr(motion, "local_body_pos"):
            target_feet_b = motion.local_body_pos[:, 0, self.feet_idx_motion, :]
        elif hasattr(motion, "body_pos_b"):
            target_feet_b = motion.body_pos_b[:, 0, self.feet_idx_motion, :]
        else:
            return torch.zeros(self.num_envs, 1, device=self.device)

        actual_feet_w = self.asset.data.body_pos_w[:, self.feet_idx_asset, :]
        root_pos = self.asset.data.root_pos_w.unsqueeze(1)
        root_quat = self.asset.data.root_quat_w.unsqueeze(1)
        actual_feet_b = quat_apply_inverse(root_quat, actual_feet_w - root_pos)

        diff = target_feet_b - actual_feet_b
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["feet"])

    @reward
    def joint_pos_tracking(self):
        actual = self.asset.data.joint_pos[:, self.joint_idx_asset]
        target = self._motion.joint_pos[:, 0, self.joint_idx_motion]
        error = (target - actual).abs().mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["joint_pos"])

    @reward
    def joint_vel_tracking(self):
        current_joint_pos = self.asset.data.joint_pos[:, self.joint_idx_asset]
        vel_diff = (current_joint_pos - self.joint_pos_last) / self.env.step_dt
        self.joint_pos_last[:] = current_joint_pos

        target = self._motion.joint_vel[:, 0, self.joint_idx_motion]
        error = (target - vel_diff).abs().mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["joint_vel"])

    def update_reward_target(self):
        delta_yaw = quat_mul(yaw_quat(self.asset.data.root_quat_w), quat_conjugate(yaw_quat(self._motion.root_quat_w[:, 0])))
        tgt_rel = self._motion.body_pos_w[:, 0] - self._motion.root_pos_w[:, 0].unsqueeze(1)
        self.reward_keypoints_w = quat_apply(delta_yaw.unsqueeze(1), tgt_rel) + self.asset.data.root_pos_w.unsqueeze(1)
        self.reward_keypoints_w[:, 2] = self._motion.body_pos_w[:, 0, 2]

        if not self.student_train:
            self.reward_root_pos_w = self._motion.root_pos_w[:, 0] + self.env.scene.env_origins
            self.reward_root_quat_w = self._motion.root_quat_w[:, 0]
        else:
            steps = 50 # calc t+50 target root pos/rot from current root pos/rot
            # prepare future root pos/rot cache
            if hasattr(self, 'ts_root_pos_w') is False:
                self.ts_root_pos_w = torch.zeros(self.num_envs, steps, 3, device=self.device, dtype=torch.float32)
                self.ts_root_quat_w = torch.zeros(self.num_envs, steps, 4, device=self.device, dtype=torch.float32)
            # update only for reset envs
            if self.last_reset_env_ids is not None:
                future_motion = self.dataset.get_slice(self.last_reset_env_ids, self.t[self.last_reset_env_ids], steps=steps)
                self.ts_root_pos_w[self.last_reset_env_ids] = future_motion.root_pos_w + self.env.scene.env_origins[self.last_reset_env_ids].unsqueeze(1)
                self.ts_root_quat_w[self.last_reset_env_ids] = future_motion.root_quat_w
            # get current root pos/rot from cache
            reward_pos = self.ts_root_pos_w[:, 0].clone()
            reward_quat = self.ts_root_quat_w[:, 0].clone()
            # roll forward the cache (clone to avoid memory overlap issue)
            self.ts_root_pos_w[:, :-1] = self.ts_root_pos_w[:, 1:].clone()
            self.ts_root_quat_w[:, :-1] = self.ts_root_quat_w[:, 1:].clone()
            # compute target root pos/rot at t+steps
            current_pos_t = self.asset.data.root_pos_w
            current_quat_t = self.asset.data.root_quat_w
            ref_motion_plus = self.dataset.get_slice(None, self.t, steps=torch.tensor([steps], device=self.device, dtype=torch.int64))
            ref_pos_t = self._motion.root_pos_w[:, 0]
            ref_pos_t_plus = ref_motion_plus.root_pos_w[:, 0]
            ref_quat_t = self._motion.root_quat_w[:, 0]
            ref_quat_t_plus = ref_motion_plus.root_quat_w[:, 0]

            delta_yaw = quat_mul(yaw_quat(current_quat_t), quat_conjugate(yaw_quat(ref_quat_t)))
            self.ts_root_pos_w[:, -1] = quat_apply(delta_yaw, (ref_pos_t_plus - ref_pos_t)) + current_pos_t
            self.ts_root_pos_w[:, -1, 2] = ref_pos_t_plus[:, 2]  # recover z axis
            self.ts_root_quat_w[:, -1] = quat_mul(delta_yaw, ref_quat_t_plus)

            self.reward_root_pos_w = reward_pos
            self.reward_root_quat_w = reward_quat

        if getattr(self, "compliance", False) and hasattr(self, "root_compliance_offset_xy"):
            self.reward_root_pos_w[:, :2] += self.update_root_compliance_offset()

    def before_update(self):
        self.t = torch.clamp_max(self.t + 1, self.lengths - 1)
        # In teleop mode, never mark as finished (disable auto-reset from motion end)
        if self.disable_motion_finish:
            self.finished[:] = False
        else:
            self.finished[:] = self.t >= self.lengths - 1
        self.boot_indicator[:] = torch.clamp_min(self.boot_indicator - 1, 0)

        self._motion = self.dataset.get_slice(None, self.t, steps=self.future_steps)

        feet_vel = (self._motion.body_pos_w[:, 1, self.feet_idx_motion] - self._motion.body_pos_w[:, 0, self.feet_idx_motion]) / ((self.future_steps[1] - self.future_steps[0]) * self.env.step_dt) # [N, 2, 3]
        self.feet_standing = (feet_vel[:, :, :2].norm(dim=-1, keepdim=False) < 0.1)

    def update(self):
        self.dataset.update()
        if self.last_reset_env_ids is not None:
            self.last_reset_env_ids = None
        # Log UDP target and actual EE position/pose, save to CSV every 10s
        try:
            if hasattr(self, "_teleop") and self._teleop is not None:
                seq, t_recv, root_pos_udp, root_quat_udp, head_pos_b, head_quat_b, l_pos_b, l_quat_b, r_pos_b, r_quat_b = self._teleop.get_latest()
                # Only log if UDP data is valid
                if seq >= 0:
                    # Target EE position/pose (UDP, left/right hand)
                    target_pos = torch.cat([l_pos_b, r_pos_b]).cpu().numpy().tolist()  # [6]
                    target_pose = torch.cat([l_quat_b, r_quat_b]).cpu().numpy().tolist()  # [8]
                    self.target_position_list.append(target_pos)
                    self.target_pose_list.append(target_pose)
                    # Actual EE position/pose (robot, left/right hand)
                    wrist_names = ["left_hand_mimic", "right_hand_mimic"]
                    try:
                        wrist_idx_asset = [self.asset.body_names.index(n) for n in wrist_names]
                        # world-frame per-env
                        l_current_w = self.asset.data.body_pos_w[:, wrist_idx_asset[0], :]  # [N,3]
                        r_current_w = self.asset.data.body_pos_w[:, wrist_idx_asset[1], :]  # [N,3]
                        l_current_quat_w = self.asset.data.body_quat_w[:, wrist_idx_asset[0], :]  # [N,4]
                        r_current_quat_w = self.asset.data.body_quat_w[:, wrist_idx_asset[1], :]  # [N,4]

                        # root state per-env
                        root_pos = self.asset.data.root_pos_w  # [N,3]
                        root_quat = self.asset.data.root_quat_w  # [N,4]

                        # convert positions to ROOT frame: p_b = quat_apply_inverse(root_quat, p_w - root_pos)
                        l_current_b = quat_apply_inverse(root_quat, l_current_w - root_pos)  # [N,3]
                        r_current_b = quat_apply_inverse(root_quat, r_current_w - root_pos)  # [N,3]

                        # convert quaternions to ROOT frame: q_b = conj(root_quat) * q_w
                        root_conj = quat_conjugate(root_quat)
                        l_quat_b = quat_mul(root_conj, l_current_quat_w)  # [N,4]
                        r_quat_b = quat_mul(root_conj, r_current_quat_w)  # [N,4]

                        # average across envs to produce a single representative sample
                        l_b_mean = l_current_b.mean(dim=0).cpu().numpy().tolist()
                        r_b_mean = r_current_b.mean(dim=0).cpu().numpy().tolist()
                        l_q_mean = l_quat_b.mean(dim=0).cpu().numpy().tolist()
                        r_q_mean = r_quat_b.mean(dim=0).cpu().numpy().tolist()

                        actual_pos = l_b_mean + r_b_mean  # [6] in ROOT frame
                        actual_pose = l_q_mean + r_q_mean  # [8] quaternions in ROOT frame
                        self.actual_position_list.append(actual_pos)
                        self.actual_pose_list.append(actual_pose)
                    except Exception:
                        pass
                # Broadcast root state via UDP if enabled (one line CSV: tstamp, x,y,z, qw,qx,qy,qz)
                try:
                    if getattr(self, "_udp_broadcast_enabled", False) and self._udp_broadcast_sock is not None:
                        # send root state for env 0 (absolute world frame)
                        rp = self.asset.data.root_pos_w[0]  # [3]
                        rq = self.asset.data.root_quat_w[0]  # [4] (w,x,y,z)
                        tstamp = time.time()
                        msg = f"{tstamp:.6f},{rp[0].item():.6f},{rp[1].item():.6f},{rp[2].item():.6f},{rq[0].item():.6f},{rq[1].item():.6f},{rq[2].item():.6f},{rq[3].item():.6f}"
                        try:
                            self._udp_broadcast_sock.sendto(msg.encode('ascii'), self._udp_broadcast_addr)
                            # increment sequence and optionally print every N sends
                            # try:
                            #     self._udp_broadcast_seq += 1
                            #     pe = getattr(self, "_udp_broadcast_print_every", 0)
                            #     if pe and (self._udp_broadcast_seq % pe) == 0:
                            #         print(f"[MOTION_TRACKING][UDP SEND #{self._udp_broadcast_seq}] -> {self._udp_broadcast_addr} msg={msg}", flush=True)
                            # except Exception:
                            #     # non-critical debug failure
                            #     pass
                        except BlockingIOError:
                            pass
                        except Exception:
                            # disable broadcaster on repeated failure
                            try:
                                self._udp_broadcast_sock.close()
                            except Exception:
                                pass
                            self._udp_broadcast_sock = None
                            self._udp_broadcast_enabled = False
                except Exception:
                    pass
            # Save to CSV every 10s
            now = time.time()
            if now - self._last_csv_save_time >= 10.0:
                import csv
                import os
                csv_path = os.path.join(os.getcwd(), "ee_tracking_log.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    # Header: target pos (Lx,Ly,Lz,Rx,Ry,Rz), target RPY (Lr,Lp,Ly, Rr,Rp,Ry),
                    # actual pos (Lx,Ly,Lz,Rx,Ry,Rz), actual RPY (Lr,Lp,Ly, Rr,Rp,Ry)
                    header = [
                        "t_lx","t_ly","t_lz","t_rx","t_ry","t_rz",
                        "t_lr","t_lp","t_ly","t_rr","t_rp","t_ry",
                        "a_lx","a_ly","a_lz","a_rx","a_ry","a_rz",
                        "a_lr","a_lp","a_ly","a_rr","a_rp","a_ry",
                    ]
                    writer.writerow(header)
                    for tp, tpo, ap, apo in zip(self.target_position_list, self.target_pose_list, self.actual_position_list, self.actual_pose_list):
                        try:
                            # target positions (6 floats)
                            tpos = list(tp)
                            # target quaternions: two quaternions concatenated (w,x,y,z, w,x,y,z)
                            tq_l = tpo[0:4]
                            tq_r = tpo[4:8]
                            t_rpy_l = quat_wxyz_to_rpy(tq_l)
                            t_rpy_r = quat_wxyz_to_rpy(tq_r)

                            # actual positions (6 floats)
                            apos = list(ap)
                            aq_l = apo[0:4]
                            aq_r = apo[4:8]
                            a_rpy_l = quat_wxyz_to_rpy(aq_l)
                            a_rpy_r = quat_wxyz_to_rpy(aq_r)

                            row = tpos + list(t_rpy_l) + list(t_rpy_r) + apos + list(a_rpy_l) + list(a_rpy_r)
                            writer.writerow(row)
                        except Exception:
                            # fallback: write raw lists if conversion fails
                            writer.writerow([tp, tpo, ap, apo])
                self._last_csv_save_time = now
        except Exception as e:
            print(f"[EE TRACKING LOG ERROR] {e}", flush=True)

    def debug_draw(self):
        root_pos = self.asset.data.root_pos_w    # [N,1,3]
        root_quat = self.asset.data.root_quat_w  # [N,1,4]
        target_root_quat = self._motion.root_quat_w[:, 0, :]
        heading_rel = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        heading_world = quat_apply(root_quat, heading_rel)
        heading_world_target = quat_apply(target_root_quat, heading_rel)

        # —— original world‐frame drawing —— 
        target_keypoints_w = self.reward_keypoints_w[:, self.keypoint_idx_motion]
        robot_keypoints_w = self.asset.data.body_pos_w[:, self.keypoint_idx_asset]

        # draw points and error vectors
        # self.env.debug_draw.point(
        #     target_keypoints_w.reshape(-1, 3), color=(1, 0, 0, 1)
        # )
        # self.env.debug_draw.point(
        #     robot_keypoints_w.reshape(-1, 3), color=(0, 1, 0, 1)
        # )
        # self.env.debug_draw.vector(
        #     robot_keypoints_w.reshape(-1, 3),
        #     (target_keypoints_w - robot_keypoints_w).reshape(-1, 3),
        #     color=(0, 0, 1, 1)
        # )
        
        # self.env.debug_draw.vector(
        #     root_pos.reshape(-1, 3),
        #     heading_world.reshape(-1, 3),
        #     color=(0, 0, 1, 2)
        # )
        
        # self.env.debug_draw.vector(
        #     root_pos.reshape(-1, 3),
        #     heading_world_target.reshape(-1, 3),
        #     color=(1, 0, 0, 2)
        # )

        robot_root_pos = self.asset.data.root_pos_w
        robot_root_quat = self.asset.data.root_quat_w
        robot_root_yaw = yaw_quat(robot_root_quat)

        command_obs = self.command()
        root_height = command_obs[:, 0]
        target_linvel_b_xy = command_obs[:, 1:3]
        target_heading_b_xy = command_obs[:, 3:5]

        target_linvel_b = torch.cat([
            target_linvel_b_xy,
            torch.zeros(self.num_envs, 1, device=self.device),
        ], dim=-1)
        target_linvel_w = quat_apply(robot_root_yaw, target_linvel_b)
        root_target_xy_w = robot_root_pos.clone()
        root_target_xy_w[:, :2] += target_linvel_w[:, :2]
        root_target_xy_w[:, 2] = root_height
        self.env.debug_draw.point(root_target_xy_w, color=(1, 0.5, 0, 1), size=15.0)

        arrow_start = robot_root_pos.clone()
        arrow_start[:, 2] = root_height
        target_heading_b = torch.cat([
            target_heading_b_xy,
            torch.zeros(self.num_envs, 1, device=self.device),
        ], dim=-1)
        arrow_dir = quat_apply(robot_root_yaw, target_heading_b)
        arrow_dir[:, 2] = 0.0
        arrow_dir = normalize(arrow_dir) * 0.5
        self.env.debug_draw.vector(arrow_start, arrow_dir, color=(1, 1, 0, 1), size=5.0)

        wrist_obs = self.root_and_wrist_6d().reshape(self.num_envs, 4, 3)
        wrist_pos_b = wrist_obs[:, 0:2]
        wrist_axis_angle_b = wrist_obs[:, 2:4]
        root_quat_for_wrist = robot_root_quat.unsqueeze(1).expand(-1, 2, -1)
        wrist_target_w = quat_apply(root_quat_for_wrist.reshape(-1, 4), wrist_pos_b.reshape(-1, 3))
        wrist_target_w = wrist_target_w.reshape(self.num_envs, 2, 3) + robot_root_pos.unsqueeze(1)

        angle = torch.linalg.norm(wrist_axis_angle_b, dim=-1)
        axis = wrist_axis_angle_b / angle.unsqueeze(-1).clamp_min(1e-6)
        wrist_quat_b = quat_from_angle_axis(angle.reshape(-1), axis.reshape(-1, 3)).reshape(self.num_envs, 2, 4)
        wrist_quat_w = quat_mul(root_quat_for_wrist.reshape(-1, 4), wrist_quat_b.reshape(-1, 4))
        wrist_quat_w = wrist_quat_w.reshape(self.num_envs, 2, 4)

        self.env.debug_draw.point(
            wrist_target_w.reshape(-1, 3), color=(0, 1, 1, 1), size=12.0
        )

        axis_len = 0.08
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 2, -1)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, 2, -1)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 2, -1)

        x_w = quat_apply(wrist_quat_w, x_axis) * axis_len
        y_w = quat_apply(wrist_quat_w, y_axis) * axis_len
        z_w = quat_apply(wrist_quat_w, z_axis) * axis_len
        wrist_target_w_flat = wrist_target_w.reshape(-1, 3)
        self.env.debug_draw.vector(wrist_target_w_flat, x_w.reshape(-1, 3), color=(1.0, 0.4, 0.4, 1.0), size=3.0)
        self.env.debug_draw.vector(wrist_target_w_flat, y_w.reshape(-1, 3), color=(0.4, 1.0, 0.4, 1.0), size=3.0)
        self.env.debug_draw.vector(wrist_target_w_flat, z_w.reshape(-1, 3), color=(0.4, 0.4, 1.0, 1.0), size=3.0)

        feet_pos_b = self.feet_pos_b().reshape(self.num_envs, -1, 3)
        num_feet = feet_pos_b.shape[1]
        root_quat_for_feet = robot_root_quat.unsqueeze(1).expand(-1, num_feet, -1)
        feet_target_w = quat_apply(root_quat_for_feet.reshape(-1, 4), feet_pos_b.reshape(-1, 3))
        feet_target_w = feet_target_w.reshape(self.num_envs, num_feet, 3) + robot_root_pos.unsqueeze(1)
        self.env.debug_draw.point(
            feet_target_w.reshape(-1, 3), color=(0.2, 1.0, 0.2, 1.0), size=10.0
        )

from .utils import TemporalLerp, clamp_norm, create_mapping, rand_points_disk, rand_points_isotropic, random_uniform
import os
from collections.abc import Sequence as SequenceABC
from omegaconf import ListConfig


def _parse_xyz_or_scalar(value, *, name: str, device: torch.device) -> tuple[torch.Tensor, bool]:
    if isinstance(value, (SequenceABC, ListConfig)) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(f"{name} must be a scalar or 3 xyz values, got {value}.")
        tensor = torch.tensor(value, dtype=torch.float32, device=device).reshape(1, 1, 3)
        directional = True
    else:
        tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
        directional = False
    if torch.any(tensor <= 0.0):
        raise ValueError(f"{name} must be positive, got {value}.")
    return tensor, directional


class MotionTrackingCommand_impedance(MotionTrackingCommand):
    def __init__(
        self,
        env,
        *args,
        max_force: float = 30.0,
        compliance: bool = True,
        net_force_limit: float = 30.0,
        net_torque_limit: float = 20.0,
        external_force_mode: str = "legacy",
        net_pull_apply_pattern: Sequence[str] | None = None,
        net_pull_force_range: Sequence[float] = (80.0, 200.0),
        net_pull_ramp_up_range: Sequence[int] = (25, 75),
        net_pull_hold_range: Sequence[int] = (100, 250),
        net_pull_ramp_down_range: Sequence[int] = (25, 75),
        net_pull_rest_range: Sequence[int] = (50, 150),
        net_pull_xy_only: bool = True,
        net_pull_zero_prob: float = 0.0,
        net_pull_bimanual_prob: float = 0.0,
        net_pull_curriculum: bool = False,
        net_pull_curriculum_end: float = 0.6,
        net_pull_zero_prob_start: float = 0.85,
        net_pull_force_scale_start: float = 0.25,
        net_pull_margin_aware: bool = False,
        net_pull_margin_min_force: float = 5.0,
        net_pull_margin_safety: float = 0.9,
        net_pull_margin_resample_attempts: int = 32,
        net_pull_margin_left_range: Sequence[Sequence[float]] | None = None,
        net_pull_margin_right_range: Sequence[Sequence[float]] | None = None,
        net_pull_ee_compliance_stiffness: float = 200.0,
        net_pull_ee_compliance_stiffness_range: Sequence[float] | None = None,
        net_pull_ee_compliance_stiffness_xyz_range: Sequence[Sequence[float]] | None = None,
        net_pull_ee_compliance_stiffness_levels: Sequence[float] | None = None,
        net_pull_ee_compliance_stiffness_levels_prob: float = 0.0,
        net_pull_ee_compliance_stiffness_xyz_anchors: Sequence[Sequence[float]] | None = None,
        net_pull_ee_compliance_stiffness_anchor_prob: float = 0.0,
        net_pull_ee_compliance_stiffness_inverse_command: bool = False,
        net_pull_ee_compliance_stiffness_per_ee: bool = False,
        net_pull_ee_compliance_max_offset: float = 0.25,
        net_pull_ee_compliance_smoothing: bool = True,
        net_pull_ee_compliance_force_deadband: float = 5.0,
        use_net_pull_ee_target_for_tracking: bool = False,
        sync_net_pull_ee_reward_targets: bool = False,
        external_force_enabled: bool = True,
        force_apply_pattern: Sequence[str] | None = None,
        force_active_pattern: Sequence[str] | None = None,
        force_safe_bounds: Sequence[float] = (5.0, 15.0),
        force_penalty_offset: float = 10.0,
        force_safe_default: float = 10.0,
        **kwargs,
    ):
        super().__init__(env, *args, **kwargs)

        force_apply_pattern = list(force_apply_pattern or [".*shoulder_yaw_link", ".*wrist_roll_link", ".*hand_mimic"])
        self.force_apply_idx_motion, self.force_apply_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            force_apply_pattern,
            name_map=self.keypoint_map,
        )
        self.force_in_keypoint_idx = create_mapping(self.keypoint_idx_asset, self.force_apply_idx_asset, self.device)
        self.num_force_bodies = self.force_apply_idx_asset.shape[0]

        self.max_force = max_force
        self.net_force_limit  = torch.as_tensor(net_force_limit,  dtype=torch.float32, device=self.device)
        self.net_torque_limit = torch.as_tensor(net_torque_limit, dtype=torch.float32, device=self.device)
        self.compliance = compliance
        self.external_force_enabled = external_force_enabled
        self.external_force_mode = external_force_mode
        if self.external_force_mode not in ("legacy", "net_pull"):
            raise ValueError(f"Unsupported external_force_mode: {self.external_force_mode}")

        # force origin samples
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "../../../..")
        hand_samples_path = os.path.join(project_root, "scripts/data_process/hand_grid_samples.pt")
        self.force_origin_samples_raw = torch.load(hand_samples_path, map_location=self.device)
        self.force_origin_samples_left = self.force_origin_samples_raw["left"].to(self.device)
        self.force_origin_samples_right = self.force_origin_samples_raw["right"].to(self.device)

        self.force_origin_sample_prob = 0.5
        self.force_origin_samples = self.force_origin_samples_left.shape[0]

        self.kp_range = (5.0, 250.0)
        self.kp_slope_range = (-5.0, 5.0)
        self.zero_kp_slope_prob = 0.5
        self.kp_time_range = (25, 100)
        self.force_time_range = (20, 200)
        self.ramping_time_range = (25, 100)
        self.force_origin_transit_time_range = (25, 100)
        self.root_compliance_force_threshold = 15.0
        self.root_compliance_gain = 0.01
        self.root_compliance_max_offset = 0.3
        self.root_compliance_ema = 0.95
        if len(force_safe_bounds) != 2:
            raise ValueError(f"force_safe_bounds must have 2 values, got {force_safe_bounds}")
        self.force_safe_bounds = tuple(float(x) for x in force_safe_bounds)
        self.force_penalty_offset = float(force_penalty_offset)
        self.force_safe_default = float(force_safe_default)

        self.skip_ref = False

        with torch.device(self.device):
            self.force_type_probs = torch.tensor([0.4, 0.15, 0.15, 0.15, 0.15]) # [zero_force, full_force, left_full, right_full, partial_force]
            force_body_names = get_items_by_index(self.asset.body_names, self.force_apply_idx_asset)
            self.left_mask = torch.tensor(
                ["left" in name for name in force_body_names],
                dtype=torch.bool,
                device=self.device,
            )[None, :, None]
            self.right_mask = torch.tensor(
                ["right" in name for name in force_body_names],
                dtype=torch.bool,
                device=self.device,
            )[None, :, None]
            if not self.left_mask.any() or not self.right_mask.any():
                raise ValueError(f"force_apply_pattern must match left/right bodies, got {force_body_names}.")
            if force_active_pattern is None:
                self.force_active_mask = torch.ones(1, self.num_force_bodies, 1, dtype=torch.bool, device=self.device)
            else:
                active_names = set(
                    get_items_by_index(
                        self.asset.body_names,
                        _match_indices(
                            self.dataset.body_names,
                            self.asset.body_names,
                            list(force_active_pattern),
                            name_map=self.keypoint_map,
                            device=self.device,
                        )[1],
                    )
                )
                self.force_active_mask = torch.tensor(
                    [name in active_names for name in force_body_names],
                    dtype=torch.bool,
                    device=self.device,
                )[None, :, None]
                if not self.force_active_mask.any():
                    raise ValueError(f"force_active_pattern matched no force bodies: {force_active_pattern}.")
            self.force_partial_single_prob = 0.5

            # force threshold sample
            self.force_safe_limit_tl = TemporalLerp(
                shape=(self.num_envs, 1),
                device=self.device,
                easing="linear",
                clamp=self.force_safe_bounds
            )
            self.force_safe_limit_sample_timer = torch.zeros(self.num_envs, dtype=torch.int32) # next time to change safe limit

            # sample control
            self.force_type = torch.zeros(self.num_envs, dtype=torch.int32)
            self.force_sample_timer = torch.zeros(self.num_envs, dtype=torch.int32)
            self.force_enable = torch.zeros(self.num_envs, self.num_force_bodies, 1, dtype=torch.bool) # enable flag

            # force spring / ramping
            self.force_kp_scaled = torch.zeros(self.num_envs, self.num_force_bodies, 1, dtype=torch.float32) # scaled spring stiffness (multiplied by alpha)
            self.force_kp_matrix = torch.zeros(self.num_envs, self.num_force_bodies, 3, 3, dtype=torch.float32)
            self.force_kp_sample_timer = torch.zeros(self.num_envs, dtype=torch.int32) # next time to change slope
            self.force_kp_ramping_down = torch.zeros(self.num_envs, dtype=torch.bool) # is ramping down
            self.force_kp_tl = TemporalLerp(
                shape=(self.num_envs, self.num_force_bodies, 1),
                device=self.device,
                easing="linear",
                clamp=self.kp_range
            )

            # admittance tracking
            self.ref_pos_b_prev = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32)
            self.force_keypoint_w = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # target keypoints in world frame
            self.force_keypoint_b = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # target keypoints in body frame
            self.force_keypoint_w_prev = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32)
            self.force_keypoint_vel_w = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32)

            # spring origin b -> origin w -> dir w
            self.force_origin_tl = TemporalLerp(
                shape=(self.num_envs, self.num_force_bodies, 3),
                device=self.device,
                easing="linear"
            )
            self.force_origin_w = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # origin position of spring in world frame
            self.force_dir_w = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32)
            # spring force
            self.force_applied_w = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # applied force
            self.force_applied_b = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # applied force in body frame
            self.force_expected_w = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # expected apply force
            self.force_expected_b = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # expected apply force in body frame
            self.root_compliance_offset_xy = torch.zeros(self.num_envs, 2, dtype=torch.float32)

            # force apply
            self.force_pos_delta = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32)
            self.force_apply_buffer = torch.zeros(self.num_envs, self.asset.num_bodies, 3, dtype=torch.float32)
            self.torque_apply_buffer = torch.zeros(self.num_envs, self.asset.num_bodies, 3, dtype=torch.float32)

            # student obs
            student_body_pattern = ["torso"]
            self.student_body_idx_motion, self.student_body_idx_asset = _match_indices(
                self.dataset.body_names,
                self.asset.body_names,
                student_body_pattern,
                debug=True
            )
            self.torso_idx_motion = self.student_body_idx_motion[0].item()
            self.torso_idx_asset = self.student_body_idx_asset[0].item()

            # baseline: random perturbation forces (non-compliant mode)
            self.perturb_force = TemporalLerp(
                shape=(self.num_envs, 2, 3),
                device=self.device,
                easing="linear",
            )

        self.force_alpha = 1.0

        self.net_pull_apply_pattern = list(net_pull_apply_pattern or [
            ".*torso.*",
            ".*shoulder_yaw_link",
            ".*wrist_roll_link",
            ".*hand_mimic",
        ])
        self.net_pull_force_range = tuple(float(x) for x in net_pull_force_range)
        self.net_pull_ramp_up_range = tuple(int(x) for x in net_pull_ramp_up_range)
        self.net_pull_hold_range = tuple(int(x) for x in net_pull_hold_range)
        self.net_pull_ramp_down_range = tuple(int(x) for x in net_pull_ramp_down_range)
        self.net_pull_rest_range = tuple(int(x) for x in net_pull_rest_range)
        self.net_pull_xy_only = net_pull_xy_only
        self.net_pull_zero_prob_target = float(net_pull_zero_prob)
        self.net_pull_zero_prob = self.net_pull_zero_prob_target
        self.net_pull_bimanual_prob = float(net_pull_bimanual_prob)
        self.net_pull_curriculum = bool(net_pull_curriculum)
        self.net_pull_curriculum_end = float(net_pull_curriculum_end)
        self.net_pull_zero_prob_start = float(net_pull_zero_prob_start)
        self.net_pull_force_scale_start = float(net_pull_force_scale_start)
        self.net_pull_force_scale = 1.0
        self.net_pull_margin_aware = bool(net_pull_margin_aware)
        self.net_pull_margin_min_force = float(net_pull_margin_min_force)
        self.net_pull_margin_safety = float(net_pull_margin_safety)
        self.net_pull_margin_resample_attempts = int(net_pull_margin_resample_attempts)
        self.net_pull_margin_left_range = self._parse_synthetic_ee_range(
            net_pull_margin_left_range or [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            name="net_pull_margin_left_range",
        )
        self.net_pull_margin_right_range = self._parse_synthetic_ee_range(
            net_pull_margin_right_range or [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            name="net_pull_margin_right_range",
        )
        if not 0.0 <= self.net_pull_zero_prob_target <= 1.0:
            raise ValueError(f"net_pull_zero_prob must be in [0, 1], got {self.net_pull_zero_prob_target}.")
        if not 0.0 <= self.net_pull_bimanual_prob <= 1.0:
            raise ValueError(f"net_pull_bimanual_prob must be in [0, 1], got {self.net_pull_bimanual_prob}.")
        if self.net_pull_curriculum_end <= 0.0:
            raise ValueError(f"net_pull_curriculum_end must be positive, got {self.net_pull_curriculum_end}.")
        if not 0.0 <= self.net_pull_zero_prob_start <= 1.0:
            raise ValueError(f"net_pull_zero_prob_start must be in [0, 1], got {self.net_pull_zero_prob_start}.")
        if not 0.0 <= self.net_pull_force_scale_start <= 1.0:
            raise ValueError(f"net_pull_force_scale_start must be in [0, 1], got {self.net_pull_force_scale_start}.")
        if self.net_pull_margin_min_force < 0.0:
            raise ValueError("net_pull_margin_min_force must be non-negative.")
        if not 0.0 < self.net_pull_margin_safety <= 1.0:
            raise ValueError("net_pull_margin_safety must be in (0, 1].")
        if self.net_pull_margin_resample_attempts <= 0:
            raise ValueError("net_pull_margin_resample_attempts must be positive.")
        (
            self.net_pull_ee_compliance_stiffness,
            stiffness_directional,
        ) = _parse_xyz_or_scalar(
            net_pull_ee_compliance_stiffness,
            name="net_pull_ee_compliance_stiffness",
            device=self.device,
        )
        (
            self.net_pull_ee_compliance_max_offset,
            offset_directional,
        ) = _parse_xyz_or_scalar(
            net_pull_ee_compliance_max_offset,
            name="net_pull_ee_compliance_max_offset",
            device=self.device,
        )
        self.net_pull_ee_compliance_directional = stiffness_directional or offset_directional
        # If disabled, use the instantaneous analytical target F/K instead of
        # filtering the target through the virtual admittance system.
        self.net_pull_ee_compliance_smoothing = bool(net_pull_ee_compliance_smoothing)
        self.net_pull_ee_compliance_force_deadband = float(net_pull_ee_compliance_force_deadband)
        self.net_pull_ee_compliance_stiffness_range = None
        if net_pull_ee_compliance_stiffness_range is not None:
            if len(net_pull_ee_compliance_stiffness_range) != 2:
                raise ValueError(
                    "net_pull_ee_compliance_stiffness_range must be [min, max]."
                )
            stiffness_min, stiffness_max = map(float, net_pull_ee_compliance_stiffness_range)
            if stiffness_min <= 0.0 or stiffness_max < stiffness_min:
                raise ValueError(
                    "net_pull_ee_compliance_stiffness_range must satisfy 0 < min <= max."
                )
            self.net_pull_ee_compliance_stiffness_range = (stiffness_min, stiffness_max)
        self.net_pull_ee_compliance_stiffness_xyz_range = None
        if net_pull_ee_compliance_stiffness_xyz_range is not None:
            if self.net_pull_ee_compliance_stiffness_range is not None:
                raise ValueError(
                    "Use either net_pull_ee_compliance_stiffness_range or "
                    "net_pull_ee_compliance_stiffness_xyz_range, not both."
                )
            xyz_range = torch.as_tensor(
                net_pull_ee_compliance_stiffness_xyz_range,
                dtype=torch.float32,
                device=self.device,
            )
            if xyz_range.shape != (2, 3):
                raise ValueError(
                    "net_pull_ee_compliance_stiffness_xyz_range must be "
                    "[[min_x, min_y, min_z], [max_x, max_y, max_z]]."
                )
            if torch.any(xyz_range <= 0.0) or torch.any(xyz_range[1] < xyz_range[0]):
                raise ValueError(
                    "net_pull_ee_compliance_stiffness_xyz_range must contain "
                    "positive per-axis bounds with min <= max."
                )
            self.net_pull_ee_compliance_stiffness_xyz_range = (
                xyz_range[0],
                xyz_range[1],
            )
            self.net_pull_ee_compliance_directional = True
        self.net_pull_ee_compliance_stiffness_levels = None
        self.net_pull_ee_compliance_stiffness_levels_prob = float(
            net_pull_ee_compliance_stiffness_levels_prob
        )
        self.net_pull_ee_compliance_stiffness_xyz_anchors = None
        self.net_pull_ee_compliance_stiffness_anchor_prob = float(
            net_pull_ee_compliance_stiffness_anchor_prob
        )
        self.net_pull_ee_compliance_stiffness_inverse_command = bool(
            net_pull_ee_compliance_stiffness_inverse_command
        )
        self.net_pull_ee_compliance_stiffness_per_ee = bool(
            net_pull_ee_compliance_stiffness_per_ee
        )
        if not 0.0 <= self.net_pull_ee_compliance_stiffness_levels_prob <= 1.0:
            raise ValueError(
                "net_pull_ee_compliance_stiffness_levels_prob must be in [0, 1]."
            )
        if not 0.0 <= self.net_pull_ee_compliance_stiffness_anchor_prob <= 1.0:
            raise ValueError(
                "net_pull_ee_compliance_stiffness_anchor_prob must be in [0, 1]."
            )
        if (
            self.net_pull_ee_compliance_stiffness_levels_prob
            + self.net_pull_ee_compliance_stiffness_anchor_prob
            > 1.0
        ):
            raise ValueError(
                "EE stiffness level and anchor probabilities must sum to at most 1."
            )
        mixed_sampling_enabled = (
            self.net_pull_ee_compliance_stiffness_levels_prob > 0.0
            or self.net_pull_ee_compliance_stiffness_anchor_prob > 0.0
        )
        if (
            mixed_sampling_enabled
            and self.net_pull_ee_compliance_stiffness_xyz_range is None
        ):
            raise ValueError(
                "Mixed EE stiffness sampling requires "
                "net_pull_ee_compliance_stiffness_xyz_range."
            )
        if net_pull_ee_compliance_stiffness_levels is not None:
            levels = torch.as_tensor(
                net_pull_ee_compliance_stiffness_levels,
                dtype=torch.float32,
                device=self.device,
            ).flatten()
            if levels.numel() == 0 or torch.any(levels <= 0.0):
                raise ValueError(
                    "net_pull_ee_compliance_stiffness_levels must contain positive values."
                )
            if self.net_pull_ee_compliance_stiffness_xyz_range is not None:
                stiffness_min, stiffness_max = (
                    self.net_pull_ee_compliance_stiffness_xyz_range
                )
                if torch.any(levels[:, None] < stiffness_min) or torch.any(
                    levels[:, None] > stiffness_max
                ):
                    raise ValueError(
                        "EE stiffness levels must lie inside every configured xyz range."
                    )
            self.net_pull_ee_compliance_stiffness_levels = levels
        if self.net_pull_ee_compliance_stiffness_levels_prob > 0.0 and (
            self.net_pull_ee_compliance_stiffness_levels is None
        ):
            raise ValueError(
                "Positive EE stiffness levels probability requires stiffness levels."
            )
        if net_pull_ee_compliance_stiffness_xyz_anchors is not None:
            anchors = torch.as_tensor(
                net_pull_ee_compliance_stiffness_xyz_anchors,
                dtype=torch.float32,
                device=self.device,
            )
            if anchors.ndim != 2 or anchors.shape[1] != 3 or anchors.shape[0] == 0:
                raise ValueError(
                    "net_pull_ee_compliance_stiffness_xyz_anchors must have shape [N, 3]."
                )
            if torch.any(anchors <= 0.0):
                raise ValueError("EE stiffness anchors must be positive.")
            if self.net_pull_ee_compliance_stiffness_xyz_range is not None:
                stiffness_min, stiffness_max = (
                    self.net_pull_ee_compliance_stiffness_xyz_range
                )
                if torch.any(anchors < stiffness_min) or torch.any(anchors > stiffness_max):
                    raise ValueError(
                        "EE stiffness anchors must lie inside the configured xyz range."
                    )
            self.net_pull_ee_compliance_stiffness_xyz_anchors = anchors
        if self.net_pull_ee_compliance_stiffness_anchor_prob > 0.0 and (
            self.net_pull_ee_compliance_stiffness_xyz_anchors is None
        ):
            raise ValueError(
                "Positive EE stiffness anchor probability requires xyz anchors."
            )
        if self.net_pull_ee_compliance_stiffness_per_ee:
            if self.net_pull_ee_compliance_stiffness.ndim == 0:
                self.net_pull_ee_compliance_stiffness = self.net_pull_ee_compliance_stiffness.reshape(1, 1).expand(2, 3).clone()
            else:
                self.net_pull_ee_compliance_stiffness = self.net_pull_ee_compliance_stiffness.reshape(1, 3).expand(2, 3).clone()
        self.net_pull_ee_compliance_stiffness_current = torch.empty(
            self.num_envs,
            2 if self.net_pull_ee_compliance_stiffness_per_ee else 1,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        self._set_net_pull_ee_compliance_stiffness_fixed()
        # Keep the legacy EE reward target by default. New net-pull baselines
        # can opt into the same compliance target used by high-level rewards.
        self.use_net_pull_ee_target_for_tracking = bool(use_net_pull_ee_target_for_tracking)
        self.sync_net_pull_ee_reward_targets = bool(sync_net_pull_ee_reward_targets)

        self.configure_admittance()
        self.configure_net_pull()
        self.configure_net_pull_ee_compliance()
        self.eval_ee_force_enabled = False
        self.eval_ee_force_body_idx_asset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.eval_ee_force_body_local_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.eval_ee_force_b = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.eval_ee_force_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.eval_bimanual_ee_force_enabled = False
        self.eval_bimanual_ee_force_b = torch.zeros(self.num_envs, 2, 3, dtype=torch.float32, device=self.device)
        self.eval_root_force_enabled = False
        self.eval_root_force_body_idx_asset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.eval_root_force_body_local_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.eval_root_force_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

    def set_external_force_enabled(self, enabled: bool):
        self.external_force_enabled = bool(enabled)
        if not self.external_force_enabled:
            self.force_apply_world = False
            self.asset.has_external_wrench = False
            if hasattr(self, "force_apply_buffer"):
                self.force_apply_buffer.zero_()
            if hasattr(self, "torque_apply_buffer"):
                self.torque_apply_buffer.zero_()
            if hasattr(self, "force_applied_w"):
                self.force_applied_w.zero_()
            if hasattr(self, "force_applied_b"):
                self.force_applied_b.zero_()
            if hasattr(self, "net_pull_force_w"):
                self.net_pull_force_w.zero_()
            if hasattr(self, "net_pull_force_b"):
                self.net_pull_force_b.zero_()
            if hasattr(self, "net_pull_ee_force_b"):
                self.net_pull_ee_force_b.zero_()
            if hasattr(self, "update_net_pull_ee_compliance_target"):
                self.update_net_pull_ee_compliance_target()

    def configure_net_pull(self):
        self.net_pull_idx_motion, self.net_pull_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.net_pull_apply_pattern,
            name_map=self.keypoint_map,
            device=self.device,
        )
        if self.net_pull_idx_asset.numel() == 0:
            raise ValueError(f"net_pull_apply_pattern matched no bodies: {self.net_pull_apply_pattern}")

        with torch.device(self.device):
            self.net_pull_num_bodies = self.net_pull_idx_asset.shape[0]
            net_pull_body_names = get_items_by_index(self.asset.body_names, self.net_pull_idx_asset)
            self.net_pull_left_body_candidates = torch.tensor(
                [i for i, name in enumerate(net_pull_body_names) if "left" in name],
                dtype=torch.long,
                device=self.device,
            )
            self.net_pull_right_body_candidates = torch.tensor(
                [i for i, name in enumerate(net_pull_body_names) if "right" in name],
                dtype=torch.long,
                device=self.device,
            )
            if self.net_pull_bimanual_prob > 0.0 and (
                self.net_pull_left_body_candidates.numel() == 0
                or self.net_pull_right_body_candidates.numel() == 0
            ):
                raise ValueError(
                    "net_pull_bimanual_prob > 0 requires both left and right bodies "
                    "in net_pull_apply_pattern."
                )
            self.net_pull_body_local_idx = torch.zeros(self.num_envs, dtype=torch.long)
            self.net_pull_bimanual_active = torch.zeros(self.num_envs, dtype=torch.bool)
            self.net_pull_bimanual_body_local_idx = torch.zeros(
                self.num_envs, 2, dtype=torch.long
            )
            self.net_pull_phase = torch.zeros(self.num_envs, dtype=torch.int32)
            self.net_pull_phase_timer = torch.zeros(self.num_envs, dtype=torch.int32)
            self.net_pull_force_dir_w = torch.zeros(self.num_envs, 3, dtype=torch.float32)
            self.net_pull_force_w = torch.zeros(self.num_envs, 3, dtype=torch.float32)
            self.net_pull_force_b = torch.zeros(self.num_envs, 3, dtype=torch.float32)
            self.net_pull_point_b = torch.zeros(self.num_envs, 3, dtype=torch.float32)
            self.net_pull_bimanual_force_dir_w = torch.zeros(
                self.num_envs, 2, 3, dtype=torch.float32
            )
            self.net_pull_bimanual_force_w = torch.zeros(
                self.num_envs, 2, 3, dtype=torch.float32
            )
            self.net_pull_bimanual_force_b = torch.zeros(
                self.num_envs, 2, 3, dtype=torch.float32
            )
            self.net_pull_bimanual_point_b = torch.zeros(
                self.num_envs, 2, 3, dtype=torch.float32
            )
            self.net_pull_force_tl = TemporalLerp(
                shape=(self.num_envs, 1),
                device=self.device,
                easing="linear",
                clamp=(0.0, self.net_pull_force_range[1]),
            )
            self.net_pull_bimanual_force_tl = TemporalLerp(
                shape=(self.num_envs, 2, 1),
                device=self.device,
                easing="linear",
                clamp=(0.0, self.net_pull_force_range[1]),
            )

    def configure_net_pull_ee_compliance(self):
        self.net_pull_ee_names = ["left_hand_mimic", "right_hand_mimic"]
        self.net_pull_ee_idx_motion = torch.tensor(
            [self.dataset.body_names.index(name) for name in self.net_pull_ee_names],
            device=self.device,
            dtype=torch.long,
        )
        self.net_pull_ee_idx_asset = torch.tensor(
            [self.asset.body_names.index(name) for name in self.net_pull_ee_names],
            device=self.device,
            dtype=torch.long,
        )
        self.net_pull_ee_admit = AdmittanceMassChain(
            num_envs=self.num_envs,
            num_points=len(self.net_pull_ee_names),
            dt=self.env.physics_dt,
            mixed_loop_steps=1,
            device=self.device,
            mass=0.1,
            damping=2.0,
            vel_clip=4.0,
            acc_clip=1000.0,
        )
        with torch.device(self.device):
            self.net_pull_ee_nominal_target_b = torch.zeros(self.num_envs, 2, 3, dtype=torch.float32)
            self.net_pull_ee_nominal_target_b_prev = torch.zeros(self.num_envs, 2, 3, dtype=torch.float32)
            self.net_pull_ee_compliance_target_b = torch.zeros(self.num_envs, 2, 3, dtype=torch.float32)
            self.net_pull_ee_force_b = torch.zeros(self.num_envs, 2, 3, dtype=torch.float32)

    def _set_net_pull_ee_compliance_stiffness_fixed(self, env_ids=None):
        if self.net_pull_ee_compliance_stiffness_per_ee:
            # Per-EE stiffness is stored as [left/right, xyz].  The legacy
            # path below stores a single shared [xyz] value.
            if self.net_pull_ee_compliance_stiffness.ndim == 0:
                fixed = self.net_pull_ee_compliance_stiffness.reshape(1, 1).expand(2, 3)
            else:
                fixed = self.net_pull_ee_compliance_stiffness.reshape(2, 3)
            fixed = fixed.unsqueeze(0)
        elif self.net_pull_ee_compliance_stiffness.ndim == 0:
            fixed = self.net_pull_ee_compliance_stiffness.reshape(1, 1, 1).expand(-1, -1, 3)
        else:
            fixed = self.net_pull_ee_compliance_stiffness.reshape(1, 1, 3)
        if env_ids is None:
            self.net_pull_ee_compliance_stiffness_current[:] = fixed
        else:
            self.net_pull_ee_compliance_stiffness_current[env_ids] = fixed

    def _sample_net_pull_ee_compliance_stiffness(self, env_ids):
        if self.net_pull_ee_compliance_stiffness_xyz_range is not None:
            stiffness_min, stiffness_max = self.net_pull_ee_compliance_stiffness_xyz_range
            sampled = sample_mixed_xyz_stiffness(
                env_ids.numel(),
                stiffness_min,
                stiffness_max,
                levels=self.net_pull_ee_compliance_stiffness_levels,
                levels_prob=self.net_pull_ee_compliance_stiffness_levels_prob,
                anchors=self.net_pull_ee_compliance_stiffness_xyz_anchors,
                anchor_prob=self.net_pull_ee_compliance_stiffness_anchor_prob,
            )
            if self.net_pull_ee_compliance_stiffness_per_ee:
                self.net_pull_ee_compliance_stiffness_current[env_ids] = sampled.reshape(-1, 1, 3).expand(-1, 2, -1)
            else:
                self.net_pull_ee_compliance_stiffness_current[env_ids, 0] = sampled
            return
        if self.net_pull_ee_compliance_stiffness_range is None:
            self._set_net_pull_ee_compliance_stiffness_fixed(env_ids)
            return
        stiffness_min, stiffness_max = self.net_pull_ee_compliance_stiffness_range
        sampled = torch.empty(env_ids.numel(), 1, 1, device=self.device).uniform_(
            stiffness_min, stiffness_max
        )
        if self.net_pull_ee_compliance_stiffness_per_ee:
            self.net_pull_ee_compliance_stiffness_current[env_ids] = sampled.expand(-1, 2, 3)
        else:
            self.net_pull_ee_compliance_stiffness_current[env_ids] = sampled.expand(-1, 1, 3)

    def get_net_pull_ee_compliance_stiffness(self):
        """Return active EE stiffness as [env, 1, xyz] or [env, 2, xyz]."""
        return self.net_pull_ee_compliance_stiffness_current

    def set_net_pull_ee_compliance_stiffness(self, stiffness):
        """Set the active EE compliance stiffness for all environments.

        This is primarily used by evaluation to select a stiffness for a
        range-trained policy. Fixed-stiffness policies keep their original
        configuration path, while range-trained policies update both the
        compliance target and the stiffness command observation.
        """
        if self.net_pull_ee_compliance_stiffness_per_ee:
            values = torch.as_tensor(stiffness, dtype=torch.float32, device=self.device).reshape(-1)
            if values.numel() == 1:
                values = values.reshape(1, 1).expand(2, 3).clone()
            elif values.numel() == 2:
                values = values.reshape(2, 1).expand(-1, 3).clone()
            elif values.numel() == 3:
                values = values.reshape(1, 3).expand(2, -1).clone()
            elif values.numel() == 6:
                values = values.reshape(2, 3).clone()
            else:
                raise ValueError(
                    "Per-EE stiffness expects one scalar, two isotropic values, "
                    "three shared xyz values, or six [left xyz, right xyz] values."
                )
            if torch.any(values <= 0.0):
                raise ValueError("Per-EE compliance stiffness must be positive.")
            if self.net_pull_ee_compliance_stiffness_xyz_range is not None:
                stiffness_min, stiffness_max = self.net_pull_ee_compliance_stiffness_xyz_range
                if torch.any(values < stiffness_min) or torch.any(values > stiffness_max):
                    raise ValueError(
                        "Requested per-EE stiffness is outside the configured xyz range "
                        f"[{stiffness_min.tolist()}, {stiffness_max.tolist()}], got {values.tolist()}."
                    )
            self.net_pull_ee_compliance_stiffness = values
            self.net_pull_ee_compliance_stiffness_current[:] = values.reshape(1, 2, 3)
            self.net_pull_ee_compliance_directional = True
            return

        parsed, directional = _parse_xyz_or_scalar(
            stiffness,
            name="net_pull_ee_compliance_stiffness",
            device=self.device,
        )
        if self.net_pull_ee_compliance_stiffness_xyz_range is not None:
            values = parsed.reshape(-1)
            if values.numel() == 1:
                values = values.expand(3)
            if values.numel() != 3:
                raise ValueError(
                    "XYZ-range EE compliance policies require one scalar or three "
                    "stiffness values."
                )
            stiffness_min, stiffness_max = self.net_pull_ee_compliance_stiffness_xyz_range
            if torch.any(values < stiffness_min) or torch.any(values > stiffness_max):
                raise ValueError(
                    "Requested EE stiffness is outside the configured per-axis range "
                    f"[{stiffness_min.tolist()}, {stiffness_max.tolist()}], got {values.tolist()}."
                )
            self.net_pull_ee_compliance_stiffness = values
            self.net_pull_ee_compliance_stiffness_current[:] = values.reshape(1, 1, 3)
            self.net_pull_ee_compliance_directional = True
            return

        if self.net_pull_ee_compliance_stiffness_range is not None:
            if directional:
                values = parsed.reshape(-1)
                if not torch.allclose(values, values[0].expand_as(values)):
                    raise ValueError(
                        "Range-trained EE compliance policies require one isotropic stiffness value."
                    )
                value = float(values[0].item())
            else:
                value = float(parsed.item())
            stiffness_min, stiffness_max = self.net_pull_ee_compliance_stiffness_range
            if not stiffness_min <= value <= stiffness_max:
                raise ValueError(
                    "Requested EE stiffness must be within the configured range "
                    f"[{stiffness_min}, {stiffness_max}], got {value}."
                )
            self.net_pull_ee_compliance_stiffness_current[:] = value
            return

        self.net_pull_ee_compliance_stiffness = parsed
        self.net_pull_ee_compliance_directional = directional or (
            self.net_pull_ee_compliance_max_offset.ndim > 0
        )
        self._set_net_pull_ee_compliance_stiffness_fixed()

    def _net_pull_ee_nominal_from_motion(self):
        if hasattr(self, "get_root_and_wrist_6d_reference"):
            reference = self.get_root_and_wrist_6d_reference()
            if reference.shape[-1] >= 6:
                return reference[:, :6].reshape(self.num_envs, 2, 3)

        motion = self._motion
        if hasattr(motion, "local_body_pos"):
            return motion.local_body_pos[:, 0, self.net_pull_ee_idx_motion, :]
        if hasattr(motion, "body_pos_b"):
            return motion.body_pos_b[:, 0, self.net_pull_ee_idx_motion, :]
        raise RuntimeError("net-pull EE compliance needs local_body_pos or body_pos_b in the motion data.")

    def _net_pull_ee_force_from_net_pull(self):
        # The dedicated bimanual evaluation path writes both EE forces into
        # eval_bimanual_ee_force_b.  Use that buffer directly when refreshing
        # the compliance target; falling through to the legacy single-force
        # buffers can leave one hand's diagnostic force stale or zero.
        if getattr(self, "eval_bimanual_ee_force_enabled", False):
            return self.eval_bimanual_ee_force_b
        force_b = torch.zeros_like(self.net_pull_ee_force_b)
        body_idx = self.net_pull_idx_asset[self.net_pull_body_local_idx]
        for ee_i, asset_i in enumerate(self.net_pull_ee_idx_asset.tolist()):
            mask = body_idx == asset_i
            force_b[mask, ee_i] = self.net_pull_force_b[mask]
        if self.net_pull_bimanual_active.any():
            force_b[self.net_pull_bimanual_active] = self.net_pull_bimanual_force_b[
                self.net_pull_bimanual_active
            ]
        return force_b

    def update_net_pull_ee_compliance_target(self):
        if not hasattr(self, "_motion") and not hasattr(self, "get_root_and_wrist_6d_reference"):
            return

        nominal_target_b = self._net_pull_ee_nominal_from_motion()
        self.net_pull_ee_nominal_target_b[:] = nominal_target_b

        if self.last_reset_env_ids is not None:
            self.net_pull_ee_nominal_target_b_prev[self.last_reset_env_ids] = nominal_target_b[self.last_reset_env_ids]
        nominal_vel_b = (nominal_target_b - self.net_pull_ee_nominal_target_b_prev) / self.env.step_dt
        self.net_pull_ee_nominal_target_b_prev[:] = nominal_target_b

        force_b = self._net_pull_ee_force_from_net_pull() if self.external_force_enabled else torch.zeros_like(self.net_pull_ee_force_b)
        active = force_b.norm(dim=-1, keepdim=True) > self.net_pull_ee_compliance_force_deadband
        force_b = torch.where(active, force_b, torch.zeros_like(force_b))
        self.net_pull_ee_force_b[:] = force_b

        if self.last_reset_env_ids is not None:
            self.net_pull_ee_admit.reset(
                self.last_reset_env_ids,
                x0_b=nominal_target_b[self.last_reset_env_ids],
                v0_b=nominal_vel_b[self.last_reset_env_ids],
            )

        if (
            self.net_pull_ee_compliance_stiffness_range is None
            and self.net_pull_ee_compliance_stiffness_xyz_range is None
        ):
            stiffness = self.net_pull_ee_compliance_stiffness
            if self.net_pull_ee_compliance_stiffness_per_ee:
                stiffness = stiffness.unsqueeze(0)
        else:
            # [1, env, hand, xyz]. Legacy tasks use hand=1 and broadcast over
            # both hands; per-EE tasks retain separate left/right stiffness.
            stiffness = self.net_pull_ee_compliance_stiffness_current.unsqueeze(0)
        if self.net_pull_ee_compliance_smoothing:
            mass = torch.as_tensor(self.net_pull_ee_admit.mass, device=self.device, dtype=torch.float32)
            damping = 2.0 * torch.sqrt(stiffness * mass)

            nominal_target_b_exp = nominal_target_b.unsqueeze(0)
            nominal_vel_b_exp = nominal_vel_b.unsqueeze(0)
            force_b_exp = force_b.unsqueeze(0)

            for _ in range(4):
                F_drive_b = (
                    stiffness * (nominal_target_b_exp - self.net_pull_ee_admit.x)
                    + damping * (nominal_vel_b_exp - self.net_pull_ee_admit.v)
                )
                self.net_pull_ee_admit.step(F_drive_b=F_drive_b, F_ext_b=force_b_exp)

            target_b = self.net_pull_ee_admit.x[0]
            offset_b = target_b - nominal_target_b
        else:
            # Direct analytical compliance target. The final max_offset clamp
            # below remains as a safety bound, but no virtual mass/damping or
            # temporal filtering is applied.
            offset_b = force_b / stiffness
        if self.net_pull_ee_compliance_directional:
            offset_b = torch.clamp(
                offset_b,
                min=-self.net_pull_ee_compliance_max_offset,
                max=self.net_pull_ee_compliance_max_offset,
            )
        else:
            offset_b = clamp_norm(offset_b, max=float(self.net_pull_ee_compliance_max_offset.item()))
        self.net_pull_ee_compliance_target_b[:] = nominal_target_b + offset_b
        self.net_pull_ee_admit.x[0] = self.net_pull_ee_compliance_target_b

    def get_net_pull_ee_compliance_target_b(self):
        return self.net_pull_ee_compliance_target_b

    def sync_net_pull_ee_reward_target_buffers(self):
        """Bridge net-pull compliance targets into legacy impedance rewards.

        The original impedance rewards consume ``force_keypoint_w`` and
        ``force_keypoint_vel_w``. Net-pull writes the applied force and point
        through a separate path, so those legacy buffers would otherwise stay
        stale. This bridge is opt-in for the new EE baseline tasks only.
        """
        if not self.sync_net_pull_ee_reward_targets:
            return
        if self.external_force_mode != "net_pull":
            return

        root_pos_w = self.asset.data.root_pos_w.unsqueeze(1)
        root_quat_w = self.asset.data.root_quat_w.unsqueeze(1)
        target_b = self.net_pull_ee_compliance_target_b
        target_w = quat_apply(root_quat_w.expand(-1, target_b.shape[1], -1), target_b) + root_pos_w

        # Keep any non-net-pull force slots at their nominal motion targets.
        nominal_w = self.reward_keypoints_w[:, self.force_apply_idx_motion]
        self.force_keypoint_w[:] = nominal_w
        self.force_keypoint_b[:] = quat_apply_inverse(
            root_quat_w.expand(-1, nominal_w.shape[1], -1),
            nominal_w - root_pos_w,
        )

        # Replace the hand slots with the dynamic compliance target.
        for ee_i, asset_i in enumerate(self.net_pull_ee_idx_asset.tolist()):
            matches = (self.force_apply_idx_asset == int(asset_i)).nonzero(as_tuple=False).flatten()
            if matches.numel() == 0:
                continue
            slot = int(matches[0].item())
            self.force_keypoint_w[:, slot] = target_w[:, ee_i]
            self.force_keypoint_b[:, slot] = target_b[:, ee_i]

        if self.last_reset_env_ids is not None:
            self.force_keypoint_w_prev[self.last_reset_env_ids] = self.force_keypoint_w[self.last_reset_env_ids]
        self.force_keypoint_vel_w[:] = (
            self.force_keypoint_w - self.force_keypoint_w_prev
        ) / self.env.step_dt
        self.force_keypoint_w_prev[:] = self.force_keypoint_w

    def get_net_pull_ee_nominal_target_b(self):
        return self.net_pull_ee_nominal_target_b

    def get_net_pull_ee_force_b(self):
        return self.net_pull_ee_force_b

    def set_eval_ee_force_b(self, ee_index: int, force_b: torch.Tensor, env_ids: torch.Tensor | None = None):
        if ee_index < 0 or ee_index >= len(self.net_pull_ee_idx_asset):
            raise ValueError(f"ee_index must be in [0, {len(self.net_pull_ee_idx_asset) - 1}], got {ee_index}.")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if force_b.ndim == 1:
            force_b = force_b.unsqueeze(0).expand(env_ids.numel(), -1)
        force_b = force_b.to(device=self.device, dtype=torch.float32)
        if force_b.shape != (env_ids.numel(), 3):
            raise ValueError(
                f"force_b must have shape {(env_ids.numel(), 3)} or (3,), got {tuple(force_b.shape)}."
            )

        body_idx = self.net_pull_ee_idx_asset[ee_index]
        matches = (self.net_pull_idx_asset == body_idx).nonzero(as_tuple=False).flatten()
        if matches.numel() == 0:
            body_name = self.asset.body_names[int(body_idx.item())]
            raise RuntimeError(f"Eval EE force body {body_name} is not present in net_pull_apply_pattern.")

        self.eval_ee_force_enabled = True
        self.eval_ee_force_body_idx_asset[env_ids] = body_idx
        self.eval_ee_force_body_local_idx[env_ids] = matches[0]
        self.eval_ee_force_b[env_ids] = force_b

    def set_eval_bimanual_ee_force_b(self, force_b: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Set simultaneous left/right EE forces for the dedicated eval mode."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if force_b.ndim == 2:
            force_b = force_b.unsqueeze(0).expand(env_ids.numel(), -1, -1)
        force_b = force_b.to(device=self.device, dtype=torch.float32)
        if force_b.shape != (env_ids.numel(), 2, 3):
            raise ValueError(
                f"Bimanual EE force must have shape {(env_ids.numel(), 2, 3)} or (2, 3), "
                f"got {tuple(force_b.shape)}."
            )
        for body_idx in self.net_pull_ee_idx_asset.tolist():
            matches = (self.net_pull_idx_asset == int(body_idx)).nonzero(as_tuple=False).flatten()
            if matches.numel() == 0:
                body_name = self.asset.body_names[int(body_idx)]
                raise RuntimeError(f"Eval EE force body {body_name} is not present in net_pull_apply_pattern.")
        self.eval_ee_force_enabled = False
        self.eval_bimanual_ee_force_enabled = True
        self.eval_bimanual_ee_force_b[env_ids] = force_b

    def clear_eval_ee_force(self):
        self.eval_ee_force_enabled = False
        self.eval_bimanual_ee_force_enabled = False
        self.eval_ee_force_body_idx_asset.zero_()
        self.eval_ee_force_body_local_idx.zero_()
        self.eval_ee_force_b.zero_()
        self.eval_ee_force_w.zero_()
        self.eval_bimanual_ee_force_b.zero_()
        if hasattr(self, "net_pull_force_w"):
            self.net_pull_force_w.zero_()
            self.net_pull_force_b.zero_()
            self.net_pull_ee_force_b.zero_()
            self.force_applied_w.zero_()
            self.force_applied_b.zero_()
            self.force_apply_buffer.zero_()
            self.torque_apply_buffer.zero_()
        if hasattr(self, "update_net_pull_ee_compliance_target"):
            self.update_net_pull_ee_compliance_target()

    def _update_eval_ee_force_target(self):
        self.net_pull_body_local_idx[:] = self.eval_ee_force_body_local_idx
        self.net_pull_force_b[:] = self.eval_ee_force_b
        self.net_pull_force_w[:] = quat_apply(self.asset.data.root_quat_w, self.eval_ee_force_b)
        self.eval_ee_force_w[:] = self.net_pull_force_w
        self.net_pull_force_dir_w[:] = normalize(self.net_pull_force_w)
        self.net_pull_force_tl.current[:] = self.net_pull_force_w.norm(dim=-1, keepdim=True)
        self.net_pull_phase[:] = 2
        self.net_pull_phase_timer[:] = 1

        env_ids = torch.arange(self.num_envs, device=self.device)
        point_w = self.asset.data.body_pos_w[env_ids, self.eval_ee_force_body_idx_asset]
        self.net_pull_point_b[:] = quat_apply_inverse(
            self.asset.data.root_quat_w,
            point_w - self.asset.data.root_pos_w,
        )
        self.force_sample_timer[:] = 1
        self.update_net_pull_ee_compliance_target()

    def _update_eval_bimanual_ee_force_target(self):
        root_quat = self.asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
        self.net_pull_ee_force_b[:] = self.eval_bimanual_ee_force_b
        force_w = quat_apply(root_quat, self.eval_bimanual_ee_force_b)
        # Keep the legacy single-force buffers meaningful for diagnostics.
        self.net_pull_force_w[:] = force_w[:, 0]
        self.net_pull_force_b[:] = self.eval_bimanual_ee_force_b[:, 0]
        self.net_pull_force_dir_w[:] = normalize(self.net_pull_force_w)
        self.net_pull_force_tl.current[:] = self.net_pull_force_w.norm(dim=-1, keepdim=True)
        self.net_pull_phase[:] = 2
        self.net_pull_phase_timer[:] = 1
        self.force_sample_timer[:] = 1
        self.update_net_pull_ee_compliance_target()

    def set_eval_root_force_w(self, body_name: str, force_w: torch.Tensor, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if force_w.ndim == 1:
            force_w = force_w.unsqueeze(0).expand(env_ids.numel(), -1)
        force_w = force_w.to(device=self.device, dtype=torch.float32)
        if force_w.shape != (env_ids.numel(), 3):
            raise ValueError(
                f"force_w must have shape {(env_ids.numel(), 3)} or (3,), got {tuple(force_w.shape)}."
            )
        if body_name not in self.asset.body_names:
            raise RuntimeError(f"Eval root force body {body_name} not found in asset bodies.")
        body_idx = torch.tensor(self.asset.body_names.index(body_name), dtype=torch.long, device=self.device)
        matches = (self.net_pull_idx_asset == body_idx).nonzero(as_tuple=False).flatten()
        if matches.numel() == 0:
            raise RuntimeError(f"Eval root force body {body_name} is not present in net_pull_apply_pattern.")

        self.eval_root_force_enabled = True
        self.eval_root_force_body_idx_asset[env_ids] = body_idx
        self.eval_root_force_body_local_idx[env_ids] = matches[0]
        self.eval_root_force_w[env_ids] = force_w

    def clear_eval_root_force(self):
        self.eval_root_force_enabled = False
        self.eval_root_force_body_idx_asset.zero_()
        self.eval_root_force_body_local_idx.zero_()
        self.eval_root_force_w.zero_()
        if hasattr(self, "net_pull_force_w"):
            self.net_pull_force_w.zero_()
            self.net_pull_force_b.zero_()
            self.net_pull_force_dir_w.zero_()
            self.net_pull_force_tl.current.zero_()
            for buffer_name in (
                "net_pull_ee_force_b",
                "force_applied_w",
                "force_applied_b",
                "force_expected_w",
                "force_expected_b",
                "force_apply_buffer",
                "torque_apply_buffer",
            ):
                if hasattr(self, buffer_name):
                    getattr(self, buffer_name).zero_()
        if hasattr(self, "update_net_pull_ee_compliance_target"):
            self.update_net_pull_ee_compliance_target()

    def _update_eval_root_force_target(self):
        self.net_pull_body_local_idx[:] = self.eval_root_force_body_local_idx
        self.net_pull_force_w[:] = self.eval_root_force_w
        self.net_pull_force_b[:] = quat_apply_inverse(self.asset.data.root_quat_w, self.net_pull_force_w)
        self.net_pull_force_dir_w[:] = normalize(self.net_pull_force_w)
        self.net_pull_force_tl.current[:] = self.net_pull_force_w.norm(dim=-1, keepdim=True)
        self.net_pull_phase[:] = 2
        self.net_pull_phase_timer[:] = 1

        env_ids = torch.arange(self.num_envs, device=self.device)
        point_w = self.asset.data.body_pos_w[env_ids, self.eval_root_force_body_idx_asset]
        self.net_pull_point_b[:] = quat_apply_inverse(
            self.asset.data.root_quat_w,
            point_w - self.asset.data.root_pos_w,
        )
        self.force_sample_timer[:] = 1
        self.update_net_pull_ee_compliance_target()

    def update_force_kp_matrix(self):
        diag = torch.cat([
            self.force_kp_scaled,
            self.force_kp_scaled,
            self.force_kp_scaled,
        ], dim=-1)
        self.force_kp_matrix[:] = torch.diag_embed(diag)

    def apply_force_kp_matrix(self, pos_diff: torch.Tensor):
        kp_matrix = self.force_kp_matrix
        while kp_matrix.dim() < pos_diff.dim() + 1:
            kp_matrix = kp_matrix.unsqueeze(0)
        return torch.matmul(kp_matrix, pos_diff.unsqueeze(-1)).squeeze(-1)

    def update_root_compliance_offset(self):
        force_net_xy = self.force_applied_w.sum(dim=1)[:, :2]
        force_mag = force_net_xy.norm(dim=-1, keepdim=True)
        force_excess_xy = (force_mag - self.root_compliance_force_threshold).clamp_min(0.0) * force_net_xy / (force_mag + 1e-6)
        target_offset_xy = clamp_norm(
            self.root_compliance_gain * force_excess_xy,
            max=self.root_compliance_max_offset,
        )
        self.root_compliance_offset_xy[:] = (
            self.root_compliance_ema * self.root_compliance_offset_xy
            + (1.0 - self.root_compliance_ema) * target_offset_xy
        )
        return self.root_compliance_offset_xy

    def configure_admittance(self):
        self.admit = AdmittanceMassChain(
            num_envs=self.num_envs,
            num_points=self.num_force_bodies,
            dt=self.env.physics_dt, # !important
            mixed_loop_steps=1,
            device=self.device,
            mass=0.1,
            damping=2.0,
            vel_clip=4.0,
            acc_clip=1000.0
        )

    def sample_init(self, env_ids: torch.Tensor):
        super().sample_init(env_ids)
        self._sample_net_pull_ee_compliance_stiffness(env_ids)
        self.force_reset(env_ids)
        return None

    def force_reset(self, env_ids: torch.Tensor, refresh_time: bool = True):
        self.force_type[env_ids] = 0
        self.force_applied_w[env_ids] = 0
        self.force_applied_b[env_ids] = 0
        self.force_enable[env_ids] = 0
        self.force_origin_tl.reset(env_ids)
        self.force_kp_tl.reset(env_ids)
        self.force_kp_matrix[env_ids] = 0.0
        self.force_kp_ramping_down[env_ids] = False
        self.perturb_force.reset(env_ids, value=0.0)
        self.root_compliance_offset_xy[env_ids] = 0.0
        self.net_pull_reset(env_ids, refresh_time=refresh_time)
        
        if refresh_time:
            self.force_sample_timer[env_ids] = random_uniform(env_ids.shape[0], 10, 60, self.device).int()
            self.force_safe_limit_tl.reset(env_ids)
            self.force_safe_limit_tl.set(env_ids, start=self.force_safe_default, end=self.force_safe_default, total_steps=1)
            self.force_safe_limit_sample_timer[env_ids] = 0

    def net_pull_reset(self, env_ids: torch.Tensor, refresh_time: bool = True):
        self.net_pull_phase[env_ids] = 0
        self.net_pull_bimanual_active[env_ids] = False
        self.net_pull_force_dir_w[env_ids] = 0.0
        self.net_pull_force_w[env_ids] = 0.0
        self.net_pull_force_b[env_ids] = 0.0
        self.net_pull_point_b[env_ids] = 0.0
        self.net_pull_bimanual_force_dir_w[env_ids] = 0.0
        self.net_pull_bimanual_force_w[env_ids] = 0.0
        self.net_pull_bimanual_force_b[env_ids] = 0.0
        self.net_pull_bimanual_point_b[env_ids] = 0.0
        self.net_pull_force_tl.reset(env_ids, value=torch.zeros(env_ids.shape[0], 1, device=self.device))
        self.net_pull_bimanual_force_tl.reset(
            env_ids, value=torch.zeros(env_ids.shape[0], 2, 1, device=self.device)
        )
        if hasattr(self, "net_pull_ee_admit") and (hasattr(self, "_motion") or hasattr(self, "get_root_and_wrist_6d_reference")):
            nominal_target_b = self._net_pull_ee_nominal_from_motion()
            self.net_pull_ee_nominal_target_b[env_ids] = nominal_target_b[env_ids]
            self.net_pull_ee_nominal_target_b_prev[env_ids] = nominal_target_b[env_ids]
            self.net_pull_ee_compliance_target_b[env_ids] = nominal_target_b[env_ids]
            self.net_pull_ee_force_b[env_ids] = 0.0
            self.net_pull_ee_admit.reset(env_ids, x0_b=nominal_target_b[env_ids])
        if refresh_time:
            self.net_pull_phase_timer[env_ids] = random_uniform(
                env_ids.shape[0],
                self.net_pull_rest_range[0],
                self.net_pull_rest_range[1],
                self.device,
            ).int()

    def _sample_net_pull_direction(self, count: int) -> torch.Tensor:
        if self.net_pull_xy_only:
            angle = random_uniform(count, 0.0, 2.0 * math.pi, self.device)
            direction = torch.zeros(count, 3, device=self.device)
            direction[:, 0] = torch.cos(angle)
            direction[:, 1] = torch.sin(angle)
            return direction
        return normalize(rand_points_isotropic(count, 1, r_max=1.0, device=self.device).squeeze(1))

    def _net_pull_margin_force_limit(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not self.net_pull_margin_aware:
            return torch.full((env_ids.numel(), 1), self.net_pull_force_range[1], device=self.device)
        if not hasattr(self, "net_pull_ee_idx_asset"):
            return torch.full((env_ids.numel(), 1), self.net_pull_force_range[1], device=self.device)

        body_idx = self.net_pull_idx_asset[self.net_pull_body_local_idx[env_ids]]
        ee_match = body_idx.unsqueeze(1) == self.net_pull_ee_idx_asset.unsqueeze(0)
        is_ee = ee_match.any(dim=1)
        limits = torch.full((env_ids.numel(), 1), self.net_pull_force_range[1], device=self.device)
        if not is_ee.any():
            return limits

        local_ids = torch.arange(env_ids.numel(), device=self.device)[is_ee]
        ee_ids = ee_match[is_ee].int().argmax(dim=1)
        active_env_ids = env_ids[is_ee]
        nominal_target_b = self._net_pull_ee_nominal_from_motion()[active_env_ids, ee_ids]
        force_dir_b = quat_apply_inverse(
            self.asset.data.root_quat_w[active_env_ids],
            self.net_pull_force_dir_w[active_env_ids],
        )

        ranges = torch.stack([
            self.net_pull_margin_left_range,
            self.net_pull_margin_right_range,
        ])
        low = ranges[ee_ids, 0]
        high = ranges[ee_ids, 1]
        if self.net_pull_ee_compliance_stiffness_xyz_range is not None:
            if self.net_pull_ee_compliance_stiffness_per_ee:
                stiffness_xyz = self.net_pull_ee_compliance_stiffness_current[active_env_ids, ee_ids, :]
            else:
                stiffness_xyz = self.net_pull_ee_compliance_stiffness_current[active_env_ids, 0, :]
        else:
            stiffness = self.net_pull_ee_compliance_stiffness
            if self.net_pull_ee_compliance_stiffness_per_ee:
                stiffness_xyz = stiffness[ee_ids]
            elif stiffness.ndim == 0:
                stiffness_xyz = torch.full_like(force_dir_b, float(stiffness.item()))
            else:
                stiffness_xyz = stiffness.reshape(1, 3).expand_as(force_dir_b)

        eps = 1e-6
        positive = force_dir_b > eps
        negative = force_dir_b < -eps
        axis_limit = torch.full_like(force_dir_b, float("inf"))
        axis_limit = torch.where(
            positive,
            ((high - nominal_target_b).clamp_min(0.0) * stiffness_xyz / force_dir_b.clamp_min(eps)),
            axis_limit,
        )
        axis_limit = torch.where(
            negative,
            ((nominal_target_b - low).clamp_min(0.0) * stiffness_xyz / (-force_dir_b).clamp_min(eps)),
            axis_limit,
        )
        force_limit = axis_limit.min(dim=-1, keepdim=True).values
        force_limit = torch.where(torch.isfinite(force_limit), force_limit, limits[local_ids])
        limits[local_ids] = force_limit.clamp_min(0.0) * self.net_pull_margin_safety
        return limits

    def _resample_net_pull_directions_for_margin(self, env_ids: torch.Tensor) -> None:
        """Resample infeasible single-hand directions instead of dropping them.

        The margin check is defined in the root frame, while the net-pull
        direction is stored in the world frame. Candidate directions are
        therefore sampled through the same world-frame path as ordinary
        net-pull directions and evaluated by ``_net_pull_margin_force_limit``.
        A bounded rejection sampler keeps the first feasible candidate. If
        no candidate reaches ``net_pull_margin_min_force``, the direction with
        the largest observed margin is retained and the caller applies its
        normal final fallback.
        """
        if not self.net_pull_margin_aware or env_ids.numel() == 0:
            return

        current_limit = self._net_pull_margin_force_limit(env_ids)
        infeasible = current_limit[:, 0] < self.net_pull_margin_min_force
        if not infeasible.any():
            return

        unresolved = env_ids[infeasible]
        best_limit = current_limit[infeasible].clone()
        best_direction = self.net_pull_force_dir_w[unresolved].clone()

        for _ in range(self.net_pull_margin_resample_attempts):
            candidate_direction = self._sample_net_pull_direction(unresolved.numel())
            self.net_pull_force_dir_w[unresolved] = candidate_direction
            candidate_limit = self._net_pull_margin_force_limit(unresolved)

            improved = candidate_limit[:, 0] > best_limit[:, 0]
            best_limit = torch.where(improved.unsqueeze(-1), candidate_limit, best_limit)
            best_direction = torch.where(
                improved.unsqueeze(-1), candidate_direction, best_direction
            )

            feasible = candidate_limit[:, 0] >= self.net_pull_margin_min_force
            if feasible.any():
                unresolved = unresolved[~feasible]
                best_limit = best_limit[~feasible]
                best_direction = best_direction[~feasible]
                if unresolved.numel() == 0:
                    break

        if unresolved.numel() > 0:
            self.net_pull_force_dir_w[unresolved] = best_direction

    def net_pull_schedule(self):
        self.net_pull_phase_timer -= 1
        phase_done = self.net_pull_phase_timer < 0
        if not phase_done.any():
            return

        env_ids = phase_done.nonzero(as_tuple=False).squeeze(-1)
        phase_at_done = self.net_pull_phase[env_ids].clone()

        rest_envs = env_ids[phase_at_done == 0]
        if rest_envs.numel() > 0:
            zero_pull = torch.rand(rest_envs.numel(), device=self.device) < self.net_pull_zero_prob
            zero_envs = rest_envs[zero_pull]
            if zero_envs.numel() > 0:
                self.net_pull_phase[zero_envs] = 0
                self.net_pull_bimanual_active[zero_envs] = False
                self.net_pull_force_dir_w[zero_envs] = 0.0
                self.net_pull_bimanual_force_dir_w[zero_envs] = 0.0
                self.net_pull_force_tl.set(zero_envs, end=0.0, total_steps=1)
                self.net_pull_bimanual_force_tl.set(zero_envs, end=0.0, total_steps=1)
                self.net_pull_phase_timer[zero_envs] = random_uniform(
                    zero_envs.numel(),
                    self.net_pull_rest_range[0],
                    self.net_pull_rest_range[1],
                    self.device,
                ).int()

            active_envs = rest_envs[~zero_pull]
            if active_envs.numel() > 0:
                self.net_pull_phase[active_envs] = 1
                bimanual = torch.rand(active_envs.numel(), device=self.device) < self.net_pull_bimanual_prob
                bimanual_envs = active_envs[bimanual]
                single_envs = active_envs[~bimanual]
                self.net_pull_bimanual_active[active_envs] = bimanual
                self.net_pull_body_local_idx[active_envs] = (
                    torch.rand(active_envs.numel(), device=self.device) * self.net_pull_num_bodies
                ).floor().long()
                self.net_pull_force_dir_w[active_envs] = self._sample_net_pull_direction(active_envs.numel())
                self.net_pull_bimanual_force_dir_w[bimanual_envs] = self._sample_net_pull_direction(
                    bimanual_envs.numel() * 2
                ).reshape(-1, 2, 3)
                if bimanual_envs.numel() > 0:
                    left = self.net_pull_left_body_candidates[
                        torch.randint(
                            self.net_pull_left_body_candidates.numel(),
                            (bimanual_envs.numel(),),
                            device=self.device,
                        )
                    ]
                    right = self.net_pull_right_body_candidates[
                        torch.randint(
                            self.net_pull_right_body_candidates.numel(),
                            (bimanual_envs.numel(),),
                            device=self.device,
                        )
                    ]
                    self.net_pull_bimanual_body_local_idx[bimanual_envs] = torch.stack(
                        (left, right), dim=-1
                    )

                force_mag = random_uniform(
                    (single_envs.numel(), 1),
                    self.net_pull_force_range[0],
                    self.net_pull_force_range[1],
                    self.device,
                ) * self.net_pull_force_scale
                if self.net_pull_margin_aware and single_envs.numel() > 0:
                    self._resample_net_pull_directions_for_margin(single_envs)
                    force_limit = self._net_pull_margin_force_limit(single_envs)
                    feasible = force_limit >= self.net_pull_margin_min_force
                    force_mag = torch.minimum(force_mag, force_limit)
                    infeasible_envs = single_envs[~feasible.squeeze(-1)]
                    if infeasible_envs.numel() > 0:
                        self.net_pull_phase[infeasible_envs] = 0
                        self.net_pull_bimanual_active[infeasible_envs] = False
                        self.net_pull_force_dir_w[infeasible_envs] = 0.0
                        self.net_pull_bimanual_force_dir_w[infeasible_envs] = 0.0
                        self.net_pull_force_tl.set(infeasible_envs, end=0.0, total_steps=1)
                        self.net_pull_bimanual_force_tl.set(infeasible_envs, end=0.0, total_steps=1)
                        self.net_pull_phase_timer[infeasible_envs] = random_uniform(
                            infeasible_envs.numel(),
                            self.net_pull_rest_range[0],
                            self.net_pull_rest_range[1],
                            self.device,
                        ).int()
                    single_envs = single_envs[feasible.squeeze(-1)]
                    force_mag = force_mag[feasible.squeeze(-1)]
                if single_envs.numel() > 0:
                    ramp_steps = random_uniform(
                        single_envs.numel(),
                        self.net_pull_ramp_up_range[0],
                        self.net_pull_ramp_up_range[1],
                        self.device,
                    ).int()
                    self.net_pull_phase_timer[single_envs] = ramp_steps
                    self.net_pull_force_tl.set(single_envs, end=force_mag, total_steps=ramp_steps)
                if bimanual_envs.numel() > 0:
                    ramp_steps = random_uniform(
                        bimanual_envs.numel(),
                        self.net_pull_ramp_up_range[0],
                        self.net_pull_ramp_up_range[1],
                        self.device,
                    ).int()
                    force_mag_bimanual = random_uniform(
                        (bimanual_envs.numel(), 2, 1),
                        self.net_pull_force_range[0],
                        self.net_pull_force_range[1],
                        self.device,
                    ) * self.net_pull_force_scale
                    self.net_pull_phase_timer[bimanual_envs] = ramp_steps
                    self.net_pull_bimanual_force_tl.set(
                        bimanual_envs, end=force_mag_bimanual, total_steps=ramp_steps
                    )

        ramp_up_envs = env_ids[phase_at_done == 1]
        if ramp_up_envs.numel() > 0:
            self.net_pull_phase[ramp_up_envs] = 2
            self.net_pull_phase_timer[ramp_up_envs] = random_uniform(
                ramp_up_envs.numel(),
                self.net_pull_hold_range[0],
                self.net_pull_hold_range[1],
                self.device,
            ).int()

        hold_envs = env_ids[phase_at_done == 2]
        if hold_envs.numel() > 0:
            self.net_pull_phase[hold_envs] = 3
            ramp_steps = random_uniform(
                hold_envs.numel(),
                self.net_pull_ramp_down_range[0],
                self.net_pull_ramp_down_range[1],
                self.device,
            ).int()
            self.net_pull_phase_timer[hold_envs] = ramp_steps
            bimanual = self.net_pull_bimanual_active[hold_envs]
            if (~bimanual).any():
                self.net_pull_force_tl.set(
                    hold_envs[~bimanual], end=0.0, total_steps=ramp_steps[~bimanual]
                )
            if bimanual.any():
                self.net_pull_bimanual_force_tl.set(
                    hold_envs[bimanual], end=0.0, total_steps=ramp_steps[bimanual]
                )

        ramp_down_envs = env_ids[phase_at_done == 3]
        if ramp_down_envs.numel() > 0:
            self.net_pull_phase[ramp_down_envs] = 0
            self.net_pull_bimanual_active[ramp_down_envs] = False
            self.net_pull_force_dir_w[ramp_down_envs] = 0.0
            self.net_pull_bimanual_force_dir_w[ramp_down_envs] = 0.0
            self.net_pull_phase_timer[ramp_down_envs] = random_uniform(
                ramp_down_envs.numel(),
                self.net_pull_rest_range[0],
                self.net_pull_rest_range[1],
                self.device,
            ).int()

    def net_pull_update_target(self):
        force_mag = self.net_pull_force_tl.current
        self.net_pull_force_w[:] = self.net_pull_force_dir_w * force_mag
        self.net_pull_force_b[:] = quat_apply_inverse(self.asset.data.root_quat_w, self.net_pull_force_w)
        bimanual = self.net_pull_bimanual_active
        if bimanual.any():
            root_quat = self.asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
            force_mag_bimanual = self.net_pull_bimanual_force_tl.current
            self.net_pull_bimanual_force_w[:] = (
                self.net_pull_bimanual_force_dir_w * force_mag_bimanual
            )
            self.net_pull_bimanual_force_b[:] = quat_apply_inverse(
                root_quat, self.net_pull_bimanual_force_w
            )
            self.net_pull_force_w[bimanual] = 0.0
            self.net_pull_force_b[bimanual] = 0.0
        self.force_sample_timer[:] = self.net_pull_phase_timer

        body_idx = self.net_pull_idx_asset[self.net_pull_body_local_idx]
        point_w = self.asset.data.body_pos_w[torch.arange(self.num_envs, device=self.device), body_idx]
        self.net_pull_point_b[:] = quat_apply_inverse(
            self.asset.data.root_quat_w,
            point_w - self.asset.data.root_pos_w,
        )
        if bimanual.any():
            env_grid = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, 2)
            body_idx_bimanual = self.net_pull_idx_asset[self.net_pull_bimanual_body_local_idx]
            point_w_bimanual = self.asset.data.body_pos_w[env_grid, body_idx_bimanual]
            self.net_pull_bimanual_point_b[:] = quat_apply_inverse(
                root_quat,
                point_w_bimanual - self.asset.data.root_pos_w.unsqueeze(1),
            )
    
    def force_schedule(self):
        # procedure:
        # 0. update force kp
        #    - if force_kp_sample_timer is done, resample force_kp_slope and force_kp_active
        # 1. check if force is ramping down, if so, update the force
        #    - if ramping down is done, reset force for these envs
        #    - if not done, update the force_kp_active and force_ramping_down
        # 2. check if (time is done), if so, start ramping down
        # 3. check if (time is done) & (ramping down is done), if so, resample force

        self.force_kp_sample_timer -= 1
        self.force_sample_timer -= 1
        self.force_safe_limit_sample_timer -= 1

        # -0 update force safe limit
        need_resample_safe_limit = self.force_safe_limit_sample_timer < 0
        if need_resample_safe_limit.any():
            resample_envs = need_resample_safe_limit.nonzero(as_tuple=False).squeeze(-1)
            self.force_safe_limit_sample_timer[resample_envs] = random_uniform(resample_envs.shape[0], 100, 200, self.device).int()
            safe_limit = random_uniform((resample_envs.shape[0], 1), self.force_safe_bounds[0], self.force_safe_bounds[1], self.device)
            delta_time = random_uniform(resample_envs.shape[0], 25, 100, self.device).int()
            self.force_safe_limit_tl.set(resample_envs, end=safe_limit, total_steps=delta_time)

        # -0 update force kp
        need_resample_kp = (self.force_kp_sample_timer < 0) & (~self.force_kp_ramping_down)
        if need_resample_kp.any():
            resample_envs = need_resample_kp.nonzero(as_tuple=False).squeeze(-1)
            kp_next_time = random_uniform(resample_envs.shape[0], self.kp_time_range[0], self.kp_time_range[1], self.device).int()
            self.force_kp_sample_timer[resample_envs] = kp_next_time
            kp_zero_slope = torch.rand(resample_envs.shape[0], 2, 1, 1, device=self.device) < self.zero_kp_slope_prob
            kp_slope = random_uniform((resample_envs.shape[0], 2, 1, 1), self.kp_slope_range[0], self.kp_slope_range[1], self.device) * (~kp_zero_slope)
            kp_delta = (kp_slope[:, 0] * self.left_mask + kp_slope[:, 1] * self.right_mask) * kp_next_time[:, None, None]  # [N, 1, 1]
            self.force_kp_tl.set(resample_envs, delta=kp_delta, total_steps=kp_next_time)

        # -1
        # deal with force ramping down
        finished_ramping_down = self.force_kp_ramping_down & self.force_kp_tl.mask_done
        if finished_ramping_down.any():
            finished_ramping_down_envs = finished_ramping_down.nonzero(as_tuple=False).squeeze(-1)
            self.force_kp_ramping_down[finished_ramping_down_envs] = False
            self.force_reset(finished_ramping_down_envs, refresh_time=False)  # reset force for these envs

        # -2
        # time done
        time_done = self.force_sample_timer < 0
        # force required
        force_required_mask = self.force_enable.any(dim=(1,2))
        
        # ramping force when (1) time is done & (2) not ramping now & (3) currently have force
        should_start_ramping_down = time_done & (~self.force_kp_ramping_down) & force_required_mask
        if should_start_ramping_down.any():
            ramping_down_envs = should_start_ramping_down.nonzero(as_tuple=False).squeeze(-1)
            self.force_kp_ramping_down[ramping_down_envs] = True
            steps = random_uniform(ramping_down_envs.shape[0], self.ramping_time_range[0], self.ramping_time_range[1], self.device).int()
            self.force_kp_tl.set(env_ids=ramping_down_envs, end=0.0, total_steps=steps)

        # -3 deal with new force sampling
        need_resample = time_done & (~self.force_kp_ramping_down) # recompute current ramping envs
        if need_resample.any():
            need_resample_envs = need_resample.nonzero(as_tuple=False).squeeze(-1)
            force_type = torch.multinomial(self.force_type_probs, need_resample_envs.shape[0], replacement=True)
            zero_force = force_type == 0
            full_force = force_type == 1
            left_full = force_type == 2
            right_full = force_type == 3
            partial_force = force_type == 4
            
            force_enable = torch.zeros(need_resample_envs.shape[0], self.num_force_bodies, 1, dtype=torch.bool, device=self.device)
            force_enable[zero_force, :] = 0
            force_enable[full_force, :] = 1
            force_enable[left_full, :] = self.left_mask
            force_enable[right_full, :] = self.right_mask
            force_enable[partial_force, :] = torch.rand_like(force_enable[partial_force, :], dtype=torch.float32) <= self.force_partial_single_prob
            force_enable &= self.force_active_mask
            self.force_enable[need_resample_envs] = force_enable

            kp_left = random_uniform((need_resample_envs.shape[0], 1, 1), self.kp_range[0], self.kp_range[1], self.device)
            kp_right = random_uniform((need_resample_envs.shape[0], 1, 1), self.kp_range[0], self.kp_range[1], self.device)
            kp = (kp_left * self.left_mask + kp_right * self.right_mask) * force_enable
            self.force_kp_tl.set(need_resample_envs, end=kp, total_steps=1)

            self.force_sample_timer[need_resample_envs] = random_uniform(need_resample_envs.shape[0], self.force_time_range[0], self.force_time_range[1], self.device).int()
            self.force_reset_origin(need_resample_envs)

            # deal with force delta pos
            self.force_pos_delta[need_resample_envs] = rand_points_isotropic(need_resample_envs.shape[0], self.num_force_bodies, r_max=0.05, device=self.device)

        self.force_kp_scaled = self.force_kp_tl.current * self.force_enable
        self.update_force_kp_matrix()

    def force_reset_origin(self, env_ids: torch.Tensor):
        pos_w = self.asset.data.body_pos_w[env_ids][:, self.force_apply_idx_asset, :]
        root_pos = self.asset.data.root_pos_w[env_ids, :]
        root_quat = self.asset.data.root_quat_w[env_ids, :]
        pos_b_start = quat_apply_inverse(root_quat.unsqueeze(1), pos_w - root_pos.unsqueeze(1))
        pos_b_end_left = pos_b_start[:, self.left_mask.reshape(-1), :]
        pos_b_end_right = pos_b_start[:, self.right_mask.reshape(-1), :]
        # in some env, we will sample a pulling force but not on partial force
        need_left_sample = ((torch.rand(env_ids.shape[0], device=self.device) < self.force_origin_sample_prob) & (self.force_type[env_ids] != 4)).nonzero(as_tuple=False).squeeze(-1)
        need_right_sample = ((torch.rand(env_ids.shape[0], device=self.device) < self.force_origin_sample_prob) & (self.force_type[env_ids] != 4)).nonzero(as_tuple=False).squeeze(-1)

        def sample_origin(source: torch.Tensor, env_ids: torch.Tensor, num_points: int):
            idx = (torch.rand(env_ids.shape[0], device=self.device) * self.force_origin_samples).floor().int()
            link_pos_in_torso = source[idx][:, :num_points]
            torso_pos = self.asset.data.body_pos_w[env_ids, self.torso_idx_asset].unsqueeze(1)
            torso_quat = self.asset.data.body_quat_w[env_ids, self.torso_idx_asset].unsqueeze(1)

            root_pos = self.asset.data.root_pos_w[env_ids, :].unsqueeze(1)
            root_quat = self.asset.data.root_quat_w[env_ids, :].unsqueeze(1)

            link_pos_w = quat_apply(torso_quat, link_pos_in_torso) + torso_pos
            return quat_apply_inverse(root_quat, link_pos_w - root_pos)

        pos_b_end_left[need_left_sample] = sample_origin(
            self.force_origin_samples_left,
            env_ids[need_left_sample],
            pos_b_end_left.shape[1],
        )
        pos_b_end_right[need_right_sample] = sample_origin(
            self.force_origin_samples_right,
            env_ids[need_right_sample],
            pos_b_end_right.shape[1],
        )

        pos_b_end = torch.zeros_like(pos_b_start)
        pos_b_end[:, self.left_mask.reshape(-1), :] = pos_b_end_left
        pos_b_end[:, self.right_mask.reshape(-1), :] = pos_b_end_right

        transit_steps = random_uniform(env_ids.shape[0], self.force_origin_transit_time_range[0], self.force_origin_transit_time_range[1], self.device).int()

        # pos_b_end += rand_points_isotropic(env_ids.shape[0], self.num_force_bodies, r_max=0.05, device=self.device)

        self.force_origin_tl.set(env_ids=env_ids, start=pos_b_start, end=pos_b_end, total_steps=transit_steps)

    def force_update_origin_and_target(self):
        root_pos_w = self.asset.data.root_pos_w.unsqueeze(1)
        root_quat = self.asset.data.root_quat_w
        root_quat_exp = root_quat.unsqueeze(1)

        # spring origin is stored in the root frame
        force_origin_b = self.force_origin_tl.current
        self.force_origin_w[:] = quat_apply(root_quat_exp, force_origin_b) + root_pos_w

        # reference keypoints expressed directly in the robot root frame
        ref_point_b = quat_apply_inverse(root_quat_exp, self.reward_keypoints_w[:, self.force_apply_idx_motion] - root_pos_w)

        # calc ref point vel
        if self.last_reset_env_ids is not None:
            self.ref_pos_b_prev[self.last_reset_env_ids] = ref_point_b[self.last_reset_env_ids]
        ref_point_vel_b = (ref_point_b - self.ref_pos_b_prev) / self.env.step_dt
        self.ref_pos_b_prev[:] = ref_point_b

        force_dir_b = normalize(ref_point_b - force_origin_b)

        self.force_dir_w[:] = quat_apply(root_quat_exp, force_dir_b)

        current_point_w = self.asset.data.body_pos_w[:, self.force_apply_idx_asset]
        current_point_b = quat_apply_inverse(root_quat_exp, current_point_w - root_pos_w)

        current_point_vel_w = self.asset.data.body_lin_vel_w[:, self.force_apply_idx_asset]
        root_lin_vel_w = self.asset.data.root_lin_vel_w.unsqueeze(1)
        rel_vel_w = current_point_vel_w - root_lin_vel_w

        root_ang_vel_w = self.asset.data.root_ang_vel_w
        omega_b = quat_apply_inverse(root_quat, root_ang_vel_w).unsqueeze(1)
        omega_b = omega_b.expand(-1, current_point_b.shape[1], -1)
        rel_vel_b = quat_apply_inverse(root_quat_exp, rel_vel_w)

        # deal with init state
        if self.last_reset_env_ids is not None:
            self.admit.reset(
                self.last_reset_env_ids,
                x0_b=ref_point_b[self.last_reset_env_ids],
                v0_b=ref_point_vel_b[self.last_reset_env_ids],
            )

        # driving force in root frame
        force_limit = self.force_safe_limit_tl.current.unsqueeze(1)         # [N, 1, 1]
        K_p_drive = (force_limit / 0.05)                                    # [N, 1, 1] 5cm
        K_d_drive = 2.0 * torch.sqrt(K_p_drive * 0.1)                       # critical damping assuming mass=0.1kg

        ref_point_b_exp = ref_point_b.unsqueeze(0)
        ref_point_vel_b_exp = ref_point_vel_b.unsqueeze(0)
        force_origin_b_exp = force_origin_b.unsqueeze(0)
        force_dir_b_exp = force_dir_b.unsqueeze(0)

        # use 4 substep to integrate in root frame
        for _ in range(4):
            admit_point_b = self.admit.x  # current integrate pos (root frame)
            admit_point_vel_b = self.admit.v  # current integrate vel (root frame)
            # driving force
            F_drive_b = clamp_norm(
                K_p_drive * (ref_point_b_exp - admit_point_b) +
                K_d_drive * (ref_point_vel_b_exp - admit_point_vel_b),
                max=force_limit
            )
            # external force
            F_ext_b = clamp_norm(
                self.apply_force_kp_matrix(self.project_pos_diff(
                    force_origin_b_exp - admit_point_b,
                    force_dir=force_dir_b_exp,
                )),
                self.max_force * self.force_alpha
            )

            self.admit.step(F_drive_b=F_drive_b, F_ext_b=F_ext_b)

        force_keypoint_b = self.admit.x[0]

        self.force_keypoint_b[:] = force_keypoint_b
        self.force_keypoint_w[:] = quat_apply(root_quat_exp, force_keypoint_b) + root_pos_w

        # get other force target
        if self.last_reset_env_ids is not None:
            self.force_keypoint_w_prev[self.last_reset_env_ids] = self.force_keypoint_w[self.last_reset_env_ids]
        self.force_keypoint_vel_w[:] = (self.force_keypoint_w - self.force_keypoint_w_prev) / self.env.step_dt
        self.force_keypoint_w_prev[:] = self.force_keypoint_w

        force_expected_b = clamp_norm(
            self.apply_force_kp_matrix(self.project_pos_diff(
                force_origin_b - force_keypoint_b,
                force_dir=force_dir_b,
            )),
            self.max_force * self.force_alpha
        )
        self.force_expected_b[:] = force_expected_b
        self.force_expected_w[:] = quat_apply(root_quat_exp, force_expected_b)

    # this function is for baseline (non-compliant perturbation)
    def force_update_perturb_and_target(self):
        self.force_sample_timer -= 1
        time_done = self.force_sample_timer <= 0
        need_resample = time_done.nonzero(as_tuple=False).squeeze(-1)
        if need_resample.numel() > 0:
            transit_time = random_uniform(need_resample.shape[0], 20, 50, self.device).int()
            hold_time = random_uniform(need_resample.shape[0], 20, 100, self.device).int()
            self.force_sample_timer[need_resample] = transit_time + hold_time

            force = rand_points_isotropic(
                need_resample.shape[0],
                2,
                r_max=float(self.max_force) * float(self.force_alpha),
                device=self.device,
            )  # [K, 2, 3]
            force_enable = (torch.rand(need_resample.shape[0], 2, device=self.device) < 0.5).unsqueeze(-1)  # [K, 2, 1]
            self.perturb_force.set(need_resample, end=force * force_enable, total_steps=transit_time)

        # force target: track the reference keypoints (no admittance)
        self.force_keypoint_w[:] = self.reward_keypoints_w[:, self.force_apply_idx_motion]

        root_pos_w = self.asset.data.root_pos_w.unsqueeze(1)
        root_quat_w = self.asset.data.root_quat_w.unsqueeze(1)
        self.force_keypoint_b[:] = quat_apply_inverse(root_quat_w, self.force_keypoint_w - root_pos_w)

        if self.last_reset_env_ids is not None:
            self.force_keypoint_w_prev[self.last_reset_env_ids] = self.force_keypoint_w[self.last_reset_env_ids]
        self.force_keypoint_vel_w[:] = (self.force_keypoint_w - self.force_keypoint_w_prev) / self.env.step_dt
        self.force_keypoint_w_prev[:] = self.force_keypoint_w

    def project_pos_diff(self, pos_diff: torch.Tensor, force_dir: torch.Tensor):
        coef = (pos_diff * force_dir).sum(dim=-1, keepdim=True).clamp_max(0.0)
        return coef * force_dir
    
    def project_vel(self, vel: torch.Tensor, force_dir: torch.Tensor):
        coef = (vel * force_dir).sum(dim=-1, keepdim=True).clamp_min(0.0)
        return coef * force_dir

    def _limit_net_wrench_about_torso(self):
        apply_idx = self.net_pull_idx_asset if self.external_force_mode == "net_pull" else self.force_apply_idx_asset
        torso_i   = self.torso_idx_asset

        pos_w_6   = self.asset.data.body_pos_w[:, apply_idx, :]    # [N, M, 3]
        F_w_6     = self.force_apply_buffer[:, apply_idx, :]       # [N, M, 3]
        Tau_w_6   = self.torque_apply_buffer[:, apply_idx, :]      # [N, M, 3]

        p_torso   = self.asset.data.body_pos_w[:, self.torso_idx_asset, :].unsqueeze(1)  # [N,1,3]
        r_w_6     = pos_w_6 - p_torso                                                    # [N, M, 3]

        F_net = F_w_6.sum(dim=1)                                     # [N,3]
        M_net = torch.cross(r_w_6, F_w_6, dim=-1).sum(dim=1) + Tau_w_6.sum(dim=1)  # [N,3]

        eps = 1e-8
        
        F_norm   = F_net.norm(dim=-1, keepdim=True).clamp_min(eps)   # [N,1]
        F_scale  = torch.clamp(self.net_force_limit / F_norm, max=1.0)
        F_allow  = F_net * F_scale                                   # [N,3]
        dF       = F_allow - F_net                                   # [N,3]
        
        M_norm   = M_net.norm(dim=-1, keepdim=True).clamp_min(eps)   # [N,1]
        M_scale  = torch.clamp(self.net_torque_limit / M_norm, max=1.0)
        M_allow  = M_net * M_scale
        dM       = M_allow - M_net

        self.force_apply_buffer[:, torso_i,  :].add_(dF)
        self.torque_apply_buffer[:, torso_i, :].add_(dM)

    def force_apply(self, substep):
        self.force_applied_w.zero_()
        self.force_applied_b.zero_()
        self.force_apply_buffer.zero_()
        self.torque_apply_buffer.zero_()

        pos_w = self.asset.data.body_pos_w[:, self.force_apply_idx_asset, :]
        quat_w = self.asset.data.body_quat_w[:, self.force_apply_idx_asset, :]

        self.force_applied_w[:] = clamp_norm(
            self.force_kp_scaled * self.project_pos_diff(self.force_origin_w - pos_w, force_dir=self.force_dir_w),
            self.max_force * self.force_alpha
        )
        self.force_applied_b[:] = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.force_applied_w
        )
        self.force_apply_buffer[:, self.force_apply_idx_asset, :] = self.force_applied_w

        delta_w = quat_apply(quat_w, self.force_pos_delta)
        self.torque_apply_buffer[:, self.force_apply_idx_asset, :] = torch.cross(delta_w, self.force_applied_w, dim=-1)

        # override original force apply
        self.asset.has_external_wrench = False
        self.force_apply_world = True

    # this function is for baseline (non-compliant perturbation)
    def force_apply_perturb(self, substep: int):
        self.force_applied_w.zero_()
        self.force_applied_b.zero_()
        self.force_apply_buffer.zero_()
        self.torque_apply_buffer.zero_()

        quat_w = self.asset.data.body_quat_w[:, self.force_apply_idx_asset, :]
        self.force_applied_w[:, -2:] = self.perturb_force.current  # [N, 2, 3]
        self.force_applied_b[:] = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.force_applied_w,
        )
        self.force_apply_buffer[:, self.force_apply_idx_asset, :] = self.force_applied_w

        delta_w = quat_apply(quat_w, self.force_pos_delta)
        self.torque_apply_buffer[:, self.force_apply_idx_asset, :] = torch.cross(delta_w, self.force_applied_w, dim=-1)

        # override original force apply
        self.asset.has_external_wrench = False
        self.force_apply_world = True

    def force_apply_net_pull(self, substep: int):
        self.force_applied_w.zero_()
        self.force_applied_b.zero_()
        self.force_apply_buffer.zero_()
        self.torque_apply_buffer.zero_()

        env_ids = torch.arange(self.num_envs, device=self.device)
        if getattr(self, "eval_bimanual_ee_force_enabled", False):
            body_idx = self.net_pull_ee_idx_asset
            root_quat = self.asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
            force_w = quat_apply(root_quat, self.eval_bimanual_ee_force_b)
            self.force_apply_buffer[env_ids[:, None], body_idx[None, :], :] = force_w
            self.asset.has_external_wrench = False
            self.force_apply_world = True
            return

        body_idx = self.net_pull_idx_asset[self.net_pull_body_local_idx]
        bimanual = self.net_pull_bimanual_active
        single_ids = env_ids[~bimanual]
        if single_ids.numel() > 0:
            self.force_apply_buffer[single_ids, body_idx[single_ids], :] = self.net_pull_force_w[single_ids]
        bimanual_ids = env_ids[bimanual]
        if bimanual_ids.numel() > 0:
            body_idx_bimanual = self.net_pull_idx_asset[
                self.net_pull_bimanual_body_local_idx[bimanual_ids]
            ]
            self.force_apply_buffer[
                bimanual_ids[:, None], body_idx_bimanual, :
            ] = self.net_pull_bimanual_force_w[bimanual_ids]

        # Keep legacy-shaped force tensors meaningful for debug paths that sum them.
        self.force_applied_w.zero_()
        self.force_applied_b.zero_()
        force_apply_idx_asset = self.force_apply_idx_asset.to(body_idx.device)
        self.force_keypoint_b.zero_()
        self.force_expected_w.zero_()
        self.force_expected_b.zero_()

        single_matches = body_idx.unsqueeze(1) == force_apply_idx_asset.unsqueeze(0)
        single_matches &= (~bimanual).unsqueeze(1)
        matched_envs = single_matches.any(dim=1)
        if matched_envs.any():
            matched_slots = single_matches[matched_envs].int().argmax(dim=1)
            matched_ids = env_ids[matched_envs]
            self.force_keypoint_b[matched_ids, matched_slots, :] = self.net_pull_point_b[matched_envs]
            self.force_applied_w[matched_ids, matched_slots, :] = self.net_pull_force_w[matched_envs]
            self.force_applied_b[matched_ids, matched_slots, :] = self.net_pull_force_b[matched_envs]
            self.force_expected_w[matched_ids, matched_slots, :] = self.net_pull_force_w[matched_envs]
            self.force_expected_b[matched_ids, matched_slots, :] = self.net_pull_force_b[matched_envs]

        for hand in range(2):
            if bimanual_ids.numel() == 0:
                break
            hand_body_idx = body_idx_bimanual[:, hand]
            hand_matches = hand_body_idx.unsqueeze(1) == force_apply_idx_asset.unsqueeze(0)
            if hand_matches.any():
                hand_slots = hand_matches.int().argmax(dim=1)
                self.force_keypoint_b[bimanual_ids, hand_slots, :] = self.net_pull_bimanual_point_b[
                    bimanual_ids, hand
                ]
                self.force_applied_w[bimanual_ids, hand_slots, :] = self.net_pull_bimanual_force_w[
                    bimanual_ids, hand
                ]
                self.force_applied_b[bimanual_ids, hand_slots, :] = self.net_pull_bimanual_force_b[
                    bimanual_ids, hand
                ]
                self.force_expected_w[bimanual_ids, hand_slots, :] = self.net_pull_bimanual_force_w[
                    bimanual_ids, hand
                ]
                self.force_expected_b[bimanual_ids, hand_slots, :] = self.net_pull_bimanual_force_b[
                    bimanual_ids, hand
                ]

        self.asset.has_external_wrench = False
        self.force_apply_world = True

    def before_update(self):
        super().before_update()
        self.update_reward_target()
        if not self.external_force_enabled:
            self.set_external_force_enabled(False)
            return
        if getattr(self, "eval_bimanual_ee_force_enabled", False):
            self._update_eval_bimanual_ee_force_target()
            return
        if getattr(self, "eval_ee_force_enabled", False):
            self._update_eval_ee_force_target()
            return
        if getattr(self, "eval_root_force_enabled", False):
            self._update_eval_root_force_target()
            return
        if self.external_force_mode == "net_pull":
            self.force_safe_limit_tl.update_time()
            self.net_pull_force_tl.update_time()
            self.net_pull_bimanual_force_tl.update_time()
            self.net_pull_schedule()
            self.net_pull_update_target()
            self.update_net_pull_ee_compliance_target()
            self.sync_net_pull_ee_reward_target_buffers()
        elif self.compliance:
            self.force_kp_tl.update_time()
            self.force_origin_tl.update_time()
            self.force_safe_limit_tl.update_time()
            self.force_schedule()
            self.force_update_origin_and_target()
        else:
            self.perturb_force.update_time()
            self.force_update_perturb_and_target()

    def step(self, substep: int):
        super().step(substep)
        if not self.external_force_enabled:
            self.set_external_force_enabled(False)
            return
        if getattr(self, "eval_ee_force_enabled", False):
            self.force_apply_net_pull(substep)
            self._limit_net_wrench_about_torso()
            return
        if getattr(self, "eval_root_force_enabled", False):
            self.force_apply_net_pull(substep)
            self._limit_net_wrench_about_torso()
            return
        if self.external_force_mode == "net_pull":
            self.force_apply_net_pull(substep)
        elif self.compliance:
            self.force_apply(substep)
        elif self.max_force > 0.0:
            self.force_apply_perturb(substep)
        self._limit_net_wrench_about_torso()
    
    def step_schedule(self, progress: float):
        self.zero_init_prob = 0.0
        self.force_alpha = 1.0
        if self.external_force_mode == "net_pull" and self.net_pull_curriculum:
            ratio = max(0.0, min(progress / self.net_pull_curriculum_end, 1.0))
            self.net_pull_zero_prob = self.net_pull_zero_prob_start + ratio * (
                self.net_pull_zero_prob_target - self.net_pull_zero_prob_start
            )
            self.net_pull_force_scale = self.net_pull_force_scale_start + ratio * (
                1.0 - self.net_pull_force_scale_start
            )
        if not self.student_train:
            ratio = max(0.25, min(progress / 0.6, 1.0))
            force_prob = 0.15 * ratio
            self.force_type_probs[1:5] = force_prob
            self.force_type_probs[0] = 1.0 - force_prob * 4

    # old command obs
    # @observation
    # def command(self):
    #     # here we use root pos instead of student body pos
    #     root_yaw_quat = yaw_quat(self._motion.root_quat_w[:, 0, :]).unsqueeze(1)
    #     root_yaw_future = yaw_quat(self._motion.root_quat_w[:, 1:, :])
    #     root_pos = self._motion.root_pos_w[:, 0, :].unsqueeze(1)
    #     root_pos_future = self._motion.root_pos_w[:, 1:, :]

    #     pos_diff_b = quat_apply_inverse(
    #         root_yaw_quat,
    #         root_pos_future - root_pos
    #     )

    #     heading = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32).reshape(1, 1, 3)
    #     target_heading = quat_apply(root_yaw_future, heading)
    #     target_heading_b = quat_apply_inverse(root_yaw_quat, target_heading)

    #     return torch.cat([
    #         self._motion.root_pos_w[:, :, 2].reshape(self.num_envs, -1),
    #         pos_diff_b[:, :, :2].reshape(self.num_envs, -1),
    #         target_heading_b[:, :, :2].reshape(self.num_envs, -1),
    #         self.force_safe_limit_tl.current
    #     ], dim=-1)

    def command_sym(self):
        # command = [root_height (1), target_linvel_b_xy (2), target_heading_b_xy (2), force_limit (1)] = 6 dims
        return sym_utils.SymmetryTransform.cat([
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]),        # root_height: no change
            sym_utils.SymmetryTransform(perm=torch.arange(2), signs=[1, -1]),   # target_linvel_b_xy: flip y
            sym_utils.SymmetryTransform(perm=torch.arange(2), signs=[1, -1]),   # target_heading_b_xy: flip y
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]),        # force_limit: no change
        ])

    def _command_from_motion(self):
        """
        Simplified command observation for velocity tracking + EE reaching tasks.
        Only provides next frame target (future step 1), not full trajectory.
        
        Output format:
        - root_height: [N, 1] - current frame root height
        - target_linvel_b: [N, 2] - target xy linear velocity in body frame
        - target_heading_b: [N, 2] - target heading direction in body frame
        - force_safe_limit: [N, 1] - force limit
        
        Total: 6 dimensions
        """
        # Get motion data for current and next frame
        # self._motion contains [N, S, ...] where S = len(future_steps)
        # future_steps = [0, 2, 4, 8, 16], so index 1 is 2 steps ahead
        
        # ---- Root height (current frame) ----
        root_height = self._motion.root_pos_w[:, 0, 2:3]  # [N, 1]
        
        # ---- Target linear velocity in body frame ----
        # Use motion's root_lin_vel_w for next frame, convert to body frame
        current_quat = self.asset.data.root_quat_w  # [N, 4]
        target_linvel_w = self._motion.root_lin_vel_w[:, 0]  # [N, 3] - current frame velocity
        target_linvel_b = quat_apply_inverse(current_quat, target_linvel_w)  # [N, 3]
        target_linvel_b_xy = target_linvel_b[:, :2]  # [N, 2] - only xy
        
        # ---- Target heading in body frame ----
        # Compute heading direction from motion's root orientation
        root_yaw_quat_current = yaw_quat(current_quat)  # [N, 4]
        root_yaw_quat_target = yaw_quat(self._motion.root_quat_w[:, 0])  # [N, 4] - current frame target
        
        heading = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        target_heading_w = quat_apply(root_yaw_quat_target, heading.unsqueeze(0).expand(self.num_envs, -1))  # [N, 3]
        target_heading_b = quat_apply_inverse(root_yaw_quat_current, target_heading_w)  # [N, 3]
        target_heading_b_xy = target_heading_b[:, :2]  # [N, 2]
        
        # ---- Force safe limit ----
        force_limit = self.force_safe_limit_tl.current  # [N, 1]
        
        out = torch.cat([
            root_height,           # [N, 1]
            target_linvel_b_xy,    # [N, 2]
            target_heading_b_xy,   # [N, 2]
            force_limit            # [N, 1]
        ], dim=-1)  # [N, 6]
        
        return out

    def _command_from_udp(self):
        """
        Simplified command observation for teleop mode.

        Output format matches _command_from_motion:
        root_height(1), target_linvel_b_xy(2), target_heading_b_xy(2), force_limit(1).
        """
        root_height = torch.full((self.num_envs, 1), 0.79, device=self.device)
        target_linvel_b_xy = torch.zeros(self.num_envs, 2, device=self.device)
        target_heading_b_xy = torch.zeros(self.num_envs, 2, device=self.device)
        target_heading_b_xy[:, 0] = 1.0

        if self._teleop is not None:
            seq, _t_recv, \
                root_pos, root_quat, \
                _head_pos_b, _head_quat_b, \
                _l_pos_b, _l_quat_b, \
                _r_pos_b, _r_quat_b = self._teleop.get_latest()

            if seq >= 0:
                root_quat_xyzw = root_quat.to(self.device)
                root_quat_wxyz = torch.stack([
                    root_quat_xyzw[3],
                    root_quat_xyzw[0],
                    root_quat_xyzw[1],
                    root_quat_xyzw[2],
                ], dim=-1)
                root_quat_wxyz = root_quat_wxyz / (torch.norm(root_quat_wxyz) + 1e-8)
                target_yaw_quat = yaw_quat(root_quat_wxyz.unsqueeze(0))
                target_yaw_quat = target_yaw_quat.expand(self.num_envs, -1)

                current_yaw_quat = yaw_quat(self.asset.data.root_quat_w)
                heading_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device)
                target_heading_w = quat_apply(
                    target_yaw_quat,
                    heading_vec.unsqueeze(0).expand(self.num_envs, -1),
                )
                target_heading_b = quat_apply_inverse(current_yaw_quat, target_heading_w)
                target_heading_b_xy = target_heading_b[:, :2]

                vel_scale = 1.0
                vel_in_target_frame = torch.zeros(self.num_envs, 3, device=self.device)
                vel_in_target_frame[:, 0] = -root_pos[0].to(self.device) * vel_scale
                vel_in_target_frame[:, 1] = -root_pos[1].to(self.device) * vel_scale
                target_linvel_w = quat_apply(target_yaw_quat, vel_in_target_frame)
                target_linvel_b = quat_apply_inverse(current_yaw_quat, target_linvel_w)
                target_linvel_b_xy = target_linvel_b[:, :2]

                if root_pos[2] > 0.1:
                    root_height = torch.full((self.num_envs, 1), root_pos[2].item(), device=self.device)

                self._command_udp_counter = getattr(self, "_command_udp_counter", -1) + 1
                if self._command_udp_counter % 200 == 0:
                    print(
                        f"[command UDP] seq={seq}, "
                        f"vel_target_frame=[{root_pos[0].item():.2f}, {root_pos[1].item():.2f}], "
                        f"linvel_b={target_linvel_b_xy[0].tolist()}, "
                        f"heading_b={target_heading_b_xy[0].tolist()}",
                        flush=True,
                    )

        force_limit = self.force_safe_limit_tl.current
        return torch.cat([
            root_height,
            target_linvel_b_xy,
            target_heading_b_xy,
            force_limit,
        ], dim=-1)

    @observation
    def command(self):
        if self.use_static_root_command:
            return self._static_root_command()
        if self.use_command_override:
            return self.command_override
        if not hasattr(self, "_printed_command_source"):
            print(
                f"[MotionTrackingCommand] command source={self.teleop_obs_source}",
                flush=True,
            )
            self._printed_command_source = True
        if self.teleop_obs_source == "udp":
            return self._command_from_udp()
        return self._command_from_motion()

    def get_root_command_reference(self):
        return self.command()[..., :5]

    def set_command_override(self, command: torch.Tensor):
        if command.shape[-1] != 6:
            raise ValueError(f"Command override must have 6 dims, got {command.shape[-1]}.")
        self.use_command_override = True
        self.command_override[:] = command

    def set_static_root_command(
        self,
        root_height: torch.Tensor | None = None,
        heading_w: torch.Tensor | None = None,
    ):
        self.use_static_root_command = True
        if root_height is None:
            root_height = self.asset.data.root_pos_w[:, 2:3]
        if heading_w is None:
            heading = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            heading_w = quat_apply(
                yaw_quat(self.asset.data.root_quat_w),
                heading.unsqueeze(0).expand(self.num_envs, -1),
            )
        self.static_root_height[:] = torch.as_tensor(root_height, device=self.device, dtype=torch.float32)
        self.static_heading_w[:] = normalize(torch.as_tensor(heading_w, device=self.device, dtype=torch.float32))
        self.static_heading_w[:, 2] = 0.0
        self.static_heading_w[:] = normalize(self.static_heading_w)

    def _static_root_command(self):
        current_yaw_quat = yaw_quat(self.asset.data.root_quat_w)
        target_heading_b = quat_apply_inverse(current_yaw_quat, self.static_heading_w)
        target_heading_b_xy = normalize(target_heading_b[:, :2])
        target_linvel_b_xy = torch.zeros(self.num_envs, 2, device=self.device)
        force_limit = self.force_safe_limit_tl.current if hasattr(self, "force_safe_limit_tl") else torch.zeros(self.num_envs, 1, device=self.device)
        return torch.cat([
            self.static_root_height,
            target_linvel_b_xy,
            target_heading_b_xy,
            force_limit,
        ], dim=-1)

    @observation
    def force_priv(self):
        return torch.cat([
            self.force_keypoint_b.reshape(self.num_envs, -1),
            self.force_applied_b.reshape(self.num_envs, -1),
            self.force_expected_b.reshape(self.num_envs, -1),
            self.force_sample_timer.reshape(self.num_envs, -1)
        ], dim=-1)
    
    def force_priv_sym(self):
        return sym_utils.SymmetryTransform.cat([
            sym_utils.cartesian_space_symmetry(self.asset,get_items_by_index(self.asset.body_names, self.force_apply_idx_asset), sign=[1, -1, 1]).repeat(3),
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1])
        ])

    @observation
    def net_pull_ee_compliance_stiffness_command(self):
        """EE stiffness command for scalar or xyz range tasks.

        New ablations can request the direct reciprocal ``1 / k`` command.
        The default path remains the original normalized stiffness command.
        """
        if self.net_pull_ee_compliance_stiffness_per_ee:
            stiffness = self.net_pull_ee_compliance_stiffness_current
            if self.net_pull_ee_compliance_stiffness_xyz_range is not None:
                stiffness_min, stiffness_max = self.net_pull_ee_compliance_stiffness_xyz_range
                command = 2.0 * (stiffness - stiffness_min) / (stiffness_max - stiffness_min).clamp_min(1e-6) - 1.0
            else:
                command = stiffness
            if self.net_pull_ee_compliance_stiffness_inverse_command:
                command = 1.0 / stiffness.clamp_min(1e-6)
            return command.reshape(self.num_envs, 6)
        if self.net_pull_ee_compliance_stiffness_xyz_range is not None:
            stiffness_min, stiffness_max = self.net_pull_ee_compliance_stiffness_xyz_range
            stiffness = self.net_pull_ee_compliance_stiffness_current[:, 0, :]
            if self.net_pull_ee_compliance_stiffness_inverse_command:
                return 1.0 / stiffness.clamp_min(1e-6)
            return 2.0 * (stiffness - stiffness_min) / (stiffness_max - stiffness_min) - 1.0
        if self.net_pull_ee_compliance_stiffness_range is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        stiffness_min, stiffness_max = self.net_pull_ee_compliance_stiffness_range
        stiffness = self.net_pull_ee_compliance_stiffness_current[:, 0, 0:1]
        if self.net_pull_ee_compliance_stiffness_inverse_command:
            return 1.0 / stiffness.clamp_min(1e-6)
        return 2.0 * (stiffness - stiffness_min) / (stiffness_max - stiffness_min) - 1.0

    def net_pull_ee_compliance_stiffness_command_sym(self):
        if self.net_pull_ee_compliance_stiffness_xyz_range is not None:
            return sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, 1, 1])
        return sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1])

    @observation
    def net_pull_ee_force_b_priv(self):
        """Per-hand net-pull force in the root frame: [left xyz, right xyz]."""
        force_scale = max(self.net_pull_force_range[1], 1e-6)
        return (self.net_pull_ee_force_b / force_scale).reshape(self.num_envs, -1)

    def net_pull_ee_force_b_priv_sym(self):
        return sym_utils.cartesian_space_symmetry(
            self.asset,
            self.net_pull_ee_names,
            sign=[1, -1, 1],
        )

    @observation
    def net_pull_force_b_priv(self):
        """Compact root-frame net-pull force plus the active body identity."""
        force_scale = max(self.net_pull_force_range[1], 1e-6)
        body_one_hot = torch.zeros(self.num_envs, self.net_pull_num_bodies, device=self.device)
        body_one_hot.scatter_(1, self.net_pull_body_local_idx.unsqueeze(1), 1.0)

        # Do not expose a stale body identity during the rest/no-force phase.
        active = (self.net_pull_force_b.norm(dim=-1, keepdim=True) > 1e-6).float()
        body_one_hot = body_one_hot * active
        return torch.cat([
            self.net_pull_force_b / force_scale,
            body_one_hot,
        ], dim=-1)

    def net_pull_force_b_priv_sym(self):
        body_names = get_items_by_index(self.asset.body_names, self.net_pull_idx_asset)
        symmetry_mapping = self.asset.cfg.spatial_symmetry_mapping
        body_perm = torch.zeros(len(body_names), dtype=torch.long)
        for i, body_name in enumerate(body_names):
            body_perm[i] = body_names.index(symmetry_mapping[body_name])

        return sym_utils.SymmetryTransform.cat([
            sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1]),
            sym_utils.SymmetryTransform(perm=body_perm, signs=torch.ones(len(body_names))),
        ])

    @observation
    def net_pull_force_priv(self):
        body_one_hot = torch.zeros(self.num_envs, self.net_pull_num_bodies, device=self.device)
        body_one_hot.scatter_(1, self.net_pull_body_local_idx.unsqueeze(1), 1.0)

        phase_one_hot = torch.zeros(self.num_envs, 4, device=self.device)
        phase_idx = self.net_pull_phase.clamp(min=0, max=3).long()
        phase_one_hot.scatter_(1, phase_idx.unsqueeze(1), 1.0)

        timer = self.net_pull_phase_timer.clamp_min(0).float().unsqueeze(-1) / 250.0
        force_mag = self.net_pull_force_tl.current / max(self.net_pull_force_range[1], 1e-6)
        return torch.cat([
            self.net_pull_point_b,
            self.net_pull_force_b / max(self.net_pull_force_range[1], 1e-6),
            self.net_pull_force_w / max(self.net_pull_force_range[1], 1e-6),
            body_one_hot,
            phase_one_hot,
            timer,
            force_mag,
        ], dim=-1)

    def net_pull_force_priv_sym(self):
        body_names = get_items_by_index(self.asset.body_names, self.net_pull_idx_asset)
        symmetry_mapping = self.asset.cfg.spatial_symmetry_mapping
        body_perm = torch.zeros(len(body_names), dtype=torch.long)
        for i, body_name in enumerate(body_names):
            body_perm[i] = body_names.index(symmetry_mapping[body_name])

        return sym_utils.SymmetryTransform.cat([
            sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1]),
            sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1]),
            sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1]),
            sym_utils.SymmetryTransform(perm=body_perm, signs=torch.ones(len(body_names))),
            sym_utils.SymmetryTransform(perm=torch.arange(4), signs=torch.ones(4)),
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]),
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]),
        ])

    @reward
    def keypoint_tracking_imp(self):
        actual = self.asset.data.body_pos_w[:, self.keypoint_idx_asset]
        target = self.reward_keypoints_w[:, self.keypoint_idx_motion].clone()
        target[:, self.force_in_keypoint_idx] = self.force_keypoint_w
        diff = target - actual
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        self._cum_error[:, 2:3] = error / self._cum_keypoint_scale
        return _calc_exp_sigma(error, self.reward_sigma["keypoint"])

    @reward
    def force_target_tracking(self):
        actual = self.asset.data.body_pos_w[:, self.force_apply_idx_asset, :]
        target = self.force_keypoint_w
        diff = target - actual
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["force_target"])

    @reward
    def force_target_vel_tracking(self):
        actual_vel = self.asset.data.body_lin_vel_w[:, self.force_apply_idx_asset, :]
        target_vel = self.force_keypoint_vel_w
        diff = target_vel - actual_vel
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["force_target_vel"])

    @reward
    def force_reward(self):
        force_norm = self.force_applied_w.norm(dim=-1, keepdim=False)
        force_norm_diff = (self.force_applied_w - self.force_expected_w).norm(dim=-1).mean(dim=-1, keepdim=True)  # [N, M]
        force_reward = _calc_exp_sigma(force_norm_diff, self.reward_sigma["force"])
        force_max_limit = self.force_safe_limit_tl.current + self.force_penalty_offset
        force_exd = (force_norm > force_max_limit).any(dim=-1, keepdim=True)
        return (force_reward * (~force_exd)).mean(dim=-1, keepdim=True)

    @reward
    def force_exd_penalty(self):
        force_norm = self.force_applied_w.norm(dim=-1, keepdim=False)
        force_exp_norm = self.force_expected_w.norm(dim=-1, keepdim=False)
        force_max_limit = self.force_safe_limit_tl.current + self.force_penalty_offset
        force_exd = ((force_norm > force_max_limit) & (force_norm > force_exp_norm + self.force_penalty_offset*0.5)).float().mean(dim=-1, keepdim=True) # [N, 1]
        return - force_exd

    def debug_draw(self):
        super().debug_draw()
        # Visualize the force buffer that is actually submitted to PhysX.  The
        # legacy force_applied_w tensor only covers force_apply_idx_asset, so it
        # can miss net-pull/eval forces applied to bodies such as torso_link.
        if getattr(self, "force_apply_world", False) and hasattr(self, "force_apply_buffer"):
            pos = self.asset.data.body_pos_w
            force_vis = self.force_apply_buffer * 0.02
            active_force = self.force_apply_buffer.norm(dim=-1) > 1e-3
        else:
            pos = self.asset.data.body_pos_w[:, self.force_apply_idx_asset, :]
            root_yaw = yaw_quat(self.asset.data.root_quat_w).unsqueeze(1)
            force_root_heading_w = quat_apply(root_yaw, self.force_applied_b)
            force_vis = force_root_heading_w * 0.02
            active_force = self.force_applied_w.norm(dim=-1) > 1e-3

        if active_force.any():
            pos_flat = pos[active_force].reshape(-1, 3)
            force_flat = force_vis[active_force].reshape(-1, 3)
            self.env.debug_draw.vector(
                pos_flat, force_flat,
                color=(1.0, 0.0, 1.0, 1.0), size=4.0
            )
            self.env.debug_draw.point(
                pos_flat, color=(1.0, 0.0, 1.0, 1.0), size=8.0
            )

        # act1 = self.asset.data.body_pos_w[:, self.force_apply_idx_asset, :].reshape(-1, 3)
        # tar1 = self.force_keypoint_w.reshape(-1, 3)
        # self.env.debug_draw.vector(
        #     act1,
        #     (tar1 - act1).reshape(-1, 3),
        #     color=(1, 0, 1, 1)
        # )
        # self.env.debug_draw.point(
        #     tar1.reshape(-1, 3), color=(1, 0, 1, 1), size=10.0
        # )
        # self.env.debug_draw.point(
        #     act1.reshape(-1, 3), color=(0, 0, 1, 1), size=10.0
        # )


class RootCommandMotionTrackingCommand_impedance(MotionTrackingCommand_impedance):
    """Motion-tracking command variant whose root command is supplied externally."""

    def __init__(
        self,
        env,
        *args,
        nominal_root_height: float = 0.79,
        **kwargs,
    ):
        super().__init__(env, *args, **kwargs)
        self.nominal_root_height = nominal_root_height
        self.root_command = torch.zeros(self.num_envs, 5, device=self.device)
        self.root_command[:, 0] = self.nominal_root_height
        self.root_command[:, 3] = 1.0
        self.use_root_and_wrist_6d_command = False
        self.root_and_wrist_6d_command = torch.zeros(self.num_envs, 12, device=self.device)
        self.use_feet_pos_b_command = False
        self.feet_pos_b_command = torch.zeros(self.num_envs, 6, device=self.device)

    def reset(self, env_ids):
        super().reset(env_ids)
        self.root_command[env_ids] = 0.0
        self.root_command[env_ids, 0] = self.nominal_root_height
        self.root_command[env_ids, 3] = 1.0
        self.root_and_wrist_6d_command[env_ids] = self.get_root_and_wrist_6d_reference()[env_ids]
        self.feet_pos_b_command[env_ids] = self.get_feet_pos_b_reference()[env_ids]

    def set_root_command(self, command: torch.Tensor):
        if command.shape[-1] != 5:
            raise ValueError(f"Root command must have 5 dims, got {command.shape[-1]}.")
        self.root_command[:] = command

    def get_root_command_reference(self):
        if self.use_static_root_command:
            return self._static_root_command()[..., :5]
        if self.use_command_override:
            return self.command_override[..., :5]
        if self.teleop_obs_source == "udp":
            return self._command_from_udp()[..., :5]
        return self._command_from_motion()[..., :5]

    @observation
    def root_command_reference(self):
        return self.get_root_command_reference()

    def root_command_reference_sym(self):
        return sym_utils.SymmetryTransform.cat([
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]),
            sym_utils.SymmetryTransform(perm=torch.arange(2), signs=[1, -1]),
            sym_utils.SymmetryTransform(perm=torch.arange(2), signs=[1, -1]),
        ])

    def get_root_and_wrist_6d_reference(self):
        if self.use_root_and_wrist_6d_reference_override:
            return self.root_and_wrist_6d_reference_override
        if self.teleop_obs_source == "udp":
            return self._root_and_wrist_6d_from_udp()
        return self._apply_synthetic_ee_target(self._root_and_wrist_6d_from_motion())

    def set_root_and_wrist_6d_command(self, command: torch.Tensor):
        if command.shape[-1] != 12:
            raise ValueError(f"Root-and-wrist command must have 12 dims, got {command.shape[-1]}.")
        self.use_root_and_wrist_6d_command = True
        self.root_and_wrist_6d_command[:] = command

    def get_feet_pos_b_reference(self):
        if self.teleop_obs_source == "udp":
            return self._feet_pos_b_from_udp()
        return self._feet_pos_b_from_motion()

    def _feet_pos_b_from_motion(self):
        motion = self._motion
        if hasattr(motion, "local_body_pos"):
            target_feet_b = motion.local_body_pos[:, 0, self.feet_idx_motion, :]
        elif hasattr(motion, "body_pos_b"):
            target_feet_b = motion.body_pos_b[:, 0, self.feet_idx_motion, :]
        else:
            return torch.zeros(self.num_envs, 6, device=self.device)
        return target_feet_b.reshape(self.num_envs, -1)

    def _feet_pos_b_from_udp(self):
        if self._teleop is not None:
            has_feet_pos_b, feet_pos_b = self._teleop.get_latest_feet_pos_b()
            if has_feet_pos_b:
                return feet_pos_b.to(self.device).unsqueeze(0).expand(self.num_envs, -1)
        return self._feet_pos_b_from_motion()

    def set_feet_pos_b_command(self, command: torch.Tensor):
        if command.shape[-1] != 6:
            raise ValueError(f"Feet position command must have 6 dims, got {command.shape[-1]}.")
        self.use_feet_pos_b_command = True
        self.feet_pos_b_command[:] = command

    @observation
    def root_and_wrist_6d(self):
        if self.use_root_and_wrist_6d_command:
            return self.root_and_wrist_6d_command
        return self.get_root_and_wrist_6d_reference()

    @observation
    def feet_pos_b(self):
        if self.use_feet_pos_b_command:
            return self.feet_pos_b_command
        return self.get_feet_pos_b_reference()

    @observation
    def command(self):
        force_limit = self.force_safe_limit_tl.current
        return torch.cat([self.root_command, force_limit], dim=-1)
