#!/usr/bin/env python3
import math
import numpy as np
import socket
import struct
import time
import argparse
import sys
import termios
import tty
import select
import pathlib
import json

np.random.seed(48)
MAGIC = b"G6D1"
JOINT_MAGIC = b"G6J1"
JOINT_TARGET_FUTURE_STEPS = (0, 2, 4, 8, 16)
# magic(4s) + seq(u32) + N float32. The first 28 floats are fixed:
# root/head/left_hand/right_hand, each pos3 + quat4. Extra keypoints append
# after that prefix; the first extension is feet_pos_b with 6 floats.
PACK_HEADER_FMT = "<4sI"
PACK_FMT = PACK_HEADER_FMT + "f" * 28


DEFAULT_BODY_NAMES_30 = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_hand_mimic",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_hand_mimic",
]


DEFAULT_BODY_NAMES_28 = [
    "world",
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "head_mimic",
    "left_hand_mimic",
    "right_hand_mimic",
]


DEFAULT_G1_JOINT_NAMES_29 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def make_packet(seq, floats):
    return struct.pack(PACK_HEADER_FMT + "f" * len(floats), MAGIC, seq, *map(float, floats))


def make_packet_with_magic(seq, floats, magic=MAGIC):
    return struct.pack(PACK_HEADER_FMT + "f" * len(floats), magic, seq, *map(float, floats))


class SignalSendLogger:
    """Append-only logger for sent robot command packets."""

    def __init__(self, mode: int, root_dir: str = "logs/signal_send", value_count: int = 28):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.dir = pathlib.Path(root_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"teleop_mode{mode}_{ts}.csv"
        self.f = self.path.open("w", buffering=1)
        header = ["timestamp", "seq", "mode"] + [f"v{i}" for i in range(value_count)]
        self.f.write(",".join(header) + "\n")

    def log(self, t_sec: float, seq: int, mode: int, values):
        vals = ",".join(f"{float(v):.6f}" for v in values)
        self.f.write(f"{t_sec:.6f},{int(seq)},{int(mode)},{vals}\n")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

def quat_from_yaw(yaw: float):
    """Return (x,y,z,w) quaternion for yaw-only rotation."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def quat_to_rpy(quat):
    """
    Convert quaternion (x, y, z, w) to Roll-Pitch-Yaw (in radians).
    Returns (roll, pitch, yaw) in radians.
    """
    qx, qy, qz, qw = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return (roll, pitch, yaw)


def euler_to_quat(roll: float, pitch: float, yaw: float):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w).
    Angles are in radians. Uses the z-y'-x'' (yaw-pitch-roll) convention.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def euler_to_target_quat(roll: float, pitch: float, yaw: float):
    """
    Convert Euler angles (roll,pitch,yaw) to a quaternion in the target
    coordinate system by first building a quaternion in the source (Euler)
    frame and then applying the same axis/flip mapping we use for VR
    quaternions (vr_to_target_quat). This keeps orientation handling
    consistent across modes when Euler angles are used.
    """
    q = euler_to_quat(roll, pitch, yaw)
    return vr_to_target_quat(q)


def quat_normalize_xyzw(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def quat_inv_xyzw(q):
    q = quat_normalize_xyzw(q)
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=float)


def quat_mul_xyzw(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return quat_normalize_xyzw(np.array([x, y, z, w], dtype=float))


def quat_normalize_wxyz(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_inv_wxyz(q):
    q = quat_normalize_wxyz(q)
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_mul_wxyz(q1, q2):
    q1 = quat_normalize_wxyz(q1)
    q2 = quat_normalize_wxyz(q2)
    return quat_normalize_wxyz(quat_mul_wxyz_raw(q1, q2))


def quat_mul_wxyz_raw(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def quat_apply_wxyz(q, v):
    q = quat_normalize_wxyz(q)
    vq = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=float)
    return quat_mul_wxyz_raw(quat_mul_wxyz_raw(q, vq), quat_inv_wxyz(q))[1:4]


def quat_wxyz_to_xyzw(q):
    q = quat_normalize_wxyz(q)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def quat_xyzw_to_wxyz(q):
    q = quat_normalize_xyzw(q)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def body_names_from_motion(m, body_count):
    if "body_names" in m:
        return [str(x) for x in m["body_names"].tolist()]
    if body_count == len(DEFAULT_BODY_NAMES_30):
        return DEFAULT_BODY_NAMES_30
    if body_count == len(DEFAULT_BODY_NAMES_28):
        return DEFAULT_BODY_NAMES_28
    raise ValueError(
        f"Motion file has {body_count} bodies and no body_names. "
        "Pass --motion-body-indices head,left_hand,right_hand,left_foot,right_foot."
    )


def resolve_motion_body_indices(body_names, override):
    if override:
        parts = [int(x.strip()) for x in override.split(",") if x.strip()]
        if len(parts) != 5:
            raise ValueError("--motion-body-indices must have 5 comma-separated indices.")
        return {
            "head": parts[0],
            "left_hand": parts[1],
            "right_hand": parts[2],
            "left_foot": parts[3],
            "right_foot": parts[4],
        }

    def find(*names):
        for name in names:
            if name in body_names:
                return body_names.index(name)
        raise ValueError(f"None of these body names were found: {names}")

    return {
        "head": find("head_mimic", "torso_link"),
        "left_hand": find("left_hand_mimic", "left_wrist_roll_link"),
        "right_hand": find("right_hand_mimic", "right_wrist_roll_link"),
        "left_foot": find("left_ankle_roll_link"),
        "right_foot": find("right_ankle_roll_link"),
    }


def joint_names_from_motion(m, joint_count):
    if "joint_names" in m:
        return [str(x) for x in m["joint_names"].tolist()]
    if joint_count == len(DEFAULT_G1_JOINT_NAMES_29):
        return DEFAULT_G1_JOINT_NAMES_29
    raise ValueError(
        f"Motion file has {joint_count} joints and no joint_names. "
        "Pass a motion file with joint_names for mode 5."
    )


def resolve_motion_joint_indices(joint_names):
    missing = [name for name in DEFAULT_G1_JOINT_NAMES_29 if name not in joint_names]
    if missing:
        raise ValueError(f"Motion joint_names missing required G1 joints: {missing}")
    return [joint_names.index(name) for name in DEFAULT_G1_JOINT_NAMES_29]


def yaw_quat_wxyz(q):
    q = quat_normalize_wxyz(q)
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def load_motion_packets(path, body_index_override=None, root_mode="velocity"):
    m = dict(np.load(path, allow_pickle=True))
    if "body_pos_w" in m and "body_quat_w" in m:
        body_pos_w = np.asarray(m["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(m["body_quat_w"], dtype=np.float32)
        root_pos_w = np.asarray(m.get("root_pos", body_pos_w[:, 0]), dtype=np.float32)
        root_quat_w = np.asarray(m.get("root_quat", body_quat_w[:, 0]), dtype=np.float32)
    elif {"local_body_pos", "local_body_rot", "root_pos", "root_rot"} <= set(m.keys()):
        local_body_pos = np.asarray(m["local_body_pos"], dtype=np.float32)
        local_body_rot_xyzw = np.asarray(m["local_body_rot"], dtype=np.float32)
        root_pos_w = np.asarray(m["root_pos"], dtype=np.float32)
        root_rot_xyzw = np.asarray(m["root_rot"], dtype=np.float32)
        body_pos_w = np.zeros_like(local_body_pos, dtype=np.float32)
        body_quat_w = np.zeros_like(local_body_rot_xyzw, dtype=np.float32)
        root_quat_w = np.zeros_like(root_rot_xyzw, dtype=np.float32)
        for t in range(local_body_pos.shape[0]):
            root_quat_w[t] = quat_xyzw_to_wxyz(root_rot_xyzw[t])
            for b in range(local_body_pos.shape[1]):
                body_pos_w[t, b] = quat_apply_wxyz(root_quat_w[t], local_body_pos[t, b]) + root_pos_w[t]
                local_quat_w = quat_xyzw_to_wxyz(local_body_rot_xyzw[t, b])
                body_quat_w[t, b] = quat_mul_wxyz(root_quat_w[t], local_quat_w)
    else:
        raise ValueError(
            "Motion file must contain either body_pos_w/body_quat_w or "
            "local_body_pos/local_body_rot/root_pos/root_rot."
        )

    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape [T, B, 3], got {body_pos_w.shape}.")
    if body_quat_w.shape[:2] != body_pos_w.shape[:2] or body_quat_w.shape[-1] != 4:
        raise ValueError(f"body_quat_w must have shape [T, B, 4], got {body_quat_w.shape}.")

    fps = float(np.asarray(m.get("fps", [30.0])).reshape(-1)[0])

    body_names = body_names_from_motion(m, body_pos_w.shape[1])
    indices = resolve_motion_body_indices(body_names, body_index_override)
    root_vel_w = np.asarray(m.get("root_lin_vel", np.zeros_like(root_pos_w)), dtype=np.float32)
    if "root_lin_vel" not in m:
        if len(root_pos_w) > 1:
            root_vel_w[:-1] = (root_pos_w[1:] - root_pos_w[:-1]) * fps
            root_vel_w[-1] = root_vel_w[-2]
        else:
            root_vel_w[:] = 0.0

    packets = []
    for t in range(body_pos_w.shape[0]):
        root_pos = root_pos_w[t].astype(float)
        root_cmd_pos = np.zeros(3, dtype=float)
        if root_mode == "velocity":
            root_yaw_inv = quat_inv_wxyz(yaw_quat_wxyz(root_quat_w[t]))
            root_vel_b = quat_apply_wxyz(root_yaw_inv, root_vel_w[t])
            root_cmd_pos[0:2] = -root_vel_b[0:2]
            root_cmd_pos[2] = root_pos[2]
        elif root_mode == "zero":
            root_cmd_pos[2] = root_pos[2]
        elif root_mode == "absolute":
            root_cmd_pos = root_pos.copy()
        else:
            raise ValueError(f"Unsupported root_mode: {root_mode}")
        root_quat_xyzw = quat_wxyz_to_xyzw(root_quat_w[t])
        root = tuple(root_cmd_pos.tolist() + root_quat_xyzw.tolist())

        root_inv = quat_inv_wxyz(root_quat_w[t])

        def local_pose(body_idx):
            pos_b = quat_apply_wxyz(root_inv, body_pos_w[t, body_idx] - root_pos_w[t])
            quat_b = quat_mul_wxyz(root_inv, body_quat_w[t, body_idx])
            return pos_b, quat_b

        head_pos_b, head_quat_b = local_pose(indices["head"])
        left_pos_b, left_quat_b = local_pose(indices["left_hand"])
        right_pos_b, right_quat_b = local_pose(indices["right_hand"])
        left_foot_b, _ = local_pose(indices["left_foot"])
        right_foot_b, _ = local_pose(indices["right_foot"])

        head = tuple(head_pos_b.tolist() + head_quat_b.tolist())
        left = tuple(left_pos_b.tolist() + left_quat_b.tolist())
        right = tuple(right_pos_b.tolist() + right_quat_b.tolist())
        feet = tuple(left_foot_b.tolist() + right_foot_b.tolist())
        packets.append(root + head + left + right + feet)

    return packets, fps, indices, body_names


def load_motion_packets_with_joint_targets(path, body_index_override=None, root_mode="velocity"):
    m = dict(np.load(path, allow_pickle=True))
    packets, fps, indices, body_names = load_motion_packets(
        path,
        body_index_override=body_index_override,
        root_mode=root_mode,
    )

    joint_key = "joint_pos" if "joint_pos" in m else "dof_pos" if "dof_pos" in m else None
    if joint_key is None:
        raise ValueError("Mode 5 requires motion file key 'joint_pos' or 'dof_pos'.")
    joint_pos = np.asarray(m[joint_key], dtype=np.float32)
    if joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must have shape [T, J], got {joint_pos.shape}.")
    if joint_pos.shape[0] != len(packets):
        raise ValueError(
            f"joint_pos frame count {joint_pos.shape[0]} does not match body frames {len(packets)}."
        )

    joint_names = joint_names_from_motion(m, joint_pos.shape[1])
    joint_indices = resolve_motion_joint_indices(joint_names)
    joint_pos = joint_pos[:, joint_indices]

    packets_with_joint = []
    num_frames = joint_pos.shape[0]
    for frame, packet in enumerate(packets):
        future_frames = [int((frame + step) % num_frames) for step in JOINT_TARGET_FUTURE_STEPS]
        joint_target = joint_pos[future_frames].reshape(-1)
        packets_with_joint.append(tuple(packet) + tuple(joint_target.tolist()))

    return packets_with_joint, fps, indices, body_names, DEFAULT_G1_JOINT_NAMES_29

class ILRecorder:
    """
    Records:
      - commands: seq + 28 floats (root/head/left/right pose) per step
      - rgb video frames (mp4)
    Writes per-episode folder with commands.npz + video_rgb.mp4 + manifest.json
    """
    def __init__(self, root_dir="dataset/episodes", hz=30.0):
        self.hz = float(hz)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.ep_dir = pathlib.Path(root_dir) / f"ep_{ts}"
        self.ep_dir.mkdir(parents=True, exist_ok=True)

        self._t = []
        self._seq = []
        self._cmd = []

        self._cap = None
        self._writer = None
        self._rgb_path = str(self.ep_dir / "video_rgb.mp4")
        self._w = None
        self._h = None

    def start_camera(self, cam_index_or_path=4, width=None, height=None, fps=None):
        import cv2
        fps = float(fps if fps is not None else self.hz)

        # prefer V4L2 on linux
        self._cap = cv2.VideoCapture(cam_index_or_path, cv2.CAP_V4L2)

        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self._cap.set(cv2.CAP_PROP_FPS, float(fps))

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {cam_index_or_path}")

        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or (width or 640))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or (height or 480))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._rgb_path, fourcc, fps, (self._w, self._h))
        if not self._writer.isOpened():
            raise RuntimeError("Failed to open video writer (mp4v).")

        return self._w, self._h, fps

    def step(self, t_sec: float, seq: int, cmd28):
        # cmd28: iterable of 28 floats
        self._t.append(float(t_sec))
        self._seq.append(int(seq))
        self._cmd.append(np.asarray(cmd28, dtype=np.float32).reshape(28))

    def write_frame(self):
        if self._cap is None or self._writer is None:
            return False
        ret, frame = self._cap.read()
        if ret:
            self._writer.write(frame)
        return bool(ret)

    def close(self, conventions=None):
        if self._cap is not None:
            self._cap.release()
        if self._writer is not None:
            self._writer.release()

        t = np.asarray(self._t, dtype=np.float64)
        seq = np.asarray(self._seq, dtype=np.int64)
        cmd = np.stack(self._cmd, axis=0) if len(self._cmd) > 0 else np.zeros((0, 28), np.float32)

        np.savez_compressed(str(self.ep_dir / "commands.npz"), t=t, seq=seq, cmd=cmd)

        manifest = {
            "episode_dir": str(self.ep_dir),
            "steps": int(cmd.shape[0]),
            "hz": self.hz,
            "commands": {
                "file": "commands.npz",
                "fields": {
                    "t": "float64 seconds",
                    "seq": "int64",
                    "cmd": "[T,28] float32 = root(7)+head(7)+left(7)+right(7)"
                },
                "frame": "WORLD",
                "quat_order": "xyzw",
            },
            "rgb": {
                "file": "video_rgb.mp4",
                "size": [int(self._h or 0), int(self._w or 0)],
                "fps": self.hz,
                "codec": "mp4v",
            },
            "conventions": conventions or {},
        }
        with open(self.ep_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest



class KeyboardController:
    """Non-blocking keyboard input handler for terminal."""
    
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
    
    def __enter__(self):
        tty.setraw(self.fd)
        return self
    
    def __exit__(self, *args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self):
        """Return key pressed or None if no key pressed. Non-blocking."""
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            # Handle arrow keys (escape sequences)
            if ch == '\x1b':
                # Wait a bit for the rest of the escape sequence
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            ch3 = sys.stdin.read(1)
                            if ch3 == 'A': return 'UP'
                            elif ch3 == 'B': return 'DOWN'
                            elif ch3 == 'C': return 'RIGHT'
                            elif ch3 == 'D': return 'LEFT'
                        return None  # incomplete sequence, ignore
                    return None  # incomplete sequence, ignore
                return 'ESC'  # only ESC pressed alone (no following chars after timeout)
            return ch
        return None


def vr_to_target_pos(vr_pos):
    """
    Convert VR coordinate system to target coordinate system.
    Based on observed mapping:
      VR X+ -> Target Y-
      VR Y+ -> Target Z+
      VR Z+ -> Target X-
    """
    vr_x, vr_y, vr_z = vr_pos
    target_x = -vr_z
    target_y = -vr_x
    target_z = vr_y
    return np.array([target_x, target_y, target_z], dtype=float)


def vr_to_target_quat(vr_quat):
    """
    Convert VR quaternion to target coordinate system.
    Rotation axes should match 1:1 (VR X->Robot X, VR Y->Robot Y, VR Z->Robot Z).
    Apply 180° rotations around Z and Y to match robot EE initial orientation,
    with axis correction and direction inversion.
    """
    qx, qy, qz, qw = vr_quat
    
    # Axis mapping with X and Z direction inverted (flip from previous)
    target_qx = -qy  # VR Y -> Target X (negated)
    target_qy = -qx  # VR X -> Target Y (negated)
    target_qz = qz   # VR Z -> Target Z
    target_qw = qw
    
    # Quaternion multiplication: q1 * q2
    # q1 = (x1, y1, z1, w1), q2 = (x2, y2, z2, w2)
    # w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    # x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    # y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    # z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    #
    # First: rotate 180° around Z axis: q_z180 = (0, 0, 1, 0)
    z180_x, z180_y, z180_z, z180_w = 0.0, 0.0, 1.0, 0.0
    
    temp_w = z180_w * target_qw - z180_x * target_qx - z180_y * target_qy - z180_z * target_qz
    temp_x = z180_w * target_qx + z180_x * target_qw + z180_y * target_qz - z180_z * target_qy
    temp_y = z180_w * target_qy - z180_x * target_qz + z180_y * target_qw + z180_z * target_qx
    temp_z = z180_w * target_qz + z180_x * target_qy - z180_y * target_qx + z180_z * target_qw
    
    # Second: rotate 180° around Y axis: q_y180 = (0, 1, 0, 0)
    y180_x, y180_y, y180_z, y180_w = 0.0, 1.0, 0.0, 0.0
    
    result_w = y180_w * temp_w - y180_x * temp_x - y180_y * temp_y - y180_z * temp_z
    result_x = y180_w * temp_x + y180_x * temp_w + y180_y * temp_z - y180_z * temp_y
    result_y = y180_w * temp_y - y180_x * temp_z + y180_y * temp_w + y180_z * temp_x
    result_z = y180_w * temp_z + y180_x * temp_y - y180_y * temp_x + y180_z * temp_w
    
    return np.array([result_x, result_y, result_z, result_w], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst_ip", type=str, default="127.0.0.1")
    ap.add_argument("--dst_port", type=int, default=15000)
    ap.add_argument("--hz", type=float, default=None)
    ap.add_argument("--mode", type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                    help="0: keyboard-controlled poses (default); 1: VR poses (requires a local `vr` module, not included); 2: random pose; 3: root follower; 4: replay 5kp; 5: replay 5kp + joint target")
    ap.add_argument("--broadcast-port", type=int, default=15001,
                    help="UDP port to listen for root state broadcasts (mode 3)")
    ap.add_argument("--motion-file", type=str, default=None,
                    help="Motion npz to replay in mode 4 or mode 5 (required for replay modes)")
    ap.add_argument("--motion-body-indices", type=str, default=None,
                    help="Optional mode-4/5 body indices: head,left_hand,right_hand,left_foot,right_foot")
    ap.add_argument("--motion-root-mode", choices=["velocity", "zero", "absolute"], default="velocity",
                    help="Mode 4/5 root command: velocity uses frame-to-frame root velocity, zero disables root xy, absolute sends root position")
    ap.add_argument("-r", "--record", action="store_true",
                    help="Record camera video in mode 0/1 (uses camera index 4)")
    args = ap.parse_args()
    if args.hz is None:
        args.hz = 0.0 if args.mode in (4, 5) else 30.0
    print_every = max(1, int(args.hz))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Fixed body offsets (used for mode=0 static sending, and mode=1 calibration)
    head_off_b = (0.0, 0.0, 0.77)
    lhand_off_b = (0.25,  0.18, 0.15)
    rhand_off_b = (0.25, -0.18, 0.15)

    # ==================== MODE 0: Interactive keyboard control ====================
    if args.mode == 0:
        print(f"[teleop_dummy_udp_sender] MODE=0 (keyboard interactive)")
        print(f"[teleop_dummy_udp_sender] sending to {args.dst_ip}:{args.dst_port} at {args.hz} Hz")
        print("[packet] bodies = root, head, left, right; each = pos(xyz) + quat(xyzw) in WORLD frame")
        send_logger = SignalSendLogger(mode=0)
        print(f"[log] signal send log -> {send_logger.path}")
        print()
        print("=== Keyboard Controls ===")
        print("  Hands:  I/K = X +/-,  J/L = Y (apart/together),  U/O = Z +/-")
        print("  Root:   W/S = X +/-,  A/D = Y +/-,  F/H = Z +/-,  Q/E = rotate left/right (10°)")
        print("  ESC or Ctrl+C: quit")
        print("=========================")
        print()

        seq = 0
        period = 1.0 / args.hz
        next_t = time.time()
        last_root_yaw = 0.0
        step = 0.01  # 1cm step size
        loco_step = 0.2
        rot_step = math.radians(10)  # 10 degrees in radians
        root_yaw = 0.0  # current root yaw angle

        rec = None
        record_enabled = False
        if args.record:
            try:
                rec = ILRecorder(root_dir="dataset/episodes", hz=args.hz)

                # RealSense color stream: /dev/video4
                # cv2.VideoCapture(4) also works; keep the stream index at 4
                w, h, fps = rec.start_camera(cam_index_or_path=4, width=1280, height=720, fps=args.hz)

                record_enabled = True
                print(f"[record] episode dir: {rec.ep_dir}")
                print(f"[record] rgb -> {rec._rgb_path} ({w}x{h}@{fps}fps)")
                print(f"[record] cmd -> {rec.ep_dir/'commands.npz'}")
            except Exception as e:
                print(f"[record] Recording unavailable: {e}")
                rec = None
                record_enabled = False

        # Fixed poses: root at origin, head/left/right at configured offsets
        # Quaternion (1, 0, 0, 0) = 180° rotation around X axis, flipping Y and Z
        root_x, root_y, root_z = (0.0, 0.0, 0.79)
        rqx, rqy, rqz, rqw = (1.0, 0.0, 0.0, 0.0)

        hx, hy, hz_pos = head_off_b
        hqx, hqy, hqz, hqw = (1.0, 0.0, 0.0, 0.0)

        lx, ly, lz = list(lhand_off_b)
        lqx, lqy, lqz, lqw = (1.0, 0.0, 0.0, 0.0)

        rx, ry, rz = list(rhand_off_b)
        rrqx, rrqy, rrqz, rrqw = (1.0, 0.0, 0.0, 0.0)

        with KeyboardController() as kb:
            try:
                while True:
                    now = time.time()
                    if now < next_t:
                        time.sleep(max(0.0, min(next_t - now, 0.01)))

                    # Check for keyboard input
                    key = kb.get_key()
                    if key:
                        if key == 'ESC' or key == '\x03':  # ESC or Ctrl+C
                            print("\r\n[quit] Exiting...\r\n")
                            break
                        # Hands control
                        elif key in ('i', 'I'):
                            lx += step
                            rx += step
                        elif key in ('k', 'K'):
                            lx -= step
                            rx -= step
                        elif key in ('j', 'J'):
                            ly += step   # left hand Y+
                            ry -= step   # right hand Y- (opposite direction)
                        elif key in ('l', 'L'):
                            ly -= step   # left hand Y-
                            ry += step   # right hand Y+ (opposite direction)
                        elif key in ('u', 'U'):
                            lz += step
                            rz += step
                        elif key in ('o', 'O'):
                            lz -= step
                            rz -= step
                        # Root control
                        elif key in ('w', 'W'):
                            root_x -= loco_step
                        elif key in ('s', 'S'):
                            root_x += loco_step
                        elif key in ('a', 'A'):
                            root_y -= loco_step
                        elif key in ('d', 'D'):
                            root_y += loco_step
                        elif key in ('f', 'F'):
                            root_z += step
                        elif key in ('h', 'H'):
                            root_z -= step
                        # Root rotation (yaw)
                        elif key in ('q', 'Q'):
                            root_yaw += rot_step  # counter-clockwise (left)
                        elif key in ('e', 'E'):
                            root_yaw -= rot_step  # clockwise (right)

                    # Update root quaternion from yaw (convert via target mapping)
                    rqx, rqy, rqz, rqw = euler_to_target_quat(0.0, 0.0, root_yaw)

                    if now >= next_t:
                        next_t += period

                        # Build the command to send (28 floats)
                        root = (root_x, root_y, root_z, rqx, rqy, rqz, rqw)
                        head = (hx, hy, hz_pos, hqx, hqy, hqz, hqw)
                        left = (lx, ly, lz, lqx, lqy, lqz, lqw)
                        right = (rx, ry, rz, rrqx, rrqy, rrqz, rrqw)
                        floats = root + head + left + right  # 28 floats

                        # Recording: save the command and RGB frame at the same timestamp
                        if record_enabled and rec is not None:
                            rec.step(t_sec=now, seq=seq, cmd28=floats)
                            rec.write_frame()   # Skip the frame on failure, but keep the command record

                        # Send packet
                        send_seq = seq
                        pkt = make_packet(send_seq, floats)
                        sock.sendto(pkt, (args.dst_ip, args.dst_port))
                        send_logger.log(t_sec=now, seq=send_seq, mode=0, values=floats)
                        seq = (send_seq + 1) & 0xFFFFFFFF


                        # low-rate print (every second)
                        if seq % print_every == 0:
                            # Clear line and print status
                            yaw_deg = math.degrees(root_yaw)
                            print(f"\r[seq={seq:6d}] root=({root_x:+.3f},{root_y:+.3f},{root_z:+.3f},yaw={yaw_deg:+.1f}°) left=({lx:+.3f},{ly:+.3f},{lz:+.3f}) right=({rx:+.3f},{ry:+.3f},{rz:+.3f})   ", end='', flush=True)
            finally:
                if record_enabled and rec is not None:
                    conventions = {
                        "packet": "root/head/left/right each: pos(xyz)+quat(xyzw) in WORLD",
                        "quat_order": "xyzw",
                        "axes": "world frame (as sent)",
                        "hz": args.hz,
                    }
                    manifest = rec.close(conventions=conventions)
                    print(f"\n[record] saved episode: {manifest['episode_dir']}")
                send_logger.close()

        return  # mode=0 exits when user presses ESC

    # ==================== MODE 1: Receive VR poses ====================
    # initialize VR subsystem (required). Do NOT fall back to static poses.
    if args.mode == 1:
        try:
            import vr
        except ImportError:
            print("[ERROR] Mode 1 requires a local `vr` module (OpenVR interface), which is not "
                  "included in this repository. Use --mode 0 (keyboard) or another mode instead.")
            raise SystemExit(2)
        try:
            vr_ctx = vr.init_vr()
        except Exception as e:
            print(f"[ERROR] Failed to initialize VR subsystem: {e}")
            print("This sender requires a working OpenVR environment. Exiting.")
            raise SystemExit(2)
        send_logger = SignalSendLogger(mode=1)
        print(f"[log] signal send log -> {send_logger.path}")

        seq = 0
        t0 = time.time()
        period = 1.0 / args.hz
        last_root_yaw = 0.0

        calibrated = False
        offsets = {
            'head': None,
            'left': None,
            'right': None,
        }
        quat_offsets = {
            'left': None,
            'right': None,
        }

        print(f"[teleop_dummy_udp_sender] MODE=1 (VR poses)")
        print(f"[teleop_dummy_udp_sender] sending to {args.dst_ip}:{args.dst_port} at {args.hz} Hz")
        print("[packet] bodies = root, head, left, right; each = pos(xyz) + quat(xyzw) in WORLD frame")
        print("[toggle] Sending the default pose initially; press the VR right-controller B button to toggle between default pose and VR control")

        # Initialize recording (mode 1 + -r: segment recording with the B button)
        rec = None
        if args.record:
            print("[record] Mode 1 recording enabled: the B button controls VR-linked recording (VR on=start, VR off=save and stop)")

        def start_new_episode_recorder():
            new_rec = ILRecorder(root_dir="dataset/episodes", hz=args.hz)
            w, h, fps = new_rec.start_camera(cam_index_or_path=4, width=1280, height=720, fps=args.hz)
            print(f"[record] started episode: {new_rec.ep_dir}")
            print(f"[record] rgb -> {new_rec._rgb_path} ({w}x{h}@{fps}fps)")
            print(f"[record] cmd -> {new_rec.ep_dir/'commands.npz'}")
            return new_rec

        def finalize_episode_recorder(active_rec):
            conventions = {
                "packet": "root/head/left/right each: pos(xyz)+quat(xyzw) in WORLD",
                "quat_order": "xyzw",
                "axes": "world frame (as sent)",
                "hz": args.hz,
            }
            manifest = active_rec.close(conventions=conventions)
            print(f"[record] saved episode: {manifest['episode_dir']}")
            return manifest

        # Support ESC to exit
        with KeyboardController() as kb:
            try:
                next_t = time.time()
                # mode=False: default pose, mode=True: VR teleop
                use_vr_mode = False
                prev_b_pressed = False
                while True:
                    now = time.time()
                    if now < next_t:
                        time.sleep(max(0.0, next_t - now))
                    next_t += period

                    key = kb.get_key()
                    if key:
                        if key == 'ESC' or key == '\x03':
                            print("\r\n[quit] Exiting...\r\n")
                            return

                    head, controllers = vr.poll_vr(vr_ctx)
                    right_c = controllers.get('R') if controllers is not None else None
                    right_inputs = right_c.get('inputs', {}) if right_c is not None else {}
                    b_pressed = bool(right_inputs.get('b_pressed', right_inputs.get('menu_pressed', False)))

                    if b_pressed and not prev_b_pressed:
                        use_vr_mode = not use_vr_mode
                        mode_name = "VR control" if use_vr_mode else "default pose"
                        print(f"[toggle] B button detected; switched to: {mode_name}")
                        # Recalibrate every time we enter VR mode, same as first entry.
                        if use_vr_mode:
                            calibrated = False
                            offsets['head'] = None
                            offsets['left'] = None
                            offsets['right'] = None
                            quat_offsets['left'] = None
                            quat_offsets['right'] = None
                            if args.record:
                                if rec is None:
                                    try:
                                        rec = start_new_episode_recorder()
                                    except Exception as e:
                                        print(f"[record] start failed: {e}")
                                        rec = None
                        else:
                            if rec is not None:
                                try:
                                    finalize_episode_recorder(rec)
                                except Exception as e:
                                    print(f"[record] save failed: {e}")
                                rec = None
                    prev_b_pressed = b_pressed

                    if (not use_vr_mode) or (head is None):
                        rqx, rqy, rqz, rqw = euler_to_target_quat(0.0, 0.0, 0.0)
                        root = (0.0, 0.0, 0.79, rqx, rqy, rqz, rqw)
                        head_tuple = head_off_b + (1.0, 0.0, 0.0, 0.0)
                        left = tuple(lhand_off_b) + (1.0, 0.0, 0.0, 0.0)
                        right = tuple(rhand_off_b) + (1.0, 0.0, 0.0, 0.0)
                        floats = root + head_tuple + left + right
                    else:
                        if not calibrated:
                            meas_head = vr_to_target_pos(head['pos'])
                            desired_head = np.array(head_off_b, dtype=float)
                            offsets['head'] = desired_head - meas_head

                            left_c0 = controllers.get('L')
                            if left_c0 is not None:
                                meas_left = vr_to_target_pos(left_c0['pos'])
                                desired_left = np.array(lhand_off_b, dtype=float)
                                offsets['left'] = desired_left - meas_left
                                # Orientation zeroing: first measured orientation -> default quat.
                                desired_left_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                                meas_left_quat = quat_normalize_xyzw(vr_to_target_quat(left_c0['quat']))
                                quat_offsets['left'] = quat_mul_xyzw(desired_left_quat, quat_inv_xyzw(meas_left_quat))
                            else:
                                offsets['left'] = None
                                quat_offsets['left'] = None

                            right_c0 = controllers.get('R')
                            if right_c0 is not None:
                                meas_right = vr_to_target_pos(right_c0['pos'])
                                desired_right = np.array(rhand_off_b, dtype=float)
                                offsets['right'] = desired_right - meas_right
                                desired_right_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                                meas_right_quat = quat_normalize_xyzw(vr_to_target_quat(right_c0['quat']))
                                quat_offsets['right'] = quat_mul_xyzw(desired_right_quat, quat_inv_xyzw(meas_right_quat))
                            else:
                                offsets['right'] = None
                                quat_offsets['right'] = None

                            calibrated = True
                            print(f"[CALIB] Computed offsets: head={offsets['head']}, left={offsets['left']}, right={offsets['right']}")

                        lx_axis = 0.0
                        ly_axis = 0.0
                        rx_axis = 0.0
                        ry_axis = 0.0
                        if controllers['L'] is not None:
                            lx_axis, ly_axis = controllers['L']['inputs'].get('axis0', (0.0, 0.0))
                        if controllers['R'] is not None:
                            rx_axis, ry_axis = controllers['R']['inputs'].get('axis0', (0.0, 0.0))

                        mag = math.hypot(lx_axis, ly_axis)
                        if mag > 1e-3:
                            angle = math.atan2(ly_axis, lx_axis) + math.pi / 2
                            root_x = math.cos(angle) * 0.5
                            root_y = math.sin(angle) * 0.5
                        else:
                            root_x = 0.0
                            root_y = 0.0
                        root_z = 0.79

                        mag_r = math.hypot(rx_axis, ry_axis)
                        if mag_r > 1e-3:
                            root_yaw = math.atan2(ry_axis, rx_axis) + math.pi / 2
                            last_root_yaw = root_yaw
                        else:
                            root_yaw = last_root_yaw
                        rqx, rqy, rqz, rqw = euler_to_target_quat(0.0, 0.0, root_yaw)

                        hx, hy, hz = (root_x, root_y, root_z)
                        hqx, hqy, hqz, hqw = (rqx, rqy, rqz, rqw)

                        left_c = controllers.get('L')
                        if left_c is not None:
                            meas_left = vr_to_target_pos(left_c['pos'])
                            if offsets['left'] is None:
                                offsets['left'] = np.array(lhand_off_b, dtype=float) - meas_left
                                print(f"[CALIB] Late-computed left offset: {offsets['left']}")
                            if quat_offsets['left'] is None:
                                desired_left_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                                meas_left_quat = quat_normalize_xyzw(vr_to_target_quat(left_c['quat']))
                                quat_offsets['left'] = quat_mul_xyzw(desired_left_quat, quat_inv_xyzw(meas_left_quat))
                                print(f"[CALIB] Late-computed left quat offset: {np.round(quat_offsets['left'], 4)}")
                            left_pos = meas_left + (offsets['left'] if offsets['left'] is not None else np.zeros(3))
                            lx, ly, lz = map(float, left_pos)
                            left_quat_raw = quat_normalize_xyzw(vr_to_target_quat(left_c['quat']))
                            if quat_offsets['left'] is not None:
                                left_quat = quat_mul_xyzw(quat_offsets['left'], left_quat_raw)
                            else:
                                left_quat = left_quat_raw
                            lqx, lqy, lqz, lqw = map(float, left_quat)
                        else:
                            lx, ly, lz = (0.0, 0.0, 0.0)
                            lqx, lqy, lqz, lqw = (1.0, 0.0, 0.0, 0.0)

                        right_c = controllers.get('R')
                        if right_c is not None:
                            meas_right = vr_to_target_pos(right_c['pos'])
                            if offsets['right'] is None:
                                offsets['right'] = np.array(rhand_off_b, dtype=float) - meas_right
                                print(f"[CALIB] Late-computed right offset: {offsets['right']}")
                            if quat_offsets['right'] is None:
                                desired_right_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                                meas_right_quat = quat_normalize_xyzw(vr_to_target_quat(right_c['quat']))
                                quat_offsets['right'] = quat_mul_xyzw(desired_right_quat, quat_inv_xyzw(meas_right_quat))
                                print(f"[CALIB] Late-computed right quat offset: {np.round(quat_offsets['right'], 4)}")
                            right_pos = meas_right + (offsets['right'] if offsets['right'] is not None else np.zeros(3))
                            rx, ry, rz = map(float, right_pos)
                            right_quat_raw = quat_normalize_xyzw(vr_to_target_quat(right_c['quat']))
                            if quat_offsets['right'] is not None:
                                right_quat = quat_mul_xyzw(quat_offsets['right'], right_quat_raw)
                            else:
                                right_quat = right_quat_raw
                            rrqx, rrqy, rrqz, rrqw = map(float, right_quat)
                        else:
                            rx, ry, rz = (0.0, 0.0, 0.0)
                            rrqx, rrqy, rrqz, rrqw = (1.0, 0.0, 0.0, 0.0)

                        # disable loco
                        rqx, rqy, rqz, rqw = euler_to_target_quat(0.0, 0.0, 0.0)
                        root_x = 0.0
                        root_y = 0.0
                        root_z = 0.79

                        root = (root_x, root_y, root_z, rqx, rqy, rqz, rqw)
                        head_tuple = (hx, hy, hz, hqx, hqy, hqz, hqw)
                        left = (lx, ly, lz, lqx, lqy, lqz, lqw)
                        right = (rx, ry, rz, rrqx, rrqy, rrqz, rrqw)
                        floats = root + head_tuple + left + right

                    if rec is not None:
                        rec.step(t_sec=now, seq=seq, cmd28=floats)
                        rec.write_frame()

                    send_seq = seq
                    pkt = make_packet(send_seq, floats)
                    sock.sendto(pkt, (args.dst_ip, args.dst_port))
                    send_logger.log(t_sec=now, seq=send_seq, mode=1, values=floats)
                    seq = (send_seq + 1) & 0xFFFFFFFF

                    if seq % print_every == 0:
                        mode_name = "VR" if use_vr_mode else "DEFAULT"
                        print(f"[send] seq={seq} mode={mode_name}")
            finally:
                if rec is not None:
                    try:
                        finalize_episode_recorder(rec)
                    except Exception as e:
                        print(f"\n[record] final save failed: {e}")
                send_logger.close()
        return

    # ==================== MODE 2: Random pose every 5s ====================
    if args.mode == 2:
        print(f"[teleop_dummy_udp_sender] MODE=2 (random pose every 5s)")
        print(f"[teleop_dummy_udp_sender] sending to {args.dst_ip}:{args.dst_port} at {args.hz} Hz")
        print("[packet] bodies = root, head, left, right; each = pos(xyz) + quat(xyzw) in WORLD frame")
        print("Left/right position range: x,y,z=[-0.1,0.1] relative to default")

        seq = 0
        period = 1.0 / args.hz
        next_t = time.time()
        last_update = time.time() - 5.0
        # Fixed root and head
        root_x, root_y, root_z = (0.0, 0.0, 0.79)
        rqx, rqy, rqz, rqw = (1.0, 0.0, 0.0, 0.0)
        hx, hy, hz_pos = (0.0, 0.0, 0.79)
        hqx, hqy, hqz, hqw = (1.0, 0.0, 0.0, 0.0)
        # Initial left/right positions (random within range) but FIXED orientations
        # for verification: roll,pitch,yaw = 0.0 and quaternion = (0,0,0,1)
        lx = 0.25 + np.random.uniform(-0.1, 0.1)
        ly = 0.18 + np.random.uniform(-0.1, 0.1)
        lz = 0.15 + np.random.uniform(-0.1, 0.1)
        # small random orientation: roll,pitch,yaw within +/-20 degrees
        l_roll = np.random.uniform(-math.radians(20), math.radians(20))
        l_pitch = np.random.uniform(-math.radians(20), math.radians(20))
        l_yaw = np.random.uniform(-math.radians(20), math.radians(20))
        # convert to target-mapped quaternion so mode2 uses same axis mapping
        lqx, lqy, lqz, lqw = euler_to_target_quat(l_roll, l_pitch, l_yaw)

        rx = 0.25 + np.random.uniform(-0.1, 0.1)
        ry = -0.18 + np.random.uniform(-0.1, 0.1)
        rz = 0.15 + np.random.uniform(-0.1, 0.1)
        r_roll = np.random.uniform(-math.radians(20), math.radians(20))
        r_pitch = np.random.uniform(-math.radians(20), math.radians(20))
        r_yaw = np.random.uniform(-math.radians(20), math.radians(20))
        rrqx, rrqy, rrqz, rrqw = euler_to_target_quat(r_roll, r_pitch, r_yaw)

        while True:
            now = time.time()
            if now < next_t:
                time.sleep(max(0.0, min(next_t - now, 0.01)))
            if now - last_update >= 5.0:
                lx = 0.25 + np.random.uniform(-0.1, 0.1)
                ly = 0.18 + np.random.uniform(-0.1, 0.1)
                lz = 0.15 + np.random.uniform(-0.1, 0.1)
                # randomize orientation within +/-20 degrees (mapped to target frame)
                l_roll = np.random.uniform(-math.radians(20), math.radians(20))
                l_pitch = np.random.uniform(-math.radians(20), math.radians(20))
                l_yaw = np.random.uniform(-math.radians(20), math.radians(20))
                lqx, lqy, lqz, lqw = euler_to_target_quat(l_roll, l_pitch, l_yaw)

                rx = 0.25 + np.random.uniform(-0.1, 0.1)
                ry = -0.18 + np.random.uniform(-0.1, 0.1)
                rz = 0.15 + np.random.uniform(-0.1, 0.1)
                r_roll = np.random.uniform(-math.radians(20), math.radians(20))
                r_pitch = np.random.uniform(-math.radians(20), math.radians(20))
                r_yaw = np.random.uniform(-math.radians(20), math.radians(20))
                rrqx, rrqy, rrqz, rrqw = euler_to_target_quat(r_roll, r_pitch, r_yaw)
                last_update = now
            next_t += period

            root = (root_x, root_y, root_z, rqx, rqy, rqz, rqw)
            head = (hx, hy, hz_pos, hqx, hqy, hqz, hqw)
            left = (lx, ly, lz, lqx, lqy, lqz, lqw)
            right = (rx, ry, rz, rrqx, rrqy, rrqz, rrqw)
            floats = root + head + left + right  # 28 floats

            pkt = make_packet(seq, floats)
            sock.sendto(pkt, (args.dst_ip, args.dst_port))
            seq = (seq + 1) & 0xFFFFFFFF

            # low-rate print (every second)
            if seq % print_every == 0:
                yaw_deg = math.degrees(0.0)  # root yaw is fixed
                print(f"\r[seq={seq:6d}] root=({root_x:+.3f},{root_y:+.3f},{root_z:+.3f},yaw={yaw_deg:+.1f}°) left=({lx:+.3f},{ly:+.3f},{lz:+.3f}) right=({rx:+.3f},{ry:+.3f},{rz:+.3f})   ", end='', flush=True)

    # ==================== MODE 4: Replay motion npz as 5-keypoint UDP stream ====
    if args.mode == 4:
        motion_path = pathlib.Path(args.motion_file).expanduser()
        packets, motion_fps, indices, body_names = load_motion_packets(
            motion_path,
            body_index_override=args.motion_body_indices,
            root_mode=args.motion_root_mode,
        )
        send_hz = args.hz if args.hz > 0 else motion_fps
        period = 1.0 / send_hz
        print(f"[teleop_dummy_udp_sender] MODE=4 (motion replay)")
        print(f"[motion] file: {motion_path}")
        print(f"[motion] frames={len(packets)}, motion_fps={motion_fps:.3f}, send_hz={send_hz:.3f}")
        print(f"[motion] root_mode={args.motion_root_mode}")
        print(f"[motion] body indices: {indices}")
        print("[packet] root/head/left/right 28-float prefix + left/right feet xyz_b extension")

        send_logger = SignalSendLogger(mode=4, value_count=len(packets[0]))
        print(f"[log] signal send log -> {send_logger.path}")

        seq = 0
        frame = 0
        next_t = time.time()
        try:
            while True:
                now = time.time()
                if now < next_t:
                    time.sleep(max(0.0, min(next_t - now, 0.01)))
                    continue
                next_t += period

                floats = packets[frame]
                pkt = make_packet(seq, floats)
                sock.sendto(pkt, (args.dst_ip, args.dst_port))
                send_logger.log(t_sec=now, seq=seq, mode=4, values=floats)

                if seq % print_every == 0:
                    left = floats[14:17]
                    right = floats[21:24]
                    lfoot = floats[28:31]
                    rfoot = floats[31:34]
                    print(
                        f"\r[seq={seq:6d} frame={frame:5d}/{len(packets)}] "
                        f"LH=({left[0]:+.3f},{left[1]:+.3f},{left[2]:+.3f}) "
                        f"RH=({right[0]:+.3f},{right[1]:+.3f},{right[2]:+.3f}) "
                        f"LF=({lfoot[0]:+.3f},{lfoot[1]:+.3f},{lfoot[2]:+.3f}) "
                        f"RF=({rfoot[0]:+.3f},{rfoot[1]:+.3f},{rfoot[2]:+.3f})   ",
                        end="",
                        flush=True,
                    )

                seq = (seq + 1) & 0xFFFFFFFF
                frame = (frame + 1) % len(packets)
        finally:
            send_logger.close()

        return

    # =========== MODE 5: Replay motion npz as 5-keypoint + joint-target stream ===
    if args.mode == 5:
        motion_path = pathlib.Path(args.motion_file).expanduser()
        packets, motion_fps, indices, body_names, joint_names = load_motion_packets_with_joint_targets(
            motion_path,
            body_index_override=args.motion_body_indices,
            root_mode=args.motion_root_mode,
        )
        send_hz = args.hz if args.hz > 0 else motion_fps
        period = 1.0 / send_hz
        print(f"[teleop_dummy_udp_sender] MODE=5 (motion replay + joint target)")
        print(f"[motion] file: {motion_path}")
        print(f"[motion] frames={len(packets)}, motion_fps={motion_fps:.3f}, send_hz={send_hz:.3f}")
        print(f"[motion] root_mode={args.motion_root_mode}")
        print(f"[motion] body indices: {indices}")
        print(f"[motion] joint target future steps: {JOINT_TARGET_FUTURE_STEPS}")
        print(f"[motion] joint order: {joint_names}")
        print("[packet] magic=G6J1, root/head/left/right 28-float prefix + feet xyz_b 6 + joint_target 145")

        send_logger = SignalSendLogger(mode=5, value_count=len(packets[0]))
        print(f"[log] signal send log -> {send_logger.path}")

        seq = 0
        frame = 0
        next_t = time.time()
        try:
            while True:
                now = time.time()
                if now < next_t:
                    time.sleep(max(0.0, min(next_t - now, 0.01)))
                    continue
                next_t += period

                floats = packets[frame]
                pkt = make_packet_with_magic(seq, floats, magic=JOINT_MAGIC)
                sock.sendto(pkt, (args.dst_ip, args.dst_port))
                send_logger.log(t_sec=now, seq=seq, mode=5, values=floats)

                if seq % print_every == 0:
                    left = floats[14:17]
                    right = floats[21:24]
                    lfoot = floats[28:31]
                    rfoot = floats[31:34]
                    joint0 = floats[34:63]
                    print(
                        f"\r[seq={seq:6d} frame={frame:5d}/{len(packets)}] "
                        f"LH=({left[0]:+.3f},{left[1]:+.3f},{left[2]:+.3f}) "
                        f"RH=({right[0]:+.3f},{right[1]:+.3f},{right[2]:+.3f}) "
                        f"LF=({lfoot[0]:+.3f},{lfoot[1]:+.3f},{lfoot[2]:+.3f}) "
                        f"RF=({rfoot[0]:+.3f},{rfoot[1]:+.3f},{rfoot[2]:+.3f}) "
                        f"J0=({min(joint0):+.3f},{max(joint0):+.3f})   ",
                        end="",
                        flush=True,
                    )

                seq = (seq + 1) & 0xFFFFFFFF
                frame = (frame + 1) % len(packets)
        finally:
            send_logger.close()

        return

    # ==================== MODE 3: Listen for simulator root-state UDP broadcast ====
    if args.mode == 3:
        b_port = args.broadcast_port
        print(f"[teleop_dummy_udp_sender] MODE=3 (listening for root-state UDP on 0.0.0.0:{b_port})")
        print("Expecting CSV lines: tstamp,x,y,z,qw,qx,qy,qz")
        try:
            b_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            b_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            b_sock.bind(("0.0.0.0", b_port))
            b_sock.settimeout(1.0)
        except Exception as e:
            print(f"[ERROR] Failed to bind UDP socket on port {b_port}: {e}")
            raise SystemExit(2)

        seq = 0
        period = 1.0 / args.hz
        # switch to non-blocking receive and keep the last seen broadcast
        b_sock.setblocking(False)
        last_msg = None
        next_send = time.time()
        try:
            while True:
                # drain all available incoming datagrams, keep the last one
                try:
                    while True:
                        data, addr = b_sock.recvfrom(4096)
                        last_msg = (data, addr)
                except BlockingIOError:
                    pass
                except Exception as e:
                    print(f"[ERROR] socket recv failed: {e}")
                    break

                now = time.time()
                if now >= next_send:
                    next_send += period
                    if last_msg is None:
                        # no data yet, nothing to do this tick
                        continue

                    data, addr = last_msg
                    try:
                        line = data.decode('ascii', errors='ignore').strip()
                        if not line:
                            continue
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) < 8:
                            print(f"[WARN] malformed line from {addr}: {line}")
                            continue
                        tstamp = float(parts[0])
                        sim_x = float(parts[1]); sim_y = float(parts[2]); sim_z = float(parts[3])
                        qw = float(parts[4]); qx = float(parts[5]); qy = float(parts[6]); qz = float(parts[7])

                        # convert incoming (w,x,y,z) -> (x,y,z,w) for our quat helpers
                        in_quat = (qx, qy, qz, qw)
                        cur_roll, cur_pitch, cur_yaw = quat_to_rpy(in_quat)

                        # target and distance
                        target_x, target_y = -0.0, -0.0
            
                        dist = math.hypot(target_x - sim_x, target_y - sim_y)

                        if dist < 0.2:
                            print(f"[root_state] arrived at target (dist={dist:.3f}), stopping mode 3")
                            break

                        desired_yaw = - math.atan2(target_y - sim_y, target_x - sim_x)
                        # compute yaw command as the angle difference between desired vector and current root yaw
                        def _wrap_to_pi(a):
                            return (a + math.pi) % (2 * math.pi) - math.pi

                        yaw_cmd = _wrap_to_pi(desired_yaw - 0.0 * math.pi / 180.0)
                        
                        # create quaternion from the relative yaw command (roll/pitch=0)
                        rqx, rqy, rqz, rqw = euler_to_target_quat(0.0, 0.0, yaw_cmd)

                        k = 1.0
                        rx_cmd = max(0.0, min(1.5, k * dist))
                        rx_cmd = -rx_cmd  # flip sign to match robot's forward convention
                        # rx_cmd = 0
                        # send commanded root as (rx_cmd, 0, 0) with desired orientation
                        root = (rx_cmd, 0.0, 0.79, rqx, rqy, rqz, rqw)
                        hx, hy, hz = (rx_cmd, 0.0, 0.0)
                        hqx, hqy, hqz, hqw = (rqx, rqy, rqz, rqw)
                        lx, ly, lz = lhand_off_b
                        lqx, lqy, lqz, lqw = (0.0, 0.0, 0.0, 1.0)
                        rx_, ry_, rz_ = rhand_off_b
                        rrqx, rrqy, rrqz, rrqw = (0.0, 0.0, 0.0, 1.0)

                        head = (hx, hy, hz, hqx, hqy, hqz, hqw)
                        left = (lx, ly, lz, lqx, lqy, lqz, lqw)
                        right = (rx_, ry_, rz_, rrqx, rrqy, rrqz, rrqw)

                        floats = root + head + left + right
                        pkt = make_packet(seq, floats)
                        sock.sendto(pkt, (args.dst_ip, args.dst_port))
                        seq = (seq + 1) & 0xFFFFFFFF

                        tstr = time.strftime('%H:%M:%S', time.localtime(tstamp)) + f".{int((tstamp%1)*1000):03d}"
                        print(f"[root_state] {tstr} from {addr[0]}:{addr[1]} -> sim_pos=({sim_x:.3f},{sim_y:.3f},{sim_z:.3f}) dist={dist:.3f} desired_yaw={math.degrees(desired_yaw):.1f}° cur_yaw={math.degrees(cur_yaw):.1f}° yaw_cmd={math.degrees(yaw_cmd):.1f}° sent_seq={seq}")
                    except Exception as e:
                        print(f"[WARN] failed to parse/process message from {addr}: {e}; raw={data}")

                # small sleep to avoid busy loop
                time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                b_sock.close()
            except Exception:
                pass

        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
