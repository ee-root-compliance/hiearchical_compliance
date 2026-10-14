import torch
import numpy as np
import json
import functools
from pathlib import Path
from typing import Callable
from tensordict import TensorClass, MemoryMappedTensor
from scipy.spatial.transform import Rotation as sRot
import os
import warnings
from active_adaptation.utils.motion_utils import rotate_to_body, select_in_order, angvel_from_rot

class MotionData(TensorClass):
    motion_id: torch.Tensor
    step: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_pos_b: torch.Tensor
    body_quat_w: torch.Tensor

from tqdm import tqdm

G1_JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]

LIMMT_KPT_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_hand_mimic",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_hand_mimic",
]


def _is_limmt_keypoint_motion(m: dict) -> bool:
    return (
        "joint_names" not in m
        and {"qpos", "qvel", "kpt2gv_pose", "gv2wrd_pose"} <= set(m.keys())
    )


RETARGETED_REQUIRED_KEYS = [
    "joint_names", "body_names", "root_pos", "root_rot",
    "dof_pos", "local_body_pos", "local_body_rot",
]
FPS_KEYS = ["mocap_framerate", "frequency", "fps"]
FPS_TOLERANCE = 0.5

# Bodies used by the tracking/compliance tasks that the LIMMT keypoint format
# does not provide; training fills missing bodies with zeros.
EXPECTED_EXTRA_BODY_NAMES = [
    "head_mimic",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
]


def _check_retargeted_motion(m: dict, path) -> None:
    missing = [k for k in RETARGETED_REQUIRED_KEYS if k not in m]
    if not any(k in m for k in FPS_KEYS):
        missing.append("one of " + "/".join(FPS_KEYS))
    if not missing:
        return
    hint = ""
    if {"poses", "trans", "betas"} <= set(m.keys()):
        hint = (
            " This looks like a raw AMASS SMPL(-X) file; it must be retargeted "
            "to the Unitree G1 first."
        )
    raise ValueError(
        f"Unsupported motion file {path}: missing keys {missing} "
        f"(found keys {sorted(m.keys())}).{hint} The converter expects either "
        "G1-retargeted motion with keys joint_names, body_names, root_pos (T,3), "
        "root_rot (T,4, xyzw), dof_pos (T,J), local_body_pos (T,B,3), "
        "local_body_rot (T,B,4, xyzw) and an fps key "
        f"({'/'.join(FPS_KEYS)}), sampled at the target fps; "
        "or the LIMMT keypoint format with keys qpos, qvel, kpt2gv_pose, gv2wrd_pose."
    )


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def _adapt_limmt_keypoint_motion(m: dict, target_fps: int) -> dict:
    qpos = m["qpos"].astype(np.float32)
    if qpos.shape[-1] != 7 + len(G1_JOINT_NAMES):
        raise ValueError(
            f"LIMMT qpos must have {7 + len(G1_JOINT_NAMES)} dims, got {qpos.shape[-1]}."
        )

    kpt2gv_pose = m["kpt2gv_pose"].astype(np.float32)
    gv2wrd_pose = m["gv2wrd_pose"].astype(np.float32)
    if kpt2gv_pose.shape[1] != len(LIMMT_KPT_BODY_NAMES):
        raise ValueError(
            f"LIMMT kpt2gv_pose must have {len(LIMMT_KPT_BODY_NAMES)} bodies, "
            f"got {kpt2gv_pose.shape[1]}."
        )

    kpt2wrd_pose = np.matmul(gv2wrd_pose[:, None], kpt2gv_pose)
    body_pos_w = kpt2wrd_pose[..., :3, 3].astype(np.float32)
    body_rot_w_xyzw = sRot.from_matrix(
        kpt2wrd_pose[..., :3, :3].reshape(-1, 3, 3)
    ).as_quat().reshape(qpos.shape[0], len(LIMMT_KPT_BODY_NAMES), 4).astype(np.float32)

    root_pos = qpos[:, :3].astype(np.float32)
    root_rot_xyzw = _quat_wxyz_to_xyzw(qpos[:, 3:7]).astype(np.float32)
    root_rot = sRot.from_quat(root_rot_xyzw)
    root_rot_inv_mat = root_rot.inv().as_matrix().astype(np.float32)

    local_body_pos = np.einsum(
        "tij,tbj->tbi",
        root_rot_inv_mat,
        body_pos_w - root_pos[:, None],
    ).astype(np.float32)
    body_rot_w_mat = sRot.from_quat(
        body_rot_w_xyzw.reshape(-1, 4)
    ).as_matrix().reshape(qpos.shape[0], len(LIMMT_KPT_BODY_NAMES), 3, 3).astype(np.float32)
    local_body_rot_mat = np.einsum("tij,tbjk->tbik", root_rot_inv_mat, body_rot_w_mat)
    local_body_rot = sRot.from_matrix(
        local_body_rot_mat.reshape(-1, 3, 3)
    ).as_quat().reshape(qpos.shape[0], len(LIMMT_KPT_BODY_NAMES), 4).astype(np.float32)

    out = dict(m)
    out["fps"] = int(m.get("fps", m.get("frequency", m.get("mocap_framerate", target_fps))))
    out["root_pos"] = root_pos
    out["root_rot"] = root_rot_xyzw
    out["dof_pos"] = qpos[:, 7:].astype(np.float32)
    out["local_body_pos"] = local_body_pos
    out["local_body_rot"] = local_body_rot
    out["joint_names"] = np.array(G1_JOINT_NAMES, dtype=object)
    out["body_names"] = np.array(LIMMT_KPT_BODY_NAMES, dtype=object)
    return out


class MotionDataset:
    def __init__(self, body_names: list, joint_names: list, starts: list, ends: list, data: MotionData, info: list, device: torch.device = torch.device('cpu')):
        self.body_names = body_names
        self.joint_names = joint_names
        self.starts = torch.as_tensor(starts, dtype=torch.int32, device=device)
        self.ends = torch.as_tensor(ends, dtype=torch.int32, device=device)
        self.lengths = self.ends - self.starts
        self.data = data
        self.info = info

    @classmethod
    def create_from_path_lazy(cls, mem_path: str, dataset_extra_keys: list[dict] = [], device: torch.device = torch.device('cpu')):
        # get mempath root
        path_root = os.environ.get("MEMPATH")
        if path_root is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, "../..")
            path_root = os.path.join(project_root, "dataset")
        mem_path = os.path.join(path_root, mem_path)

        data = MotionData.load(mem_path)
        data = data.to(device)
        with open(mem_path + "/meta_motion.json", "r") as f:
            meta = json.load(f)
        
        infos = {}
        for k in dataset_extra_keys:
            k_name = k['name']
            k_shape = k['shape']
            if k_name not in meta['info'].keys():
                infos[k_name] = torch.zeros((len(meta['starts']), k_shape), dtype=k['dtype'], device=device)
            else:
                infos[k_name] = torch.tensor(meta['info'][k_name], dtype=k['dtype'], device=device)
            if infos[k_name].shape != (len(meta['starts']), k_shape):
                raise ValueError(f"Shape of {k_name} does not match: {infos[k_name].shape} != {len(meta['starts'])}, {k_shape}")

        return cls(body_names=meta['body_names'], joint_names=meta['joint_names'], starts=meta['starts'], ends=meta['ends'], data=data, info=infos, device=device)

    @classmethod
    def create_from_path(
        cls,
        root_path: str,
        target_fps: int = 50,
        mem_path: str | None = None,
        motion_processer: Callable | None = None,
        motion_filter: Callable | None = None,
        callback: Callable | None = None,
        pad_before: int = 0,
        pad_after: int = 0,
        segment_len: int = 1000,
        *,
        build_dataset: bool = True,
        storage_float_dtype: torch.dtype = torch.float16,
        storage_int_dtype: torch.dtype = torch.int32,
    ):
        root = Path(root_path)
        # Support single file or directory
        meta = None
        if root.is_file() and root.suffix == '.npz':
            paths = [root]
        else:
            paths = list(root.rglob('*.npz'))
        if not paths:
            raise RuntimeError(f"No motions found in {root_path}")

        motions = [] if build_dataset else None
        total = 0
        
        pb = tqdm(paths)
        # Read (input must already be at target_fps; no resampling is done)
        # Calculate velocities and world frame poses based on the following keys before segmentation
        preserved_keys = ['fps', 'qpos', 'qvel', 'xpos', 'xquat']
        SEGMENT_LEN = segment_len
        
        foot_names = ['left_ankle_roll_link', 'right_ankle_roll_link']
        foot_idx = None
        
        joint_names_keep = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        body_names_keep = ["world", "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link", "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link", "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link", "head_mimic", "left_hand_mimic", "right_hand_mimic"]
        for p in pb:
            m = dict(np.load(p, allow_pickle=True))
            if _is_limmt_keypoint_motion(m):
                m = _adapt_limmt_keypoint_motion(m, target_fps=target_fps)
            _check_retargeted_motion(m, p)

            # Select joints/bodies by name for every file
            joint_names = m["joint_names"].tolist()
            body_names = m["body_names"].tolist()

            sel_joint_names, sel_joint_idx = select_in_order(joint_names, joint_names_keep, return_missing=False)
            sel_body_names, sel_body_idx = select_in_order(body_names, body_names_keep, return_missing=False)

            # Initialize metadata from first motion
            if meta is None:
                meta = {"joint_names": sel_joint_names,
                        "body_names": sel_body_names}
                meta['info'] = []
                missing_bodies = [n for n in EXPECTED_EXTRA_BODY_NAMES if n not in sel_body_names]
                if missing_bodies:
                    warnings.warn(
                        f"\n{'!' * 80}\nMotion dataset is missing bodies {missing_bodies} "
                        f"(first file: {p}). Training silently fills missing bodies with zeros, "
                        "so rewards/observations referencing them will be wrong. "
                        "The LIMMT keypoint format only provides "
                        f"{len(LIMMT_KPT_BODY_NAMES)} bodies; use G1-retargeted motion "
                        f"with full body_names instead.\n{'!' * 80}",
                        stacklevel=2,
                    )
            elif sel_joint_names != meta['joint_names'] or sel_body_names != meta['body_names']:
                raise ValueError(
                    f"Motion file {p} provides a different joint/body set than the first file.\n"
                    f"  joints: {sel_joint_names}\n  expected: {meta['joint_names']}\n"
                    f"  bodies: {sel_body_names}\n  expected: {meta['body_names']}\n"
                    "Do not mix motion formats (e.g. LIMMT keypoint and G1-retargeted files) in one dataset."
                )
            if foot_idx is None:
                foot_idx = [meta['body_names'].index(n) for n in foot_names]
            
            # Extract standard field
            raw_fps = float(m.get('mocap_framerate', m.get('frequency', m.get('fps', 0))))
            if abs(raw_fps - target_fps) > FPS_TOLERANCE:
                raise ValueError(
                    f"Motion file {p} is sampled at {raw_fps:g} Hz, but input motion must already be "
                    f"at {target_fps} Hz (tolerance +/-{FPS_TOLERANCE} Hz). The converter does not "
                    "resample; resample the retargeted motion before conversion."
                )
            m['fps'] = int(m.get('mocap_framerate', m.get('frequency', m.get('fps', 0))))

            # Pad data with specified frames before and after
            def pad_data(data, pad_before, pad_after):
                slot = np.zeros((pad_before + pad_after + data.shape[0], *data.shape[1:]), dtype=data.dtype)
                slot[pad_before:pad_before + data.shape[0], :] = data
                if pad_before > 0:
                    slot[:pad_before] = slot[pad_before:pad_before+1]
                if pad_after > 0:
                    slot[-pad_after:] = slot[-pad_after-1:-pad_after]
                return slot
            
            m["root_pos"] = pad_data(m["root_pos"], pad_before, pad_after)
            m["root_rot"] = pad_data(m["root_rot"], pad_before, pad_after)
            m["dof_pos"] = pad_data(m["dof_pos"][:, sel_joint_idx], pad_before, pad_after)
            m["local_body_pos"] = pad_data(m["local_body_pos"][:, sel_body_idx, :], pad_before, pad_after)
            m["local_body_rot"] = pad_data(m["local_body_rot"][:, sel_body_idx, :], pad_before, pad_after)

            # Calculate velocities and world frame poses based on new data format
            # (completed before segmentation to avoid boundary artifacts)
            T_full = m['root_pos'].shape[0]
            fps = int(m['fps'])
            # Root position and orientation
            root_pos = m['root_pos'].astype(np.float32)
            root_quat_xyzw = m['root_rot'].astype(np.float32)
            # Convert to SciPy Rotation
            R_root = sRot.from_quat(root_quat_xyzw)  # expects (x,y,z,w)
            R_root_m = R_root.as_matrix()            # (T,3,3)
            # Local body poses (relative to root)
            local_pos = m['local_body_pos'].astype(np.float32)     # (T,B,3)
            local_quat_xyzw = m['local_body_rot'].astype(np.float32)  # (T,B,4)
            # World frame body poses
            body_pos_w = np.einsum('tij,tbj->tbi', R_root_m, local_pos) + root_pos[:, None, :]
            R_local_m = sRot.from_quat(local_quat_xyzw.reshape(-1, 4)).as_matrix().reshape(T_full, -1, 3, 3)
            R_world_m = np.einsum('tij,tbjk->t bik', R_root_m, R_local_m)
            body_quat_w_xyzw = sRot.from_matrix(R_world_m.reshape(-1, 3, 3)).as_quat().reshape(T_full, -1, 4)
            # Root linear velocity (world frame)
            root_lin_vel = np.zeros_like(root_pos, dtype=np.float32)
            if T_full >= 2:
                root_lin_vel[1:-1] = (root_pos[2:] - root_pos[:-2]) * (fps / 2.0)
                root_lin_vel[0] = (root_pos[1] - root_pos[0]) * fps
                root_lin_vel[-1] = (root_pos[-1] - root_pos[-2]) * fps
            # Joint velocities
            dof_pos = m['dof_pos'].astype(np.float32)
            joint_vel = np.zeros_like(dof_pos, dtype=np.float32)
            if T_full >= 2:
                joint_vel[1:-1] = (dof_pos[2:] - dof_pos[:-2]) * (fps / 2.0)
                joint_vel[0] = (dof_pos[1] - dof_pos[0]) * fps
                joint_vel[-1] = (dof_pos[-1] - dof_pos[-2]) * fps
            # Root angular velocity (world frame): using central difference of ΔR = R_{t+1} @ R_{t-1}^T
            root_ang_vel = angvel_from_rot(R_root, fps=fps)

            # Assemble qpos, qvel (quaternions in qpos are arranged as wxyz to match subsequent usage)
            # Convert xyzw -> wxyz
            root_quat_wxyz = np.concatenate([root_quat_xyzw[:, 3:4], root_quat_xyzw[:, :3]], axis=-1)
            body_quat_w_wxyz = np.concatenate([body_quat_w_xyzw[..., 3:4], body_quat_w_xyzw[..., :3]], axis=-1)
            qpos = np.concatenate([root_pos, root_quat_wxyz, dof_pos], axis=-1)
            qvel = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=-1)

            # Store back to motion dictionary for subsequent segmentation and downstream use
            m['xpos'] = body_pos_w.astype(np.float32)
            m['xquat'] = body_quat_w_wxyz.astype(np.float32)
            m['qpos'] = qpos.astype(np.float32)
            m['qvel'] = qvel.astype(np.float32)
            
            # Process metadata and save motion
            T = m['root_pos'].shape[0];

            for start_idx in range(0, T, SEGMENT_LEN):
                end_idx   = min(start_idx + SEGMENT_LEN, T)
                m_seg = {}
                for k in preserved_keys:
                    m_seg[k] = m[k] if k == 'fps' else m[k][start_idx:end_idx]
                m_seg["joint_names"] = meta["joint_names"]
                m_seg["body_names"] = meta['body_names']

                if motion_filter is not None:
                    try:
                        ok = motion_filter(m_seg, foot_idx, p, start_idx, end_idx)
                    except TypeError:
                        ok = motion_filter(m_seg, foot_idx, p)
                    if not ok:
                        continue

                if motion_processer is not None:
                    try:
                        m_seg = motion_processer(m_seg, foot_idx, p, start_idx, end_idx)
                    except TypeError:
                        m_seg = motion_processer(m_seg, foot_idx)

                total += m_seg["qpos"].shape[0]  # Accumulate accepted frame count

                if callback is not None:
                    ctx = locals()
                    callback(ctx, m_seg)

                if build_dataset:
                    motions.append(m_seg)  # Append segmented motion
                    meta['info'].append(m_seg.get('metadata') or {})  # Synchronize metadata
            pb.set_postfix(total=total)

        if not build_dataset:
            return None, meta

        if not motions:
            raise RuntimeError(
                f"No motion segments were accepted from {root_path} ({len(paths)} files read); "
                "all segments were rejected by motion_filter. If an allowlist is used, check that "
                "--dataset-root points at the directory the allowlist paths are relative to and "
                "that pad/segment parameters match."
            )

        # Stack metadata
        meta_keys = []

        for k in meta['info'][0].keys():
            meta_keys.append(k)
        meta_keys = list(set(meta_keys))
        
        info = {}
        for k in meta_keys:
            info[k] = []
        for m in meta['info']:
            for k in meta_keys:
                info[k].append(m[k])
        for k in meta_keys:
            tmp = np.array(info[k])
            tmp = tmp.reshape(len(motions), -1)
            info[k] = tmp.tolist()
        
        meta['info'] = info

        # Pre-allocate memory-mapped tensors
        mm = {}
        mm['motion_id']      = MemoryMappedTensor.empty(total, dtype=storage_int_dtype)
        mm['step']           = MemoryMappedTensor.empty(total, dtype=storage_int_dtype)
        mm['root_pos_w']     = MemoryMappedTensor.empty(total, 3, dtype=storage_float_dtype)
        mm['root_quat_w']    = MemoryMappedTensor.empty(total, 4, dtype=storage_float_dtype)
        mm['root_lin_vel_w'] = MemoryMappedTensor.empty(total, 3, dtype=storage_float_dtype)
        mm['root_ang_vel_w'] = MemoryMappedTensor.empty(total, 3, dtype=storage_float_dtype)
        mm['joint_pos']      = MemoryMappedTensor.empty(total, len(meta['joint_names']), dtype=storage_float_dtype)
        mm['joint_vel']      = MemoryMappedTensor.empty(total, len(meta['joint_names']), dtype=storage_float_dtype)
        mm['body_pos_w']     = MemoryMappedTensor.empty(total, len(meta['body_names']), 3, dtype=storage_float_dtype)
        mm['body_pos_b']     = MemoryMappedTensor.empty(total, len(meta['body_names']), 3, dtype=storage_float_dtype)
        mm['body_quat_w']    = MemoryMappedTensor.empty(total, len(meta['body_names']), 4, dtype=storage_float_dtype)

        cursor = 0
        starts = []
        ends = []
        for i, m in enumerate(motions):
            T = m['qpos'].shape[0]
            # Root position and velocity
            root_pos_w     = m['qpos'][:, :3]
            root_quat_w    = m['qpos'][:, 3:7]
            root_lin_vel_w = m['qvel'][:, :3]
            root_ang_vel_w = m['qvel'][:, 3:6]
            # Joints
            J = len(meta['joint_names'])
            joint_pos = m['qpos'][:, 7:7+J]
            joint_vel = m['qvel'][:, 6:6+J]
            # Bodies
            B = len(meta['body_names'])
            body_pos_w     = m['xpos']
            # Body frame
            body_pos_b     = rotate_to_body(root_quat_w, body_pos_w - root_pos_w[:, None]).astype(np.float32)
            body_quat_w    = m['xquat']
            # Fill tensors
            mm['step'          ][cursor:cursor+T] = torch.arange(T, dtype=storage_int_dtype)
            mm['motion_id'     ][cursor:cursor+T] = i
            mm['root_pos_w'    ][cursor:cursor+T] = torch.as_tensor(root_pos_w, dtype=storage_float_dtype)
            mm['root_quat_w'   ][cursor:cursor+T] = torch.as_tensor(root_quat_w, dtype=storage_float_dtype)
            mm['root_lin_vel_w'][cursor:cursor+T] = torch.as_tensor(root_lin_vel_w, dtype=storage_float_dtype)
            mm['root_ang_vel_w'][cursor:cursor+T] = torch.as_tensor(root_ang_vel_w, dtype=storage_float_dtype)
            mm['joint_pos'     ][cursor:cursor+T] = torch.as_tensor(joint_pos, dtype=storage_float_dtype)
            mm['joint_vel'     ][cursor:cursor+T] = torch.as_tensor(joint_vel, dtype=storage_float_dtype)
            mm['body_pos_w'    ][cursor:cursor+T] = torch.as_tensor(body_pos_w, dtype=storage_float_dtype)
            mm['body_pos_b'    ][cursor:cursor+T] = torch.as_tensor(body_pos_b, dtype=storage_float_dtype)
            mm['body_quat_w'   ][cursor:cursor+T] = torch.as_tensor(body_quat_w, dtype=storage_float_dtype)

            starts.append(cursor)
            cursor += T
            ends.append(cursor)
        data = MotionData(**mm, batch_size=[total])
        
        # Save to mem_path
        if mem_path is not None:
            path = mem_path
            data.memmap(path)
            # Write metadata
            dump_data = {
                "body_names": meta['body_names'],
                "joint_names": meta['joint_names'],
                "starts": starts,
                "ends": ends,
                "info": meta['info']
            }
            with open(path + "/meta_motion.json", "w") as f:
                json.dump(dump_data, f)
        
        return data, meta

    @property
    def num_motions(self):
        return len(self.starts)

    @property
    def num_steps(self):
        return len(self.data)

    def get_slice(self, motion_ids: torch.Tensor, starts: torch.Tensor, steps: int = 1) -> MotionData:
        motion_ids = motion_ids.to(dtype=torch.long, device=self.starts.device)
        starts = starts.to(dtype=torch.long, device=self.starts.device)
        if isinstance(steps, int):
            idx = (self.starts[motion_ids].to(torch.long) + starts).unsqueeze(1) + torch.arange(
                steps, device=self.starts.device, dtype=torch.long
            )
        else:
            idx = (self.starts[motion_ids].to(torch.long) + starts).unsqueeze(1) + steps.to(
                dtype=torch.long, device=self.starts.device
            )
        ends_per_motion = (self.ends[motion_ids].to(torch.long) - 1).unsqueeze(1)
        idx = idx.clamp(max=ends_per_motion)
        return self.data[idx]
