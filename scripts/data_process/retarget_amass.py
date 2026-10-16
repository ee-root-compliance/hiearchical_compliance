# Copyright 2025 Yanjieze
# Adapted compatibility behavior from the GentleHumanoid GMR fork.
# MIT notice: docs/licenses/gmr.txt
"""AMASS SMPL-X -> CEER G1 NPZ using a pinned official GMR installation.

Compatibility behavior follows the GentleHumanoid GMR data pipeline:
10 Hz human pose sampling, independent IK per clip, and 50 Hz PCHIP/SLERP
export. See docs/retarget_compatibility.md for provenance and validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL', 'disable')
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp
from convert_gmr_to_npz import convert_arrays, load_model

GMR_REVISION = '12a1d14b049e358dc182de9c679b30b96a215191'


def sample_times(count, source_fps, target_fps):
    if count < 2 or not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError('Motion must contain >=2 frames and a positive finite FPS')
    duration = (count - 1) / source_fps
    return np.linspace(0, duration, count), np.linspace(
        0, duration, int(np.round(duration * target_fps)) + 1)


def human_frames(raw, body_model):
    """Resample SMPL-X poses before body-model evaluation, as in the reference."""
    import torch
    from smplx.joint_names import JOINT_NAMES
    keys = ['mocap_frame_rate', 'root_orient', 'pose_body', 'trans', 'betas']
    missing = [k for k in keys if k not in raw]
    if missing:
        raise ValueError(f'Expected AMASS SMPL-X data; missing {missing}')
    if str(raw.get('surface_model_type', 'smplx')).lower() != 'smplx':
        raise ValueError('Only SMPL-X AMASS is supported; convert SMPL separately')
    count = len(raw['root_orient'])
    root = np.asarray(raw['root_orient'])
    pose = np.asarray(raw['pose_body']).reshape(count, 21, 3)
    trans = np.asarray(raw['trans'])
    if root.shape != (count, 3) or trans.shape != (count, 3):
        raise ValueError('Inconsistent root/translation shapes')
    if not all(np.isfinite(v).all() for v in [root, pose, trans, raw['betas']]):
        raise ValueError('Non-finite human motion')
    t, u = sample_times(count, float(raw['mocap_frame_rate']), 10)
    root = Slerp(t, Rotation.from_rotvec(root))(u).as_rotvec()
    pose = np.stack([Slerp(t, Rotation.from_rotvec(pose[:, j]))(u).as_rotvec()
                     for j in range(21)], axis=1).reshape(len(u), 63)
    trans = PchipInterpolator(t, trans, axis=0)(u)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32)
    zeros = lambda n: torch.zeros(len(u), n)
    with torch.no_grad():
        output = body_model(betas=tensor(raw['betas']).view(1, -1),
            global_orient=tensor(root), body_pose=tensor(pose), transl=tensor(trans),
            left_hand_pose=zeros(45), right_hand_pose=zeros(45), jaw_pose=zeros(3),
            leye_pose=zeros(3), reye_pose=zeros(3), return_full_pose=True, return_vert=False)
    positions = output.joints.detach().cpu().numpy()
    poses = output.full_pose.detach().cpu().numpy().reshape(len(u), -1, 3)
    parents = body_model.parents.tolist()
    frames = []
    for f in range(len(u)):
        rotations, frame = [], {}
        for j, name in enumerate(JOINT_NAMES[:len(parents)]):
            r = Rotation.from_rotvec(root[f]) if j == 0 else rotations[parents[j]] * Rotation.from_rotvec(poses[f, j])
            rotations.append(r)
            frame[name] = (positions[f, j], r.as_quat()[[3, 0, 1, 2]])
        frames.append(frame)
    return frames


def robot_arrays(qpos):
    """Reference timing convention: rounded counts, endpoint-inclusive linspace."""
    t, u = sample_times(len(qpos), 10, 50)
    return dict(fps=50, root_pos=PchipInterpolator(t, qpos[:, :3], axis=0)(u),
        root_rot=Slerp(t, Rotation.from_quat(qpos[:, [4, 5, 6, 3]]))(u).as_quat(),
        dof_pos=PchipInterpolator(t, np.unwrap(qpos[:, 7:], axis=0), axis=0)(u))


def retarget_file(source, model_dir):
    import smplx
    from general_motion_retargeting import GeneralMotionRetargeting
    with np.load(source, allow_pickle=False) as data:
        raw = {k: data[k] for k in ['gender', 'surface_model_type', 'mocap_frame_rate',
               'root_orient', 'pose_body', 'trans', 'betas'] if k in data}
    gender = str(raw.get('gender', 'neutral'))
    body_model = smplx.create(str(model_dir), 'smplx', gender=gender, use_pca=False)
    frames = human_frames(raw, body_model)
    height = 1.66 + 0.1 * float(np.asarray(raw['betas']).reshape(-1)[0])
    # A new retargeter per clip avoids state/height leaking between sequences.
    engine = GeneralMotionRetargeting(src_human='smplx', tgt_robot='unitree_g1',
        actual_human_height=height, verbose=False, use_velocity_limit=True)
    qpos = np.array([engine.retarget(frame).copy() for frame in frames])
    model, names = load_model(engine.xml_file)
    motion, _ = convert_arrays(robot_arrays(qpos), model, names)
    # Match the reference's clip-wide lowest body height and initial XY origin.
    rotations = Rotation.from_quat(motion['root_rot'])
    world = np.stack([rotations.apply(motion['local_body_pos'][:, b])
                      for b in range(len(motion['body_names']))], axis=1)
    world += motion['root_pos'][:, None, :]
    motion['root_pos'][:, 2] -= world[:, :, 2].min()
    motion['root_pos'][:, :2] -= motion['root_pos'][0, :2].copy()
    return motion, dict(source_frames=len(raw['root_orient']), ik_frames=len(frames),
        output_frames=len(qpos)*5-4, output_fps=50,
        xml_sha256=hashlib.sha256(Path(engine.xml_file).read_bytes()).hexdigest())


def check_gmr():
    import general_motion_retargeting
    root = Path(general_motion_retargeting.__file__).resolve().parents[1]
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != GMR_REVISION:
        raise ValueError(f'Expected official GMR {GMR_REVISION}, got {revision}')
    dirty = subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--',
        'general_motion_retargeting', 'assets/unitree_g1/g1_mocap_29dof.xml'], text=True)
    if dirty:
        raise ValueError('GMR core/model has tracked modifications; use a clean pinned checkout')
    return revision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='SMPL-X AMASS file or root directory')
    p.add_argument('--output', required=True, type=Path, help='New NPZ file or directory')
    p.add_argument('--body-models', required=True, type=Path, help='Parent of smplx/SMPLX_*.npz')
    p.add_argument('--allowlist', type=Path, help='Select files; segment filtering is performed by the packer')
    p.add_argument('--report', required=True, type=Path)
    args = p.parse_args()
    revision = check_gmr()
    directory = args.input.is_dir()
    files = sorted(args.input.rglob('*_stageii.npz')) if directory else [args.input]
    allowed = None
    if args.allowlist:
        allowed = set(json.loads(args.allowlist.read_text())['segments'])
        files = [f for f in files if (f.relative_to(args.input) if directory else Path(f.name)).as_posix() in allowed]
    if not files or not all(f.is_file() for f in files):
        p.error('No matching input motions')
    jobs = [(f, args.output/f.relative_to(args.input) if directory else args.output) for f in files]
    if args.report.exists() or any(out.exists() or out.suffix != '.npz' for _, out in jobs):
        p.error('Outputs/report must be new; existing files will not be overwritten')
    manifest = dict(gmr_revision=revision, profile='ceer-smplx-10hz-pchip50-v1', files=[])
    if allowed is not None:
        present = {(f.relative_to(args.input) if directory else Path(f.name)).as_posix() for f in files}
        manifest['missing_allowlist_files'] = sorted(allowed-present)
    try:
        for source, out in jobs:
            motion, stats = retarget_file(source, args.body_models)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, **motion)
            manifest['files'].append(dict(source=str(source), output=str(out), **stats))
            print(f'Saved {out}: {len(motion["root_pos"])} frames')
    except Exception as error:
        manifest['error'] = str(error)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
