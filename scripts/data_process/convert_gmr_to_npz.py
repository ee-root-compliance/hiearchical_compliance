"""Convert trusted GMR pickle files to CEER NPZ using G1 forward kinematics.

Dependencies: numpy, scipy, mujoco. No Isaac Sim initialization is needed.
This adapter does not retarget human motion; inputs are already G1 motions.
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


JOINTS = [
    *(f"{s}_{j}_joint" for s in ("left", "right") for j in
      ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *(f"{s}_{j}_joint" for s in ("left", "right") for j in
      ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
       "wrist_roll", "wrist_pitch", "wrist_yaw")),
]
# Rigid virtual frames matching the existing CEER G1 motion data.
VIRTUAL = {
    "head_mimic": ("torso_link", [0.01, 0.0, 0.41]),
    "left_hand_mimic": ("left_wrist_yaw_link", [0.116, 0.0, 0.0]),
    "right_hand_mimic": ("right_wrist_yaw_link", [0.116, 0.0, 0.0]),
}


def load_model(xml):
    model = mujoco.MjModel.from_xml_path(str(xml))
    if model.nq != 36 or model.njnt != 30 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected G1 with one root free joint and 29 scalar joints")
    names = [model.joint(i).name for i in range(1, model.njnt)]
    if set(names) != set(JOINTS):
        raise ValueError("Robot XML does not contain the required G1 29 joints")
    for name in names:
        if model.joint(name).type[0] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Expected hinge joint: {name}")
    # Root must coincide with pelvis for root-relative FK below.
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_kinematics(model, data)
    if not np.allclose(data.xpos[model.body('pelvis').id], 0):
        raise ValueError("XML pelvis does not coincide with the floating root")
    return model, names


def convert(source, model, model_joint_names, target_fps=50.0):
    # Pickle executes code: only convert data from trusted sources.
    with Path(source).open("rb") as f:
        raw = pickle.load(f)
    return convert_arrays(raw, model, model_joint_names, target_fps)


def convert_arrays(raw, model, model_joint_names, target_fps=50.0):
    """Convert in-memory robot motion, without a pickle round trip."""
    fps = float(raw['fps'])
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("Source and target FPS must be finite and positive")
    arrays = {k: np.asarray(raw[k], dtype=np.float64) for k in ('root_pos', 'root_rot', 'dof_pos')}
    count = len(arrays['root_pos'])
    for key, width in [('root_pos', 3), ('root_rot', 4), ('dof_pos', 29)]:
        if arrays[key].shape != (count, width) or count < 2 or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid {key}: expected finite (T,{width}), T >= 2")
    norms = np.linalg.norm(arrays['root_rot'], axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("Zero root quaternion")
    arrays['root_rot'] /= norms[:, None]
    # Official GMR exports root_rot in xyzw order. Unnamed dofs follow the
    # retargeting XML's qpos order; that exact XML must be supplied by the user.
    names = list(raw.get('joint_names', model_joint_names))
    if len(names) != 29 or set(names) != set(JOINTS):
        raise ValueError("Input joint_names do not match G1")
    arrays['dof_pos'] = arrays['dof_pos'][:, [names.index(n) for n in JOINTS]]
    if not np.isclose(fps, target_fps, rtol=0, atol=1e-9):
        times = np.arange(count) / fps
        sample_times = np.arange(int(np.floor(times[-1] * target_fps + 1e-9)) + 1) / target_fps
        sample_times = np.minimum(sample_times, times[-1])
        if len(sample_times) < 2:
            raise ValueError("Motion too short at target FPS")
        for key in ('root_pos', 'dof_pos'):
            arrays[key] = np.stack([np.interp(sample_times, times, col) for col in arrays[key].T], axis=1)
        arrays['root_rot'] = Slerp(times, Rotation.from_quat(arrays['root_rot']))(sample_times).as_quat()
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    if any(not n for n in body_names):
        raise ValueError("All robot bodies must be named")
    virtual_names = [n for n in VIRTUAL if n not in body_names]
    output_names = body_names + virtual_names
    pos = np.empty((len(arrays['dof_pos']), len(output_names), 3))
    rot = np.empty((len(pos), len(output_names), 4))
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    addresses = [int(model.joint(n).qposadr[0]) for n in JOINTS]
    for t, q in enumerate(arrays['dof_pos']):
        data.qpos[addresses] = q
        mujoco.mj_kinematics(model, data)
        pos[t, :len(body_names)] = data.xpos[1:]
        rot[t, :len(body_names)] = data.xquat[1:, [1, 2, 3, 0]]
        for index, name in enumerate(virtual_names, len(body_names)):
            parent, offset = VIRTUAL[name]
            bid = model.body(parent).id
            pos[t, index] = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ offset
            rot[t, index] = data.xquat[bid, [1, 2, 3, 0]]
    output = {k: v.astype(np.float32) for k, v in arrays.items()}
    output.update(fps=np.float64(target_fps), joint_names=np.array(JOINTS),
                  body_names=np.array(output_names), local_body_pos=pos.astype(np.float32),
                  local_body_rot=rot.astype(np.float32))
    return output, {'source_frames': count, 'source_fps': fps,
                    'output_frames': len(pos), 'output_fps': target_fps,
                    'resampled': not np.isclose(fps, target_fps, rtol=0, atol=1e-9)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Trusted GMR .pkl file or directory')
    parser.add_argument('--output', required=True, type=Path, help='New NPZ file or output directory')
    parser.add_argument('--robot-xml', required=True, type=Path, help='Exact G1 XML used for retargeting')
    parser.add_argument('--target-fps', type=float, default=50)
    parser.add_argument('--allowlist', type=Path, help='Select source files only; segments are filtered later')
    parser.add_argument('--report', required=True, type=Path, help='New JSON conversion manifest')
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report exists; choose a new report path')
    model, names = load_model(args.robot_xml)
    directory = args.input.is_dir()
    files = sorted(args.input.rglob('*.pkl')) if directory else [args.input]
    allowed = None
    if args.allowlist:
        allowed = set(json.loads(args.allowlist.read_text())['segments'])
        files = [p for p in files if (p.relative_to(args.input) if directory else Path(p.name)).with_suffix('.npz').as_posix() in allowed]
    if not files or any(not p.is_file() or p.suffix != '.pkl' for p in files):
        parser.error('No matching GMR .pkl files')
    targets = [(p, args.output / p.relative_to(args.input).with_suffix('.npz') if directory else args.output) for p in files]
    if any(dst.exists() or dst.suffix != '.npz' for _, dst in targets):
        parser.error('Output must be a new .npz file; existing outputs will not be overwritten')
    report = {'robot_xml': str(args.robot_xml),
              'robot_xml_sha256': hashlib.sha256(args.robot_xml.read_bytes()).hexdigest(),
              'quaternion_order': 'xyzw', 'virtual_frames': VIRTUAL, 'files': []}
    if allowed is not None:
        present = {(p.relative_to(args.input) if directory else Path(p.name)).with_suffix('.npz').as_posix() for p in files}
        report['missing_allowlist_files'] = sorted(allowed - present)
    try:
        for source, dst in targets:
            output, stats = convert(source, model, names, args.target_fps)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, **output)
            report['files'].append(dict(source=str(source), output=str(dst), **stats))
            print(f'{source} -> {dst} ({stats["output_frames"]} frames)')
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
