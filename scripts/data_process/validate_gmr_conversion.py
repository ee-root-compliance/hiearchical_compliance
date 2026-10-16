"""Validate conversion against paired local GMR PKL / reference NPZ motions."""
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from convert_gmr_to_npz import convert, load_model, JOINTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--robot-xml', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--count', type=int, default=3)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    allow = json.loads((repo / 'scripts/data_process/allowlist_limmt_no_foot_compliance.json').read_text())
    candidates = [rel for rel in sorted(allow['segments'])
                  if (args.dataset_root / rel).is_file()
                  and (args.dataset_root / Path(rel).with_suffix('.pkl')).is_file()]
    if args.count < 1 or len(candidates) < args.count:
        raise ValueError('Not enough paired motions')
    # Spread samples over the corpus rather than choosing neighboring files.
    selected = [candidates[i] for i in np.linspace(0, len(candidates)-1, args.count, dtype=int)]
    model, names = load_model(args.robot_xml)
    report = {'motions': [], 'checks': []}
    for rel in selected:
        source = args.dataset_root / Path(rel).with_suffix('.pkl')
        out, stats = convert(source, model, names)
        ref = np.load(args.dataset_root / rel, allow_pickle=True)
        ref_names = list(ref['body_names'])
        shared = sorted(set(ref_names) & set(out['body_names']))
        ai = [list(out['body_names']).index(n) for n in shared]
        bi = [ref_names.index(n) for n in shared]
        assert out['root_pos'].shape == ref['root_pos'].shape, 'Frame count mismatch'
        joint_error = float(np.max(np.abs(out['dof_pos'] - ref['dof_pos'][:, [list(ref['joint_names']).index(n) for n in JOINTS]])))
        pos_error = float(np.max(np.linalg.norm(out['local_body_pos'][:, ai] - ref['local_body_pos'][:, bi], axis=-1)))
        a = Rotation.from_quat(out['local_body_rot'][:, ai].reshape(-1, 4))
        b = Rotation.from_quat(ref['local_body_rot'][:, bi].reshape(-1, 4))
        angle_error = float(np.max((a.inv()*b).magnitude()))
        root_error = float(np.max(np.abs(out['root_pos'] - ref['root_pos'])))
        root_angle = float(np.max((Rotation.from_quat(out['root_rot']).inv()*Rotation.from_quat(ref['root_rot'])).magnitude()))
        assert max(joint_error, pos_error, angle_error, root_error, root_angle) < 1e-4
        dst = args.output_dir / 'npz' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **out)
        report['motions'].append(dict(path=rel, shared_bodies=len(shared),
            joint_max_rad=joint_error, body_position_max_m=pos_error,
            body_rotation_max_rad=angle_error, root_position_max_m=root_error,
            root_rotation_max_rad=root_angle, **stats))
    subset = dict(params=allow['params'], segments={r:allow['segments'][r] for r in selected})
    subset_path = args.output_dir / 'allowlist.json'
    subset_path.write_text(json.dumps(subset, indent=2))
    subprocess.run([sys.executable, str(repo/'scripts/data_process/generate_dataset.py'),
                    '--dataset-root', str(args.output_dir/'npz'), '--allowlist', str(subset_path),
                    '--mem-path', str(args.output_dir/'memmap')], check=True, cwd=repo)
    sys.path.insert(0, str(repo))
    from active_adaptation.utils.motion import MotionDataset
    dataset = MotionDataset.create_from_path_lazy(str((args.output_dir/'memmap').resolve()))
    # Validate the actual generated metadata, including every requested segment.
    meta = json.loads((args.output_dir/'memmap/meta_motion.json').read_text())
    assert dataset.joint_names == JOINTS
    report['packed_segments'] = len(meta['starts'])
    assert len(meta['starts']) == sum(len(v) for v in subset['segments'].values())
    for n in ['head_mimic','left_hand_mimic','right_hand_mimic']:
        assert n in meta['body_names']
    report['checks'].append('allowlist segment counts and required bodies in packed memmap')
    # Controlled resampling fixture derived from real data: 31 frames at 30 Hz
    # should span exactly one second, yielding 51 frames at 50 Hz.
    with (args.dataset_root/Path(selected[0]).with_suffix('.pkl')).open('rb') as f:
        raw = pickle.load(f)
    fixture = {k: np.asarray(raw[k])[:31] for k in ['root_pos','root_rot','dof_pos']}
    fixture['fps'] = 30
    fixture_path = args.output_dir/'resample_fixture.pkl'
    with fixture_path.open('wb') as f:
        pickle.dump(fixture, f)
    sampled, _ = convert(fixture_path, model, names)
    assert sampled['root_pos'].shape[0] == 51
    assert np.allclose(sampled['root_pos'][[0,-1]], fixture['root_pos'][[0,-1]])
    assert np.allclose(np.linalg.norm(sampled['root_rot'],axis=1),1)
    report['checks'].append('30 to 50 Hz resampling: 51 frames, endpoints and quaternion norms')
    # Named joints may be in any order.
    fixture['joint_names'] = names[::-1]
    fixture['dof_pos'] = fixture['dof_pos'][:, ::-1]
    with fixture_path.open('wb') as f:
        pickle.dump(fixture, f)
    reordered, _ = convert(fixture_path, model, names)
    assert np.allclose(sampled['dof_pos'], reordered['dof_pos'])
    report['checks'].append('explicit reversed joint order')
    fixture['root_rot'][:] = 0
    with fixture_path.open('wb') as f:
        pickle.dump(fixture, f)
    try:
        convert(fixture_path, model, names)
    except ValueError:
        report['checks'].append('zero quaternion rejected')
    else:
        raise AssertionError('Invalid quaternion accepted')
    report['status'] = 'PASS'
    (args.output_dir/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
