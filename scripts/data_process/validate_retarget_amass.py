"""Run raw SMPL-X through the compatibility entry and compare a reference NPZ.

Run in the GMR environment; optionally provide --packer-python for the training
environment. Reference may be historical data or output of the pinned fork.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from retarget_amass import check_gmr, retarget_file


def compare(actual, reference):
    if len(actual['root_pos']) != len(reference['root_pos']):
        raise AssertionError('Frame counts differ')
    if float(actual['fps']) != float(reference['fps']):
        raise AssertionError('Frame rates differ')
    metrics = {}
    for field in ['root_pos', 'dof_pos']:
        other = reference[field]
        if field == 'dof_pos':
            other = other[:, [list(reference['joint_names']).index(n) for n in actual['joint_names']]]
        metrics[field + '_max_abs'] = float(np.max(np.abs(actual[field] - other)))
    metrics['root_rotation_max_rad'] = float(np.max((
        Rotation.from_quat(actual['root_rot']).inv() * Rotation.from_quat(reference['root_rot'])).magnitude()))
    common = sorted(set(actual['body_names']) & set(reference['body_names']))
    assert set(reference['body_names']) <= set(actual['body_names']), 'Missing reference bodies'
    ai = [list(actual['body_names']).index(n) for n in common]
    bi = [list(reference['body_names']).index(n) for n in common]
    metrics['body_position_max_m'] = float(np.max(np.linalg.norm(
        actual['local_body_pos'][:, ai] - reference['local_body_pos'][:, bi], axis=-1)))
    a = Rotation.from_quat(actual['local_body_rot'][:, ai].reshape(-1, 4))
    b = Rotation.from_quat(reference['local_body_rot'][:, bi].reshape(-1, 4))
    metrics['body_rotation_max_rad'] = float(np.max((a.inv()*b).magnitude()))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--body-models', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--packer-python', help='Python in the installed CEER training environment')
    args = p.parse_args()
    revision = check_gmr()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    actual, stats = retarget_file(args.input, args.body_models)
    output = args.output_dir/'motion.npz'
    np.savez_compressed(output, **actual)
    with np.load(args.reference, allow_pickle=True) as ref:
        metrics = compare(actual, ref)
    report = dict(gmr_revision=revision, input=str(args.input), reference=str(args.reference),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        metrics=metrics, **stats)
    # Same numerical tolerance in metres/radians as the adapter checks.
    report['status'] = 'PASS' if max(metrics.values()) < 1e-4 else 'FAIL'
    if args.packer_python:
        root = Path(__file__).resolve().parents[2]
        subprocess.run([args.packer_python, str(root/'scripts/data_process/generate_dataset.py'),
            '--dataset-root', str(output), '--mem-path', str(args.output_dir/'memmap')], check=True, cwd=root)
        metadata = json.loads((args.output_dir/'memmap/meta_motion.json').read_text())
        report['packed_segments'] = len(metadata['starts'])
        report['packed_frames'] = sum(b-a for a,b in zip(metadata['starts'], metadata['ends']))
    (args.output_dir/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
