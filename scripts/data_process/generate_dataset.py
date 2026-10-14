import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from active_adaptation.utils.motion import MotionDataset


def preprocess_motion(motion, foot_idx, always_on_ground: bool = False):
    root_pos = motion["qpos"][:, :3]  # (T,3)
    offset_xy = root_pos[0, :2].copy()  # x and y coordinates of the first frame
    motion["qpos"][:, 0] -= offset_xy[0]
    motion["qpos"][:, 1] -= offset_xy[1]
    motion["xpos"][:, :, 0] -= offset_xy[0]
    motion["xpos"][:, :, 1] -= offset_xy[1]

    z_l = motion["xpos"][:, foot_idx[0], 2]
    z_r = motion["xpos"][:, foot_idx[1], 2]

    if not always_on_ground:
        z_min = float(min(z_l.min(), z_r.min()))
        target_z0 = 0.0
        dz = target_z0 - z_min
        motion["qpos"][:, 2] += dz
        motion["xpos"][:, :, 2] += dz
    else:
        z_min = np.min(
            np.concatenate([z_l.reshape(-1, 1), z_r.reshape(-1, 1)], axis=1),
            axis=-1,
            keepdims=True,
        )
        target_z0 = 0.0
        dz = target_z0 - z_min
        motion["qpos"][:, 2] += dz.reshape(-1)
        motion["xpos"][:, :, 2] += dz
    return motion


def none_callback(_ctx, m):
    m["metadata"] = None


def _default_relpath(dataset_root: Path, p: Path) -> str:
    if dataset_root.is_file():
        return p.name
    try:
        return str(p.relative_to(dataset_root))
    except ValueError:
        return str(p)


def load_allowlist(path: str) -> set[tuple[str, int, int]]:
    payload = json.loads(Path(path).read_text())
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError("allowlist json missing required object key: params")
    segs = payload.get("segments", {})
    out: set[tuple[str, int, int]] = set()
    for fname, spans in segs.items():
        for start, end in spans:
            out.add((str(fname), int(start), int(end)))
    required_params = ["target_fps", "pad_before", "pad_after", "segment_len"]
    missing = [k for k in required_params if k not in params]
    if missing:
        raise ValueError(f"allowlist json missing required params keys: {missing}")
    return out, {
        "target_fps": int(params["target_fps"]),
        "pad_before": int(params["pad_before"]),
        "pad_after": int(params["pad_after"]),
        "segment_len": int(params["segment_len"]),
    }


def make_allowlist_filter(allow: set[tuple[str, int, int]], dataset_root: Path, matched: set | None = None):
    def _filter(_motion, _foot_idx, p, start_idx, end_idx):
        rel = _default_relpath(dataset_root, Path(p))
        key = (rel, int(start_idx), int(end_idx))
        ok = key in allow
        if ok and matched is not None:
            matched.add(key)
        return ok

    return _filter


def report_allowlist_matches(allow: set[tuple[str, int, int]], matched: set, max_list: int = 10):
    missing = sorted(allow - matched)
    print(f"[generate_dataset] matched {len(matched)} / {len(allow)} allowlist segments")
    if missing:
        print(f"[generate_dataset] {len(missing)} allowlist segments not found, e.g.:")
        for rel, start, end in missing[:max_list]:
            print(f"    {rel} [{start}, {end})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="NPZ file or directory to convert")
    ap.add_argument(
        "--allowlist",
        default=None,
        help="Optional allowlist json (e.g. scripts/data_process/allowlist_limmt_no_foot_compliance.json); "
        "its segment paths are relative to --dataset-root and its params override the CLI values",
    )
    ap.add_argument("--mem-path", required=True, help="Output memmap directory")
    ap.add_argument("--target-fps", type=int, default=50)
    ap.add_argument("--pad-before", type=int, default=25)
    ap.add_argument("--pad-after", type=int, default=50)
    ap.add_argument("--segment-len", type=int, default=1000)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    params = {
        "target_fps": args.target_fps,
        "pad_before": args.pad_before,
        "pad_after": args.pad_after,
        "segment_len": args.segment_len,
    }
    allow_filter = None
    allow = None
    matched: set[tuple[str, int, int]] = set()
    if args.allowlist is not None:
        allow, params = load_allowlist(args.allowlist)
        allow_files = sorted({rel for rel, _, _ in allow})
        if not allow_files:
            raise ValueError(f"allowlist {args.allowlist} contains no segments")
        if dataset_root.is_file():
            found = [rel for rel in allow_files if rel == dataset_root.name]
        else:
            found = [rel for rel in allow_files if (dataset_root / rel).is_file()]
        print(f"[generate_dataset] found {len(found)} / {len(allow_files)} allowlist files under {dataset_root}")
        if not found:
            raise FileNotFoundError(
                f"None of the {len(allow_files)} allowlist files exist under --dataset-root {dataset_root} "
                f"(e.g. {dataset_root / allow_files[0]}). Allowlist paths are relative to the dataset root "
                f"(e.g. {allow_files[0]}); check that --dataset-root points at that directory level."
            )
        allow_filter = make_allowlist_filter(allow, dataset_root, matched)

    try:
        MotionDataset.create_from_path(
            str(dataset_root),
            target_fps=params["target_fps"],
            mem_path=args.mem_path,
            callback=none_callback,
            motion_processer=partial(preprocess_motion, always_on_ground=True),
            motion_filter=allow_filter,
            pad_before=params["pad_before"],
            pad_after=params["pad_after"],
            segment_len=params["segment_len"],
            storage_float_dtype=torch.float16,
            storage_int_dtype=torch.int32,
        )
    finally:
        if allow is not None:
            report_allowlist_matches(allow, matched)


if __name__ == "__main__":
    main()
