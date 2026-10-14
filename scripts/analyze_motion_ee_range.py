#!/usr/bin/env python3
"""Summarize real motion EE target ranges from cached motion datasets.

The EE targets used by root_and_wrist_6d_reference come from
MotionDataset.data.body_pos_b for left_hand_mimic/right_hand_mimic. This script
loads the same memmapped datasets referenced by a task config and reports the
root-frame xyz distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


DEFAULT_BODIES = ("left_hand_mimic", "right_hand_mimic")
AXES = ("x", "y", "z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_repo_root()))

from active_adaptation.utils.motion import MotionDataset  # noqa: E402


def _dataset_root() -> Path:
    return Path(os.environ.get("MEMPATH", _repo_root() / "dataset")).resolve()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _percentile_key(value: float) -> str:
    return f"{float(value):g}"


def _load_task_dataset(task: str) -> tuple[list[str], list[float]]:
    config_dir = str(_repo_root() / "cfg")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    OmegaConf.resolve(cfg)
    dataset_cfg = cfg.task.command.dataset
    mem_paths = list(dataset_cfg.mem_paths)
    path_weights = list(getattr(dataset_cfg, "path_weights", [1.0] * len(mem_paths)))
    if len(mem_paths) != len(path_weights):
        raise ValueError(f"mem_paths and path_weights length mismatch: {mem_paths} vs {path_weights}")
    return mem_paths, [float(w) for w in path_weights]


def _resolve_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze root-frame EE target ranges from motion mem datasets."
    )
    parser.add_argument(
        "--task",
        default="G1/hl/ee/G1_hl_ee_compliance_pos_delta_student",
        help="Hydra task config used to find task.command.dataset.mem_paths.",
    )
    parser.add_argument(
        "--mem-paths",
        nargs="*",
        default=None,
        help="Override dataset mem paths, e.g. limmt_no_foot_compliance_full amass_full.",
    )
    parser.add_argument(
        "--path-weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional weights matching --mem-paths. Used only for metadata in the report.",
    )
    parser.add_argument(
        "--body-names",
        nargs="+",
        default=list(DEFAULT_BODIES),
        help="Motion body names to summarize.",
    )
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--hist-bins", type=int, default=32)
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=[0.5, 1, 5, 25, 50, 75, 95, 99, 99.5],
        help="Percentiles to report in [0, 100].",
    )
    parser.add_argument(
        "--max-percentile-samples",
        type=int,
        default=2_000_000,
        help="Evenly sample this many frames per dataset for percentile estimates. "
        "Min/max/mean/std/histograms remain exact.",
    )
    parser.add_argument(
        "--output",
        default="outputs/ee_motion_range.json",
        help="Output JSON path. A .md summary with the same stem is also written.",
    )
    return parser.parse_args()


def _empty_stats(num_bodies: int) -> dict[str, torch.Tensor | int]:
    return {
        "count": 0,
        "min": torch.full((num_bodies, 3), float("inf"), dtype=torch.float64),
        "max": torch.full((num_bodies, 3), -float("inf"), dtype=torch.float64),
        "sum": torch.zeros((num_bodies, 3), dtype=torch.float64),
        "sum_sq": torch.zeros((num_bodies, 3), dtype=torch.float64),
    }


def _update_stats(stats: dict[str, torch.Tensor | int], chunk: torch.Tensor) -> None:
    values = chunk.to(torch.float64)
    stats["count"] = int(stats["count"]) + values.shape[0]
    stats["min"] = torch.minimum(stats["min"], values.amin(dim=0))
    stats["max"] = torch.maximum(stats["max"], values.amax(dim=0))
    stats["sum"] = stats["sum"] + values.sum(dim=0)
    stats["sum_sq"] = stats["sum_sq"] + (values * values).sum(dim=0)


def _finalize_stats(stats: dict[str, torch.Tensor | int]) -> dict[str, Any]:
    count = int(stats["count"])
    mean = stats["sum"] / max(count, 1)
    var = stats["sum_sq"] / max(count, 1) - mean * mean
    var = torch.clamp(var, min=0.0)
    return {
        "count": count,
        "min": stats["min"],
        "max": stats["max"],
        "mean": mean,
        "std": torch.sqrt(var),
        "range": stats["max"] - stats["min"],
    }


def _collect_percentile_sample(
    ds: MotionDataset,
    body_ids: list[int],
    chunk_size: int,
    max_samples: int,
) -> tuple[torch.Tensor, bool]:
    total = ds.num_steps
    stride = max(1, math.ceil(total / max_samples))
    sampled = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        local = torch.arange(start, end, stride, dtype=torch.long)
        if local.numel() > 0:
            sampled.append(ds.data.body_pos_b[local][:, body_ids, :].to(torch.float32).cpu())
    if not sampled:
        return torch.empty(0, len(body_ids), 3), False
    return torch.cat(sampled, dim=0), stride > 1


def _summarize_dataset(
    mem_path: str,
    body_names: list[str],
    percentiles: list[float],
    hist_bins: int,
    chunk_size: int,
    max_percentile_samples: int,
) -> dict[str, Any]:
    ds = MotionDataset.create_from_path_lazy(mem_path, dataset_extra_keys=[], device=torch.device("cpu"))
    missing = [name for name in body_names if name not in ds.body_names]
    if missing:
        raise ValueError(
            f"Dataset {mem_path} is missing bodies {missing}. Available bodies: {ds.body_names}"
        )
    body_ids = [ds.body_names.index(name) for name in body_names]

    stats = _empty_stats(len(body_names))
    total = ds.num_steps
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = ds.data.body_pos_b[start:end][:, body_ids, :].to(torch.float32).cpu()
        _update_stats(stats, chunk)

    finalized = _finalize_stats(stats)
    samples, sampled_percentiles = _collect_percentile_sample(
        ds, body_ids, chunk_size, max_percentile_samples
    )
    q = torch.tensor(percentiles, dtype=torch.float32) / 100.0
    quantiles = torch.quantile(samples, q, dim=0) if samples.numel() else torch.empty(0)

    histograms: dict[str, dict[str, dict[str, Any]]] = {}
    mins = finalized["min"].to(torch.float32)
    maxs = finalized["max"].to(torch.float32)
    for body_i, body_name in enumerate(body_names):
        histograms[body_name] = {}
        for axis_i, axis in enumerate(AXES):
            axis_min = float(mins[body_i, axis_i])
            axis_max = float(maxs[body_i, axis_i])
            if axis_min == axis_max:
                axis_max = axis_min + 1e-6
            counts = torch.zeros(hist_bins, dtype=torch.int64)
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                values = ds.data.body_pos_b[start:end, body_ids[body_i], axis_i].to(torch.float32).cpu()
                counts += torch.histc(values, bins=hist_bins, min=axis_min, max=axis_max).to(torch.int64)
            edges = torch.linspace(axis_min, axis_max, hist_bins + 1)
            histograms[body_name][axis] = {"bin_edges": edges, "counts": counts}

    body_reports = {}
    for body_i, body_name in enumerate(body_names):
        body_reports[body_name] = {
            "min": finalized["min"][body_i],
            "max": finalized["max"][body_i],
            "range": finalized["range"][body_i],
            "mean": finalized["mean"][body_i],
            "std": finalized["std"][body_i],
            "percentiles": {
                _percentile_key(p): quantiles[p_i, body_i] for p_i, p in enumerate(percentiles)
            },
            "histogram": histograms[body_name],
        }

    return {
        "mem_path": mem_path,
        "dataset_root": str(_dataset_root()),
        "num_motions": ds.num_motions,
        "num_frames": total,
        "body_names": body_names,
        "body_ids": body_ids,
        "percentile_sample_frames": int(samples.shape[0]),
        "percentiles_are_sampled": sampled_percentiles,
        "bodies": body_reports,
    }


def _summarize_combined(dataset_reports: list[dict[str, Any]], body_names: list[str]) -> dict[str, Any]:
    combined = {}
    for body_name in body_names:
        counts = torch.tensor([r["num_frames"] for r in dataset_reports], dtype=torch.float64)
        mins = torch.stack(
            [r["bodies"][body_name]["min"].to(torch.float64) for r in dataset_reports]
        )
        maxs = torch.stack(
            [r["bodies"][body_name]["max"].to(torch.float64) for r in dataset_reports]
        )
        means = torch.stack(
            [r["bodies"][body_name]["mean"].to(torch.float64) for r in dataset_reports]
        )
        stds = torch.stack(
            [r["bodies"][body_name]["std"].to(torch.float64) for r in dataset_reports]
        )
        total = counts.sum().clamp_min(1.0)
        mean = (means * counts[:, None]).sum(dim=0) / total
        second = ((stds * stds + means * means) * counts[:, None]).sum(dim=0) / total
        var = torch.clamp(second - mean * mean, min=0.0)
        combined[body_name] = {
            "count": int(total.item()),
            "min": mins.amin(dim=0),
            "max": maxs.amax(dim=0),
            "range": maxs.amax(dim=0) - mins.amin(dim=0),
            "mean": mean,
            "std": torch.sqrt(var),
        }
    return combined


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# EE Motion Target Range",
        "",
        f"- task: `{report['task']}`",
        f"- mem_paths: `{', '.join(report['mem_paths'])}`",
        f"- bodies: `{', '.join(report['body_names'])}`",
        f"- frame count: `{report['total_frames']}`",
        "",
        "Values are root-frame xyz positions from `MotionDataset.data.body_pos_b`.",
        "",
        "## Combined Exact Stats",
        "",
        "| Body | Axis | min | p1 | p5 | mean | std | p95 | p99 | max | range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for body_name in report["body_names"]:
        exact = report["combined"][body_name]
        percentiles = report["datasets"][0]["bodies"][body_name]["percentiles"]
        if len(report["datasets"]) > 1:
            percentiles = None
        for axis_i, axis in enumerate(AXES):
            p1 = percentiles["1"][axis_i] if percentiles and "1" in percentiles else None
            p5 = percentiles["5"][axis_i] if percentiles and "5" in percentiles else None
            p95 = percentiles["95"][axis_i] if percentiles and "95" in percentiles else None
            p99 = percentiles["99"][axis_i] if percentiles and "99" in percentiles else None
            lines.append(
                "| {body} | {axis} | {min:.4f} | {p1} | {p5} | {mean:.4f} | {std:.4f} | "
                "{p95} | {p99} | {max:.4f} | {range:.4f} |".format(
                    body=body_name,
                    axis=axis,
                    min=exact["min"][axis_i],
                    p1=f"{p1:.4f}" if p1 is not None else "-",
                    p5=f"{p5:.4f}" if p5 is not None else "-",
                    mean=exact["mean"][axis_i],
                    std=exact["std"][axis_i],
                    p95=f"{p95:.4f}" if p95 is not None else "-",
                    p99=f"{p99:.4f}" if p99 is not None else "-",
                    max=exact["max"][axis_i],
                    range=exact["range"][axis_i],
                )
            )

    if len(report["datasets"]) > 1:
        lines += [
            "",
            "Percentiles are written per dataset in the JSON report. Combined percentiles are omitted here.",
        ]
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = _resolve_args()
    if args.mem_paths:
        mem_paths = args.mem_paths
        path_weights = args.path_weights or [1.0] * len(mem_paths)
        if len(path_weights) != len(mem_paths):
            raise ValueError("--path-weights must match --mem-paths length")
    else:
        mem_paths, path_weights = _load_task_dataset(args.task)

    reports = []
    for mem_path in mem_paths:
        print(f"Loading {mem_path} ...")
        reports.append(
            _summarize_dataset(
                mem_path=mem_path,
                body_names=args.body_names,
                percentiles=args.percentiles,
                hist_bins=args.hist_bins,
                chunk_size=args.chunk_size,
                max_percentile_samples=args.max_percentile_samples,
            )
        )

    report = {
        "task": args.task,
        "mem_paths": mem_paths,
        "path_weights": path_weights,
        "body_names": args.body_names,
        "total_frames": sum(r["num_frames"] for r in reports),
        "hist_bins": args.hist_bins,
        "percentiles": args.percentiles,
        "datasets": reports,
        "combined": _summarize_combined(reports, args.body_names),
    }
    report = _to_builtin(report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    md_path = out_path.with_suffix(".md")
    _write_markdown(md_path, report)

    print(f"Wrote {out_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
