#!/usr/bin/env python3
"""Validate the real XR1 JsonDataset epoch length and the exact smoke sample mix.

This complements validate_mix_dataset.py: the latter mirrors Dwell Filter logic,
whereas this script instantiates the actual JsonDataset used by BaseDataModule.
It reads JSON metadata and builds the sample index; it does not start a DataLoader,
load images, initialize a model, or write files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from torch.utils.data.distributed import DistributedSampler
import yaml

XR1_ROOT = Path(os.environ.get("XR1_ROOT", "/mnt/panhe/xr1"))
if str(XR1_ROOT) not in sys.path:
    sys.path.insert(0, str(XR1_ROOT))

from mibot.data.datasets.json_dataset import JsonDataset
from mibot.utils.ef_align import is_piper_traj


def classify_sources(dataset: JsonDataset):
    source_by_path = {}
    for path in dataset.files:
        traj = dataset._read_json(path)
        source_by_path[path] = "piper" if is_piper_traj(traj) else "nero"
        if source_by_path[path] == "piper" and (traj.get("source") or {}).get("split") == "val":
            raise ValueError(f"Piper validation episode leaked into training: {path}")
    counts = Counter(source_by_path[item["file"]] for item in dataset.samples)
    return source_by_path, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instantiate the real XR1 JsonDataset and verify epoch steps or smoke source coverage."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("epoch", "smoke"), required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--expected-effective-frames", type=int, default=486500)
    parser.add_argument("--expected-episodes", type=int, default=3611)
    parser.add_argument("--expected-steps-per-epoch", type=int, default=6082)
    parser.add_argument("--expected-max-steps", type=int, default=18246)
    parser.add_argument("--allow-plan-drift", action="store_true")
    args = parser.parse_args()

    for name in ("world_size", "batch_size", "accumulation", "epochs"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.mode == "smoke" and args.smoke_steps <= 0:
        raise ValueError("smoke_steps must be positive")

    os.environ["WORLD_SIZE"] = str(args.world_size)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    params = dict(raw["data"]["params"])
    params["train_datasets"] = dict(params["train_datasets"])
    params["train_datasets"]["batch_size"] = args.batch_size

    if args.mode == "epoch":
        params["finite_epoch"] = True
        params["max_steps"] = 1
    else:
        params["finite_epoch"] = False
        params["max_steps"] = args.smoke_steps

    dataset = JsonDataset(params)
    source_by_path, source_counts = classify_sources(dataset)
    raw_episodes = len(dataset.files)
    effective = len(dataset)
    dwell_removed = int(sum(dataset.dwell_removed.values()))

    result = {
        "mode": args.mode,
        "episodes": raw_episodes,
        "effective_samples": effective,
        "dwell_removed": dwell_removed,
        "sample_sources": dict(source_counts),
        "world_size": args.world_size,
        "batch_size_per_device": args.batch_size,
        "accumulation": args.accumulation,
    }

    errors = []
    if args.mode == "epoch":
        if raw_episodes != args.expected_episodes:
            errors.append(f"episode count {raw_episodes} != expected {args.expected_episodes}")
        if effective != args.expected_effective_frames:
            errors.append(
                f"actual JsonDataset samples {effective} != expected {args.expected_effective_frames}"
            )
        samples_per_rank = math.ceil(effective / args.world_size)
        batches_per_epoch = math.ceil(samples_per_rank / args.batch_size)
        steps_per_epoch = math.ceil(batches_per_epoch / args.accumulation)
        planned_max_steps = steps_per_epoch * args.epochs
        result.update({
            "samples_per_rank": samples_per_rank,
            "batches_per_epoch": batches_per_epoch,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "epochs": args.epochs,
            "planned_max_steps": planned_max_steps,
            "sampler_padding_per_epoch": samples_per_rank * args.world_size - effective,
        })
        if steps_per_epoch != args.expected_steps_per_epoch:
            errors.append(
                f"steps_per_epoch {steps_per_epoch} != expected {args.expected_steps_per_epoch}"
            )
        if planned_max_steps != args.expected_max_steps:
            errors.append(
                f"planned max_steps {planned_max_steps} != expected {args.expected_max_steps}"
            )
    else:
        expected_samples = args.smoke_steps * args.batch_size * args.world_size
        rank_source_counts = {}
        for rank in range(args.world_size):
            sampler = DistributedSampler(
                dataset, num_replicas=args.world_size, rank=rank,
                shuffle=True, seed=42, drop_last=False
            )
            rank_counts = Counter(
                source_by_path[dataset.samples[index]["file"]] for index in sampler
            )
            rank_source_counts[str(rank)] = dict(rank_counts)
        result.update({
            "per_rank_sample_sources": rank_source_counts,
            "requested_smoke_steps": args.smoke_steps,
            "expected_indexed_samples": expected_samples,
        })
        if effective != expected_samples:
            errors.append(f"smoke sample index length {effective} != expected {expected_samples}")
        if args.accumulation != 1:
            errors.append("smoke validation currently expects accumulation=1")
        if source_counts.get("nero", 0) == 0 or source_counts.get("piper", 0) == 0:
            errors.append("the selected smoke sample index does not contain both NERO and Piper")
        ranks_missing_source = [
            rank for rank, counts in rank_source_counts.items()
            if counts.get("nero", 0) == 0 or counts.get("piper", 0) == 0
        ]
        if ranks_missing_source:
            errors.append(f"DistributedSampler ranks missing one source: {ranks_missing_source}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors and not args.allow_plan_drift:
        raise SystemExit("[FAIL] " + "; ".join(errors))
    if errors:
        print("[WARN] " + "; ".join(errors), file=sys.stderr)
    else:
        print("[OK] Runtime JsonDataset checks passed.")


if __name__ == "__main__":
    main()
