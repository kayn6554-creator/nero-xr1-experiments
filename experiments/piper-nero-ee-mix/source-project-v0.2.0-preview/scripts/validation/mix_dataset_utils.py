from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import yaml

PIPER_GRIPPER_TO_NERO_M = 0.1


def load_train_dataset_config(config_path: str) -> dict:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    try:
        return raw["data"]["params"]["train_datasets"]
    except (TypeError, KeyError) as exc:
        raise ValueError(f"{config_path} does not contain data.params.train_datasets") from exc


def collect_json_files(paths: list[str]) -> list[str]:
    files: set[str] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".json":
            files.add(str(path))
        elif path.is_dir():
            # Keep path strings rather than resolving symlink targets. This is
            # intentional: nero_r1 and nero_r2 must count as two dataset roots.
            files.update(glob.glob(str(path / "**" / "*.json"), recursive=True))
        else:
            raise FileNotFoundError(f"dataset path is missing or invalid: {raw}")
    if not files:
        raise ValueError("dataset paths contain no JSON files")
    return sorted(files)


def _trajectory_arrays(traj: dict, path: str):
    n = int(traj.get("num_frames") or 0)
    if n <= 0:
        raise ValueError(f"{path}: num_frames must be positive")
    proprios = traj.get("proprios") or {}
    actions = traj.get("actions") or {}

    if "left_arm_joint" in proprios and "arm_joint" not in proprios:
        robot = "piper"
        q = np.asarray(proprios.get("left_arm_joint"), dtype=np.float64)
        g = np.asarray(actions.get("left_gripper_pos"), dtype=np.float64).reshape(n, -1)[:, 0]
        g = g * PIPER_GRIPPER_TO_NERO_M
    elif "arm_joint" in proprios:
        robot = "nero"
        q = np.asarray(proprios.get("arm_joint"), dtype=np.float64)
        g = np.asarray(actions.get("gripper_pos"), dtype=np.float64).reshape(n, -1)[:, 0]
    else:
        raise ValueError(f"{path}: cannot identify NERO or Piper trajectory")

    if q.ndim != 2 or len(q) != n or len(g) != n:
        raise ValueError(f"{path}: joint/gripper arrays do not match num_frames={n}")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError(f"{path}: joint/gripper arrays contain NaN or Inf")
    return robot, q, g, n


def count_dwell_kept(traj: dict, q: np.ndarray, g: np.ndarray, n: int, robot: str, data_cfg: dict):
    dwell = data_cfg.get("dwell_keep")
    if not dwell:
        return n, 0

    dwell = {str(k): int(v) for k, v in dict(dwell).items()}
    tail = int(data_cfg.get("dwell_tail", 10))
    speed_threshold = float(data_cfg.get("dwell_speed_deg", 2.0))
    gripper_threshold = float(data_cfg.get("dwell_gripper_m", 0.001))

    source = traj.get("source") or {}
    steps = list(source.get("step_index") or [0] * n)
    if len(steps) != n:
        raise ValueError("source.step_index length does not match num_frames")

    fps = float(traj.get("fps", 10))
    speed = np.r_[np.inf, np.abs(np.diff(q, axis=0)).max(1) * (180.0 / np.pi) * fps]
    grip_delta = np.r_[np.inf, np.abs(np.diff(g))]
    static = (speed < speed_threshold) & (grip_delta < gripper_threshold)

    keep = np.ones(n, dtype=bool)
    removed = 0
    i = 0
    while i < n:
        if not static[i]:
            i += 1
            continue
        j = i
        while j < n and static[j] and steps[j] == steps[i]:
            j += 1
        head = int(dwell.get(str(steps[i]), dwell.get("default", 10**9)))
        if j - i > head + tail:
            start, stop = i + head, j - tail
            keep[start:stop] = False
            removed += stop - start
        i = j

    return int(keep.sum()), int(removed)


def inspect_dataset(config_path: str) -> dict:
    data_cfg = load_train_dataset_config(config_path)
    paths = [str(p) for p in data_cfg.get("paths", [])]
    files = collect_json_files(paths)

    report = {
        "config": str(Path(config_path)),
        "action_layout": data_cfg.get("action_layout"),
        "action_length": int(data_cfg.get("action_length", 30)),
        "batch_size": int(data_cfg.get("batch_size", 16)),
        "episodes": len(files),
        "raw_frames": 0,
        "effective_frames": 0,
        "dwell_removed": 0,
        "robots": {
            "nero": {"episodes": 0, "raw_frames": 0, "effective_frames": 0},
            "piper": {"episodes": 0, "raw_frames": 0, "effective_frames": 0},
        },
        "val_split_episodes": 0,
        "path_counts": {},
    }

    for root in paths:
        root_path = Path(root)
        if root_path.is_file():
            count = int(root_path.suffix == ".json")
        else:
            count = len(glob.glob(str(root_path / "**" / "*.json"), recursive=True))
        report["path_counts"][root] = count

    for path in files:
        traj = json.loads(Path(path).read_text(encoding="utf-8"))
        robot, q, g, n = _trajectory_arrays(traj, path)
        src = traj.get("source") or {}
        if robot == "piper" and src.get("split") == "val":
            report["val_split_episodes"] += 1
            raise ValueError(f"{path}: Piper validation split must not enter training")

        if robot == "nero":
            step_index = src.get("step_index")
            if step_index is None or len(step_index) != n:
                raise ValueError(f"{path}: NERO trajectory needs a num_frames-length source.step_index")

        kept, removed = count_dwell_kept(traj, q, g, n, robot, data_cfg)
        report["raw_frames"] += n
        report["effective_frames"] += kept
        report["dwell_removed"] += removed
        report["robots"][robot]["episodes"] += 1
        report["robots"][robot]["raw_frames"] += n
        report["robots"][robot]["effective_frames"] += kept

    return report


def calculate_steps(effective_frames: int, world_size: int, batch_size: int, accumulation: int, epochs: int):
    if min(effective_frames, world_size, batch_size, accumulation, epochs) <= 0:
        raise ValueError("effective_frames, world_size, batch_size, accumulation and epochs must be positive")
    samples_per_rank = math.ceil(effective_frames / world_size)
    batches_per_epoch = math.ceil(samples_per_rank / batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    return {
        "effective_frames": effective_frames,
        "world_size": world_size,
        "batch_per_device": batch_size,
        "accumulate_grad_batches": accumulation,
        "epochs": epochs,
        "samples_per_rank": samples_per_rank,
        "batches_per_epoch": batches_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "max_steps": epochs * steps_per_epoch,
        "sampler_padding_per_epoch": samples_per_rank * world_size - effective_frames,
    }
