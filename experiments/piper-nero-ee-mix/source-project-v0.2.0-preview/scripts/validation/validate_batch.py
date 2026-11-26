#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

XR1_ROOT = Path(os.environ.get("XR1_ROOT", "/mnt/panhe/xr1"))
if str(XR1_ROOT) not in sys.path:
    sys.path.insert(0, str(XR1_ROOT))

from mibot.data.datasets.json_dataset import JsonDataset
from mibot.utils.ef_align import is_piper_traj


def make_single_source_dataset(data_cfg: dict, path: str):
    subset = dict(data_cfg)
    subset["paths"] = [path]
    subset["batch_size"] = 1
    params = {
        "train_datasets": subset,
        "max_steps": 1,
        "finite_epoch": False,
    }
    return JsonDataset(params)


def check_one(robot: str, data_cfg: dict, path: str):
    dataset = make_single_source_dataset(data_cfg, path)
    if not dataset.samples:
        raise RuntimeError(f"{robot}: no sample was generated from {path}")

    item = dataset[0]
    action = item["action"]
    state = item["state"]
    mask = item["action_mask"]

    if tuple(action.shape) != (30, 60):
        raise AssertionError(f"{robot}: action shape {tuple(action.shape)} != (30,60)")
    if tuple(state.shape) != (1, 60):
        raise AssertionError(f"{robot}: state shape {tuple(state.shape)} != (1,60)")
    if tuple(mask.shape) != (30, 60):
        raise AssertionError(f"{robot}: action_mask shape {tuple(mask.shape)} != (30,60)")
    if not torch.isfinite(action).all() or not torch.isfinite(state).all():
        raise AssertionError(f"{robot}: action/state contains NaN or Inf")
    if not torch.all(mask[:, :7] >= 0):
        raise AssertionError(f"{robot}: invalid base action mask")

    if robot == "nero":
        if not torch.any(mask[:, 7]) or not torch.any(mask[:, 8]):
            raise AssertionError("NERO sample should supervise elbow slot 7 and Step Flag slot 8")
    elif robot == "piper":
        if torch.any(mask[:, 7]) or torch.any(mask[:, 8]):
            raise AssertionError("Piper sample must mask NERO-only elbow and Step Flag slots")

    sample = dataset.samples[0]
    traj = dataset._read_json(sample["file"])
    detected = "piper" if is_piper_traj(traj) else "nero"
    if detected != robot:
        raise AssertionError(f"Expected {robot}, but selected sample is {detected}")

    print(
        f"[OK] {robot}: file={sample['file']} frame={sample['frame_index']} "
        f"action={tuple(action.shape)} state={tuple(state.shape)} mask={tuple(mask.shape)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Instantiate JsonDataset separately for NERO and Piper and inspect one item from each.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = cfg["data"]["params"]["train_datasets"]
    paths = list(data_cfg["paths"])
    if len(paths) < 3:
        raise ValueError("Expected mix roots ordered as nero_r1, nero_r2, piper_v2")

    check_one("nero", data_cfg, paths[0])
    check_one("piper", data_cfg, paths[-1])
    print("[OK] Both robot sources produce compatible nero_ee_sb samples.")


if __name__ == "__main__":
    main()
