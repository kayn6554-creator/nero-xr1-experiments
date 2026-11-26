#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from mix_dataset_utils import calculate_steps, inspect_dataset


def main():
    parser = argparse.ArgumentParser(description="Plan finite epochs using actual JsonDataset dwell_keep rules.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--counts", action="store_true")
    args = parser.parse_args()

    report = inspect_dataset(args.config)
    plan = calculate_steps(
        report["effective_frames"], args.world_size, args.batch_size,
        args.accumulation, args.epochs
    )
    combined = {"dataset": report, "plan": plan}
    if args.counts:
        print(
            report["effective_frames"],
            plan["steps_per_epoch"],
            plan["max_steps"],
            report["episodes"],
        )
    else:
        print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
