#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mix_dataset_utils import inspect_dataset


def main():
    parser = argparse.ArgumentParser(description="Validate NERO + Piper mixed JSON dataset and dwell-filter counts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = inspect_dataset(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote manifest: {args.output}")


if __name__ == "__main__":
    main()