#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Inspect a Lightning/DeepSpeed checkpoint directory without loading weights.")
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    root = Path(args.path).expanduser()
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    print("input:", root)
    print("resolved:", root.resolve())
    if root.is_symlink():
        print("symlink target:", os.readlink(root))

    if root.is_file():
        print("file:", root, "bytes:", root.stat().st_size)
        return

    last = root / "last.ckpt"
    if last.exists() or last.is_symlink():
        print("last.ckpt exists:", True)
        print("last.ckpt resolved:", last.resolve())
        if last.is_symlink():
            print("last.ckpt link:", os.readlink(last))

    matches = sorted(root.rglob("mp_rank_00_model_states.pt"))
    print("mp_rank_00_model_states.pt count:", len(matches))
    for item in matches:
        print("model shard:", item, "bytes:", item.stat().st_size)

    if not matches:
        print("WARNING: no model shard found; this may be a file-format/layout mismatch or an incomplete checkpoint.")


if __name__ == "__main__":
    main()