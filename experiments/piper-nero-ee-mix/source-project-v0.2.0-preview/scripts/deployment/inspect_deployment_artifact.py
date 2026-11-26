#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description="Verify an XR1/NERO deployment artifact and write a provenance manifest.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--checkpoint", default="epoch3-step4026.ckpt")
    p.add_argument("--expected-layout", default="nero_ee_sb")
    p.add_argument("--output", default="")
    args = p.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    ckpt_dir = model_dir / args.checkpoint
    config = model_dir / "config.py"
    weights = ckpt_dir / "checkpoint/mp_rank_00_model_states.pt"

    missing = [str(x) for x in (config, weights) if not x.is_file()]
    if missing:
        raise SystemExit("Missing deployment files:\n" + "\n".join(missing))
    if weights.stat().st_size <= 0:
        raise SystemExit(f"Checkpoint weight file is empty: {weights}")

    config_text = config.read_text(encoding="utf-8", errors="replace")
    if args.expected_layout not in config_text:
        raise SystemExit(
            f"config.py does not visibly contain expected action layout {args.expected_layout!r}; "
            "do not deploy until the final config is inspected."
        )

    manifest = {
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_dir": str(model_dir),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(re.search(r"epoch(\d+)", args.checkpoint).group(1))
            if re.search(r"epoch(\d+)", args.checkpoint) else None,
        "expected_action_layout": args.expected_layout,
        "config_path": str(config),
        "config_bytes": config.stat().st_size,
        "config_sha256": sha256(config),
        "weights_path": str(weights),
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": sha256(weights),
        "note": "Hash proves file identity only; it does not prove successful model loading or real-robot safety."
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
        print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
