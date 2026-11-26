# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import argparse
import os
import sys
from os.path import join as osp

from mibot.utils.compat import patch_torch_pytree

patch_torch_pytree()

if os.environ.get("XR1_DEVICE_TYPE", "cuda").lower() == "npu":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

import torch
from mmengine import Config

from mibot.models import MIMODEL
from mibot.server.runtime.server import Server
from mibot.utils.io import build_action_mask, validate_quantiles, validate_stats


def strip_prefix(state_dict, prefix):
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


def load_model(model_dir, device, checkpoint="last.ckpt"):
    cfg = Config.fromfile(osp(model_dir, "config.py"))
    model = MIMODEL.build(cfg.model.params.model)
    prepare_for_load = getattr(model, "prepare_for_checkpoint_load", None)
    if prepare_for_load is not None:
        prepare_for_load()
    prepare = getattr(model, "prepare_posttrain_adaptation", None)
    if prepare is not None:
        prepare()
    ckpt = torch.load(
        osp(model_dir, checkpoint, "checkpoint", "mp_rank_00_model_states.pt"),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )["module"]
    print(model.load_state_dict(strip_prefix(ckpt, "model."), strict=True))
    return cfg, model.eval().to(device)


def load_stats(cfg, device):
    data = cfg.data.params.train_datasets
    action_length = int(data.get("action_length", cfg.data.params.get("action_length", 30)))
    mean, std = validate_stats(data.mean, data.std, action_length)
    q01, q99 = validate_quantiles(data.q01, data.q99)
    return (
        torch.tensor(mean, device=device),
        torch.tensor(std, device=device),
        torch.tensor(q01, device=device),
        torch.tensor(q99, device=device),
        torch.from_numpy(
            build_action_mask(
                action_length,
                action_layout=str(data.get("action_layout", "ee")),
            )
        ).to(device),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model dir.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint directory under --model (default: last.ckpt).",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10086)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda:0"
    cfg, model = load_model(args.model, device, checkpoint=args.checkpoint)
    mean, std, q01, q99, action_mask = load_stats(cfg, device)

    try:
        server = Server(args.host, args.port, model, mean, std, q01, q99, action_mask, device)
        print(f"Starting server on {args.host}:{args.port}")
        server.run()
    except OSError as error:
        if error.errno == 98:
            print(f"Error: Port {args.port} is already in use. Please choose a different port.")
            sys.exit(1)
        raise
    except KeyboardInterrupt:
        print("Server interrupted")
