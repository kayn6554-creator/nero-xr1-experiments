#!/usr/bin/env python3
# Copyright (C) 2026 Xiaomi Corporation.
"""Serve an XR-1 Piper or Nero checkpoint through the OpenPI websocket protocol.

The request and response schema is intentionally identical to OpenPI's
``WebsocketClientPolicy`` contract.  XR-1 predicts current-observation-relative
joint deltas, so this adapter restores absolute 14-D Piper or 8-D Nero targets
before sending ``{"actions": ...}`` to the client. Nero uses only front/wrist
RGB observations and retains its recorded physical gripper units.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import msgpack
import numpy as np
from PIL import Image
import torch

# A directly executed script receives ``scripts/`` rather than the XR-1 root
# on sys.path.  Keep the documented command independent of caller PYTHONPATH.
XR1_ROOT = Path(__file__).resolve().parents[1]
if str(XR1_ROOT) not in sys.path:
    sys.path.insert(0, str(XR1_ROOT))

from mibot.utils.io import ACTION_EPS, compose_piper_left_state, compose_state, denormalize_action, resize_image


LOGGER = logging.getLogger("xr1.openpi_server")


def _pack_array(value: Any) -> Any:
    """Encode NumPy values exactly like ``openpi_client.msgpack_numpy``."""
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_array(value: dict) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


Packer = functools.partial(msgpack.Packer, default=_pack_array)


def unpackb(payload: bytes) -> Any:
    return msgpack.unpackb(payload, object_hook=_unpack_array)


def image_to_pil(value: Any, name: str) -> Image.Image:
    """Validate an OpenPI RGB uint8 CHW/HWC image without channel swapping."""
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"images.{name} must be rank 3, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"images.{name} must be RGB CHW or HWC, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"images.{name} must have dtype uint8, got {image.dtype}")
    return Image.fromarray(np.ascontiguousarray(image), mode="RGB")


def piper_delta_to_absolute(delta: np.ndarray, state14: np.ndarray) -> np.ndarray:
    """Map XR-1's packed 60-D deltas to absolute OpenPI-style 14-D joints.

    Every horizon element is relative to ``state14`` at request time.  This is
    deliberately a broadcasted addition, not a cumulative sum.
    """
    delta = np.asarray(delta, dtype=np.float32)
    state14 = np.asarray(state14, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[1] != 60:
        raise ValueError(f"expected XR-1 delta shape [T, 60], got {delta.shape}")
    if state14.shape != (14,):
        raise ValueError(f"expected OpenPI state shape [14], got {state14.shape}")
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(state14)):
        raise ValueError("state/actions contain NaN or Inf")

    actions = np.empty((delta.shape[0], 14), dtype=np.float32)
    actions[:, :6] = state14[None, :6] + delta[:, :6]
    actions[:, 6] = state14[6] + delta[:, 6]
    actions[:, 7:13] = state14[None, 7:13] + delta[:, 8:14]
    actions[:, 13] = state14[13] + delta[:, 14]
    return actions


def piper_absolute_to_normalized_prefix(prefix, state14, mean, std):
    """Re-anchor future absolute targets to THIS request's state, then normalize.

    History of already executed targets is not a future RTC prefix. The input
    uses the same absolute 14-D/radian/normalized-gripper schema as actions.
    """
    prefix = np.asarray(prefix, dtype=np.float32)
    state14 = np.asarray(state14, dtype=np.float32)
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if prefix.ndim != 2 or prefix.shape[1] != 14 or state14.shape != (14,):
        raise ValueError("RTC prefix/state must have shapes [K,14] and [14]")
    if mean.ndim != 2 or mean.shape[1] != 60 or std.shape != mean.shape:
        raise ValueError("RTC statistics must have matching [T,60] shapes")
    if not 0 <= len(prefix) <= 6 or len(prefix) >= len(mean):
        raise ValueError("RTC prefix must contain 0..6 steps and leave a generated suffix")
    if not all(np.isfinite(x).all() for x in (prefix, state14, mean, std)) or np.any(std < 0):
        raise ValueError("RTC prefix/state/stats must be finite with nonnegative std")
    packed = np.zeros((len(prefix), 60), dtype=np.float32)
    packed[:, :7] = prefix[:, :7] - state14[None, :7]
    packed[:, 8:15] = prefix[:, 7:] - state14[None, 7:]
    return (packed - mean[:len(prefix)]) / (std[:len(prefix)] + ACTION_EPS)


def nero_delta_to_absolute(delta: np.ndarray, state8: np.ndarray) -> np.ndarray:
    """Restore all seven Nero joints and the physical gripper from chunk deltas."""
    delta, state8 = np.asarray(delta, dtype=np.float32), np.asarray(state8, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[1] != 60 or state8.shape != (8,):
        raise ValueError("Nero requires delta [T,60] and state [8]")
    if not np.isfinite(delta).all() or not np.isfinite(state8).all():
        raise ValueError("state/actions contain NaN or Inf")
    return delta[:, :8] + state8[None, :]


class XR1OpenPIPolicy:
    """XR-1 inference wrapper implementing OpenPI's ``infer`` contract."""

    CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __init__(
        self,
        model_dir: str,
        device: str,
        processor_path: str,
        default_prompt: str | None,
        checkpoint: str = "last.ckpt",
    ) -> None:
        from mibot.server.deploy import load_model, load_stats

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        self.cfg, self.model = load_model(
            model_dir,
            str(self.device),
            checkpoint=checkpoint,
        )
        data_cfg = self.cfg.data.params.train_datasets
        action_layout = str(data_cfg.get("action_layout", "ee"))
        if action_layout not in {"piper_joint", "piper_left_joint", "nero_joint"}:
            raise ValueError(f"OpenPI serving does not support action_layout={action_layout!r}")
        self.action_layout = action_layout
        self.action_dim = 8 if action_layout == "nero_joint" else 14
        self.CAMERA_KEYS = ("front", "wrist") if action_layout == "nero_joint" else type(self).CAMERA_KEYS
        if action_layout == "piper_left_joint":
            self.action_dim = 7
            self.CAMERA_KEYS = type(self).CAMERA_KEYS[:2]

        self.mean, self.std, self.q01, self.q99, self.action_mask = load_stats(
            self.cfg, str(self.device)
        )
        self.state_valid = self.q99 > self.q01
        # M2: gripper predicted as an absolute target (see JsonDataset.gripper_absolute).
        self.gripper_absolute = bool(data_cfg.get("gripper_absolute", False))
        self.default_prompt = default_prompt
        self.processor = self._build_processor(processor_path)

        self.metadata = {
            "model": "Xiaomi-Robotics-1",
            "protocol": "openpi-websocket-v1",
            "action_horizon": int(self.mean.shape[0]),
            "action_dim": self.action_dim,
            "action_layout": self.action_layout,
            "camera_keys": list(self.CAMERA_KEYS),
            "action_semantics": "absolute_joint",
            "rtc": {"mode": "native_future_action_prefix", "max_prefix_steps": 6},
            "state_layout": "joint7,gripper" if action_layout == "nero_joint" else "left_joint6,left_gripper,right_joint6,right_gripper",
        }

        if action_layout == "nero_joint":
            self.metadata["rtc"] = {"mode": "disabled", "max_prefix_steps": 0}
            self.metadata["gripper_range"] = [0.0, 0.1]
            self.metadata["gripper_mode"] = "absolute" if self.gripper_absolute else "delta"
        elif action_layout == "piper_left_joint":
            self.metadata.update(state_layout="left_joint6,left_gripper",
                                 gripper_range=[0.0, 1.0], rtc={"mode": "disabled", "max_prefix_steps": 0})

    @staticmethod
    def _build_processor(processor_path: str):
        from transformers import AutoProcessor

        from mibot.models.VLM.qwen3vl import ACTION_START_ID, SCORE_ID, STATE_ID

        special_tokens = {"score": "<score>", "state": "<state>"}
        special_tokens.update({f"a_{index}": f"<a_{index}>" for index in range(60)})
        processor = AutoProcessor.from_pretrained(
            processor_path,
            use_fast=True,
            extra_special_tokens=special_tokens,
        )
        processor.tokenizer.padding_side = "right"
        actual = processor.tokenizer.convert_tokens_to_ids(
            ["<score>", "<state>", "<a_0>", "<a_59>"]
        )
        expected = [SCORE_ID, STATE_ID, ACTION_START_ID, ACTION_START_ID + 59]
        if actual != expected:
            raise ValueError(
                f"unexpected processor special-token ids {actual}; expected {expected}"
            )
        return processor

    @staticmethod
    def _messages(instruction: str, images: list[Image.Image], action_layout="piper_joint") -> list[dict[str, Any]]:
        if action_layout == "nero_joint":
            front, wrist = images
            # Match the two-image training prompt; no invented third view.
            return [
                {"role": "user", "content": [
                    {"type": "image", "image": front}, {"type": "text", "text": "\n"},
                    {"type": "image", "image": wrist},
                    {"type": "text", "text": "\n" + instruction + " /no_cot"},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
            ]
        ego, left = images[:2]
        right_view = [] if action_layout == "piper_left_joint" else [
            {"type": "text", "text": "\n# Right-Wrist View\n"},
            {"type": "image", "image": images[2]},
        ]
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The following observations are captured from multiple views.\n# Ego View\n",
                    },
                    {"type": "image", "image": ego},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": left},
                    *right_view,
                    {
                        "type": "text",
                        "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot",
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    def _prepare(self, obs: dict[str, Any]) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        if not isinstance(obs, dict):
            raise TypeError(f"observation must be a dict, got {type(obs).__name__}")
        state14 = np.asarray(obs.get("state"), dtype=np.float32)
        layout = getattr(self, "action_layout", "piper_joint")
        dimension = {"nero_joint": 8, "piper_left_joint": 7, "piper_joint": 14}[layout]
        if state14.shape != (dimension,):
            raise ValueError(f"observation state must have shape [{dimension}], got {state14.shape}")
        if layout == "nero_joint" and "rtc_prefix" in obs:
            raise ValueError("RTC prefix is disabled for this Nero service")
        if layout == "piper_left_joint" and "rtc_prefix" in obs:
            raise ValueError("RTC prefix is disabled for single-arm Piper")
        if not np.all(np.isfinite(state14)):
            raise ValueError("observation state contains NaN or Inf")

        image_dict = obs.get("images")
        if not isinstance(image_dict, dict):
            raise ValueError("observation must contain an images dict")
        images = []
        camera_keys = self.CAMERA_KEYS
        if layout == "nero_joint":
            camera_keys = ("front", "wrist")
            if not all(k in image_dict for k in camera_keys):
                aliases = ("cam_high", "cam_left_wrist")
                if all(k in image_dict for k in aliases):
                    camera_keys = aliases
        for name in camera_keys:
            if name not in image_dict:
                raise KeyError(f"observation images is missing {name!r}")
            image = image_to_pil(image_dict[name], name)
            images.append(resize_image(image, factor=32, max_pixels=160000))

        prompt = obs.get("prompt") or self.default_prompt
        self.last_prompt = prompt.strip() if isinstance(prompt, str) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("observation prompt is empty and no --default-prompt was set")

        payload = self.processor.apply_chat_template(
            [self._messages(prompt.strip(), images, layout) if layout != "piper_joint"
             else self._messages(prompt.strip(), images)],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={"do_resize": False},
        )
        if layout == "nero_joint":
            packed_state = np.zeros((1, 60), dtype=np.float32)
            packed_state[0, :8] = state14
        elif layout == "piper_left_joint":
            packed_state = compose_piper_left_state(state14[:6], state14[6])
        else:
            packed_state = compose_state(
                left_joint=state14[:6], left_gripper=state14[6:7],
                right_joint=state14[7:13], right_gripper=state14[13:14],
            )
        payload = dict(payload)
        payload["state"] = torch.from_numpy(packed_state)[None]
        return payload, state14

    @torch.no_grad()
    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        batch, state14 = self._prepare(obs)
        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        batch_size = int(batch["input_ids"].shape[0])
        mask = self.action_mask.unsqueeze(0).expand(batch_size, -1, -1)
        batch["action_mask"] = mask
        batch["action"] = torch.zeros(
            (batch_size, *self.mean.shape),
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )

        rtc_prefix = None
        if "rtc_prefix" in obs:
            rtc_prefix = np.asarray(obs["rtc_prefix"], dtype=np.float32)
            normalized = piper_absolute_to_normalized_prefix(
                rtc_prefix, state14, self.mean.detach().float().cpu().numpy(),
                self.std.detach().float().cpu().numpy(),
            )
            length = len(rtc_prefix)
            batch["prefix_length"] = length
            batch["action"][:, :length] = torch.as_tensor(
                normalized, device=self.device, dtype=batch["action"].dtype,
            )[None]

        state = batch["state"]
        normalized_state = torch.zeros_like(state)
        valid = self.state_valid
        normalized_state[..., valid[0]] = (
            2.0
            * (state[..., valid[0]] - self.q01[..., valid[0]])
            / (self.q99[..., valid[0]] - self.q01[..., valid[0]] + ACTION_EPS)
            - 1.0
        )
        batch["state"] = normalized_state.clamp(-1.0, 1.0)

        delta = self.model.generate(batch)
        delta = denormalize_action(delta * mask, self.mean, self.std) * mask
        delta = delta[0].float().cpu().numpy()
        recover = nero_delta_to_absolute if getattr(self, "action_layout", "piper_joint") == "nero_joint" else piper_delta_to_absolute
        if getattr(self, "action_layout", "piper_joint") == "piper_left_joint":
            actions = state14[None, :] + delta[:, :7]
        else:
            actions = recover(delta, state14)
        if getattr(self, "gripper_absolute", False) and getattr(self, "action_layout", "") == "nero_joint":
            actions[:, 7] = delta[:, 7]
        result = {"actions": actions, "prompt_used": getattr(self, "last_prompt", None)}
        if rtc_prefix is not None:
            # Keep committed targets bit-exact despite BF16 conditioning roundoff.
            actions[:len(rtc_prefix)] = rtc_prefix
            result["rtc_prefix_length"] = len(rtc_prefix)
        return result


class OpenPIWebsocketServer:
    def __init__(self, policy: XR1OpenPIPolicy, host: str, port: int) -> None:
        self.policy = policy
        self.host = host
        self.port = port

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        try:
            import websockets.asyncio.server as websocket_server
        except ImportError as exc:
            raise RuntimeError(
                "websockets is required; install assets/stage2_requirements.txt"
            ) from exc

        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            LOGGER.info("XR-1 OpenPI-compatible server listening on ws://%s:%d", self.host, self.port)
            await server.serve_forever()

    async def _handler(self, websocket) -> None:
        import websockets

        LOGGER.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self.policy.metadata))
        previous_total = None
        while True:
            try:
                started = time.monotonic()
                obs = unpackb(await websocket.recv())
                infer_started = time.monotonic()
                result = self.policy.infer(obs)
                infer_ms = (time.monotonic() - infer_started) * 1000.0
                result["server_timing"] = {"infer_ms": infer_ms}
                if previous_total is not None:
                    result["server_timing"]["prev_total_ms"] = previous_total * 1000.0
                await websocket.send(packer.pack(result))
                previous_total = time.monotonic() - started
            except websockets.ConnectionClosed:
                LOGGER.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                LOGGER.exception("Inference request failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error")
                raise

    @staticmethod
    def _health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve XR-1 with the OpenPI WebsocketClientPolicy protocol"
    )
    parser.add_argument("--model", "--model-dir", dest="model_dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default="last.ckpt",
        help="Checkpoint directory under --model (default: last.ckpt).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5695)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--processor", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--default-prompt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = XR1OpenPIPolicy(
        model_dir=args.model_dir,
        device=args.device,
        processor_path=args.processor,
        default_prompt=args.default_prompt,
        checkpoint=args.checkpoint,
    )
    try:
        OpenPIWebsocketServer(policy, args.host, args.port).serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("XR-1 OpenPI-compatible server stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
