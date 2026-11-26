# Copyright (C) 2026 Xiaomi Corporation.
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from transformers.utils import logging

from mibot.models import MIMODEL

logger = logging.get_logger(__name__)

_DIAGNOSTIC_VALUE_KEYS = {
    "action_segments",
    "action_vlm_condition_segments",
    "cu_seq_lens_k",
    "cu_seq_lens_q",
    "image_grid_thw",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def summarize_diagnostic_batch(batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
    """Capture small, JSON-safe metadata without retaining training tensors."""
    summary: Dict[str, Any] = {"batch_idx": int(batch_idx), "tensors": {}}
    sample = batch.get("_diagnostic_sample")
    if sample is not None:
        summary["samples"] = _json_safe(sample)

    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "numel": int(value.numel()),
        }
        if key in _DIAGNOSTIC_VALUE_KEYS and value.numel() <= 512:
            tensor["values"] = value.detach().cpu().tolist()
        summary["tensors"][key] = tensor
    return summary


@MIMODEL.register_module()
class BaseRunner(LightningModule):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self._pretrained: Optional[str] = params.get("pretrained")
        self._model: Dict[str, Any] = params.get("model")
        self._optimizer: Dict[str, Any] = params.get("optimizer")
        self._scheduler: Optional[Dict[str, Any]] = params.get("scheduler")

    def configure_model(self) -> None:
        self.model = MIMODEL.build(self._model)

        prepare_for_load = getattr(self.model, "prepare_for_checkpoint_load", None)
        if prepare_for_load is not None:
            prepare_for_load()

        if self._pretrained:
            model_state_path = self.resolve_model_state_path(self._pretrained)
            ckpt = torch.load(
                model_state_path,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            state_dict = self.extract_model_state_dict(ckpt, model_state_path)
            # strict=False is used only to obtain both diagnostic lists in one
            # pass. Any mismatch is fatal; nothing is silently accepted.
            info = self.load_state_dict(state_dict, strict=False)
            if info.missing_keys or info.unexpected_keys:
                raise RuntimeError(f"Checkpoint does not match the configured model: {info}")
            logger.info(f"Loaded pretrained model from {model_state_path}: {info}")
            del state_dict, ckpt

        prepare = getattr(self.model, "prepare_posttrain_adaptation", None)
        if prepare is not None:
            prepare()

        statistics = getattr(self.model, "parameter_statistics", None)
        if statistics is not None:
            logger.info("Parameter statistics: %s", statistics())

    @staticmethod
    def resolve_model_state_path(path: str) -> Path:
        """Resolve a model-only file without consulting optimizer shards."""
        source = Path(path).expanduser().resolve()
        if source.is_file():
            return source
        if not source.is_dir():
            raise FileNotFoundError(f"pretrained checkpoint does not exist: {source}")

        candidates = [
            source / "checkpoint" / "mp_rank_00_model_states.pt",
            source / "mp_rank_00_model_states.pt",
        ]
        latest = source / "latest"
        if latest.is_file():
            tag = latest.read_text(encoding="utf-8").strip()
            if tag:
                candidates.insert(0, source / tag / "mp_rank_00_model_states.pt")
        candidates = list(dict.fromkeys(candidates))
        existing = [candidate for candidate in candidates if candidate.is_file()]
        if len(existing) != 1:
            rendered = ", ".join(str(candidate) for candidate in candidates)
            raise RuntimeError(
                f"expected exactly one DeepSpeed model-state file under {source}; "
                f"found {len(existing)}. Checked: {rendered}"
            )
        return existing[0]

    @staticmethod
    def extract_model_state_dict(ckpt: Any, path: Path) -> Dict[str, torch.Tensor]:
        if not isinstance(ckpt, dict):
            raise TypeError(f"checkpoint {path} must be a dict, got {type(ckpt).__name__}")
        if isinstance(ckpt.get("module"), dict):
            return ckpt["module"]
        if isinstance(ckpt.get("state_dict"), dict):
            return ckpt["state_dict"]
        raise KeyError(
            f"checkpoint {path} contains neither a 'module' nor a 'state_dict' mapping; "
            f"top-level keys={list(ckpt)[:30]}"
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.build_optimizer(self._optimizer, self.named_parameters())
        scheduler = self.build_scheduler(self._scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @staticmethod
    def build_optimizer(cfg: Dict[str, Any], parameters: Iterator[Tuple[str, torch.Tensor]]) -> Optimizer:
        module_params = cfg.get("params", {})
        parameters = list(parameters)

        no_decay = [
            "bias",
            "norm",
            "Norm",
            "ln",
            "Ln",
            "rotary_emb",
            "adaln",
        ]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in parameters if not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
                "weight_decay": module_params.get("weight_decay", 0.1),
            },
            {
                "params": [p for n, p in parameters if any(nd in n.lower() for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        module_type = cfg["type"]
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        optimizer_class = getattr(module, class_name)

        return optimizer_class(optimizer_grouped_parameters, **module_params)

    @staticmethod
    def build_scheduler(cfg: Optional[Dict[str, Any]], optimizer: Optimizer) -> Optional[LRScheduler]:
        if cfg is None:
            return None

        module_type = cfg["type"]
        module_params = cfg.get("params", {})
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        scheduler_class = getattr(module, class_name)

        return scheduler_class(optimizer, **module_params)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        self._last_diagnostic_batch = summarize_diagnostic_batch(batch, batch_idx)
        batch.pop("_diagnostic_sample", None)
        self.log("train/token", batch["input_ids"].shape[1])
        loss_dict = self.model(batch, return_loss=True)
        for loss_name, loss_value in loss_dict.items():
            if not torch.isfinite(loss_value.detach()).all():
                raise FloatingPointError(
                    f"non-finite training value at {loss_name}: {loss_value.detach()}"
                )
            self.log(f"train/{loss_name}", loss_value.detach(), prog_bar=loss_name == "loss")
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        return loss_dict