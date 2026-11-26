# Copyright (C) 2026 Xiaomi Corporation.
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple

from mibot.utils.compat import patch_torch_pytree

patch_torch_pytree()

if os.environ.get("XR1_DEVICE_TYPE", "cuda").lower() == "npu":
    import torch
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    _npu_capability = torch.cuda.get_device_capability

    def _get_device_capability(device=None):
        return _npu_capability(device) or (8, 0)

    torch.cuda.get_device_capability = _get_device_capability

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, ModelSummary, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from mmengine import Config, DATASETS
from omegaconf import DictConfig
from mibot.models import MIMODEL

from mibot.utils.cfg_utils import helper

import mibot.data

warnings.filterwarnings("ignore")


class GlobalStepTQDMProgressBar(TQDMProgressBar):
    """Display cumulative optimizer steps against ``Trainer.max_steps``.

    Lightning's default bar is epoch/dataloader based.  It can therefore show
    an unknown total for step-bounded jobs and becomes misleading when a run
    resumes with a different batch size.  This bar is display-only: stopping,
    checkpointing, data loading, and optimization still belong to Trainer.
    """

    @property
    def total_train_batches(self) -> int | float:
        max_steps = int(getattr(self.trainer, "max_steps", -1))
        return max_steps if max_steps > 0 else super().total_train_batches

    def on_train_start(self, trainer: Trainer, *_: Any) -> None:
        super().on_train_start(trainer)
        self._reset_to_global_step(trainer)
        if trainer.is_global_zero:
            logging.getLogger(__name__).info(
                "Training global-step progress: start=%d, max=%d, remaining=%d",
                trainer.global_step,
                trainer.max_steps,
                max(0, trainer.max_steps - trainer.global_step),
            )

    def _reset_to_global_step(self, trainer: Trainer) -> None:
        total = self.total_train_batches
        self.train_progress_bar.reset(total=total)
        step = min(int(trainer.global_step), int(total))
        self.train_progress_bar.initial = step
        self.train_progress_bar.n = step
        self.train_progress_bar.set_description(self.train_description)
        self.train_progress_bar.refresh()

    def on_train_epoch_start(self, trainer: Trainer, *_: Any) -> None:
        if self._leave:
            self.train_progress_bar = self.init_train_tqdm()
        self._reset_to_global_step(trainer)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        total = self.train_progress_bar.total
        step = int(trainer.global_step)
        if total is not None:
            step = min(step, int(total))
        if self._should_update(step, total):
            self.train_progress_bar.n = step
            self.train_progress_bar.refresh()
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))


class InterconnectBandwidthCallback(Callback):
    """Benchmark the initialized NCCL group once and persist it to W&B.

    This reports collective effective bandwidth, not a theoretical PCIe link
    rate. Every rank participates, while only global rank zero writes W&B.
    """

    def on_train_start(self, trainer: Trainer, *_: Any) -> None:
        import torch
        import torch.distributed as dist

        if os.environ.get("XR1_DEVICE_TYPE", "cuda").lower() != "cuda":
            raise RuntimeError("XR1_LOG_INTERCONNECT_BW is only supported on CUDA")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("XR1_LOG_INTERCONNECT_BW requires initialized torch.distributed")
        world_size = dist.get_world_size()
        if world_size < 2:
            raise RuntimeError("XR1_LOG_INTERCONNECT_BW requires at least two ranks")

        size_mib = int(os.environ.get("XR1_INTERCONNECT_BW_MIB", "64"))
        warmup = int(os.environ.get("XR1_INTERCONNECT_BW_WARMUP", "5"))
        repeats = int(os.environ.get("XR1_INTERCONNECT_BW_REPEATS", "20"))
        if size_mib <= 0 or warmup < 0 or repeats <= 0:
            raise ValueError("interconnect MiB/repeat settings must be positive (warmup may be zero)")

        size_bytes = size_mib * 1024 * 1024
        tensor = torch.zeros(
            size_bytes // 2,
            dtype=torch.float16,
            device=torch.cuda.current_device(),
        )
        dist.barrier()
        for _ in range(warmup):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        for _ in range(repeats):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=tensor.device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        elapsed = float(elapsed_tensor.item())
        algorithm_gbps = size_bytes * repeats / elapsed / 1e9
        bus_gbps = algorithm_gbps * (2.0 * (world_size - 1) / world_size)
        latency_ms = elapsed * 1000.0 / repeats
        del tensor, elapsed_tensor

        if trainer.is_global_zero:
            metrics = {
                "system/interconnect/all_reduce_algorithm_GBps": algorithm_gbps,
                "system/interconnect/all_reduce_bus_GBps": bus_gbps,
                "system/interconnect/all_reduce_latency_ms": latency_ms,
            }
            experiment = trainer.logger.experiment
            experiment.log(metrics, step=int(trainer.global_step))
            for name, value in metrics.items():
                experiment.summary[name] = value
            experiment.config.update(
                {
                    "interconnect_benchmark/world_size": world_size,
                    "interconnect_benchmark/tensor_MiB": size_mib,
                    "interconnect_benchmark/warmup": warmup,
                    "interconnect_benchmark/repeats": repeats,
                },
                allow_val_change=True,
            )
            logging.getLogger(__name__).info(
                "NCCL all-reduce benchmark: world=%d, tensor=%d MiB, "
                "algorithm=%.3f GB/s, bus=%.3f GB/s, latency=%.3f ms",
                world_size,
                size_mib,
                algorithm_gbps,
                bus_gbps,
                latency_ms,
            )


def save_failure_diagnostics(
    root: str, trainer: Trainer, model: LightningModule, error: Exception
) -> None:
    """Persist the last batch metadata per failed rank without logging to stdout."""
    try:
        rank = int(getattr(trainer, "global_rank", int(os.environ.get("RANK", 0))))
        step = int(getattr(trainer, "global_step", -1))
        output_dir = Path(root) / "diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rank": rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
            "pid": os.getpid(),
            "global_step": step,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "last_batch": getattr(model, "_last_diagnostic_batch", None),
        }
        filename = f"failure-step{step}-rank{rank}-pid{os.getpid()}.json"
        temporary = output_dir / f".{filename}.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output_dir / filename)
    except Exception:
        # Diagnostics must never hide or replace the original training failure.
        pass


def prepare(cfg: Dict[str, Any]) -> Tuple[Config, LightningDataModule, LightningModule, List[Any]]:
    cfg: Config = helper(cfg)
    datamodule: LightningDataModule = DATASETS.build(cfg.data)
    model: LightningModule = MIMODEL.build(cfg.model)
    callbacks = [GlobalStepTQDMProgressBar(), ModelSummary(max_depth=2)]
    if os.environ.get("XR1_LOG_INTERCONNECT_BW", "0").lower() in {"1", "true", "yes"}:
        callbacks.append(InterconnectBandwidthCallback())
    save_interval = cfg.trainer.pop("save_interval", 10000)
    disable_checkpoints = os.environ.get("XR1_DISABLE_CHECKPOINTS", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    save_top_k = int(os.environ.get("XR1_SAVE_TOP_K", "1"))
    if save_top_k not in {-1, 0, 1}:
        raise ValueError(
            "XR1_SAVE_TOP_K must be -1 (keep all), 0 (periodic saves disabled), "
            "or 1 (keep only the latest periodic checkpoint) when monitor is unset"
        )
    if disable_checkpoints:
        cfg.trainer["enable_checkpointing"] = False
    elif os.environ.get("XR1_SAVE_EPOCHS"):
        from mibot.utils.epoch_checkpoint import SelectedEpochCheckpoint
        callbacks.append(SelectedEpochCheckpoint(
            os.environ["XR1_SAVE_EPOCHS"], dirpath=cfg.trainer.default_root_dir
        ))
    else:
        callbacks.append(
            ModelCheckpoint(
                save_top_k=save_top_k,
                save_last="link",
                every_n_train_steps=save_interval,
                dirpath=cfg.trainer.default_root_dir,
                filename="epoch{epoch}-step{step}",
                auto_insert_metric_name=False,
                enable_version_counter=False,
            )
        )
    cfg.trainer["callbacks"] = callbacks

    logger: List[Any] = [
        WandbLogger(
            **({"save_dir": os.environ["XR1_WANDB_SAVE_DIR"]} if os.environ.get("XR1_WANDB_SAVE_DIR") else {}),
            project=cfg.trainer.pop("project"),
            name=cfg.trainer.pop("exp_name"),
            config=cfg,
        )
    ]

    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    return (cfg, datamodule, model, logger)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg, datamodule, model, logger = prepare(cfg)
    ckpt_path: Optional[str] = cfg.trainer.pop("ckpt_path", None)
    diagnostic_root = str(cfg.trainer.default_root_dir)
    trainer = Trainer(**cfg.trainer, logger=logger)
    try:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    except Exception as error:
        save_failure_diagnostics(diagnostic_root, trainer, model, error)
        raise


if __name__ == "__main__":
    main()
