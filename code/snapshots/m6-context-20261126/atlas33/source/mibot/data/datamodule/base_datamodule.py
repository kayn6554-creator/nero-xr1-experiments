# Copyright (C) 2026 Xiaomi Corporation.
from copy import deepcopy
import os

from lightning import LightningDataModule
from mmengine import Config, DATASETS
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from mibot.data.collate.custom_collate import CustomCollate
from mibot.data.datasets.json_dataset import JsonDataset


@DATASETS.register_module()
class BaseDataModule(LightningDataModule):
    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params: Config = params
        self.batch_size: int = params.train_datasets.get("batch_size", 16)
        self.collate_fn = CustomCollate()
        self.train_set = None

    def setup(self, stage=None) -> None:
        if stage in (None, "fit") and self.train_set is None:
            self.train_set = JsonDataset(deepcopy(self.params))
            if "replay_dataset" in self.params:
                if not self.params.get("finite_epoch", False):
                    raise ValueError("replay_dataset requires finite_epoch=true")
                replay_params = deepcopy(self.params)
                replay_params.train_datasets = replay_params.pop("replay_dataset")
                replay = JsonDataset(replay_params)
                if set(self.train_set.files).intersection(replay.files):
                    raise ValueError("Training and replay datasets contain overlapping episodes")
                self.train_set = ConcatDataset([self.train_set, replay])

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            self.setup("fit")
        sampler = DistributedSampler(self.train_set, shuffle=True, seed=42)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=int(os.environ.get("XR1_NUM_WORKERS", 8)),
            prefetch_factor=4,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )