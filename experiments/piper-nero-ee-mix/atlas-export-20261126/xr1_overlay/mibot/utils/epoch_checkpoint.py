"""Collective Lightning checkpointing after selected one-based epochs."""
from lightning.pytorch.callbacks import ModelCheckpoint


class SelectedEpochCheckpoint(ModelCheckpoint):
    def __init__(self, epochs, dirpath):
        self.selected_epochs = frozenset(int(x) for x in epochs.split(","))
        if not self.selected_epochs or min(self.selected_epochs) < 1:
            raise ValueError("XR1_SAVE_EPOCHS must contain positive one-based epochs")
        super().__init__(
            dirpath=dirpath, save_top_k=-1, save_last=False,
            every_n_train_steps=0, every_n_epochs=1,
            save_on_train_epoch_end=True,
            filename="epoch{completed_epoch:.0f}-step{step}",
            auto_insert_metric_name=False, enable_version_counter=False,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 in self.selected_epochs:
            super().on_train_epoch_end(trainer, pl_module)

    def _monitor_candidates(self, trainer):
        candidates = super()._monitor_candidates(trainer)
        candidates["completed_epoch"] = trainer.current_epoch + 1
        return candidates
