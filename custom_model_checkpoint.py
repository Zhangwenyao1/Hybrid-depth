import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class CustomModelCheckpoint(ModelCheckpoint):

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the training epoch."""
        if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the validation stage."""
        pass

    def _should_skip_saving_checkpoint(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        return (
            bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
            or self._last_global_step_saved == trainer.global_step  # already saved at the last step
        )

    def _should_save_on_train_epoch_end(self, trainer: "pl.Trainer") -> bool:
        return True