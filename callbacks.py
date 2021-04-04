import torch
import pytorch_lightning as pl
import wandb

from typing import Optional, Dict, Any
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger


class EqualModelCheckpoint(ModelCheckpoint):
    def check_monitor_top_k(
        self, trainer, current: Optional[torch.Tensor] = None
    ) -> bool:
        """
        Use torch.le, torch.ge instead of torch.lt, torch.gt
        """
        if current is None:
            return False

        if self.save_top_k == -1:
            return True

        less_than_k_models = len(self.best_k_models) < self.save_top_k  # type: ignore
        if less_than_k_models:
            return True

        if not isinstance(current, torch.Tensor):
            rank_zero_warn(
                f"{current} is supposed to be a `torch.Tensor`. Saving checkpoint may "
                "not work correctly."
                f" HINT: check the value of {self.monitor} in your validation loop",
                RuntimeWarning,
            )
            current = torch.tensor(current)

        monitor_op = {"min": torch.le, "max": torch.ge}[self.mode]
        should_update_best_and_save = monitor_op(
            current, self.best_k_models[self.kth_best_model_path]
        )

        # If using multiple devices, make sure all processes are unanimous
        # on the decision.
        should_update_best_and_save = (
            trainer.training_type_plugin.reduce_boolean_decision(
                should_update_best_and_save
            )
        )

        return should_update_best_and_save  # type: ignore


class EqualNonZeroModelCheckpoint(EqualModelCheckpoint):
    def check_monitor_top_k(
        self, trainer, current: Optional[torch.Tensor] = None
    ) -> bool:
        """
        If current is 0, don't save
        """
        if current is not None and current == 0:
            return False
        return super().check_monitor_top_k(trainer, current=current)


class RLEarlyStopping(Callback):
    def __init__(
        self,
        val_monitor: str,
        train_monitor: str,
        threshold: float,
        patience: int = 3,
    ):
        super().__init__()
        self.val_monitor = val_monitor
        self.train_monitor = train_monitor
        self.threshold = threshold
        self.patience = patience
        self.wait_count = 0
        self.stopped_epoch = 0

    def on_save_checkpoint(
        self, trainer, pl_module, checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "patience": self.patience,
        }

    def on_load_checkpoint(self, callback_state: Dict[str, Any]):
        self.wait_count = callback_state["wait_count"]
        self.stopped_epoch = callback_state["stopped_epoch"]
        self.patience = callback_state["patience"]

    def on_validation_end(self, trainer, pl_module):
        if trainer.running_sanity_check:
            return

        self._run_early_stopping_check(trainer, pl_module)

    def _run_early_stopping_check(self, trainer, pl_module):
        """
        Stop if the validation score is 1.0 and the train score is greater than
        or equal to the threshold. Also, if the traing score is above the threshold for
        `self.patience` times, stop.
        """
        if trainer.fast_dev_run:  # disable early_stopping with fast_dev_run
            return  # short circuit if metric not present

        logs = trainer.callback_metrics

        val_current = logs.get(self.val_monitor)
        train_current = logs.get(self.train_monitor)
        if val_current == 1.0 and train_current >= self.threshold:
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True

        if train_current >= self.threshold:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True
        else:
            self.wait_count = 0

        # stop every ddp process if any world process decides to stop
        trainer.should_stop = trainer.training_type_plugin.reduce_boolean_decision(
            trainer.should_stop
        )


class WandbSaveCallback(Callback):
    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> dict:
        if isinstance(trainer.logger, WandbLogger):
            wandb.save(f"gata/{trainer.logger.version}/checkpoints/*.ckpt")
        return {}
