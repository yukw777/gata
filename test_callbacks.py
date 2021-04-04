from pytorch_lightning import Trainer

from train_gata import GATADoubleDQN
from callbacks import RLEarlyStopping


def test_rl_early_stopping():
    gata_double_dqn = GATADoubleDQN()
    trainer = Trainer()
    es = RLEarlyStopping("val_monitor", "train_monitor", 0.95, patience=3)

    # if val score and train score are all below the threshold 0.95, don't stop
    trainer.callback_metrics = {"val_monitor": 0.1, "train_monitor": 0.1}
    es._run_early_stopping_check(trainer, gata_double_dqn)
    assert not trainer.should_stop

    # if val score is 1.0 and train score is above the threshold, stop
    trainer.callback_metrics = {"val_monitor": 1.0, "train_monitor": 0.95}
    trainer.current_epoch = 1
    es._run_early_stopping_check(trainer, gata_double_dqn)
    assert trainer.should_stop
    assert es.stopped_epoch == 1

    # if train score is above the threshold for `patience` times,
    # but val score is not 1.0, stop
    trainer.should_stop = False
    es.wait_count = 0
    es.stopped_epoch = 0
    for i in range(3):
        trainer.current_epoch = i
        trainer.callback_metrics = {"val_monitor": 0.9, "train_monitor": 0.95}
        es._run_early_stopping_check(trainer, gata_double_dqn)
        if i == 2:
            assert trainer.should_stop
            assert es.stopped_epoch == 2
        else:
            assert not trainer.should_stop
            assert es.stopped_epoch == 0
