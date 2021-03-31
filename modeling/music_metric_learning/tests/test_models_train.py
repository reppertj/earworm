import pytest
import torch
from music_metric_learning.losses.cross_entropy import MoCoCrossEntropyLoss
from music_metric_learning.models.train_music_learner import MusicMetricLearner, DEFAULT_HPARAMS
from pytorch_lightning import Trainer

"""These tests only verify that various training configurations run;
they don't check for correctness or performance.
"""

def test_trainer_config(tmpdir, datamodule):
    hparams = DEFAULT_HPARAMS
    hparams["batch_size"] = 16
    hparams["m_per_class"] = 2
    hparams["checkpoint_path"] = str(tmpdir)
    learner = MusicMetricLearner(datamodule=datamodule, conf=hparams)
    trainer = Trainer(gpus=0, fast_dev_run=True)
    trainer.fit(learner)

def test_track_loss_only(tmpdir, datamodule):
    hparams = DEFAULT_HPARAMS
    hparams["batch_size"] = 16
    hparams["m_per_class"] = 2
    hparams["checkpoint_path"] = str(tmpdir)
    hparams["mode"] = "track"
    learner = MusicMetricLearner(datamodule=datamodule, conf=hparams)
    trainer = Trainer(gpus=0, fast_dev_run=True)
    trainer.fit(learner)

def test_train_with_moco_loss(tmpdir, datamodule):
    hparams = DEFAULT_HPARAMS
    hparams["batch_size"] = 16
    hparams["m_per_class"] = 2
    hparams["checkpoint_path"] = str(tmpdir)    
    hparams["loss_func"] = "moco"
    hparams["loss_params"]["temperature"] = 0.7
    learner = MusicMetricLearner(datamodule=datamodule, conf=hparams)
    assert isinstance(learner.criterion, MoCoCrossEntropyLoss)
    assert learner.criterion.temperature == 0.7
    trainer = Trainer(gpus=0, fast_dev_run=True)
    trainer.fit(learner)

def test_inception_encoder(tmpdir, datamodule):
    hparams = DEFAULT_HPARAMS
    hparams["batch_size"] = 16
    hparams["m_per_class"] = 2
    hparams["checkpoint_path"] = str(tmpdir)
    hparams["encoder"] = "inception"
    learner = MusicMetricLearner(datamodule=datamodule, conf=hparams)
    trainer = Trainer(gpus=0, fast_dev_run=True)
    trainer.fit(learner)    
    

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision(tmpdir, datamodule):
    hparams = DEFAULT_HPARAMS
    hparams["batch_size"] = 16
    hparams["m_per_class"] = 2
    hparams["checkpoint_path"] = str(tmpdir)
    learner = MusicMetricLearner(datamodule=datamodule, conf=hparams)
    trainer = Trainer(gpus=1, precision=16, fast_dev_run=True)
    trainer.fit(learner)
