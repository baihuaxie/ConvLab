"""
    test inspection utilities' running & output, no unit-testing
"""

import os.path as op
import pytest

import torch.nn as nn

from stages.utils.inspection_utils import InspectionTrainer
from stages.utils.dataset_utils import Dataset
from common.utils import Params


@pytest.fixture
def run_dir():
    return '../../tests/02_inspection/'

@pytest.fixture()
def params(run_dir):
    """
    """
    json_path = op.join(run_dir, 'runset.json')
    return Params(json_path)

@pytest.fixture()
def dataloader(run_dir, params):
    """
    """
    return Dataset(params=params, run_dir=run_dir).dataloader()


def test_run_trainer_init(run_dir, params, dataloader):
    """
    Test Trainer class initialization
    """
    trainer = InspectionTrainer(params, run_dir=run_dir)
    assert isinstance(trainer._model, nn.Module)
    trainloader, _ = dataloader
    images, _ = next(iter(trainloader))
    trainer.net_summary(images)

def test_run_trainer_train(run_dir, params, dataloader):
    """
    """
    trainer = InspectionTrainer(params, run_dir=run_dir)
    trainloader, _ = dataloader
    trainer.train(trainloader, iterations=5)

def test_run_trainer_eval(run_dir, params, dataloader):
    """
    """
    trainer = InspectionTrainer(params, run_dir=run_dir)
    _, valloader = dataloader
    # test eval current model
    trainer.eval(valloader, iterations=1)
    # test eval pretrained model
    trainer.eval(valloader, iterations=3, \
        restore_file='test_train_01142021')
    
    