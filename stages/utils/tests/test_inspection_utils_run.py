"""
    test inspection utilities' running & output, no unit-testing
"""

import os.path as op
import pytest

import torch.nn as nn

from stages.utils.inspection_utils import InspectionTrainer, batch_loader
from stages.utils.dataset_utils import Dataset
from common.utils import Params


@pytest.fixture()
def params():
    """
    """
    json_path = op.join('../../tests/directory/02_inspection/', \
        'runset.json')
    return Params(json_path)

@pytest.fixture()
def dataloader():
    """
    """
    return Dataset('../../tests/directory/02_inspection/').dataloader()


def test_run_trainer_init(params, dataloader):
    """
    Test Trainer class initialization
    """
    trainer = InspectionTrainer(params, \
        run_dir='../../tests/directory/02_inspection/')
    assert isinstance(trainer._model, nn.Module)
    trainloader, _ = dataloader
    images, _ = next(iter(trainloader))
    trainer.net_summary(images)

def test_run_trainer_train(params, dataloader):
    """
    """
    trainer = InspectionTrainer(params, \
        run_dir='../../tests/directory/02_inspection/')
    trainloader, _ = dataloader
    trainer.train(trainloader, iterations=5)

def test_run_trainer_eval(params, dataloader):
    """
    """
    trainer = InspectionTrainer(params, \
        run_dir='../../tests/directory/02_inspection/')
    _, valloader = dataloader
    # test eval current model
    trainer.eval(valloader, iterations=1)
    # test eval pretrained model
    trainer.eval(valloader, iterations=3, \
        restore_file='test_train_01142021')
    
    