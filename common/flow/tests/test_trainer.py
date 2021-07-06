"""
    test trainer classes and functions
"""

import os.path as op
import pytest

import torch.nn as nn

from common.flow.trainer import Trainer
from common.utils.misc_utils import Params
from stages.utils.dataset_utils import Dataset


@pytest.fixture()
def params():
    """
    """
    json_path = op.join('./', \
        'runset.json')
    return Params(json_path)

@pytest.fixture()
def dataloader(params):
    """
    """
    return Dataset(params, './').dataloader()


def test_trainer_init(params, dataloader):
    """
    Test Trainer class init
    """
    trainer = Trainer(params, \
        run_dir='./')
    assert isinstance(trainer._model, nn.Module)
    print(trainer._optimizer)
    trainloader, _ = dataloader
    images, _ = next(iter(trainloader))
    trainer.net_summary(images)