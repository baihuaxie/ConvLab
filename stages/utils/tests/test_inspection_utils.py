"""
    unit-test inspection utilities
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


def test_batch_loader(dataloader):
    """
    """
    trainloader, _ = dataloader
    batch_dl = batch_loader(trainloader, length=6, samples=3)
    batch_iter = iter(batch_dl)
    data, labels = next(batch_iter)
    for idx in range(5):
        data_idx, labels_idx = next(batch_iter)
        assert (data == data_idx).all()
        assert (labels == labels_idx).all()
    
    