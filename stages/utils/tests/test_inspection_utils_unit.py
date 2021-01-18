"""
    unit-test inspection utilities
"""

import os.path as op
import pytest

from stages.utils.inspection_utils import batch_loader
from stages.utils.dataset_utils import Dataset
from common.utils import Params


@pytest.fixture()
def params():
    """
    """
    json_path = op.join('../../tests/02_inspection/', \
        'runset.json')
    return Params(json_path)

@pytest.fixture()
def dataloader(params):
    """
    """
    return Dataset(params=params, run_dir='../../tests/02_inspection/').dataloader()


@pytest.mark.parametrize("length, samples", [
    (6, 5)
])
def test_batch_loader(dataloader, length, samples):
    """
    """
    trainloader, _ = dataloader
    batch_dl = batch_loader(trainloader, length=length, samples=samples)
    batch_iter = iter(batch_dl)
    data, labels = next(batch_iter)
    
    # check number of samples
    assert data.shape[0] == samples
    assert labels.shape[0] == samples
    # check number of batches
    assert len(batch_iter) == length
    # check all batches are the same
    for idx in range(5):
        data_idx, labels_idx = next(batch_iter)
        assert (data == data_idx).all()
        assert (labels == labels_idx).all()
    
    