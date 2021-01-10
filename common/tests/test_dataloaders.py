"""
    test dataloader functions
"""

import pytest
import os.path as op

from common.utils import Params, match_dict_by_value
from common.dataloader import fetch_dataloader, \
    select_n_random, fetch_subset_dataloader


@pytest.fixture(params=[
    {
        'dataset':  'CIFAR100',
        'datadir':  '../../data/',
        'num_classes': 100
    },
    {
        'dataset':  'Imagenette',
        'datadir':  '../../data/Imagenette',
        'num_classes': 10
    }
])
def setup(request):
    """
    Setup dataset & datadir

    Note:
    - use fixture(params=[list of parameters]) & return request.param
      syntax to enable parametrized fixtures
    - if each fixture is parametrized by multiple parameters, they must
      be grouped (e.g., as a dictionary) & only one parametrized fixture
      function should be used; otherwise, pytest treat each parameter as
      a distinct case
      - e.g., if a setup is parametrized by dataset and datadir, and there
        are two setup's to be used for the test;
        if use two fixture functions for each parameter, pytest would
        generate four different setup cases instead of two
    """
    return request.param

@pytest.fixture()
def dataset(setup):
    """
    dataset
    """
    return setup['dataset']

@pytest.fixture()
def datadir(setup):
    """
    dataset directory
    """
    return setup['datadir']

@pytest.fixture()
def dataset_num_classes(setup):
    return setup['num_classes']


@pytest.fixture()
def kwargs(dataset):
    """
    Get default dataset parameters from .json file
    """
    json_path = op.join('../', 'parameters.json')
    params = Params(json_path)

    # get dataset keyword arguments from default parameter file
    dataset_dict = match_dict_by_value(params.data, 'dataset', dataset)
    dataset = dataset_dict['dataset']
    trainloader_kwargs = dataset_dict['trainloader-kwargs']
    trainset_kwargs = dataset_dict['trainset-kwargs']
    valloader_kwargs = dataset_dict['valloader-kwargs']
    valset_kwargs = dataset_dict['valset-kwargs']

    return trainloader_kwargs, trainset_kwargs, valloader_kwargs, \
        valset_kwargs


def test_fetch_dataloaders(dataset, datadir, kwargs):
    """
    """
    trainloader_kwargs, trainset_kwargs, valloader_kwargs, \
        valset_kwargs = kwargs
    dataloaders = fetch_dataloader(['train'], datadir, dataset, \
        trainloader_kwargs, trainset_kwargs, valloader_kwargs, \
        valset_kwargs, balanced=True)


def test_fetch_subset_dataloaders(dataset, datadir, kwargs):
    """
    """
    trainloader_kwargs, trainset_kwargs, valloader_kwargs, \
        valset_kwargs = kwargs
    dataloaders = fetch_subset_dataloader(['train'], datadir,\
        dataset, trainloader_kwargs, trainset_kwargs, valloader_kwargs,\
        valset_kwargs, batchsz=trainloader_kwargs['batch_size'])


def test_select_n_random(dataset, datadir, kwargs):
    """
    Test select_n_random() function

    Note that this is a legacy function that does not work on
    ImageFolder datasets
    """
    if dataset in ['CIFAR10', 'CIFAR100']:
        num = 10
        _, trainset_kwargs, _, valset_kwargs = kwargs
        images, labels = select_n_random('train', datadir, \
            trainset_kwargs, valset_kwargs, n=num)
        assert images.shape[0] == num
        assert labels.shape[0] == num
    else:
        pass

