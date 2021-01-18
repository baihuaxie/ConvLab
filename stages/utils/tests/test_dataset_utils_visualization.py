"""
test dataset inspection utilities
"""
import os.path as op
import pytest
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
import torchvision.datasets as ds

from common.dataloader import select_n_random, fetch_dataloader
from common.utils import Params, match_dict_by_value

from stages.utils.dataset_utils import show_images, \
    show_labelled_images, get_classes, get_labels_counts


@pytest.fixture(params=[
    {
        'dataset':  'CIFAR100',
        'datadir':  '../../../data/',
        'num_classes': 100
    },
    {
        'dataset':  'Imagenette',
        'datadir':  '../../../data/Imagenette',
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

@pytest.fixture(params=[transforms.Compose([
        transforms.RandomResizedCrop(160),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
])
def transform(request):
    """
    dataset transformation
    """
    return request.param

@pytest.fixture()
def kwargs(dataset):
    """
    Get default dataset parameters from .json file
    """
    json_path = op.join('../../../common/', 'parameters.json')
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


@pytest.fixture
def classes(dataset, datadir):
    """
    """
    return get_classes(dataset, datadir)

@pytest.fixture
def samples(dataset, datadir, kwargs, transform):
    """
    Get 1 batch of random or label-balanced samples from training set

    Used to test visualization functions

    Returns a tuple of (images, labels); both are tensors
    """
    if dataset in ['CIFAR10', 'CIFAR100']:
        trainset_kwargs = {}
        valset_kwargs = {}
        return select_n_random('train', datadir, trainset_kwargs, \
            valset_kwargs, dataset, n=20)

    elif dataset in ['Imagenette', 'Imagewoof']:
        trainloader_kwargs, trainset_kwargs, valloader_kwargs, \
            valset_kwargs = kwargs
        dataloaders = fetch_dataloader(['train'], datadir, dataset, \
            trainloader_kwargs, trainset_kwargs, valloader_kwargs, \
            valset_kwargs, transform, transform, balanced=True)
        trainloader = dataloaders['train']
        # return a single batch of data only
        return next(iter(trainloader))

def test_imshow(samples):
    """
    Print images
    """
    images, _ = samples
    print(images.shape)
    # make_grid input must be BxCxHxW in shape
    img_grid = make_grid(images)
    show_images(img_grid)
    plt.show()

def test_show_labelled_images(samples, classes, dataset):
    """
    Print images along with labels from dataset
    """
    images, labels = samples
    show_labelled_images(dataset, classes, images, labels, nrows=4, ncols=8, \
        savepath='./samples/')



