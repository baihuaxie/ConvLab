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

from common.data_loader import fetch_dataloader, select_n_random
from common.utils import Params, match_dict_by_value

from stages.utils.dataset_utils import show_images, \
    show_labelled_images, get_classes, get_labels_counts


@pytest.fixture(params=[
    {
        'dataset':  'CIFAR100',
        'datadir':  '../../../data/'
    },
    {
        'dataset':  'Imagenette',
        'datadir':  '../../../data/Imagenette'
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

@pytest.fixture(params=[transforms.Compose([transforms.ToTensor()])])
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
def dataloaders(dataset, datadir, transform, kwargs):
    """
    Fetch and load dataset into dataloader object
    """
    print(dataset)

    trainloader_kwargs, trainset_kwargs, valloader_kwargs, \
        valset_kwargs = kwargs

    # CIFAR
    if dataset in ['CIFAR10', 'CIFAR100']:
        traindir = datadir
        valdir = datadir
    # imagenet
    if dataset in ['ImageNet', 'Imagenette', 'Imagewoof']:
        dataset = 'ImageFolder'
        traindir = op.join(datadir, 'train')
        valdir = op.join(datadir, 'val')

    # fetch dataset
    trainset = getattr(ds, dataset)(traindir, transform=transform, \
                    **trainset_kwargs)
    valset = getattr(ds, dataset)(valdir, transform=transform, \
                    **valset_kwargs)

    # dataloaders
    train_loader = DataLoader(trainset, **trainloader_kwargs)
    val_loader = DataLoader(valset, **valloader_kwargs)

    return train_loader, val_loader


@pytest.fixture
def classes(dataset, datadir):
    """
    """
    return get_classes(dataset, datadir)


@pytest.mark.skip()
def select_random_train(dataset, datadir):
    """
    select n random data points from training set
    """
    trainset_kwargs = {}
    valset_kwargs = {}
    return select_n_random('train', datadir, trainset_kwargs, \
        valset_kwargs, dataset, n=20)


@pytest.mark.skip()
def test_imshow(select_random_train):
    """
    print images
    """
    images, _ = select_random_train
    print(images.shape)
    # make_grid input must be BxCxHxW in shape
    img_grid = make_grid(images)
    show_images(img_grid)
    plt.show()


@pytest.mark.skip()
def test_show_labelled_images(select_random_train, classes, dataset):
    """
    """
    images, labels = select_random_train
    show_labelled_images(images, labels, classes, nrows=4, ncols=4, \
        savepath='./samples/'+dataset)

def test_get_labels_counts(dataloaders):
    """
    """
    train_loader, val_loader = dataloaders
    get_labels_counts(train_loader)


