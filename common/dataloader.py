"""
    Reads raw dataset and returns dataloader(s)

    This file should take the following as inputs:
    - raw dataset -> url for download; file path for local
    - hyper-parameters, e.g., num_workers, pinning memeory, batch_size, etc.

    This file should return:
    - torch.utils.data.DataLoader objects -> conceptually, should distinct dataloaders for train, val, test datasets

    The following items need to be implemented in this file:

    1) a class definition for dataset
    -- if using pytorch built-in datasets, e.g., torchvision.datasets, can directly call the dataset class and omit custom definition
    -- if using custom datasets, need to define a class for that dataset
       - class needs to inherit from either Dataset or IterableDataset pytorch classes
       - for Dataset class, need to define __getitem__() and __len__() methods
       - for IterableDataset class, need to define __iter__() method
    -- class input: directory, tansforms
       - needs to apply transforms to raw dataset before return

    2) a set of transform pipelines
    -- for each dataloader type (train, val, test, etc.), may need to define different transform pipelines
    -- usually uses torchvision.transforms.Compose() to construct the pipelines

    3) a function / method to return the corresponding DataLoader object for each type
    -- need to instantiate the defined or built-in dataset class
    -- usually uses torch.utils.data.DataLoader() to build DataLoader objects
    -- this function / method is then called in main scripts to load dataset

"""

import os.path as op

import torch
import torchvision.transforms as transforms
import torchvision.datasets as ds
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from common.utility.dataloader_utils import make_weights_for_balanced_classes

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
# default transforms pipeline for train set
# standard ImageNet practice
default_train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
)

# default transforms pipeline for val set
# standard ImageNet practice
default_val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]
)

def fetch_dataset(types, datadir, dataset=None, trainset_kwargs=None, valset_kwargs=None, \
    train_transform=None, val_transform=None):
    """
    Fetches the dataset objects from torchvision.datasets

    Used subsequently to fetch dataloader objects or fetch random samples

    Args:
        types: (list) has one or more of "train, val, test" depending on the dataset
        datadir: (str) file path to the dataset
        dataset: (str) name of the dataset to be loaded
        trainset_kwargs: (dict) keyword arguments passed to torchvision.dataset.Dataset() function call
                         in call to fetch_dataset() for training set
        valset_kwargs: (dict) keyword arguments passed to torchvision.dataset.Dataset() function call
                       in call to fetch_dataset() for validation set
        train_transform: (tochvision.transforms) object for train set data augmentation pipeline
        val_transform: (tochvision.transforms) object for validation set data augmentation pipeline

    Returns:
        dataset: (dict) contains the dataset object for each type in types
    """

    datasets = {}

    # by default
    traindir = datadir
    valdir = datadir

    # transforms
    if train_transform is None:
        train_transform = default_train_transform
    if val_transform is None:
        val_transform = default_val_transform

    # default dataset
    if dataset is None:
        dataset = 'CIFAR10'
    # imagenet
    if dataset in ['ImageNet', 'Imagenette', 'Imagewoof']:
        dataset = 'ImageFolder'
        traindir = op.join(datadir, 'train')
        valdir = op.join(datadir, 'val')

    for split in ['train', 'val']:
        if split in types:

            # load train set
            if split == 'train':
                data = getattr(ds, dataset)(traindir, transform=train_transform, \
                    **trainset_kwargs)

            # load val set
            if split == 'val':
                data = getattr(ds, dataset)(valdir, transform=val_transform, \
                    **valset_kwargs)

            datasets[split] = data

    return datasets


def fetch_dataloader(types, datadir, dataset, trainloader_kwargs, trainset_kwargs, valloader_kwargs, \
    valset_kwargs, train_transform=None, val_transform=None, sampler=None, balanced=False):
    """
    Fetches the dataloader objects for each type from datadir.

    Note: by default load full training / validation / test datasets

    Args:
        types: (list) has one or more of "train, val, test" depending on the dataset
        datadir: (str) file path containing the raw dataset
        dataset: (str) dataset name; e.g., CIFAR10
        trainloader_kwargs: (dict) keyword arguments passed to Dataloader() function for training set
        trainset_kwargs: (dict) keyword arguments passed to torchvision.dataset.Dataset() function call
                         in call to fetch_dataset() for training set
        valloader_kwargs: (dict) keyword arguments passed to Dataloader() function for validation set
        valset_kwargs: (dict) keyword arguments passed to torchvision.dataset.Dataset() function call
                       in call to fetch_dataset() for validation set
        train_transform: (tochvision.transforms) object for train set data augmentation pipeline
        val_transform: (tochvision.transforms) object for validation set data augmentation pipeline
        sampler: (torch.utils.data.sampler object) if specified, use to draw samples from dataset
        balanced: (bool) if True use label-balanced sampler

    Returns:
        dataloadr: (dict) contains the DataLoader objects for each type in types
    """

    dataloaders = {}

    # fetch dataset
    fetched_data = fetch_dataset(types, datadir, dataset, trainset_kwargs, valset_kwargs, \
        train_transform, val_transform)

    for split in types:
        if split in ['train', 'val', 'test']:

            data = fetched_data[split]

            # dataloader kwargs
            loader_kwargs = trainloader_kwargs if split == 'train' else valloader_kwargs

            # add label-balanced sampler if 'balanced' flag is set to True
            if balanced:
                # the following code snippet only works for ImageFolder calss
                assert isinstance(data, ds.ImageFolder), "dataset type {} not supported by label-balanced sampling"\
                    .format(type(data))
                weights = make_weights_for_balanced_classes(data.imgs, len(data.classes))
                weights = torch.DoubleTensor(weights)
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

            # load dataset into DataLoader object
            if sampler is not None:
                dataloader = DataLoader(data, sampler=sampler, **loader_kwargs)
            else:
                if split == 'train':
                    shuffle = True
                else:
                    shuffle = False
                dataloader = DataLoader(data, shuffle=shuffle, **loader_kwargs)

            # add split
            dataloaders[split] = dataloader

    return dataloaders


def fetch_subset_dataloader(types, datadir, dataset, trainloader_kwargs, trainset_kwargs, \
    valloader_kwargs, valset_kwargs, batchsz=32, batch_num=10):
    """
    Fetches dataloader objects for a subset of each type of data

    This might be helpful in case I want test-train a few batches before the full sweep

    Args:
        types: (list) has one or more of "train, val, test" depending on the dataset
        datadir: (str) file path containing the raw dataset
        dataset: (str) dataset name; e.g., CIFAR10
        trainloader_kwargs: (dict) keyword arguments passed to Dataloader() function for training set
        trainset_kwargs: (dict) keyword arguments passed to torchvision.dataset.Dataset() function call
                         in call to fetch_dataset() for training set
        valloader_kwargs: (dict) keyword arguments passed to Dataloader() function for validation set
        valset_kwargs: (dict) keyword arguments passed to torchvision.dataset.Dataset() function call
                       in call to fetch_dataset() for validation set
        batchsz: (int) batch size
        batch_num: (int) number of batches in the subset

    Returns:
        dataloadr: (dict) contains the DataLoader objects for each type in types
    """

    dataloaders = {}

    # fetch dataset
    fetched_dataset = fetch_dataset(types, datadir, dataset, trainset_kwargs, valset_kwargs)

    for split in types:
        if split in ['train', 'val', 'test']:

            data = fetched_dataset[split]

            # dataloader kwargs
            loader_kwargs = trainloader_kwargs if split == 'train' else valloader_kwargs

            # get subset indices
            subset_indices = range(batchsz * batch_num)
            data_subset = Subset(data, subset_indices)

            # load dataset into DataLoader object
            if split == 'train':
                shuffle = True
            else:
                shuffle = False
            dataloader = DataLoader(data_subset, shuffle=shuffle, **loader_kwargs)

            dataloaders[split] = dataloader

    return dataloaders


def select_n_random(dataset_type, datadir, trainset_kwargs, valset_kwargs, dataset=None, n=1):
    """
    Select n random [data, label] points from dataset.
    Note: do not support ImageFolder class; use fetch_dataloaders with balanced=True instead

    Args:
        dataset_type: (str) a string of either 'train', 'val' or 'test'
        datadir: (str) file path to the raw dataset
        trainset_kwargs: (dict) keyword arguments passed to torchvision.dataset.Dataset() function call
                         in call to fetch_dataset() for training set
        valset_kwargs: (dict) keyword arguments passed to torchvision.dataset.Dataset() function call
                       in call to fetch_dataset() for validation set
        dataset: (str) name of dataset
        n: (int) number of selected data samples

    Return:
        n data points + corresponding labels (both tensors)
        data: (tensor)
        labels: (tensor)

    """
    dataset = fetch_dataset(dataset_type, datadir, dataset, trainset_kwargs, valset_kwargs)[dataset_type]

    assert not isinstance(dataset, ds.ImageFolder), "dataset type {} not supported by label-balanced sampling"\
        .format(type(dataset))
    data = torch.from_numpy(dataset.data).permute(0, 3, 1, 2)
    labels = torch.Tensor(dataset.targets)

    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]