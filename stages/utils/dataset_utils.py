"""
utilities for dataset inspection stage
"""
import pickle
import json
import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.utils.misc_utils import set_logger
from common.dataset.dataloader import fetch_dataloader, select_n_random


class Dataset(object):
    """
    An object to fetch and load dataset into DataLoader objects as configured
    """
    def __init__(self, params, run_dir=None):
        """
        Initialize dataset configurations from .json file

        Args:
            params: (Params object) a dict-like object with the runset parameters
            run_dir: (str) directory for storing output files
        """
        # set run directory
        if run_dir is None:
            self.run_dir = os.getcwd()
        assert os.path.exists(run_dir), "Directory {} does not exist!".format(run_dir)
        self.run_dir = run_dir

        # get dataset parameters from runset.json
        self.dataset = params.data['dataset']
        self.datadir = params.data['datadir']
        self.num_classes = params.data['num_classes']
        self.trainloader_kwargs = params.data['trainloader-kwargs']
        self.trainset_kwargs = params.data['trainset-kwargs']
        self.valloader_kwargs = params.data['valloader-kwargs']
        self.valset_kwargs = params.data['valset-kwargs']

        # set the logger
        set_logger(os.path.join(self.run_dir, 'dataset.log'))


    def dataloader(self, transforms=None):
        """
        Fetch and Load full dataset into train / val DataLoader objects

        Args:
            transforms: (torchvision.transforms) data augmentation; if None, applies
                        default ImageNet standard augmentation
        """
        logging.info('Loading dataset...')

        # fetch data into DataLoader
        dataloaders = fetch_dataloader(['train', 'val'], self.datadir, self.dataset, \
                self.trainloader_kwargs, self.trainset_kwargs, self.valloader_kwargs, \
                self.valset_kwargs, transforms, transforms)
        trainloader = dataloaders['train']
        valloader = dataloaders['val']

        logging.info('Done.')

        return trainloader, valloader


    def sampler(self, transform=None, sampler=None, balanced=False):
        """
        Return a list of [data, labels] sampled at random or by a specific sampler

        Args:
            sampler: (torch.utils.data.sampler object)
            balanced: (bool) if True use label-balanced sampling; this is the default sampler
                      for ImageFolder class datasets
        Returns:
            list [data, labels]; same as next(iter(DataLoader object))
        """
        if self.dataset in ['CIFAR10', 'CIFAR100']:
            return select_n_random('train', self.datadir, {}, {}, self.dataset, \
                n=self.trainloader_kwargs['batch_size'])

        elif self.dataset in ['Imagenette', 'Imnagewoof']:
            dataloaders = fetch_dataloader(['train'], self.datadir, self.dataset, \
                self.trainloader_kwargs, self.trainset_kwargs, self.valloader_kwargs, \
                self.valset_kwargs, transform, transform, sampler=sampler, balanced=balanced)
            trainloader = dataloaders['train']
            # return a single batch of data only
            return next(iter(trainloader))


def get_labels_counts(dataloader, num_classes):
    """
    Return label counts for all samples in the dataloader

    Args:
        dataloader: (DataLoader object)
        num_classes: (int) number of classes

    Returns:
        counts: (np.ndarray) a numpy array of integers; counts[i] represents counts of
                samples which have the label indexed by i in the data set

    Notes:
    - this function is for label statistics, e.g., cdf plots
    - labels must be integer id's in the dataloader
    """
    counts = np.zeros(num_classes)
    for _, (_, labels_batch) in enumerate(dataloader):
        for label in labels_batch.tolist():
            # label must be an integer id
            assert(isinstance(label, int)), "label {} is not indexed!".format(label)
            counts[label-1] += 1

    return counts


global meta_mapping
meta_mapping = {
    'CIFAR10': 'cifar-10-batches-py/batches.meta',
    'CIFAR100': 'cifar-100-python/meta',
    'Imagenette': 'labels.json'

}

def get_classes(dataset, datadir):
    """
    return class names from dataset meta file / json file as a list of strings

    Args:
        dataset: (str) dataset name
        datadir: (str) dataset directory

    Note:
    - in dataloader object, the labels are usually integer id's
    - get_classes() returns a list of strings, where list[i] == class name string for
      label id == i; this is an assumption, not a guarantee by this function
    - get_classes() obtains the class namestrings from dataset meta file; while dataloaders
      typically assign the labels when fetching & loading the data samples; it is up to
      the user of this function to ensure that the dataset meta file has matching label id
      & class namestrings;
      for ImageFolder class datasets, this usually can be done by ordering the class
      namestrings in the meta file with the same order of the sub folders on disk
    """
    try:
        meta_path = os.path.join(datadir, meta_mapping[dataset])
    except KeyError:
        raise "dataset {} meta path not registered".format(dataset)

    assert os.path.isfile(meta_path), "No meta file found at {} for dataset {}".format( \
        meta_path, dataset)

    with open(meta_path, 'rb') as fo:
        # for .json meta files
        if '.json' in meta_path:
            dct = json.load(fo)
        # for other format use pickle
        else:
            dct = pickle.load(fo, encoding='ASCII')
    # for CIFAR-10/100
    if dataset in ['CIFAR10', 'CIFAR100']:
        labels = []
        for key, value in dct.items():
            if 'fine' in key and isinstance(value, (list, tuple)):
                # CIFAR-10/100 labels are divided into fine & coarse; usually use only fine labels
                # e.g., CIFAR100 has 100 fine labels + 20 coarse labels
                labels.extend(value)
        return labels
    # for ImageFolder dataset with .json label files
    elif '.json' in meta_path:
        return [value for value in dct.values()]

    raise ValueError("labels not found in meta file {}!".format(meta_path))


def show_images(img):
    """
    print a single image

    Args:
        img: (np.ndarray or tensor) image object
    """
    if isinstance(img, torch.Tensor):
        npimg = img.numpy()
    elif isinstance(img, np.ndarray):
        npimg = img
    else:
        raise TypeError("Image type {} not recognized".format(type(img)))
    # assumes npimg shape = CxHxW; transpose to HxWxC
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show_labelled_images(dataset, classes, img, labels, predicted=None, nrows=8, \
    ncols=8, savepath=None):
    """
    print images in a grid with actual labels as image titles

    Args:
        dataset: (str) dataset name; for saving images
        classes: (list of str) a list of strings for class names
        img: (tensor or np.ndarray) images from dataset; shape = BxCxHxW
        labels: (tensor or np.ndarray) label indices from dataset; shape = Bx1
        predicted: (tensor or np.ndarray) predicted label indices from model output
        nrows: (int) # of rows in the grid; default=8
        ncols: (int) # of cols in the grid; default=8
        savepath: (str) path to save figures

    Note:
    - by default, # of grids to be saved = B / (nrows * ncols)
    """
    if isinstance(img, torch.Tensor):
        npimg = img.numpy()
    elif isinstance(img, np.ndarray):
        npimg = img
    else:
        raise TypeError("Image type {} not recognized".format(type(img)))

    if savepath is None:
        # by default set current working directory as savepath
        savepath = os.getcwd()

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    grid_sz = ncols * nrows
    fig = plt.figure(figsize=(ncols*2, nrows*2))
    for idx in range(0, npimg.shape[0]):
        ax = fig.add_subplot(nrows, ncols, idx % grid_sz + 1, xticks=[], yticks=[])
        # add image
        show_images(npimg[idx])

        # add label
        try:
            # improve: better aesthetics
            ax.text(0.5, 0.6, classes[int(labels[idx].item())], ha="center", va="bottom", \
                size="medium", color="red", transform=ax.transAxes)
            if predicted is not None:
                assert len(labels) == len(predicted)
                ax.text(0.5, 0.4, classes[int(predicted[idx].item())], ha="center", va="bottom", \
                    size="medium", color="green", transform=ax.transAxes)
        except IndexError:
            raise "label index {} out of range for {} number of \
                classes".format(labels[idx], len(classes))
        # save figure when current grid is full or when end of loop reached
        # create a new fig object once current grid is full
        if (idx + 1) % grid_sz == 0 or idx == npimg.shape[0]-1:
            fig.subplots_adjust(hspace=0.8)
            plt.savefig(savepath + "/{}_{}.png".format(dataset, idx // grid_sz))
            plt.show()
            if (idx + 1) % grid_sz == 0:
                fig = plt.figure(figsize=(ncols*2, nrows*2))
