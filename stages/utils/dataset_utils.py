"""
utilities for dataset inspection stage
"""
import pickle
import os
import os.path as op
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.utils import Params, set_logger
from common.dataloader import fetch_dataloader

global meta_mapping
meta_mapping = {
    'CIFAR10': 'cifar-10-batches-py/batches.meta',
    'CIFAR100': 'cifar-100-python/meta'
}


class Dataset(object):
    """
    An object to fetch and load dataset into DataLoader objects as configured
    """
    def __init__(self, json_path=None):
        """
        Initialize dataset configurations from .json file

        Args:
            json_path: (str) path to runset.json file; default is the current working
                       diretory by os.getcwd()
        """
        # run_directory
        if json_path is None:
            run_dir = os.path.join(os.getcwd(), '01_dataset')
        else:
            run_dir = json_path

        # read .json
        json_path = os.path.join(run_dir, 'runset.json')
        assert os.path.isfile(json_path), "No json configuration file found at \
            {}".format(json_path)
        params = Params(json_path)

        # get dataset parameters from runset.json
        self.dataset = params.data['dataset']
        self.datadir = params.data['datadir']
        self.num_classes = params.data['num_classes']
        self.trainloader_kwargs = params.data['trainloader-kwargs']
        self.trainset_kwargs = params.data['trainset-kwargs']
        self.valloader_kwargs = params.data['valloader-kwargs']
        self.valset_kwargs = params.data['valset-kwargs']

        # set the logger
        set_logger(os.path.join(run_dir, 'dataset.log'))


    def dataloader(self):
        """
        Fetch and Load dataset into train / val DataLoader objects
        """

        logging.info('Loading dataset...')

        # fetch data into DataLoader
        data_loaders = fetch_dataloader(['train', 'val'], self.datadir, self.dataset, \
                self.trainloader_kwargs, self.trainset_kwargs, self.valloader_kwargs, \
                self.valset_kwargs)
        train_dl = data_loaders['train']
        val_dl = data_loaders['val']

        logging.info('Done.')

        return train_dl, val_dl



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


def get_classes(dataset, datadir):
    """
    return class names from dataset meta file as a list of strings
    """
    try:
        meta_path = op.join(datadir, meta_mapping[dataset])
    except KeyError:
        raise "dataset {} meta path not registered".format(dataset)

    assert op.isfile(meta_path), "No meta file found at {} for dataset {}".format( \
        meta_path, dataset)
    with open(meta_path, 'rb') as fo:
        dct = pickle.load(fo, encoding='ASCII')
    for key, value in dct.items():
        if 'label' in key and isinstance(value, (list, tuple)):
            return value
    raise ValueError("labels not found!")


def show_images(img):
    """
    print images

    Args:
        img: (np.ndarray or tensor) images
    """
    if isinstance(img, torch.Tensor):
        npimg = img.numpy()
    elif isinstance(img, np.ndarray):
        npimg = img
    else:
        raise TypeError("Image type {} not recognized".format(type(img)))
    # assumes npimg shape = CxHxW; transpose to HxWxC
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show_labelled_images(img, labels, classes, nrows=8, ncols=8, savepath=None):
    """
    print images in a grid with actual labels as image titles

    Args:
        img: (tensor or np.ndarray) images; shape = BxCxHxW
        labels: (tensor or np.ndarray) label indices; shape = Bx1
        classes: (list of str) a list of strings for class names
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

    grid_sz = ncols * nrows
    fig = plt.figure(figsize=(ncols*1.5, nrows*1.5))
    for idx in range(0, npimg.shape[0]):
        ax = fig.add_subplot(nrows, ncols, idx % grid_sz + 1, xticks=[], yticks=[])
        # add image
        show_images(npimg[idx])
        # add label
        try:
            ax.set_title(classes[int(labels[idx].item())])
        except IndexError:
            raise "label index {} out of range for {} number of \
                classes".format(labels[idx], len(classes))
        # save figure when current grid is full or when end of loop reached
        # create a new fig object once current grid is full
        if (idx + 1) % grid_sz == 0 or idx == npimg.shape[0]-1:
            fig.subplots_adjust(hspace=0.5)
            plt.savefig(savepath + "_{}.png".format(idx // grid_sz))
            plt.show()
            if (idx + 1) % grid_sz == 0:
                fig = plt.figure(figsize=(ncols*1.5, nrows*1.5))
