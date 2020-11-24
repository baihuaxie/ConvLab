"""
dataset inspection
"""

import argparse
import os.path as op

from common.utils_dataset_inspection import show_labelled_images, get_classes
from common.data_loader import select_n_random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CIFAR10', \
    help='name of dataset')
parser.add_argument('--datadir', default='./data/', \
    help='parent directory containing dataset folder')
parser.add_argument('--savepath', default='./data/samples/', \
    help='path to save sampled data points')


if __name__ == '__main__':
    args = parser.parse_args()
    images, labels = select_n_random('train', args.datadir, args.dataset, n=20)
    classes = get_classes(args.dataset, args.datadir)
    savepath = op.join(args.savepath, args.dataset)
    show_labelled_images(images, labels, classes, nrows=4, ncols=4, savepath=savepath)
