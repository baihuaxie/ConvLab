"""
    test dataset.py
"""

import pytest
import os.path as op

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from stages.dataset import label_cdf


@pytest.fixture
def dataloaders():
    """
    return dataloaders for train/val sets

    - use ImageNette-160 as example dataset for testing purpose
    """
    # dataset directories
    data_dir = "../../data/Imagenette/"
    train_dir = op.join(data_dir, 'train')
    val_dir = op.join(data_dir, 'val')

    # transforms - naive
    train_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
    val_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
    
    # fetch dataset
    train_set = ImageFolder(train_dir, transform=train_transforms)
    val_set = ImageFolder(val_dir, transform=val_transforms)

    # dataloader
    train_loader = DataLoader(train_set, batch_size=256, \
        num_workers=6, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, \
        num_workers=6, pin_memory=True, shuffle=False)

    # return
    return train_loader, val_loader


def test_label_cdf(dataloaders):
    """
    test plotting cdf vs. label id
    """
    train_loader, val_loader = dataloaders
    label_cdf(labels)