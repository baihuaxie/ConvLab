"""
test memory hook functions
"""

import pytest

import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset

from model.build_models import get_network_builder
from common.memprof.hooks import _get_activations, activations, \
    _add_module_hooks, mem_hook

@pytest.fixture
def model():
    return get_network_builder('resnet18')()

@pytest.fixture
def input():
    """ get one batch from CIFAR10 as input tensor """
    dataset = CIFAR10('../../../data/', transform=transforms.ToTensor(), \
        download=False, train=False)
    subset_indices = range(64)
    subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=64, num_workers=6, \
        pin_memory=True, shuffle=True)
    images, _ = next(iter(dataloader))
    return images

@pytest.mark.skip()
def test_get_activations(model, input):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            h = module.register_forward_hook(_get_activations(name))
    out = model(input)
    print(activations)


def test_get_cuda_stats(model, input):
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_module_hooks(hr, module, mem_hook)

    output = model(input)


