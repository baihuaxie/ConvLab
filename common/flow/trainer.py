"""
    Classes and functions for training
"""

import os
from abc import abstractmethod
import logging

import torch
import torch.optim as optim

from common.utils.misc_utils import print_net_summary, load_checkpoint
from common.objectives import loss_fn, metrics
from model.build_models import get_network_builder


class Trainer(object):
    """
    A trainer object to provide launch interface for training / evaluation flows

    Trainer() contains a collection of instances:
    - model
    - optimizer
    - learning rate scheduler
    - metrics
    - loss function
    """
    def __init__(self, params, seed=200, run_dir=None):
        """
        Constructor

        Args:
            params: (Params object) a dict-like object with the runset parameters
            seed: (int) random manual seed
            run_dir: (str) directory for storing output files
        """
        # set run directory
        if run_dir is None:
            self.run_dir = os.getcwd()
        assert os.path.exists(run_dir), "Directory {} does not exist!".format(run_dir)
        self.run_dir = run_dir

        # get keyword parameters for model (if any)
        try:
            model_kwargs = params.model['kwargs']
        except KeyError:
            model_kwargs = {}
            print("Model keyword argument is not specified!")
        # get keyword parameters for optimizer and scheduler
        optim_type = params.optimizer['type']
        optim_kwargs = params.optimizer['kwargs']
        lr_type = params.scheduler['type']
        lr_kwargs = params.scheduler['kwargs']

        # store runset parameters
        self._params = params

        # get device
        self._cuda = torch.cuda.is_available()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # set random seed for reproducible experiments
        torch.manual_seed(seed)
        if self._cuda:
            torch.cuda.manual_seed(seed)

        # build model
        self._net_type = params.model['network']
        # later: may need to refactor code; .to(device) creates a cpu copy first?
        self._model = get_network_builder(self._net_type)(**model_kwargs).to(self._device)

        # build the optimizer
        self._optimizer = getattr(optim, optim_type)(self._model.parameters(), **optim_kwargs)

        # build learning rate scheduler
        self._scheduler = getattr(optim.lr_scheduler, lr_type)(self._optimizer, **lr_kwargs)

        # get loss function -> refactor loss_fn.py into registry
        self._loss_fn = loss_fn

        # get metrics -> refactor metrics.py into registry
        self._metrics = metrics

    @property
    def get_model(self):
        """
        Return model instance
        """
        return self._model


    def net_summary(self, x):
        """
        Get a summary for blocks, layers and dimensions in the network

        Args:
            x: (torch.tensor) input data
        """
        x = x.to(self._device)
        # write architecture to log file using torchsummary package
        print_net_summary(self.run_dir+'/{}_summary.log'.format(self._net_type), self._model, x)

    @abstractmethod
    def train(self):
        """
        Abstract method to run training
        """

    @abstractmethod
    def eval(self):
        """
        Abstract method to evaluate a pretrained model
        """

    @abstractmethod
    def save(self):
        """
        Save a checkpoint of (model, optimizer, lr_scheduler)
        """

    def load(self, restore_file=None):
        """
        Load (model, optimizer, lr_scheduler) states from .pth.zip file

        Args:
            restore_file: (str) file path to .pth.zip file = run_dir/restore_file.pth.zip
        """
        # reload the weights from restore_file if specified
        if restore_file is not None:
            restore_path = os.path.join(self.run_dir, restore_file+'.pth.zip')
            if os.path.exists(restore_path):
                logging.info("Restoring weights from {}".format(restore_path))
                load_checkpoint(restore_path, self._model, self._optimizer, self._scheduler)
