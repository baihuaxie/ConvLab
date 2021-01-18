"""
    Utilities for stage-03 Overfit
"""

import os
import logging
from tqdm import tqdm
import numpy as np
import wandb

from common.utils.misc_utils import set_logger, RunningAverage, save_checkpoint
from common.logging.plots import save_batch_summary
from common.flow.trainer import Trainer

class OverfitTrainer(Trainer):
    """
    A trainer object to launch training / evaluation runs

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
        super().__init__(params, seed, run_dir)

        self.seed = seed

        # set the logger
        set_logger(os.path.join(self.run_dir, '03_overfit.log'))


    def train(self, dataloader=None, restore_file=None, num_epochs=1, save_summary=False):
        """
        Train model

        Args:
            dataloader: (DataLoader object) iterator to dataset
            restore_file: (str) if specified, evaluated the pretrained model;
                          by default evalaute the current model state
            num_epochs: (int) number of epochs to train
            save_summayr: (bool) if True save batch summaries into a .csv file
        """
        if restore_file is not None:
            self.load(restore_file)

        for epoch in range(num_epochs):

            logging.info("Epoch {} / {}".format(epoch+1, num_epochs))

            # logging current learning rate
            for i, param_group in enumerate(self._optimizer.param_groups):
                logging.info("learning rate = {} for parameter group {}".format(param_group['lr'], i))

            # train
            train_metrics, batch_summ = self._train(dataloader, epoch, self._params.save_summary_steps)

            # schedule learning rate
            if self._scheduler is not None:
                self._scheduler.step()

            self.save(epoch)

            # save batch summaries
            if save_summary:
                save_batch_summary(self.run_dir, batch_summ)


    def _train(self, dataloader=None, epoch=0, save_summ_steps=1):
        """
        Train the model for one epoch

        Args:
            dataloader: (DataLoader)
            epochs: (int) epoch index
            save_summ_steps: (int) save training summary every number of steps

        Returns:
            metrics_mean: (dict) contains metric - mean value pair for epoch
            summ: (list of dict) each dict element contains metric-value pairs saved at
                  corresponding iteration index
        """
        self._model.train()

        # summ: a list containing for each element a dictionary object
        # stores metric-value pairs with iteration index
        summ = []

        # initialize a running average object for loss
        loss_avg = RunningAverage()

        # number of batches per epoch
        num_batches = len(dataloader)

        # progress bar
        with tqdm(total=num_batches) as prog:

            # standard way to access DataLoader object for iteration over dataset
            for i, (train_batch, labels_batch) in enumerate(dataloader):

                # move to GPU if available
                train_batch, labels_batch = train_batch.to(self._device, \
                    non_blocking=True), labels_batch.to(self._device, non_blocking=True)

                output_batch = self._model(train_batch)

                loss = self._loss_fn(output_batch, labels_batch)
                # .detach() to save cuda memory
                loss_detach = loss.detach().item()

                # clear previous gradients, then backprop
                self._optimizer.zero_grad()
                loss.backward()

                # update weights
                self._optimizer.step()

                # save training summaries at certain iterations
                if (epoch*num_batches + i) % save_summ_steps == 0:

                    summary_batch = {}
                    summary_batch.update({metric: metric_fn(output_batch.to('cpu'), \
                        labels_batch.to('cpu')) for metric, metric_fn in self._metrics.items()})
                    summary_batch.update({'loss': loss_detach})
                    summary_batch.update({'iteration': epoch*num_batches + i})
                    summary_batch.update({'epoch': epoch+ 1})

                    # append summary
                    summ.append(summary_batch)

                    wandb.log(summary_batch)

                # update the running average loss
                loss_avg.update(loss_detach)

                # update progress bar to show running average for loss
                prog.set_postfix(avg_loss='{:05.3f}'.format(loss_avg()), \
                    last_loss='{:05.3f}'.format(loss_detach))
                prog.update()

        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0].keys()}
        metrics_string = ' ; '.join('{}: {:5.03f}'.format(k, v) for k, v in metrics_mean.items())

        logging.info("- Train metrics: {}".format(metrics_string))

        return metrics_mean, summ


    def save(self, epoch=0, checkpoint_name=None):
        """
        Overload method to save a checkpoint of (model, optimizer, lr_scheduler)

        Args:
            epoch: (int) epoch index
            checkpoint_name: (str) file name for saved checkpoint; default='last'
        """
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': self._model.state_dict(),
                'optim_dict': self._optimizer.state_dict(),
                'scheduler_dict': self._scheduler.state_dict(),
            },
            checkpoint=self.run_dir,
            checkpoint_name=checkpoint_name
        )
        logging.info("Saved checkpoint: {}".format(os.path.join(self.run_dir, checkpoint_name)))
