"""
    Utility for inspection.py
"""

import os
import logging

import torch

from common.utils import set_logger, save_checkpoint, load_checkpoint
from common.trainer import Trainer


class InspectionTrainer(Trainer):
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
        set_logger(os.path.join(self.run_dir, '02_inspection.log'))


    def train(self, dataloader=None, epochs=0, batches=1):
        """
        Run Training on 'dataloader' as input for several batches

        Args:
            dataloader: (DataLoader object) iterator to dataset
            epochs: (int) number of training epochs
            batches: (int) number of batches to train; this option should only be used when
                     'epochs' is set to 0, i.e., train for partial epoch

        Returns:
            summ: (list of dicts) a list of dictionaries, each contains training summary
                  for one iteration; a summary consists of metric: value pairs

        Note:
        - number of iterations = 'epochs' * len(dataloader) + 'batches'
        """
        return self._train(dataloader, epochs, batches)


    def eval(self, dataloader=None, epochs=0, batches=1, restore_file=None):
        """
        Evaluate the model

        Args:
            dataloader: (DataLoader) iterator to evaluation dataset
            restore_file: (str) if specified, evaluated the pretrained model;
                          by default evalaute the current model state
        """
        return self._train(dataloader, epochs, batches, restore_file, evaluate=True)


    def _train(self, dataloader=None, epochs=0, batches=1, restore_file=None, \
        evaluate=False):
        """
        Run Training / evaluation on 'dataloader' as input for several batches

        Args:
            dataloader: (DataLoader object) iterator to dataset
            epochs: (int) number of training epochs
            batches: (int) number of batches to train; this option should only be used when
                     'epochs' is set to 0, i.e., train for partial epoch
            evaluate: (bool) if True use evaluation mode, if False (default) use train mode

        Returns:
            summ: (list of dicts) a list of dictionaries, each contains training summary
                  for one iteration; a summary consists of metric: value pairs

        Note:
        - number of iterations = 'epochs' * len(dataloader) + 'batches'
        """
        if evaluate:
            if restore_file is not None:
                self.load(restore_file)
            self.model.eval()
        else:
            self.model.train()

        # training summary
        summ = []

        # number of training iterations
        num_batches = epochs * len(dataloader) + batches

        # use next() to iterate over dataloader; this is more efficient than
        # standard enumerate(dataloader) if only a few batches are needed
        for idx in range(num_batches):

            train_batch, labels_batch = next(iter(dataloader))

            # move to GPU if available
            if self.cuda:
                train_batch, labels_batch = train_batch.to(self.device, \
                    non_blocking=True), labels_batch.to(self.device, non_blocking=True)

            # compute model output
            if evaluate:
                with torch.no_grad():
                    output_batch = self.model(train_batch)
            else:
                output_batch = self.model(train_batch)

            # compute loss
            loss = self.loss_fn(output_batch, labels_batch)

            summary_batch = {}
            # add 'loss'; detach() to save GPU memory
            summary_batch.update({'loss': loss.detach().item()})
            # add metrics
            summary_batch.update({metric: metric_fn(output_batch.to('cpu'), \
                labels_batch.to('cpu')) for metric, metric_fn in self.metrics.items()})

            # display training progress
            print_summary_msg(summary_batch, idx+1, num_batches)

            # add 'iteration' as index
            summary_batch.update({'iteration': idx})

            summ.append(summary_batch)

            if evaluate:
                continue

            # clear previous gradients, back-propagate gradients of loss w.r.t. all parameters
            self.optimizer.zero_grad()
            loss.backward()

            # update weights
            self.optimizer.step()

        return summ


    def save(self, checkpoint_name=None):
        """
        Overload method to save a checkpoint of (model, optimizer, lr_scheduler)

        Args:
            checkpoint_name: (str) file name for saved checkpoint; default='last'
        """
        save_checkpoint(
            {
                'state_dict': self.model.state_dict(),
                'optim_dict': self.optimizer.state_dict(),
                'scheduler_dict': self.scheduler.state_dict(),
            },
            checkpoint=self.run_dir,
            checkpoint_name=checkpoint_name
        )


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
                load_checkpoint(restore_path, self.model, self.optimizer, self.scheduler)


def print_summary_msg(summary_batch, prog, tot):
    """
    Display training metrics

    Args:
        summary_batch: (dict) a dict object that contains metric: value pairs
        prog: (int) training progress index; e.g., iteration, epoch, etc.
        tot: (int) total number of indices
    """
    metrics_string = ' ; '.join('{}: {:5.03f}'.format(k, v) for k, v in \
        summary_batch.items())
    logging.info("- Iteration {}/{} Train metrics: {}".format(prog, tot, metrics_string))


if __name__ == '__main__':
    from stages.utils.dataset_utils import Dataset
    from common.utils import Params
    import os.path as op

    json_path = op.join('../tests/directory/02_inspection/', \
        'runset.json')
    params = Params(json_path)
    trainer = InspectionTrainer(params, \
        run_dir='../tests/directory/02_inspection/')
    dataloader = Dataset('../tests/directory/02_inspection/').dataloader()
    trainloader, _ = dataloader
    images, _ = next(iter(trainloader))
    #trainer.net_summary(images)

    trainer.train(trainloader)