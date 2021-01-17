"""
    Utility for inspection.py
"""

import os
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from common.utils import set_logger, save_checkpoint, load_checkpoint, RunningAverage
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

    
    def get_model(self):
        """
        Return model instance
        """
        return self._model


    def train(self, dataloader=None, iterations=1, full_epoch=False, restore_file=None, ground_input=False):
        """
        Run Training on 'dataloader' as input for several batches

        Args:
            dataloader: (DataLoader object) iterator to dataset
            iterations: (int) number of training iterations
            full_epoch: (bool) if True train for a full epoch
            restore_file: (str) if specified, evaluated the pretrained model;
                          by default evalaute the current model state
            ground_input: (bool) if True, tie data batches to zeros; this is for training
                         an input-independent baseline
        """
        if restore_file is not None:
            self.load(restore_file)
        return self._train(dataloader, iterations, full_epoch, ground_input=ground_input)


    def eval(self, dataloader=None, iterations=1, full_epoch=False, restore_file=None):
        """
        Evaluate the model

        Args:
            dataloader: (DataLoader) iterator to evaluation dataset
            iterations: (int) number of training iterations
            full_epoch: (bool) if True train for a full epoch
            restore_file: (str) if specified, evaluated the pretrained model;
                          by default evalaute the current model state
        """
        return self._train(dataloader, iterations, full_epoch, restore_file, evaluate=True)


    def _train(self, dataloader=None, iterations=1, full_epoch=False, restore_file=None, \
        evaluate=False, ground_input=False):
        """
        Run Training / evaluation on 'dataloader' as input for several batches

        Args:
            dataloader: (DataLoader object) iterator to dataset
            iterations: (int) number of training iterations
            full_epoch: (bool) if True train for a full epoch
            evaluate: (bool) if True use evaluation mode, if False (default) use train mode
            ground_input: (bool) if True, tie data batches to zeros; this is for training
                         an input-independent baseline

        Returns:
            summ: (list of dicts) a list of dictionaries, each contains training summary
                  for one iteration; a summary consists of metric: value pairs

        Note:
        - this function does not support multi-epoch training
        """
        if evaluate:
            if restore_file is not None:
                self.load(restore_file)
            self._model.eval()
        else:
            self._model.train()

        # training summary
        summ = []

        # loss running avg
        loss_avg = RunningAverage()

        # FLAG: ground input
        if ground_input:
            logging.info("Training batches have been grounded...")

        # use progress bar & running avg loss for larger training runs
        display_thres = 50
        if iterations > display_thres:
            prog = tqdm(total=iterations)

        # train for a full epoch
        if full_epoch:
            iterations = len(dataloader)

        # get an iterator to dataloader
        # this code should be placed outside any loop, as it creates a new
        # instance each time iter() is called, would be very slow
        dataloader_iter = iter(dataloader)

        # use next() to iterate over dataloader; this is more efficient than
        # standard enumerate(dataloader) if training is expected to be <= 1 epoch
        for idx in range(iterations):

            train_batch, labels_batch = next(dataloader_iter)

            # ground input
            if ground_input:
                train_batch = torch.zeros(train_batch.shape, dtype=torch.float32)

            # move to GPU if available
            if self._cuda:
                train_batch, labels_batch = train_batch.to(self._device, \
                    non_blocking=True), labels_batch.to(self._device, non_blocking=True)

            # compute model output
            if evaluate:
                with torch.no_grad():
                    output_batch = self._model(train_batch)
            else:
                output_batch = self._model(train_batch)

            # compute loss
            loss = self._loss_fn(output_batch, labels_batch)
            loss_detached = loss.detach()

            if not evaluate:
                # clear previous gradients, back-propagate gradients of loss w.r.t. all parameters
                self._optimizer.zero_grad()
                loss.backward()
                # update weights
                self._optimizer.step()

            summary_batch = {}
            # add 'loss'; detach() to save GPU memory
            summary_batch.update({'loss': loss_detached.item()})
            # add metrics
            summary_batch.update({metric: metric_fn(output_batch.to('cpu'), \
                labels_batch.to('cpu')) for metric, metric_fn in self._metrics.items()})

            # add 'iteration' as index
            summary_batch.update({'iteration': idx})

            # update stats
            summ.append(summary_batch)
            loss_avg.update(loss_detached.item())

            # display training stats
            if iterations <= display_thres:
                # print as a list for smaller training runs
                print_summary_msg(summary_batch, idx+1, iterations, evaluate=evaluate)
            else:
                # for larger training runs display a progress bar with running avg
                prog.set_postfix(avg_loss='{:05.3f}'.format(loss_avg()), last_loss='{:05.3f}'.format(loss_detached.item()))
                prog.update()
                # add: also display loss / accuracy at last iteration

        return summ, loss_avg()


    def save(self, checkpoint_name=None):
        """
        Overload method to save a checkpoint of (model, optimizer, lr_scheduler)

        Args:
            checkpoint_name: (str) file name for saved checkpoint; default='last'
        """
        save_checkpoint(
            {
                'state_dict': self._model.state_dict(),
                'optim_dict': self._optimizer.state_dict(),
                'scheduler_dict': self._scheduler.state_dict(),
            },
            checkpoint=self.run_dir,
            checkpoint_name=checkpoint_name
        )
        logging.info("Saved checkpoint: {}".format(os.path.join(self.run_dir, checkpoint_name)))


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


def print_summary_msg(summary_batch, prog, tot, evaluate=False):
    """
    Display training metrics

    Args:
        summary_batch: (dict) a dict object that contains metric: value pairs
        prog: (int) training progress index; e.g., iteration, epoch, etc.
        tot: (int) total number of indices
        evaluate: (bool) if True use evaluation mode, if False (default) use train mode
    """
    metrics_string = ' ; '.join('{}: {:5.03f}'.format(k, v) for k, v in \
        summary_batch.items())
    session = 'Eval' if evaluate else 'Train'
    logging.info("- Iteration {}/{} {} metrics: {}".format(prog, tot, session, metrics_string))


def batch_loader(dataloader, length=1, samples=1):
    """
    Returns a DataLoader object that iterates over a single batch from 'dataloader' for 'len' times

    Args:
        dataloader: (DataLoader)
        len: (int) length of returned DataLoader object; default=1
        samples: (int) number of samples in the returned batch; default=1

    Note:
    - used for overfitting / evaluation anchored on a single batch of data
    """
    batch = next(iter(dataloader))

    sampled_batch = []

    import numpy as np

    # get a randomized mask
    mask = np.zeros(batch[0].shape[0], dtype=bool)
    mask[:samples] = 1
    np.random.shuffle(mask)

    for item in batch:
        # tensor dimension would not be reduced even if samples=0
        if item.dim() == 1:
            sampled_batch.append(item[mask.tolist()])
        else:
            sampled_batch.append(item[mask.tolist(), :(item.dim() - 1)])

    return DataLoader([sampled_batch for _ in range(length)], batch_size=None)




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
