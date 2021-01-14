"""
    Stage-2: validate an end-to-end training/evaluate skeleton & obtain baselines

    Tricks: (based on Andrej Karpathy's recipe: http://karpathy.github.io/2019/04/25/recipe/)
    - fix random seed (guarantees if an experiment is run multiple times, outputs would be the same)
    - disable any data augment; use most basic (e.g., ToTensor()) transformations when loading data
        - data augmentation is a form of regularization
    - plot val / test loss by running evaluation on full val / test set, instead of min-batching
        - mini-batching induces variations
    - verify loss @ initialization; e.g., if last layer is initialized properply, for soft-max outputs,
      should always get -log(1/num_classes) as the output after initialization
    - init the last layer's weights properly (verified by loss @init) that reflects the dataset's
      distributions
    - monitor metrics other than loss that are human interpretable (e.g., accuracy)
    - input-independent baseline: train a model that is independent of input, e.g., tie all input to
      zero; this serves as a baseline to check that, after plugging in real data, performance should
      be better => model actually learns sth. from the data
    - overfit one batch of a few examples: can be as little as two examples; overfit to lowest
      possible loss (e.g., zero); can do so by increasing model capacity; also plot sample with
      predicted label + ground truth label after overfit & verify that the labels are exact match
      when loss is reduced to lowest
    - verify that training loss decreases with increasing model capacity when the model is under-fit
    - visualize the exact input (data + label tensors) into the model; this can catch bugs that have
      happened during data augmentation / pre-processing
    - visualize model's prediction dynamics on a fixed val / test batch: this helps to see how the
      training is progressing; visualization can be to plot loss on this batch vs. training iterations;
      can also play with varying learning rates, optimizers, schedulers, etc.
    - use backprop to chart dependencies in the network
      e.g., to check that there is no cross-batch dependencies, can tie the i-th loss to a trivial function
      that depends only on the i-th input (e.g., sum of all outputs of i-th input) & backprop the loss;
      then check that only i-th input has non-zero gradients;
      this method is in general helpful to verify model dependencies
    - coding tip: write prototyping codes, refactor later; at this stage do not write codes with
      abstraction / generalization in mind; instead write codes for the specific case at hand, get
      it to working properply; write tests; then refactor the codes later & ensure tests are passed
    - should re-do this stage for any new model structure, optimizer, scheduler, etc.
"""

import os
import typer

from common.utils import Params
from stages.utils.inspection_utils import InspectionTrainer
from stages.utils.dataset_utils import Dataset


app = typer.Typer()

@app.command()
def check_seed(
    runs: int = typer.Option(3, help="number of reproducible runs")
):
    """
    Run multiple training with fixed random seed for reproducible results
    """
    trainer = InspectionTrainer(params, run_dir=run_dir)
    trainer.save(checkpoint_name='init')
    trainloader, _ = Dataset(run_dir).dataloader()
    for idx in range(runs):
        # change echo() to logging.info()
        typer.echo("Fixed seed={} run {}/{}".format(trainer.seed, idx+1, runs))
        # for some reason this code doesn't reproduce exact outputs. why?
        # note: if I re-instantiate a new Trainer() object each run then it's ok
        trainer.load(restore_file='init')
        trainer.train(trainloader)


@app.command()
def check_init_loss():
    """
    Check the loss after initialization
    """
    trainer = InspectionTrainer(params, run_dir=run_dir)
    _, valloader = Dataset(run_dir).dataloader()
    # Q: for this check do I need to input a batch from val set?
    # Q: how to obtain the correct loss value @ init for x-ent loss function?
    trainer.eval(valloader, batches=1)


@app.command()
def check_underfit():
    """
    Check training loss decreasing with increasing model capacity
    """


@app.command()
def check_dependency():
    """
    Use backprop to check network's dependencies are correct
    """


@app.command()
def train_input_grounded():
    """
    Train a baseline with all inputs grounded to zero
    """


@app.command()
def train_batch_overfit():
    """
    Train a baseline that overfits a single batch with several samples
    """


@app.command()
def evaluate_anchored():
    """
    Anchor evaluation on one val / test batch to visualize prediction dynamics
    """


@app.command()
def evaluate_nobatch():
    """
    Run evaluation on full val / test set, no mini-batching

    Train a baseline to one epoch if no pretrained model provided
    """


@app.callback()
def main():
    """
    Set up an end-to-end skeleton for training and evaluation
    """
    # set running directory for dataset sub-command
    global run_dir
    run_dir = os.path.join(os.getcwd(), '02_inspection')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    typer.echo("Running directory for dataset.py: {}".format(run_dir))

    # get runset parameters
    global params
    json_path = os.path.join(run_dir, 'runset.json')
    params = Params(json_path)


if __name__ == '__main__':
    app()