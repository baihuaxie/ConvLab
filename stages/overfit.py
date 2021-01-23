"""
    Overfit model on training set.
"""

import os
import typer
import wandb

from common.utils.misc_utils import Params
from stages.utils.overfit_utils import OverfitTrainer
from stages.utils.dataset_utils import Dataset

app = typer.Typer()


@app.command()
def train(
    epochs: int = typer.Option(1, help="number of epochs to train")
):
    """
    Train model
    """
    wandb.init(project=proj['proj_name'])
    trainer = OverfitTrainer(params, run_dir=run_dir)
    trainloader, _ = Dataset(params, run_dir=run_dir).dataloader()
    trainer.train(trainloader, num_epochs=epochs, restore_file='last')


@app.callback()
def main(
    ctx: typer.Context
):
    """
    Overfit model on training set
    """
    # get project context
    global proj
    proj = ctx.obj

    # get running directory
    global run_dir
    run_dir = os.path.join(os.getcwd(), '03_overfit')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    typer.echo("Running directory for dataset.py: {}".format(run_dir))

    # get runset parameters
    global params
    json_path = os.path.join(run_dir, 'runset.json')
    params = Params(json_path)


if __name__ == '__main__':
    app()
