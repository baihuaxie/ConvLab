"""
dataset inspection
"""

import os
import typer
import logging

from stages.utils.dataset_utils import get_labels_counts
from common.data_loader import fetch_dataloader
from common.utils import Params, set_logger

app = typer.Typer()


@app.command()
def get_label_stats():
    """
    Print out statistics on the distribution of labels
    """
    num_classes, train_dl, val_dl = dataloader()
    typer.echo(get_labels_counts(train_dl, num_classes))


def dataloader():
    """
    Inspect dataset
    """
    # run_directory
    run_dir = os.path.join(os.getcwd(), '01_dataset')

    # read .json
    json_path = os.path.join(run_dir, 'runset.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # get dataset parameters from runset.json
    dataset = params.data['dataset']
    datadir = params.data['datadir']
    num_classes = params.data['num_classes']
    trainloader_kwargs = params.data['trainloader-kwargs']
    trainset_kwargs = params.data['trainset-kwargs']
    valloader_kwargs = params.data['valloader-kwargs']
    valset_kwargs = params.data['valset-kwargs']

    # set the logger
    set_logger(os.path.join(run_dir, 'dataset.log'))

    logging.info('Loading dataset...')

    # fetch data into DataLoader
    data_loaders = fetch_dataloader(['train', 'val'], datadir, dataset, \
            trainloader_kwargs, trainset_kwargs, valloader_kwargs, valset_kwargs)
    train_dl = data_loaders['train']
    val_dl = data_loaders['val']

    logging.info('Done.')

    return num_classes, train_dl, val_dl



if __name__ == '__main__':
    app()

