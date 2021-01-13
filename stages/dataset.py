"""
dataset inspection
"""

import os
import typer

import torchvision.transforms as transforms

from stages.utils.dataset_utils import Dataset, get_labels_counts, get_classes, \
    show_labelled_images


app = typer.Typer()

@app.command()
def get_label_stats():
    """
    Print out statistics on the distribution of labels
    """
    myData = Dataset(run_dir)
    train_dl, _ = myData.dataloader()
    typer.echo(get_labels_counts(train_dl, myData.num_classes))


default_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(160),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
)

@app.command()
def get_samples(
    nrows: int = typer.Option(8, help="number of columns in sample grid"),
    ncols: int = typer.Option(8, help="number of rows in sample grid"),
):
    """
    Print selected training samples with ground truth labels

    Note:
    - the samples are taken at random or in label-balanced fashion
    - by default the samples are saved under './samples/' of current run directory
    """
    myData = Dataset(run_dir)
    images, labels = myData.sampler(transform=default_transform, balanced=True)
    classes = get_classes(myData.dataset, myData.datadir)
    show_labelled_images(myData.dataset, images, labels, classes, nrows=nrows, ncols=ncols, \
        savepath=os.path.join(run_dir, 'samples'))


@app.callback()
def main():
    """
    Inspect dataset
    """
    # set running directory for dataset sub-command
    global run_dir
    run_dir = os.path.join(os.getcwd(), '01_dataset')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    typer.echo("Running directory for dataset.py: {}".format(run_dir))


if __name__ == '__main__':
    app()

