"""
dataset inspection
"""

import os
import typer


from stages.utils.dataset_utils import Dataset, get_labels_counts


app = typer.Typer()

@app.command()
def get_label_stats():
    """
    Print out statistics on the distribution of labels
    """
    myData = Dataset(run_dir)
    train_dl, _ = myData.dataloader()
    typer.echo(get_labels_counts(train_dl, myData.num_classes))


@app.callback()
def main():
    """
    Inspect dataset
    """
    global run_dir
    run_dir = os.path.join(os.getcwd(), '01_dataset')
    typer.echo("Running directory for dataset.py: {}".format(run_dir))


if __name__ == '__main__':
    app()

