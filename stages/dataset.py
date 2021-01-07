"""
dataset inspection
"""

import typer

app = typer.Typer()

@app.command()
def label_cdf(labels):
    """
    Plot cumulative distribution function (cdf) of class labels vs. label id's

    Args:
        labels: (list) a list object that contains class labels for all samples in the set
    """



if __name__ == '__main__':
    app()

