"""
    main script to launch training stages
"""

import os
import typer


from stages import dataset, inspection

app = typer.Typer()

# add CLI sub-commands
app.add_typer(dataset.app, name='dataset')
app.add_typer(inspection.app, name='inspection')


@app.callback()
def main(rundir: str = './stages/tests/'):
    """
    Initializes launch directory, etc.
    """
    # change to run directory
    os.chdir(rundir)

if __name__ == '__main__':
    app()


    