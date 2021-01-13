"""
    main script to launch training stages
"""

import os
import typer


from stages import dataset

app = typer.Typer()

# add CLI sub-commands
app.add_typer(dataset.app, name='dataset')

@app.callback()
def main(rundir: str = './stages/tests/directory/'):
    """
    Initializes launch directory, etc.
    """
    # change to run directory
    os.chdir(rundir)

if __name__ == '__main__':
    app()


    