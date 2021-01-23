"""
    main script to launch training stages
"""

import os
import typer

from stages import dataset, inspection, overfit

app = typer.Typer()

# add CLI sub-commands
app.add_typer(dataset.app, name='dataset')
app.add_typer(inspection.app, name='inspection')
app.add_typer(overfit.app, name='overfit')


@app.callback()
def main(
    ctx: typer.Context,
    rundir: str = './stages/tests/',
    project: str = 'myTestProj'
):
    """
    Initializes launch directory, etc.
    """
    # change to run directory
    os.chdir(rundir)
    context = {
        'rundir': rundir,
        'proj_name': project
    }
    ctx.obj = context

if __name__ == '__main__':
    app()


    