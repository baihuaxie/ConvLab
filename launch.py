"""
    main script to launch training stages
"""

import typer

from stages import inspection


app = typer.Typer()

# add CLI sub-commands
app.add_typer(inspection.app, name='inspection')


if __name__ == '__main__':
    print("I am launch.py")
    app()


    