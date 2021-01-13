"""
    Stage-2: validate an end-to-end training/evaluate skeleton & obtain baselines
"""

import typer

app = typer.Typer()

@app.command()
def init():
    """
    """
    print("do sth...")

@app.callback()
def callback():
    """
    """
    print("I am inspection.py callback!")


if __name__ == '__main__':
    print("I am inspection.py!")