"""
    test inspection.py
"""

from typer.testing import CliRunner

from stages.inspection import app

runner = CliRunner()


def test_check_seed():
    results = runner.invoke(app, ['check-seed'])
    assert results.exit_code == 0

def test_check_init_loss():
    results = runner.invoke(app, ['check-init-loss'])
    assert results.exit_code == 0

def test_check_under_fit():
    results = runner.invoke(app, ['check-underfit'])
    assert results.exit_code == 0

def test_train_input_grounded():
    results = runner.invoke(app, ['train-input-grounded'])
    assert results.exit_code == 0

def test_overfit_batch():
    results = runner.invoke(app, ['overfit-batch'])
    assert results.exit_code == 0