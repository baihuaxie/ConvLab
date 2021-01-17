"""
    test dataset.py
"""

from typer.testing import CliRunner

from stages.dataset import app


runner = CliRunner()

def test_get_label_stats():
    """
    """
    results = runner.invoke(app, ['get-label-stats'])
    assert results.exit_code == 0

def test_get_samples():
    """
    """
    results = runner.invoke(app, ['get-samples'])
    assert results.exit_code == 0