"""
test plotting utilities
"""

import pytest
import seaborn as sns

from common.plots import save_batch_summary, read_csv, \
    plot_batch_summary, read_rc_from_json

@pytest.mark.skip()
def test_save_batch_summary():
    """
    test save batch summaries into csv file
    """
    run_dir = './test_plots/'
    for idx in range(0, 5):
        summ = [{'a':idx}, {'a':idx*10}]
        save_batch_summary(run_dir, summ)

@pytest.mark.skip()
@pytest.mark.parametrize('index_col', \
    [
        'iteration',
        'batch'     # check exception handling
    ])
def test_read_csv(index_col):
    """
    test read csv statistics into pandas DataFrame object
    """
    csv_path = './test_plots/batch_summary.csv'
    df = read_csv(csv_path, index_col=index_col)
    print(df)

def test_plot_batch_summary():
    """
    test plot metric vs iteration
    """
    csv_path = './test_plots/batch_summary.csv'
    df = read_csv(csv_path, index_col='iteration')
    params = read_rc_from_json('./test_plots')
    sns.set(context=params.plotting_context, \
        style=params.axes_style)
    for metric in list(df):
        plot_batch_summary('./test_plots', df[metric], params, \
            metric=metric, prefix='train_resnet18')