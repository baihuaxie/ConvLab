"""
script to plot standard statistics after training
"""

import os
import json
import argparse

# commandline arguments; normally just use defaults
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='./experiments/', \
    help='Parent directory for all experiments')
parser.add_argument('--jobs_dir', default='./', \
    help='Directory containing jobs.json file')

if __name__ == '__main__':
    args = parser.parse_args()
    # read parameters of all job runs from jobs.json
    json_path = os.path.join(args.jobs_dir, 'jobs.json')
    jobs = Params(json_path)


        args = parser.parse_args()

    json_path = args.params
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    # find dataset loading keyword arguments
    kwargs = match_dict_by_value(params.data, 'dataset', args.dataset)

    images, labels = select_n_random('train', args.datadir, args.dataset, kwargs['trainset-kwargs'], \
        kwargs['valset-kwargs'], n=20)
    classes = get_classes(args.dataset, args.datadir)
    savepath = op.join(args.savepath, args.dataset)
    show_labelled_images(images, labels, classes, nrows=4, ncols=4, savepath=savepath)