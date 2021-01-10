"""
    Utility functions for dataloader.py
"""

def make_weights_for_balanced_classes(images, nclasses):
    """
    Get a set of weights for label-balanced sampling

    Args:
        images: (tensor) a tensor object that stores images in dataset
        nclasses: (int) number of classes in the dataset

    Copied from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    """
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight
