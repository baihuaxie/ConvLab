"""
utilities for dataset inspection
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

def show_images(img):
    """
    print images

    Args:
        img: (np.ndarray or tensor) images
    """
    if isinstance(img, torch.Tensor):
        npimg = img.numpy()
    elif isinstance(img, np.ndarray):
        npimg = img
    else:
        raise TypeError("Image type {} not recognized".format(type(img)))
    # assumes npimg shape = CxHxW; transpose to HxWxC
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



