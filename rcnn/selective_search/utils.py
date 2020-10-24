import numpy as np
from skimage.util import img_as_float
from skimage.segmentation import felzenszwalb


def graph_based_segmentation(image, scale, sigma, min_size):
    return np.dstack([
        image, felzenszwalb(
            img_as_float(image), scale=scale,
            sigma=sigma, min_size=min_size
        )
    ])
