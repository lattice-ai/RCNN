import numpy as np
from skimage.util import img_as_float
from skimage.segmentation import felzenszwalb
from skimage.feature import local_binary_pattern


def graph_based_segmentation(image, scale, sigma, min_size):
    return np.dstack([
        image, felzenszwalb(
            img_as_float(image), scale=scale,
            sigma=sigma, min_size=min_size
        )
    ])


def get_color_histogram(image, bins=25):
    histogram = np.array([])
    for channel in (0, 1, 2):
        # extracting one colour channel
        c = image[:, channel]
        # calculate histogram for each colour and join to the result
        histogram = np.concatenate(
            [histogram] + [np.histogram(
                c, bins, (0.0, 255.0)
            )[0]]
        )
    # L1 normalize
    histogram = histogram / len(image)
    return histogram


def get_texture_gradient(image):
    gradient = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    gradient[:, :, 0] = local_binary_pattern(image[:, :, 0], 8, 1.0)
    gradient[:, :, 1] = local_binary_pattern(image[:, :, 1], 8, 1.0)
    gradient[:, :, 2] = local_binary_pattern(image[:, :, 2], 8, 1.0)
    return gradient


def get_texture_histogram(image, bins=10):
    histogram = np.array([])
    for channel in (0, 1, 2):
        # mask by the colour channel
        c = image[:, channel]
        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        histogram = np.concatenate(
            [histogram] + [np.histogram(c, bins, (0.0, 1.0))[0]])
    # L1 Normalize
    histogram = histogram / len(image)
    return histogram
