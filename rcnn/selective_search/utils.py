import numpy as np
from skimage.color import rgb2hsv
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


def extract_regions(image):
    regions = {}
    hsv = rgb2hsv(image[:, :, :3])
    # count pixel positions
    for y, i in enumerate(image):
        for x, (r, g, b, l) in enumerate(i):
            # initialize a new region
            if l not in regions:
                regions[l] = {
                    "min_x": 0xffff,
                    "min_y": 0xffff,
                    "max_x": 0,
                    "max_y": 0,
                    "labels": [l]
                }
            # bounding box
            if regions[l]["min_x"] > x:
                regions[l]["min_x"] = x
            if regions[l]["min_y"] > y:
                regions[l]["min_y"] = y
            if regions[l]["max_x"] < x:
                regions[l]["max_x"] = x
            if regions[l]["max_y"] < y:
                regions[l]["max_y"] = y
    # calculate texture gradient
    texture_gradient = get_texture_gradient(image)
    # calculate colour histogram of each region
    for k, v in list(regions.items()):
        # colour histogram
        masked_pixels = hsv[:, :, :][image[:, :, 3] == k]
        regions[k]["size"] = len(masked_pixels / 4)
        regions[k]["hist_c"] = get_color_histogram(masked_pixels)
        # texture histogram
        regions[k]["hist_t"] = get_texture_histogram(
            texture_gradient[:, :][image[:, :, 3] == k]
        )
    return regions


def intersect(a, b):
    if (a["min_x"] < b["min_x"] < a["max_x"]
            and a["min_y"] < b["min_y"] < a["max_y"]) or (
        a["min_x"] < b["max_x"] < a["max_x"]
            and a["min_y"] < b["max_y"] < a["max_y"]) or (
        a["min_x"] < b["min_x"] < a["max_x"]
            and a["min_y"] < b["max_y"] < a["max_y"]) or (
        a["min_x"] < b["max_x"] < a["max_x"]
            and a["min_y"] < b["min_y"] < a["max_y"]):
        return True
    return False


def extract_neighbours(regions):
    R = list(regions.items())
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))
    return neighbours


def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


def get_color_similarity(r1, r2):
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def get_texture_similarity(r1, r2):
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def get_size_similarity(r1, r2, imsize):
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def get_fill_similarity(r1, r2, imsize):
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def get_similarity(r1, r2, imsize):
    color_sim = get_color_similarity(r1, r2)
    texture_sim = get_texture_similarity(r1, r2)
    size_sim = get_size_similarity(r1, r2, imsize)
    fill_sim = get_fill_similarity(r1, r2, imsize)
    return color_sim, texture_sim, size_sim, fill_sim
