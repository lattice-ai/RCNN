import cv2
from .utils import *


cv2.setUseOptimized(True)
cv2.setNumThreads(4)


def selective_search(original_image, scale, sigma, min_size):
    # [r,g,b,(region)]
    img = graph_based_segmentation(original_image, scale, sigma, min_size)
    if img is None:
        return None, {}
    imsize = img.shape[0] * img.shape[1]
    R = extract_regions(img)
    # extract neighbouring information
    neighbours = extract_neighbours(R)
    # calculate initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = get_similarity(ar, br, imsize)
    # hierarchal search
    while S != {}:
        # get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]
        # merge corresponding regions
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])
        # mark similarities for regions to be removed
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)
        # remove old similarities of related regions
        for k in key_to_delete:
            del S[k]
        # calculate similarity set with the new region
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = get_similarity(R[t], R[n], imsize)
    regions = []
    for k, r in list(R.items()):
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                # r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
                r['max_x'] , r['max_y'] ),
            'size': r['size'],
            'labels': r['labels']
        })
    return img, regions


def opencv_selective_search(image):
    segmentation = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    segmentation.setBaseImage(image)
    segmentation.switchToSelectiveSearchFast()
    return segmentation.process()
