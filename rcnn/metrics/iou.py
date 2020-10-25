import numpy as np


def get_bbox_dimensions(xmin1, xmax1, ymin1, ymax1):
    width1 = xmax1 - xmin1
    height1 = ymax1 - ymin1
    area1 = width1 * height1
    return width1, height1, area1


def get_iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    width1, height1, area1 = get_bbox_dimensions(
        xmin1, xmax1, ymin1, ymax1
    )
    width2, height2, area2 = get_bbox_dimensions(
        xmin2, xmax2, ymin2, ymax2
    )
    int_xmin = np.max([xmin1, xmin2])
    int_ymin = np.max([ymin1, ymin2])
    int_xmax = np.min([xmax1, xmax2])
    int_ymax = np.min([ymax1, ymax2])
    int_width = int_xmax - int_xmin
    int_height = int_ymax - int_ymin
    int_area = int_width * int_height
    return 0 if (int_width < 0) or (int_height < 0) \
        else int_area / float(area1 + area2 - int_area)
