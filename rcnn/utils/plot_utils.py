import cv2
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from ..metrics import get_iou
from collections import Counter
from plotly import express as px
from matplotlib import pyplot as plt
from ..selective_search.utils import (
    graph_based_segmentation, get_color_histogram,
    get_texture_gradient, get_texture_histogram, extract_regions
)
from ..selective_search import selective_search, opencv_selective_search


# https://xkcd.com/color/rgb/
common_colors = sns.xkcd_rgb.values()


def plot_objects_per_image(dataframe):
    fig = px.histogram(
        dataframe, x='n_objects', nbins=100,
        title='Histogram of the number of objects per image'
    )
    fig.show()


def plot_class_frequency_distribution(dataframe):
    all_class_names = []
    for i in range(1, np.max(dataframe['n_objects']) + 1):
        classes = list(dataframe['class_' + str(i)])
        all_class_names += classes
    counter = dict(Counter(all_class_names))
    classes_col = list(counter.keys())
    classes_col.pop(-1)  # removing NaN
    counts_col = [counter[key] for key in classes_col]
    frequency_df = pd.DataFrame(
        {
            'classes': classes_col,
            'counts': counts_col
        }
    )
    fig = px.bar(
        frequency_df, x='classes', y='counts',
        title='Frrequency distribution of classes'
    )
    fig.show()


def plot_bbox(label, xmin, ymin, xmax, ymax, backgroundcolor='pink', apply_label=True, line_color='yellow'):
    if apply_label:
        plt.text(xmin, ymin, label, fontsize=20, backgroundcolor=backgroundcolor)
    plt.plot([xmin, xmin], [ymin, ymax], linewidth=3, color=line_color)
    plt.plot([xmax, xmax], [ymin, ymax], linewidth=3, color=line_color)
    plt.plot([xmin, xmax], [ymin, ymin], linewidth=3, color=line_color)
    plt.plot([xmin, xmax], [ymax, ymax], linewidth=3, color=line_color)


def plot_from_dataframe(dataframe, index):
    data  = dataframe.iloc[index, :]
    image_path = './VOCdevkit/VOC2012/JPEGImages/' + data['image_id'] + ".jpg"
    image = Image.open(image_path)
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.title(data['image_id'] + ".jpg")
    for index in range(data['n_objects']):
        plot_bbox(
            data['class_' + str(index + 1)],
            data['xmin_' + str(index + 1)],
            data['ymin_' + str(index + 1)],
            data['xmax_' + str(index + 1)],
            data['ymax_' + str(index + 1)]
        )
    plt.show()


def plot_segmentation_samples(images_list, scale, sigma, min_size):
    for index, image_path in enumerate(images_list):
        original_image = Image.open(image_path)
        segmentation_result = graph_based_segmentation(
            np.array(original_image), scale, sigma, min_size
        )[:, :, -1]
        fig = plt.figure(figsize=(12, 24))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(original_image)
        ax.set_title('Image_{}'.format(index + 1))
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(segmentation_result)
        ax.set_title(
            'Mask_{}\nscale = {}, sigma = {}, min_size = {} \nUnique Regions: {}'.format(
                index + 1, scale, sigma, min_size, len(np.unique(segmentation_result))
            )
        )
        plt.show()


def plot_color_histogram(images_list):
    for index, image_path in enumerate(images_list):
        original_image = np.array(Image.open(image_path))
        color_histogram = get_color_histogram(original_image)
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(original_image)
        ax.set_title('Image_{}'.format(index + 1))
        ax = fig.add_subplot(1, 2, 2)
        ax.hist(color_histogram, bins=25)
        ax.set_title('Color_Histogram_{}'.format(index + 1))
        plt.show()


def plot_texture_gradients(images_list):
    for index, image_path in enumerate(images_list):
        original_image = np.array(Image.open(image_path))
        texture_gradient = get_texture_gradient(original_image)
        fig = plt.figure(figsize=(12, 24))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(original_image)
        ax.set_title('Image_{}'.format(index + 1))
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(texture_gradient.astype(np.uint8))
        ax.set_title('Texture_Gradient_{}'.format(index + 1))
        plt.show()


def plot_texture_histogram(images_list):
    for index, image_path in enumerate(images_list):
        original_image = np.array(Image.open(image_path))
        texture_histogram = get_texture_histogram(original_image)
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(original_image)
        ax.set_title('Image_{}'.format(index + 1))
        ax = fig.add_subplot(1, 2, 2)
        ax.hist(texture_histogram, bins=10)
        ax.set_title('Texture_Histogram_{}'.format(index + 1))
        plt.show()


def plot_all_regions(image_path, scale, sigma, min_size, figsize=(15, 15)):
    plt.figure(figsize=figsize)
    image = Image.open(image_path)
    segmentation_result = graph_based_segmentation(image, 1.0, 0.8, 500)
    regions = extract_regions(segmentation_result)
    plt.imshow(image)
    plt.xlabel('Regions: {}'.format(len(regions.values())))
    for item, color in zip(regions.values(), common_colors):
        x1 = item["min_x"]; y1 = item["min_y"]
        x2 = item["max_x"]; y2 = item["max_y"]
        label = item["labels"][0]
        plot_bbox(label, x1, y1, x2, y2, backgroundcolor=color)
    plt.show()


def plot_selected_regions(image_path, scale, sigma, min_size, figsize=(15, 15)):
    plt.figure(figsize=figsize)
    image = Image.open(image_path)
    regions = selective_search(image, scale, sigma, min_size)[1]
    plt.imshow(image)
    plt.xlabel('Regions: {}'.format(len(regions)))
    for item, color in zip(regions, common_colors):
        x1 = item['rect'][0]; y1 = item['rect'][1]
        # x2 = item['rect'][0] + item['rect'][2]
        # y2 = item['rect'][1] + item['rect'][3]
        x2 = item['rect'][2]
        y2 = item['rect'][3]
        label = item["labels"][0]
        plot_bbox(label, x1, y1, x2, y2, backgroundcolor=color)
    plt.show()


def plot_iou(image_path, actual_box, scale, sigma, min_size, figsize=(12, 12)):
    sample_image = Image.open(image_path)
    _, regions = selective_search(
        sample_image, scale=scale,
        sigma=sigma, min_size=min_size
    )
    proposed_bboxes = [region['rect'] for region in regions]
    plt.figure(figsize=figsize)
    for box in proposed_bboxes:
        iou = get_iou(
            actual_box[1], actual_box[3],
            actual_box[0], actual_box[2],
            box[0], box[1], box[2], box[3]
        )
        plt.imshow(sample_image)
        if iou > 0.5:
            plot_bbox(
                'iou={}'.format(iou), box[0], box[1],
                box[2], box[3], backgroundcolor='yellow'
            )
            plot_bbox(
                '', actual_box[1], actual_box[3],
                actual_box[0], actual_box[2], line_color='pink'
            )


def opencv_plot_iou(image_path, actual_box, figsize=(12, 12)):
    sample_image = Image.open(image_path)
    proposed_bboxes = opencv_selective_search(cv2.imread(image_path))
    plt.figure(figsize=figsize)
    for box in proposed_bboxes:
        iou = get_iou(
            actual_box[1], actual_box[3],
            actual_box[0], actual_box[2],
            box[0], box[1], box[2] + box[0],
            box[3] + box[1]
        )
        plt.imshow(sample_image)
        if iou > 0.5:
            plot_bbox(
                'iou={}'.format(iou), box[0], box[1],
                box[2], box[3], backgroundcolor='yellow'
            )
            plot_bbox(
                '', actual_box[1], actual_box[3],
                actual_box[0], actual_box[2], line_color='pink'
            )
