import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from plotly import express as px
from matplotlib import pyplot as plt


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
