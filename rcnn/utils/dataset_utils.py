import numpy as np
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm
import xml.etree.ElementTree as ET
from collections import OrderedDict


def get_file_list():
    image_files = glob('./VOCdevkit/VOC2012/JPEGImages/*.jpg')
    annotation_files = glob('./VOCdevkit/VOC2012/Annotations/*.xml')
    return image_files, annotation_files


def parse_xml(file_name):
    root = ET.parse(file_name)
    n_objects = 0
    image_data = OrderedDict()
    image_data['image_id'] = file_name.split('/')[-1].split('.')[0]
    for elements in root.iter():
        # size tag
        if elements.tag == 'size':
            for element in elements:
                image_data[element.tag] = int(element.text)
        # object tag
        elif elements.tag == 'object':
            for element in elements:
                if element.tag == 'name':
                    image_data[
                        'class_' + str(n_objects + 1)
                    ] = str(element.text)
                elif element.tag == 'bndbox':
                    for bbox_element in element:
                        image_data[
                            bbox_element.tag + '_' + str(n_objects + 1)
                        ] = float(bbox_element.text)
                    n_objects += 1
    image_data['n_objects'] = n_objects
    return image_data


def get_dataframe(annotation_files, save_location):
    dataframe = []
    for file_path in tqdm(annotation_files):
        image_data = parse_xml(file_path)
        dataframe.append(image_data)
    dataframe = pd.DataFrame(dataframe)
    if save_location is not None:
        dataframe.to_csv(save_location, index=False)
    return dataframe
