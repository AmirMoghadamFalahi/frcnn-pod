"""
in this script, we calculate the bbox common ratio
deviation in the training set, do not calculate the statistics on the
whole dataset.
"""

import numpy as np
import cv2
import timeit
from os import listdir
from os.path import isdir
from bs4 import BeautifulSoup
from tqdm import tqdm


def calculate_mean_ratio(xml_path: str):

    if not xml_path.endswith('/'):
        xml_path += '/'

    xml_files = sorted(listdir(xml_path))
    for i in tqdm(range(len(xml_files)), desc='finding common ratios'):

        xml_file = open(xml_path + xml_files[i])
        soup = BeautifulSoup(xml_file.read(), 'xml')
        annotation = soup.annotation
        if not annotation:
            continue

        objs = soup.findAll('object')

        for obj in objs:
            x1 = int(obj.find('xmin').text)
            x2 = int(obj.find('xmax').text)
            y1 = int(obj.find('ymin').text)
            y2 = int(obj.find('ymax').text)

            object_type = obj.find('name').text

            w = x2 - x1
            h = y2 - y1

            print(x1, y1, x2, y2, object_type, w, h)

        if i >= 25:
            break


if __name__ == '__main__':
    # The script assumes that under train_root, there are separate directories for each class
    # of training images.
    train_root = "training_images/"
    start = timeit.default_timer()
    calculate_mean_ratio(train_root)
