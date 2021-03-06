"""
in this script, we calculate the image per channel mean and standard
deviation in the training set, do not calculate the statistics on the
whole dataset.
"""

import numpy as np
import cv2
import timeit
from os import listdir
from os.path import isdir
from tqdm import tqdm

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3


def cal_dir_stat(root: str):

    if not root.endswith('/'):
        root += '/'

    file_paths = [root + f for f in listdir(root) if not isdir(f)]
    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    for file_path in tqdm(file_paths):
        im = cv2.imread(file_path)  # image in M*N*CHANNEL_NUM shape, channel in BGR order
        # im = im / 255.0
        pixel_num += (im.size / CHANNEL_NUM)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    return rgb_mean, rgb_std


# The script assumes that under train_root, there are separate directories for each class
# of training images.
train_root = "training_images/"
start = timeit.default_timer()
mean, std = cal_dir_stat(train_root)
end = timeit.default_timer()
print("elapsed time: {}".format(end - start))
print("mean:{}\nstd:{}".format(mean, std))
