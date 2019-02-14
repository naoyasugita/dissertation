import os
import sys

import cv2
import numpy as np
from tqdm import tqdm_notebook as tqdm


"""
x, y = load_data("Roads" or "Buildings", "train", "test", "valid")
"""


def load_data(contents, type_, stop_num=0):
    index = 0
    data = "./dataset/data/{}/{}/".format(contents, type_)
    x_datas = data + "x_" + type_ + "/"
    y_datas = data + "y_" + type_ + "/"
    x, y = [], []
    x_dir = os.listdir(x_datas)
    y_dir = os.listdir(y_datas)
    for x_data, y_data in tqdm(zip(x_dir, y_dir), desc=type_):
        if index < stop_num:
            index += 1
            x.append(cv2.imread(x_datas + x_data))
            y.append(cv2.imread(y_datas + y_data))
        else:
            break
    return tuple(map(np.array, (x, y)))
