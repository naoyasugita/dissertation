import os

import cv2
import numpy as np
from tqdm import tqdm

raw_datas = {
    "train_data": "./dataset/raw_data/Roads/Training_Set/",
    "val_data": "./dataset/raw_data/Roads/Validation_Set/",
    "test_data": "./dataset/raw_data/Roads/Test_Set/"
}

type_names = ["./dataset/data/Roads/"]
set_names = ["train/", "valid/", "test/"]
dir_names = ["x_{}", "y_{}"]

datas = []
for type_name in type_names:
    for set_name in set_names:
        for dir_name in dir_names:
            datas.append(type_name + set_name + dir_names.format(set_name))


def makedir(datas):
    for data in datas:
        os.makedirs(data, exist_ok=True)


# 1500*1500の画像を読み込む際に使う.(.tif(tiff) -> np.array)
def load_raw_data(data):
    file_list = os.listdir(data)
    x, y = [], []  # 訓練データ, 教師データ
    x_name, y_name = [], []  # 画像の名前
    for file_name in tqdm(file_list, desc="load_raw_data"):
        img = cv2.imread(data + file_name)
        root, ext = os.path.splitext(file_name)
        if ext == ".tiff":
            x.append(img)
            x_name.append(root)
        else:
            y.append(img)
            y_name.append(root)
    return ([np.array(x), x_name], [np.array(y), y_name])


# 1500*1500 -> 256*256に分割する
def data_split(image, path, file_name):
    DIV = 256
    index = 0
    for i in range(len(image) // DIV):
        for j in range(len(image) // DIV):
            clp = image[DIV * i:DIV * (i + 1), DIV * j:DIV * (j + 1)]
            cv2.imwrite(path + file_name + "_" +
                        str(index).zfill(3) + ".jpg", clp)
            index += 1


if __name__ == '__main__':
    makedir(datas)
    for var, data in tqdm(raw_datas.items(), desc="main processing"):
        (x, y) = load_raw_data(data)
        for i in range(x[0].shape[0]):
            if var == "train_data":
                data_split(x[0][i], datas[0], x[1][i])
                data_split(y[0][i], datas[1], y[1][i])
            elif var == "val_data":
                data_split(x[0][i], datas[2], x[1][i])
                data_split(y[0][i], datas[3], y[1][i])
            elif var == "test_data":
                data_split(x[0][i], datas[4], x[1][i])
                data_split(y[0][i], datas[5], y[1][i])