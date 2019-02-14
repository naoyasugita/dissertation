import os
import sys

import cv2
from tqdm import tqdm


# ファイル名から画像をピクセル単位で扱えるように変換して、配列に入れる
def image_read(path, array):
    empty_arr = [[], [], [], [], []]
    for i, arr in enumerate(array):
        for name in arr:
            empty_arr[i].append(cv2.imread(path + "/" + name))
    return empty_arr


# 分割された画像(256*256)を結合させて一枚(1280*1280)にする
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


# 直前の関数を使って実際に結合処理を行う
def make_originalsize_img(path):
    pred_dir = os.listdir(path)
    img_arr = []
    for i in range(len(pred_dir) // 25):
        img_arr.append([[], [], [], [], []])

    for i, images in enumerate(pred_dir):
        a = i // 25
        b = (i // 5) % 5
        img_arr[a][b].append(images)
    for i, local_path in enumerate(img_arr):
        arr = image_read(path, local_path)
        img = concat_tile(arr)
        cv2.imwrite(path.split("images")[0] + "large/" +
                    pred_dir[i * 25].split("_")[0] + ".jpg", img)
