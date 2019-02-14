import os
import glob
from PIL import Image
from tqdm import tqdm

basedir = "./dataset/raw_data/"
data_types = os.listdir(basedir)
datasets = [os.listdir(basedir + data_type) for data_type in data_types][0]


# オリジナルの1500*1500の画像から、検証に用いるための1280*1280の画像を保存するディレクトリを作成
for data_type in data_types:
    for dataset in datasets:
        os.makedirs("./dataset/base_data/" + data_type +
                    "/" + dataset, exist_ok=True)


# オリジナルの1500*1500の画像から、検証に用いるための1280*1280の画像を作成
for data_type in data_types:
    for dataset in tqdm(datasets, desc=data_type):
        path = basedir + data_type + "/" + dataset + "/"
        new_path = "./dataset/base_data/" + data_type + "/" + dataset + "/"
        for f in tqdm(os.listdir(path), desc=dataset):
            img = Image.open(path + f)
            img.crop((0, 0, 1280, 1280)).save(new_path + f)
