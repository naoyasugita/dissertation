import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

root = "../dataset/learning_process_graph/"
pathes = os.listdir(root)
csv_arr = []

for path in pathes:
    csv_arr.append(glob.glob(root + path + "/*.csv"))


def csv2img(_file):
    df = pd.read_csv(_file)
    x = df["Step"]
    y = df["Value"]
    label_name = _file.split("-")[-1].split(".")[0]
    file_name = _file.split("/")[-2]
    plt.plot(x, y, label=label_name)
    plt.title(file_name + "_" + label_name)
    plt.xlabel('epoch')
    if "acc" in label_name:
        plt.ylabel('accuracy')
    else:
        plt.ylabel('loss')
    plt.legend()
    plt.savefig(root + "/" + file_name + "/" + label_name + ".png")
    plt.close()


for data in csv_arr:
    for j in range(len(data)):
        csv2img(data[j])
