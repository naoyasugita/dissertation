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

# for i in range(len(csv_arr)):
#     x = pd.read_csv(csv_arr[i][0])["Step"]
#     print(x)
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
#     for j in range(len(csv_arr[i])):
#         df = pd.read_csv(csv_arr[i][j])
#         y = df["Value"]
#         a, b = list(map(int, format(j, '02b')))
#         axes[a, b].plot(x, y, linewidth=2)
#         axes[a, b].set_title("test")
#         axes[a, b].set_xlabel('epoch')
#         axes[a, b].set_xlim(0, 1)
#         axes[a, b].grid(True)
#     fig.show()
