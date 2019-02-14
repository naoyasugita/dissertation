import csv
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm_notebook as tqdm


# 混同行列を作成
def plot_confusion_matrix(cm, title="Confusion matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["p", "n"])
    plt.yticks(tick_marks, ["Back", "Road"])
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# 作成した図を表示して保存
def do_plot(test, pred, path, title="Normalized Confusion matrix"):
    cm = confusion_matrix(test, pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    cm_normalized = [[cm_normalized[1][1], cm_normalized[1][0]],
                     [cm_normalized[0][1], cm_normalized[0][0]]]

    plt.figure()
    plot_confusion_matrix(cm_normalized, title=title)
    plt.savefig(path + "figure.png")  # learned_model_path
    plt.show()


# 差異を出力するために必要なデータを作成
def get_diff_data(pred_path):
    bases = []
    preds = []
    filenames = []
    base_dir = "./dataset/base_data/Roads/Test_Set/"
    pred_dir = "./dataset/eva_imgs/{}/large/".format(pred_path)
    for base, pred in tqdm(zip(glob.glob(base_dir + "*.tif"), os.listdir(pred_dir)), desc="loading... "):
        filenames.append(os.path.splitext(pred)[0])
        preds.append(pred_dir + pred)
        bases.append(base.replace('\\', '/'))
    return bases, preds, filenames


# 出力画像とテスト画像の差異を表示
def show_diff(pred_path):
    diff_arr = []
    bases, preds, filenames = get_diff_data(pred_path)
    for num, filename in tqdm(enumerate(filenames), desc="images ", total=len(filenames)):
        match_count = 0
        base = cv2.imread(bases[num])
        pred = cv2.imread(preds[num])
        for i in tqdm(range(1280), desc="check match ", leave=False):
            for j in range(1280):
                # 予測済みデータには0, 255以外に半端な値が多く含まれているためそれらを除いた値でカウントしている
                if sum(base[i][j]) == sum(pred[i][j]):
                    match_count += 1
        diff_arr.append([filename, match_count])
    # print(diff_arr)
    columns = ["image_name", "match_count"]
    df = pd.DataFrame(diff_arr, columns=columns)
    return df


# テスト画像と予測済みデータを比較して誤っているデータの散布図を出力
def plot_diff_scat(test, pred):
    df = df.set_index("image_name")
    plt.scatter(df.index, df['match_count'])
    plt.xticks(rotation=70)
    plt.show()


# テスト画像と予測済みデータを比較して一致しているピクセル数を棒グラフとして出力
def plot_diff_bar(df, path, num=10, asce=False):
    df_asce = df.sort_values(by="match_count", ascending=asce)
    plt.bar(df_asce['image_name'][:num], df_asce['match_count'][:num])
    plt.xticks(rotation=90)
    if (asce):
        plt.savefig(path + "bar_graph.png")  # learned_model_path
    else:
        plt.savefig(path + "arc_bar_graph.png")  # learned_model_path
    plt.show()


# 出力のネガポジを判定
def make_diff_image(pred_path):
    WHITE = 765
    BLACK = 0
    img_size = 1280
    diff_arr = []
    bases, preds, filenames = get_diff_data(pred_path)
    validation_data = [filenames]
    for num, name in tqdm(enumerate(filenames)):
        diff_arr.append([])
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        base = cv2.imread(bases[num])
        pred = cv2.imread(preds[num])
        for i in range(img_size):
            for j in range(img_size):
                # 予測済みデータには0, 255以外に半端な値が多く含まれているためそれらを除いた値でカウントしている
                if abs(sum(base[i][j]) - WHITE) < 100 and abs(sum(pred[i][j]) - WHITE) < 100:
                    diff_arr[num].append([255, 255, 255])  # 白
                    TP += 1
                elif abs(sum(base[i][j]) - BLACK) < 100 and abs(sum(pred[i][j]) - BLACK) < 100:
                    diff_arr[num].append([0, 0, 0])  # 黒
                    TN += 1
                elif sum(base[i][j]) == BLACK and sum(pred[i][j]) != BLACK:
                    diff_arr[num].append([0, 128, 0])  # 緑
                    FP += 1
                elif sum(base[i][j]) == WHITE and sum(pred[i][j]) != WHITE:
                    diff_arr[num].append([255, 0, 0])  # 赤
                    FN += 1
        validation_data.append([TP, TN, FP, FN])
    return diff_arr, filenames, validation_data


# ネガポジ判定した値を画像として出力(赤:PN, 緑:NP)
def plot_diff_image(now):
    save_dir = "./dataset/pn_imgs/" + str(now) + "/"
    os.makedirs(save_dir, exist_ok=True)
    arr, filenames, validation_data = make_diff_image(now)
    for i, name in tqdm(enumerate(filenames), total=len(arr)):
        img = np.uint8(arr[i]).reshape(1280, 1280, 3)
        img = Image.fromarray(img)
        img.save(save_dir + name + ".jpg")
    return validation_data


# 混同行列と評価値を画像ごとにCSVとして出力
def evaluation(validation_data, now):
    completeness = []
    correctness = []
    quality = []
    count = 0
    filename = validation_data[0]
    validation_data = validation_data[1:]
    save_dir = "./dataset/evaluation/"
    os.makedirs(save_dir, exist_ok=True)
    for name, data in tqdm(zip(filename, validation_data)):
        TP = data[0]
        TN = data[1]
        FP = data[2]
        FN = data[3]
        completeness = '%.4f' % (TP / (TP + FN))
        correctness = '%.4f' % (TP / (TP + FP))
        quality = '%.4f' % (TP / (TP + FN + FP))
        with open(save_dir + str(now) + ".csv", 'a') as f:
            fieldnames = ['image_name', 'TruePositive', 'FalsePositive', 'TrueNegative',
                          'FalseNegative', 'Completeness', 'Correctness', 'Quality']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if count == 0:
                writer.writeheader()
            writer.writerow({'image_name': name, 'TruePositive': TP, 'FalsePositive': TN, 'TrueNegative': FP, 'FalseNegative': FN,
                             'Completeness': completeness, 'Correctness': correctness, 'Quality': quality})
        count += 1


# evaluation('01172024')

# filename = "./dataset/evaluation/01161510.csv"
# f = pd.read_csv(filename)
# y = f['Quality']
# x_name = f['image_name']
# xx = [i for i in range(49)]
# plt.barh(xx, y, height=0.3, align='center')
# plt.yticks(xx, x_name)
# plt.show()
