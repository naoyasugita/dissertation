import os
import shutil
import sys

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

set_names = ["train", "valid", "test"]
set_types = ["sat", "map"]
dir_names = ["/Training_Set/", "/Validation_Set/", "/Test_Set/"]

# 大学内からプロキシを通すための設定
proxies = {
    'http': 'http://wwwproxy.cc.sojo-u.ac.jp:3128',
    'https': 'https://wwwproxy.cc.sojo-u.ac.jp:3128'
}

road_path = "Roads"


def check_dirs(path):
    os.makedirs("./dataset/raw_data/Roads/" + path, exist_ok=True)


# データセットの一覧ページから各データセットのリンクのみを配列で取得.
def create_url():
    BASE_URLS = []
    BASE_URL = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/{}/{}/index.html"
    for i in set_names:
        for j in set_types:
            BASE_URLS.append(BASE_URL.format(i, j))
    return BASE_URLS


# それぞれのリンク内に含まれる画像のリンクを配列で取得.
def get_url(url):
    urls = []
    html = requests.get(url, proxies=proxies).text
    soup = BeautifulSoup(html)
    list_a = soup.find_all("a")
    for a_tag in list_a:
        filename = a_tag.getText()
        if filename != "index.html":
            url = str(a_tag).split('"')[1]
            urls.append([filename, url])
    return urls


# リンクを指定したディレクトリにダウンロードする.
def download(dir_name, filename, url):
    res = requests.get(url, stream=True)
    with open('./dataset/raw_data/Roads/' + dir_name + filename, 'wb') as download_path:
        res.raw.decode_content = True
        shutil.copyfileobj(res.raw, download_path)


if __name__ == '__main__':
    for i in range(len(dir_names)):
        check_dirs(road_path + dir_names[i])
    print("=========== Download ===========")
    BASE_URLS = create_url()
    for i, url in tqdm(enumerate(BASE_URLS), desc="download"):
        urls = get_url(url)
        for filename, image_path in tqdm(urls):
            if 'train' in url:
                download("/Training_Set/", filename, image_path)
            if 'valid' in url:
                download("/Validation_Set/", filename, image_path)
            if 'test' in url:
                download("/Test_Set/", filename, image_path)
