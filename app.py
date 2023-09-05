from urllib.parse import urljoin, quote
import requests
from bs4 import BeautifulSoup
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import imgsim

import urllib
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse
import time
import tempfile
import schedule
import pickle
import os
import datetime
from datetime import time
import re

import streamlit as st

print("モジュールをインポートしました")

parser = argparse.ArgumentParser(description='コマンドライン引数の説明')

parser.add_argument('--word', type=str, help='検索ワードを入力')
parser.add_argument('--path', type=str, help='画像のパスを指定')

args = parser.parse_args()

# Load the VGG16 model only once
# model = VGG16(weights='imagenet', include_top=False)
    # Load the ResNet50 model with pre-trained ImageNet weights
# model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Define the lock for multithreading
# lock = threading.Lock()

def process_page(input_image_path, page_url, site_type):
    # 各ページの内容を取得
    page_response = requests.get(page_url)
    page_soup = BeautifulSoup(page_response.content, 'html.parser')

    img_tags = ""
    if site_type == "ヤフオク":
        tag = page_soup.find('h1', {'class': 'ProductTitle__text'})
        if tag:
            title = tag.text
            img_tags = page_soup.find_all('img',alt=title)
    elif site_type == "メルカリ":
        img_tags = page_soup.find_all('img',alt=re.compile(r'\d+のサムネイル'))

    distances = []
    # if input_text in title:
    for img in img_tags:
        img_url = img.get('src')
        if img_url.startswith('//'):
            img_url = 'http:' + img_url
        # 各画像の特徴量を計算
        try:
            with urllib.request.urlopen(img_url) as url:
                with open('temp.jpg', 'wb') as f:
                    f.write(url.read())
        except:
            continue

        # score = calculate_image_similarity(input_image_path,'temp.jpg')
        score = image_similarity(input_image_path,'temp.jpg')
        distances.append(score)
        # print(f"検索対象画像：{img_url}\nスコア：{score}")

    N = 0#ここは144~203のいずれか。そのため間を取って、175とする。
    if distances == []:
        return None
    elif max(distances) > N:
        print(f"URL:{page_url}スコア：{max(distances)}")
        return [page_url, max(distances)]
    else:
        return None

def find_similar_image_pages(input_text, input_image_path):
    log = st.empty()

    page_urls = get_url_from_yahoo(input_text)
    # page_urls2 = get_url_from_mercari(input_text)
    # page_urls.extend(page_urls2)
    # テキストと画像がマッチするページURLを格納するリスト
    matched_pages = []
    scores = []

    #同期処理
    with st.spinner(f"検索ワード{input_text}の全{len(page_urls)}個のURL処理中…"):
        # ページのスクレイピングと画像の特徴量抽出を同期処理で行うコード
        for i,url in enumerate(page_urls):
            if "yahoo" in url:
                site_type = "ヤフオク"
            elif "mercari" in url:
                site_type = "メルカリ"
            else:
                continue    
            result = process_page(input_image_path, url, site_type)
            if result:
                page_url, distance = result
                if page_url not in matched_pages:
                    matched_pages.append(page_url)
                    scores.append(distance)
            if len(page_urls) > 0:
                log.write(f"{i+1}/{len(page_urls)}")

    if matched_pages == []:
        st.error(f"{input_text}のURL取得失敗…")
    else:
        st.success(f"{input_text}のURL取得成功！")
    
    # ソートされたインデックスのリストを取得
    # sorted_indices = np.argsort(scores)[::-1]
    sorted_indices = np.argsort(scores)

    # 別のリストをソートされたインデックスの順に並び替え
    matched_pages = np.array(matched_pages)[sorted_indices].tolist()

    scores = np.array(scores)[sorted_indices].tolist()

    return matched_pages ,scores

def get_url_from_yahoo(input_text):
    # Yahoo AuctionのURL
    keyword = quote(input_text)
    b = 1
    page_urls = []
    while True:
        base_url = 'https://auctions.yahoo.co.jp/search/search?auccat=&tab_ex=commerce&ei=utf-8&aq=-1&oq=&sc_i=&fr=auc_top&p=' + keyword + '&x=0&y=0&b=' + str(b) + '&n=50'
        b += 50
        # ウェブページの内容を取得します。
        response = requests.get(base_url)
        # BeautifulSoupでHTMLを解析します。
        soup = BeautifulSoup(response.content, 'html.parser')
        # 各ページのURLを取得（href属性をチェックし、指定のURLで始まるものだけを取得）
        urls = [urljoin(base_url, a['href']) for a in soup.find_all('a', href=True) if a['href'].startswith('https://page.auctions.yahoo.co.jp/jp/auction/')]
        if urls == []:
            break
        else:
            page_urls.extend(urls)
            break
    return page_urls

def get_url_from_mercari(input_text):
    # Yahoo AuctionのURL
    keyword = quote(input_text)
    i = 0
    page_urls = []
    while True:
        print(i)
        base_url = "https://jp.mercari.com/search?keyword=" + keyword + '&page_token=v1%3A' + str(i)
        i += 1
        # ウェブページの内容を取得します。
        response = requests.get(base_url)
        # BeautifulSoupでHTMLを解析します。
        soup = BeautifulSoup(response.content, 'html.parser')
        # 各ページのURLを取得（href属性をチェックし、指定のURLで始まるものだけを取得）
        urls = [urljoin(base_url, a['href']) for a in soup.find_all('a', href=True) if a['href'].startswith('https://mercari-shops.com/products/')]
        if urls == []:
            break
        else:
            page_urls.extend(urls)
            # break
    return page_urls

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def are_same_product(image_path1, image_path2):
    # 画像をグレースケールで読み込む
    img1 = imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # 画像の読み込みに失敗した場合はエラーメッセージを表示
    if img1 is None:
        print("Could not open or find the image: ", image_path1)
        return
    if img2 is None:
        print("Could not open or find the image: ", image_path2)
        return

    # ORB detectorを作成する
    orb = cv2.ORB_create()

    # 各画像のkeypointsとdescriptorsを見つける
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force matcherを作成する
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # descriptorsをマッチングする
    matches = bf.match(des1, des2)

    return len(matches)

def image_similarity(image_path1, image_path2):
    img0 = imread(image_path1, cv2.IMREAD_COLOR)  # use cv2.IMREAD_COLOR
    img1 = imread(image_path2, cv2.IMREAD_COLOR)  # use cv2.IMREAD_COLOR

    # データ型を uint8 に変換します
    img0 = img0.astype(np.uint8)
    img1 = img1.astype(np.uint8)

    vtr = imgsim.Vectorizer()
    vec0 = vtr.vectorize(img0)
    vec1 = vtr.vectorize(img1)

    dist = imgsim.distance(vec0, vec1)

    return dist

# def calculate_image_similarity(image_path1, image_path2):
#     # Load and preprocess the images
#     img1 = image.load_img(image_path1, target_size=(224, 224))
#     img2 = image.load_img(image_path2, target_size=(224, 224))

#     # Convert the images to numpy arrays and preprocess them for ResNet50
#     img1_array = np.expand_dims(image.img_to_array(img1), axis=0)
#     img2_array = np.expand_dims(image.img_to_array(img2), axis=0)

#     img1_array = preprocess_input(img1_array)
#     img2_array = preprocess_input(img2_array)

#     # Use the ResNet50 model to extract features from the images
#     img1_features = model.predict(img1_array)
#     img2_features = model.predict(img2_array)

#     # Calculate the cosine similarity between the feature vectors
#     similarity = cosine_similarity(img1_features, img2_features)

#     # Return the similarity score
#     return similarity[0][0]

# コサイン類似度を計算する関数
# def cosine_similarity(a, b):
#     dot_product = np.sum(a * b)
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)
#     similarity = dot_product / (norm_a * norm_b)
#     return similarity

def input_word_and_image():
    image = st.file_uploader("検索画像")
    query = st.text_input("検索ワード")
    tfile = tempfile.NamedTemporaryFile(delete=False)
    if image and query:
        tfile.write(image.read())
        return tfile.name, query
    else:
        return None, None
    
def auto_scraping(img_paths, queries):
    new_result ={}
    for img_path, query in zip(img_paths, queries):
        item = find_similar_image_pages(query, img_path)
        new_result[query] = [item[0],item[1]]
    result_path = "result.pkl"
    if os.path.exists(result_path):
        with open(result_path, "rb") as f:
            old_result = pickle.load(f)
            new_result = merge_list(old_result, new_result)
        with open(result_path , "wb") as f:
            pickle.dump(new_result, f)
    else:
        with open(result_path, "wb") as f:
            pickle.dump(new_result, f)
    log = st.empty()
    current_time = datetime.datetime.now()
    log.write(f"現在時刻: {current_time.strftime('%Y-%m-%d %H:%M:%S')}プログラム終了")

def merge_list(old_result, new_result):
    result = {}
    new_keys = new_result.keys()
    for new_key in new_keys:
        if new_key in old_result:
            old_urls = list(old_result[new_key][0])
            old_scores = list(old_result[new_key][1])
        else:
            old_urls = []
            old_scores = []
        new_urls = new_result[new_key][0]
        new_scores = new_result[new_key][1]
        max_len = max([len(old_urls),len(old_scores),len(new_urls),len(new_scores)])
        for i in range(max_len):
            for new_url, new_score in zip(new_urls, new_scores):
                if new_url not in old_urls:
                    old_urls.append(new_url)
                    old_scores.append(new_score)
            result[new_key] = [old_urls, old_scores]
    result = sort_results(result)
    return result

def sort_results(results):
    sorted_results = {}

    for key, values in results.items():
        urls, scores = values
        combined = list(zip(urls, scores))
        combined.sort(key=lambda x: x[1])  # sort by score
        sorted_results[key] = list(zip(*combined))  # unzip to separate lists

    return sorted_results

def main():
    st.title("フリマ自動URL取得アプリ")
    if os.path.exists('scheduled_time.pkl'):
        os.remove('scheduled_time.pkl')

    num_uploads = st.number_input('何回アップロードしますか？', min_value=1, value=1, step=1)
    display = st.button("結果表示")
    if display:
        result_path = "result.pkl"
        if not os.path.exists(result_path):
            st.error("まだプログラムが走っていません。")
        else:
            with open(result_path, "rb") as f:
                result = pickle.load(f)
            with st.expander(f"{str(len(list(result.keys())))} 個の検索"):
                i = 0
                for key in result.keys():
                    st.write(f"検索ワード：{key}")
                    if result[key][0] == []:
                        st.write("urlを取得できませんでした。")
                    else:
                        for url, score in zip(result[key][0], result[key][1]):
                            st.write(f"URL：{url} スコア：{score}")
                    i += 1
        del st.session_state["img_paths"]
        del st.session_state["queries"]
        if os.path.exists('scheduled_time.pkl'):
            os.remove('scheduled_time.pkl')
    delete = st.button("結果削除")
    if delete:
        result_path = "result.pkl"
        if not os.path.exists(result_path):
            st.error("まだプログラムが走っていません。")
        else:
            os.remove(result_path)
            st.success("削除完了！")
    if st.button("終了"):
        st.experimental_rerun()

    img_paths = []
    queries = []
    for i in range(num_uploads):
        with st.expander(f"{i+1}個目"):
            image = st.file_uploader(f"検索画像{i+1}個目")
            query = st.text_input(f"検索ワード{i+1}個目")

            if image and query:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(image.read())
                if len(img_paths) <= num_uploads:
                    img_paths.append(tfile.name)
                if len(queries) <= num_uploads:
                    queries.append(query)
    st.session_state["img_paths"] = img_paths
    st.session_state["queries"] = queries

    if len(st.session_state["img_paths"]) == num_uploads and len(st.session_state["queries"]) == num_uploads:
        img_paths = st.session_state["img_paths"]
        queries = st.session_state["queries"]
        if st.button("今すぐスクレイピング開始"):
            # find_similar_image_pagesは画像とクエリを引数に取り、
            # スクレイピングした結果のURLリストを返す関数と仮定します
            new_result = {}
            for img_path, query in zip(img_paths, queries):
                item = find_similar_image_pages(query, img_path)
                new_result[query] = [item[0], item[1]]
            result_path = "result.pkl"
            if os.path.exists(result_path):
                with open(result_path, "rb") as f:
                    old_result = pickle.load(f)
                    new_result = merge_list(old_result, new_result)
                with open(result_path , "wb") as f:
                    pickle.dump(new_result, f)
            else:
                with open(result_path, "wb") as f:
                    pickle.dump(new_result, f)

            with st.expander(f"{len(list(new_result.keys()))} 個の検索"):
                i = 0
                for key in new_result.keys():
                    st.write(f"検索ワード：{key}")
                    if new_result[key][0] == []:
                        st.write("urlを取得できませんでした。")
                    else:
                        for url, score in zip(new_result[key][0], new_result[key][1]):
                            st.write(f"URL：{url} スコア：{score}")
                    i += 1
        if st.button("オートスクレイピング開始") or "scheduled_time" in st.session_state:
            st.session_state["scheduled_time"] = st.time_input('スケジューリングする時間を設定してください', value=time(12, 0))  # デフォルト値として12:00を設定
            # 時間をファイルに保存します。
            with open('scheduled_time.pkl', 'wb') as f:
                pickle.dump(st.session_state["scheduled_time"], f)
            determine = st.button("スケージュリング時間決定")
            if determine:
                with open('scheduled_time.pkl', 'rb') as f:
                    loaded_time = pickle.load(f)
                time_string = loaded_time.strftime('%H:%M')
                time_string = "23:00"
                schedule.every().day.at(time_string).do(auto_scraping,img_paths, queries)
                log = st.empty()
                log.write(f"{time_string}にプログラム開始")
                while True:
                    schedule.run_pending()  # スケジュールされたタスクがあれば実行
                    if display:
                        break

if __name__ == "__main__":
    main()