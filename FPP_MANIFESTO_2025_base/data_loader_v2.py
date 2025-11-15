#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能亮点（新版）：
1. 每条宣言保存为单独的TXT文件 -> manifesto_text/party_date.txt
2. 自动创建manifesto_text文件夹
3. 断点续传：自动检测已存在的TXT文件并跳过
4. 错误日志记录 + 容错重试 + 限速保护
5. tqdm进度条展示抓取进度

输出：
    manifesto_text/11220_196009.txt
日志：
    manifesto_loader_errors.log
"""

import pandas as pd
import requests
import time
from tqdm import tqdm
import os
from datetime import datetime


# ===== 配置区 =====
API_KEY = "Your_API_Key"
BASE_URL = "https://manifesto-project.wzb.eu/api/v1/texts_and_annotations"
LOG_FILE = "manifesto_loader_errors.log"
VERSION = "2024-1"  # corpus版本号
TRANSLATION = "en"  # 可改为 "original" 获取原文
SAVE_DIR = "manifesto_text"  # 存储TXT文件的文件夹


def log_error(message: str):
    """将错误信息写入日志文件"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {message}\n")


def normalize_date(date_val):
    """确保日期为YYYYMM格式"""
    date_str = str(date_val).replace("-", "").replace("/", "").strip()
    if len(date_str) == 4:  # 仅有年份
        date_str += "00"
    elif len(date_str) == 5:  # 可能缺前导0
        date_str = date_str[:4] + "0" + date_str[4:]
    return date_str


def fetch_manifesto_text(session, party: str, date: str, retries=3) -> str:
    """
    调用 Manifesto API 获取政党宣言文本
    若宣言不存在或多次失败则返回 None
    """
    key = f"{party}_{date}"
    params = {
        "api_key": API_KEY,
        "keys[]": key,
        "version": VERSION,
        "translation": TRANSLATION
    }

    for attempt in range(1, retries + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=15)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if "items" in data and data["items"]:
                        return data["items"][0].get("text", None)
                    return None
                except Exception as e:
                    log_error(f"JSON decode error for key={key}: {e}")
                    return None

            elif resp.status_code == 404:
                return None

            else:
                log_error(f"HTTP {resp.status_code} for key={key}")
                time.sleep(2)

        except requests.RequestException as e:
            log_error(f"Attempt {attempt} failed for key={key}: {e}")
            time.sleep(3)

    return None


def save_manifesto_text(party, date, text):
    """保存宣言文本到 manifesto_text/party_date.txt"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    filename = f"{party}_{date}.txt"
    filepath = os.path.join(SAVE_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text if text else "[No Text Available]")


def load_manifesto_and_save_texts(path="MPDataset_MPDS2025a.csv", sample_size=None):
    """读取CSV，通过API获取宣言并保存为TXT"""
    df = pd.read_csv(path)
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)

    os.makedirs(SAVE_DIR, exist_ok=True)
    existing_files = set(os.listdir(SAVE_DIR))
    session = requests.Session()
    session.headers.update({"User-Agent": "ManifestoLoader/2.0"})

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching Manifestos"):
        party = str(row["party"]).strip()
        date = normalize_date(row["date"])
        key = f"{party}_{date}"
        filename = f"{key}.txt"

        # 若文件已存在则跳过
        if filename in existing_files:
            continue

        text = fetch_manifesto_text(session, party, date)
        save_manifesto_text(party, date, text)
        time.sleep(0.25)  # 限速保护

    print(f"\n所有宣言已保存至文件夹：{SAVE_DIR}")
    print(f"错误日志记录在：{LOG_FILE}")


if __name__ == "__main__":
    # sample_size=None 表示处理全部数据
    load_manifesto_and_save_texts(path="MPDataset_MPDS2025a.csv", sample_size=None)
