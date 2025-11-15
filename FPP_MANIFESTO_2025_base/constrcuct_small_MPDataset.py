#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：
从原始 Manifesto 数据集中筛选出特定国家的行，
生成一个新的小规模数据集 `MPDataset_MPDS2025_small.csv`
"""

import pandas as pd

# ===== 输入输出文件路径 =====
INPUT_FILE = "MPDataset_MPDS2025a.csv"
OUTPUT_FILE = "MPDataset_MPDS2025_small.csv"

# ===== 要保留的国家列表 =====
TARGET_COUNTRIES = {
    "United States",
    "United Kingdom",
    "France",
    "Germany",
    "Spain",
    "Canada",
    "Australia",
    "Brazil"
}

def create_small_dataset(input_path=INPUT_FILE, output_path=OUTPUT_FILE):
    """根据国家名筛选出特定数据行并保存"""
    # 读取原始CSV
    df = pd.read_csv(input_path)

    # 筛选行
    df_small = df[df["countryname"].isin(TARGET_COUNTRIES)]

    # 输出结果
    df_small.to_csv(output_path, index=False, encoding="utf-8")

    # 打印结果信息
    print(f"已成功生成小规模数据集：{output_path}")
    print(f"原始数据行数：{len(df)}")
    print(f"筛选后数据行数：{len(df_small)}")
    print(f"包含国家：{', '.join(sorted(TARGET_COUNTRIES))}")

if __name__ == "__main__":
    create_small_dataset()
