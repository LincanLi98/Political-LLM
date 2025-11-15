# manifesto.py
import pandas as pd

def load_manifesto_data(
    path="MPDataset_MPDS2025a.csv",
    sample_size=500,
    drop_na=True,
    min_rile=-100,
    max_rile=100
):
    """
    加载 Manifesto Project Database 2025a 主数据文件 (MPDataset_MPDS2025a.csv)。

    功能：
      1️. 读取并筛选关键信息列；
      2️. 清理异常或缺失 RILE；
      3️. 采样部分数据以控制规模；
      4️. 返回一个 DataFrame 供 LLM 实验使用。

    输出字段：
      ['country', 'countryname', 'party', 'partyname',
       'partyabbrev', 'date', 'rile', 'planeco', 'markeco',
       'welfare', 'intpeace', 'id_perm']
    """

    # 1.加载文件
    df = pd.read_csv(path, low_memory=False)

    # 2.仅保留关键列（实验主表）
    keep_cols = [
        "country", "countryname", "party", "partyname", "partyabbrev",
        "date", "rile", "planeco", "markeco", "welfare", "intpeace", "id_perm"
    ]
    #Manifesto数据集中包含大量的`perXXX`列, 例如: per101, per102, per201, per301, per401, ...
    #这些列代表人工编码的政策主题比例(quasi-sentences 比例), 每个值表示该政党宣言中该主题的百分比, 因此不作保留.
    df = df[[col for col in keep_cols if col in df.columns]]

    # 3. 清洗：去除缺失 RILE 或非法值
    df = df.dropna(subset=["rile"])
    df = df[(df["rile"] >= min_rile) & (df["rile"] <= max_rile)]

    # 4. 移除重复行（防止同一政党同年多宣言）
    df = df.drop_duplicates(subset=["country", "party", "date"], keep="first")

    # 5. 排序与采样
    df = df.sort_values(by=["country", "date"]).reset_index(drop=True)
    df = df.sample(min(sample_size, len(df)), random_state=42)

    # 6. 强制类型转换
    df["date"] = df["date"].astype(str)
    df["rile"] = df["rile"].astype(float)

    # 7. 输出统计信息
    print(f"Loaded {len(df)} manifesto samples "
          f"from {df['countryname'].nunique()} countries "
          f"and {df['partyname'].nunique()} parties.")

    return df
