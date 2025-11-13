import pandas as pd

# 定义 fips_state_map
fips_state_map = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 12: "Florida", 13: "Georgia",
    15: "Hawaii", 16: "Idaho", 17: "Illinois", 18: "Indiana", 19: "Iowa",
    20: "Kansas", 21: "Kentucky", 22: "Louisiana", 23: "Maine", 24: "Maryland",
    25: "Massachusetts", 26: "Michigan", 27: "Minnesota", 28: "Mississippi",
    29: "Missouri", 30: "Montana", 31: "Nebraska", 32: "Nevada", 33: "New Hampshire",
    34: "New Jersey", 35: "New Mexico", 36: "New York", 37: "North Carolina",
    38: "North Dakota", 39: "Ohio", 40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania",
    44: "Rhode Island", 45: "South Carolina", 46: "South Dakota", 47: "Tennessee",
    48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington",
    54: "West Virginia", 55: "Wisconsin", 56: "Wyoming"
}

# 定义字段映射关系
fields_of_interest = {
    'V161310x': {"valmap": {1: 'white', 2: 'black', 3: 'asian', 4: 'native American', 5: 'hispanic'}},
    'V162174': {"valmap": {1: 'I like to discuss politics with my family and friends.', 2: 'I never discuss politics with my family or friends.'}},
    'V161126': {"valmap": {1: "extremely liberal", 2: "liberal", 3: "slightly liberal", 4: "moderate", 5: "slightly conservative", 6: "conservative", 7: "extremely conservative"}},
    'V161158x': {"valmap": {1: "a strong Democrat", 2: "a weak Democrat", 3: "an independent who leans Democratic", 4: "an independent", 5: "an independent who leans Republican", 6: "a weak Republican", 7: "a strong Republican"}},
    'V161244': {"valmap": {1: "attend church", 2: "do not attend church"}},
    'V161267': {"valmap": {}},  # 年龄，直接显示数值
    'V161342': {"valmap": {1: "man", 2: "woman"}},
    'V162256': {"valmap": {1: "very", 2: "somewhat", 3: "not very", 4: "not at all"}},
    'V162125x': {"valmap": {1: "extremely good", 2: "moderately good", 3: "a little good", 4: "neither good nor bad", 5: "a little bad", 6: "moderately bad", 7: "extremely bad"}},
    'V161010d': {"valmap": fips_state_map}
}

# 读取 CSV 文件
data = pd.read_csv('full_results_2016_2.csv')

# 筛选感兴趣的列
filtered_data = data[fields_of_interest.keys()]

# 转换函数
def convert_row_to_description(row):
    description_parts = []
    for col in filtered_data.columns:
        value = row[col]
        if pd.isnull(value):
            description_parts.append("Unknown")
        elif value in fields_of_interest[col]["valmap"]:
            description_parts.append(fields_of_interest[col]["valmap"][value])
        else:
            # 如果没有映射（比如年龄字段），直接使用原值
            description_parts.append(str(value))
    
    # 创建完整描述
    description = (
        f"You are {description_parts[6]}, {description_parts[0]} of age {description_parts[5]}, "
        f"identify as {description_parts[3]}. "  # 对应party identification
        f"You {description_parts[4]}, "  # attend church状态
        f"and {description_parts[1]} "  # 讨论政治的习惯
        f"You feel {description_parts[7]} about the American flag, and you live in {description_parts[9]}. The current year is 2016. "
    )
    return description

# 转换数据
identities = filtered_data.apply(convert_row_to_description, axis=1).tolist()