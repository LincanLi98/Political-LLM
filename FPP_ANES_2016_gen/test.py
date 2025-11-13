# main.py
from anes import identities

# 打印 identities 列表的前 5 行
for i, identity in enumerate(identities[:5]):
    print(f"{i + 1}: {identity}")