import json

# with open("./oracle_bone_script/oracle2radical.json", "r", encoding="utf-8") as f:
#     file =  json.load(f)

# print(len(file))  # Output: 0

import pandas as pd

# Read the CSV file
df = pd.read_csv('IDS.csv')
# Preview the first few rows
print(df.head(20))

# 去除异体字
# def devarient_characters(s):
#     if "→" in s or "←" in s:
#         char = s[1:]
#         if char in df["char"].values:
#             return df[df["char"] == char]["decomposition"].values[0]
#         else:
#             return None
#     else:
#         return s

# df["decomposition"] = df["decomposition"].apply(devarient_characters)

# print(df.head(20))


# counter_1 = 0
# counter_2 = 0
# for char, radical in file.items():
#     if radical[0] is not None:
#         counter_1 += 1
#     if radical[1] is not None:
#         counter_2 += 1

# print("total_len: ", len(file))
# print(f"modern: {counter_2}")  # Output: seal: 0, trad: 0


import json
from itertools import chain
from cjklib.characterlookup import CharacterLookup

CJK = list(chain(range(0x4E00, 0xA000), range(0x3400, 0x4DC0)))  # 27k chars