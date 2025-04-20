import json
from itertools import chain
import pandas as pd

NUM_RADICAL = 194

with open("oracle_radical_CJK.json", "r", encoding="utf-8") as f:
    oracle_dict =  json.load(f)

def list_to_one_hot(alist, num_classes=NUM_RADICAL):
    one_hot = [0] * num_classes
    if alist == None:
        return one_hot
    for item in alist:
        one_hot[item] = 1
    return one_hot

for key in oracle_dict.keys():
    oracle_dict[key] = list_to_one_hot(oracle_dict[key])

df = pd.DataFrame(list(oracle_dict.items()), columns=["char", "radical"])

df.to_csv("oracle_radical_one_hot.csv", index=False, encoding="utf-8")

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
