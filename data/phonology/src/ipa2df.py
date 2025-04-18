import pandas as pd
import torch
import pickle

# Read TSV file
df = pd.read_csv("./raw/distincfeature.tsv", delimiter="\t", header=0, dtype=str)
print(df.info())
print(df.head(5))
df.iloc[:, 1:] = df.iloc[:, 1:].replace({"+": 1, "0": 0, "-": -1, "+,-":0.5, "-,+":-0.5, "+,-,+":0.75, "-,+,-":-0.75, "+,-,-": 0.25, "-,+,+":-0.25, "+,-,+,-":0}).astype(float)
print(df.head(5))

# Convert to dictionary with first column as keys and the rest as tensor values
ipa2df = {row.iloc[0]: torch.tensor(row.iloc[1:].values.astype(float), dtype=torch.float32) for _, row in df.iterrows()}

# save the dict
with open("ipa2df.pkl", "wb") as f:
    pickle.dump(ipa2df, f)

with open("ipa2df.pkl", "rb") as f:
    loaded_dict = pickle.load(f)

print(loaded_dict)