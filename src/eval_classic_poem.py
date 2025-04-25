import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer
import pandas as pd
from model_train import LIBGEWrapper, make_linguistic_dict, MODEL_NAME, ContrastiveLossModel, tokenize_with_text,MODERN,CLASSIC
import tqdm
from safetensors.torch import load_file
import json
from sklearn.metrics import f1_score

# Load data
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
    
test = load_jsonl("ccpm_split_test.jsonl")

# Load model
ling_map, dim = make_linguistic_dict(CLASSIC)
model = ContrastiveLossModel(LIBGEWrapper(model_name=MODEL_NAME, linguistic_map=ling_map, feature_dims=dim))
states = load_file("checkpoints_poem_classic/checkpoint-4900/model.safetensors", device="cpu")  # replace XXXX with actual number
model.load_state_dict(states)  # replace XXXX with actual number
model.eval()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Evaluation loop
preds = []
trues = []
for row in tqdm.tqdm(test):
    translation = row["translation"]
    choices = row["choices"]

    scores = []
    for choice in choices:
        tokenized = tokenize_with_text({"sentence1": translation, "sentence2": choice, "score": None})
        scores.append(model.forward(
            torch.tensor([tokenized["input_ids1"]]),
            torch.tensor([tokenized["attention_mask1"]]),
            [tokenized["raw_text1"]],
            torch.tensor([tokenized["input_ids2"]]),
            torch.tensor([tokenized["attention_mask2"]]),
            [tokenized["raw_text2"]]
        )["logits"].item())
    pred = scores.index(max(scores))
    true = row["answer"]
    preds.append(pred)
    trues.append(true)

# Calculate accuracy
accuracy = sum(p == t for p, t in zip(preds, trues)) / len(trues)
print(f"Accuracy: {accuracy:.4f}")

