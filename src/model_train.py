## 1. Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import ast
import gc
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import mean_squared_error

import os
os.environ["WANDB_DISABLED"] = "true"

DIM_LING_INFO = 2007
MODEL_NAME = "BAAI/bge-base-zh"
MAX_LEN = 128

## 2. Custom Model Definition
class LIBGEWrapper(nn.Module):
    def __init__(self, model_name, pooling="mean", linguistic_map=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden_dim = self.model.config.hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(DIM_LING_INFO, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.Linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linguistic_map = linguistic_map

    def char_to_vector(self, char):
        vec = self.linguistic_map.get(char)
        if vec is not None:
            return torch.tensor(vec, dtype=torch.float32)
        return torch.zeros(DIM_LING_INFO, dtype=torch.float32)

    def forward(self, input_ids, attention_mask, raw_text):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        batch_vectors = []
        for sentence in raw_text:
            char_vecs = [self.char_to_vector(c).to(input_ids.device) for c in sentence]
            char_vecs += [torch.zeros(DIM_LING_INFO).to(input_ids.device)] * (input_ids.shape[1] - len(char_vecs))
            char_vecs = char_vecs[:input_ids.shape[1]]
            batch_vectors.append(torch.stack(char_vecs))
        raw_text_vectors = torch.stack(batch_vectors)

        raw_text_vectors = self.fusion(raw_text_vectors)
        combined = torch.cat((output.last_hidden_state, raw_text_vectors), dim=-1)
        fused = self.Linear(combined)

        if self.pooling == "cls":
            return fused[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(fused.size()).float()
            return (fused * mask).sum(1) / mask.sum(1)
        else:
            raise ValueError("Unsupported pooling type")

## 3. Contrastive Training Wrapper
class ContrastiveLossModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids1, attention_mask1, raw_text1,
                      input_ids2, attention_mask2, raw_text2,
                      labels=None):
        emb1 = self.encoder(input_ids1, attention_mask1, raw_text1)
        emb2 = self.encoder(input_ids2, attention_mask2, raw_text2)
        logits = F.cosine_similarity(emb1, emb2)
        if labels is not None:
            loss = F.mse_loss(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

## 4. Tokenization & Data Processing
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_with_text(example):
    tok1 = tokenizer(example["sentence1"], truncation=True, padding="max_length", max_length=MAX_LEN)
    tok2 = tokenizer(example["sentence2"], truncation=True, padding="max_length", max_length=MAX_LEN)
    return {
        "input_ids1": tok1["input_ids"],
        "attention_mask1": tok1["attention_mask"],
        "raw_text1": list(example["sentence1"]),
        "input_ids2": tok2["input_ids"],
        "attention_mask2": tok2["attention_mask"],
        "raw_text2": list(example["sentence2"]),
        "label": float(example["score"]),
    }

def collate_fn(batch):
    def stack(key): return torch.tensor([item[key] for item in batch])
    return {
        "input_ids1": stack("input_ids1"),
        "attention_mask1": stack("attention_mask1"),
        "raw_text1": [item["raw_text1"] for item in batch],
        "input_ids2": stack("input_ids2"),
        "attention_mask2": stack("attention_mask2"),
        "raw_text2": [item["raw_text2"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float)
    }
## 5. Load Linguistic Map
def make_linguistic_dict():
    df = pd.read_csv("Chinese_linguistic_data.csv")
    one_hot_columns = df.columns[2:]
    for col in one_hot_columns:
        df[col] = df[col].apply(ast.literal_eval)
    df['full_vector'] = df[one_hot_columns].apply(lambda row: list(np.concatenate(row.values)), axis=1)
    char_to_vector = dict(zip(df['char'], df['full_vector']))
    del df
    gc.collect()
    return char_to_vector

## 6. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    mse = mean_squared_error(labels, logits)
    return {"mse": mse}

def test_ling_map():
    print("Testing linguistic map...")
    ling_map = make_linguistic_dict()
    print(f"Loaded {len(ling_map)} characters in linguistic map.")
    test_char = "汉"
    if test_char in ling_map:
        print(f"Vector for '{test_char}': {ling_map[test_char]}")
    else:
        print(f"Character '{test_char}' not found in linguistic map.")

## 7. Training Setup
DATASET = "C-MTEB/STSB"
if __name__ == "__main__":
    ds = load_dataset(DATASET)
    ds = ds.map(tokenize_with_text)
    print(f"Dataset size: {len(ds['train'])} training samples, {len(ds['validation'])} validation samples")
    ling_map = make_linguistic_dict()

    model = ContrastiveLossModel(LIBGEWrapper(model_name=MODEL_NAME, linguistic_map=ling_map))

    args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=16,
        num_train_epochs=1,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        max_grad_norm=1.0,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    trainer.train()
