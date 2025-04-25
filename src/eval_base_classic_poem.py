import json
import tqdm
import torch
from sentence_transformers import SentenceTransformer, util

# Load dataset
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

test = load_jsonl("ccpm_split_test.jsonl")

# Load pretrained BGE model
model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
model.eval()

# Evaluate
preds = []
trues = []

for row in tqdm.tqdm(test):
    translation = row["translation"]
    choices = row["choices"]
    true = row["answer"]

    # Get embeddings
    inputs = [translation] + choices
    embeddings = model.encode(inputs, convert_to_tensor=True)

    translation_embedding = embeddings[0]
    choices_embeddings = embeddings[1:]

    # Compute cosine similarities
    similarities = util.cos_sim(translation_embedding, choices_embeddings)[0]
    pred = torch.argmax(similarities).item()

    preds.append(pred)
    trues.append(true)

# Accuracy
accuracy = sum(p == t for p, t in zip(preds, trues)) / len(trues)
print(f"Accuracy: {accuracy:.4f}")
