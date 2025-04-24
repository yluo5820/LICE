import torch
from transformers import AutoTokenizer
from model_train import LIBGEWrapper, DIM_LING_INFO, MODEL_NAME, make_linguistic_dict
from safetensors.torch import load_file
from torch import nn
import torch
from transformers import AutoTokenizer
from mteb import MTEB, get_tasks, get_model

import os
os.environ["WANDB_DISABLED"] = "true"

class BGE_C_MTEB_Wrapper(nn.Module):
    def __init__(self, libge_model, tokenizer_name="BAAI/bge-base-zh", max_length=128, pooling="mean"):
        super().__init__()
        self.encoder = libge_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.pooling = pooling

    def encode(self, sentences, batch_size=32, **kwargs):
        self.eval()
        device = next(self.encoder.parameters()).device
        all_embeddings = []

        with torch.no_grad():
            for start_idx in range(0, len(sentences), batch_size):
                batch_sentences = sentences[start_idx:start_idx + batch_size]
                tokenized = self.tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)

                input_ids = tokenized["input_ids"].to(device)
                attention_mask = tokenized["attention_mask"].to(device)
                raw_text = [list(sent) for sent in batch_sentences]

                embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask, raw_text=raw_text)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

def load_model():
    checkpoint_path = "checkpoints_frozen/checkpoint-31000/model.safetensors"
    full_state_dict = load_file(checkpoint_path)

    # Filter state dict for LIBGEWrapper only
    filtered_state_dict = {}
    for k, v in full_state_dict.items():
        if k.startswith("encoder.model."):
            new_key = k[len("encoder."): ]  # Strip "encoder."
            filtered_state_dict[new_key] = v
        elif k.startswith("encoder.fusion") or k.startswith("encoder.Linear"):
            new_key = k[len("encoder."):]
            filtered_state_dict[new_key] = v

    # Build model
    ling_map = make_linguistic_dict()
    libge_model = LIBGEWrapper(model_name=MODEL_NAME, linguistic_map=ling_map)
    libge_model.load_state_dict(filtered_state_dict)

    model = BGE_C_MTEB_Wrapper(libge_model, tokenizer_name=MODEL_NAME, max_length=128, pooling="mean")
    return model

def benchmark_model(model):
    tasks = ["IFlyTek", "ThuNewsClusteringS2S.v2", "Ocnli", "MMarcoReranking", "BQ"]
    evaluation = MTEB(tasks=tasks)
    evaluation.run(model, output_folder="mteb_results/frozen_4epochs")
    
if __name__ == "__main__":
    model = load_model()
    benchmark_model(model)
    print("Benchmarking completed.")