import pandas as pd
from datasets import DatasetDict
from model_train import ContrastiveLossModel, LIBGEWrapper, make_linguistic_dict, MODEL_NAME, tokenize_with_text, collate_fn
from transformers import AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import Dataset

dataset = DatasetDict({
        "train": Dataset.from_pandas(pd.read_csv("ccpm_split_train.csv")),
        "validation": Dataset.from_pandas(pd.read_csv("ccpm_split_val.csv"))
    }).map(tokenize_with_text)

linguistic_dict, dim = make_linguistic_dict()
model = ContrastiveLossModel(LIBGEWrapper(model_name=MODEL_NAME, linguistic_map=linguistic_dict, feature_dims=dim))

args = TrainingArguments(
    output_dir="./checkpoints_poem",
    per_device_train_batch_size=16,
    num_train_epochs=1,
    learning_rate=1e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False  # required for non-standard inputs like raw_text
)

trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn
    )

trainer.train()
