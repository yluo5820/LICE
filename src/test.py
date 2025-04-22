from datasets import load_dataset
from uniem.finetuner import FineTuner

dataset = load_dataset("clue/clue", "afqmc", cache_dir='cache')
dataset = dataset.remove_columns("idx")
finetuner = FineTuner.from_pretrained('moka-ai/m3e-small', dataset=dataset)
finetuned_model = finetuner.run(epochs=1, batch_size=64, lr=3e-5)