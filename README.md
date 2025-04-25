## 📂 LICE Repository Structure

*LICE (Linguistically Informed Chinese Embedding)* is organized into the following main directories, each serving a specific role:

### `data/`
Contains all datasets and linguistic resources used for model training and evaluation:
- **glyph/** — Data related to Chinese character **glyphs** (visual structure information, such as components and radicals).
- **phonology/** — Data related to Chinese **phonology** (pronunciations, phonetic encodings, and phonological features).
- **local dataset/** — Local training and evaluation datasets (e.g., splits of classical Chinese poetry or related corpora).

### `src/`
Holds source code, training scripts, saved model checkpoints, and evaluation results:
- **finetuned-model/** — Checkpoints and configurations of the **fine-tuned embedding model**, after additional training on domain-specific data.
- **model_trained/** — Checkpoints from the **initial base model training** prior to fine-tuning.
- **mteb_results/** — Evaluation results from benchmarking on the **MTEB (Massive Text Embedding Benchmark)**, including task scores and metrics.

In addition to these subdirectories, `src/` contains the main scripts for model development:
- `model_train.py` — Script for training the base model.
- `train_classic_poem.py` — Script for fine-tuning the model on classical Chinese poem datasets.
- `eval_classic_poem.py`, `eval_base_classic_poem.py` — Scripts for evaluating model performance on classical Chinese tasks.
- `train_classic_chinese.ipynb` — Jupyter notebook demonstrating the training workflow on classical Chinese datasets.
