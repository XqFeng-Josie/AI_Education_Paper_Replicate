# SLAM: Second Language Acquisition Modeling

Replication of the SLAM shared task experiments for predicting token-level learner errors in second-language learning using Duolingo traces.

**Paper**: [Second Language Acquisition Modeling](https://aclanthology.org/W18-0506.pdf)  
**Dataset**: [Duolingo SLAM Dataset](https://doi.org/10.7910/DVN/8SWHNO)

## Overview

The SLAM task predicts token-level learner errors using large-scale Duolingo traces. The corpus contains >7M tokens from ~6.4k beginners of English, Spanish, and French collected over learners' first 30 days.

This repository includes:
- **Baseline**: Logistic regression model (replicates paper results)
- **LLM Models**: Frozen-backbone classifiers using Llama, Mistral, and Qwen

## Environment Setup

### Requirements

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `scikit-learn>=1.3.0`
- `datasets>=2.12.0`

### Model Access

Some models (especially Llama) require HuggingFace authentication:
```bash
huggingface-cli login
```

## Data

### Dataset Structure

The dataset includes three language tracks:
- `data_en_es/`: English learners from Spanish
- `data_es_en/`: Spanish learners from English  
- `data_fr_en/`: French learners from English

Each track contains:
- `{track}.slam.20190204.train`: Training data
- `{track}.slam.20190204.dev`: Development data
- `{track}.slam.20190204.test`: Test data

Answer keys are in `keys/` directory.

### Data Format

Each instance contains:
- Token-level features: token, POS, morphological features, dependency labels
- Exercise metadata: user ID, format, session type
- Binary label: 0 = incorrect, 1 = correct

## Scripts

### 1. Baseline Model

Run the logistic regression baseline:

```bash
# Test set
./run_baseline.sh en_es test

# Dev set
./run_baseline.sh en_es dev

# Other tracks
./run_baseline.sh es_en test
./run_baseline.sh fr_en test
```

Output: `baseline_{track}_{split}.pred` with predictions and evaluation metrics.

### 2. LLM Training

Train frozen-backbone classifiers (only classification head is trained):

```bash
python train.py \
    --data_dir data_en_es \
    --model llama-3.1-8b \
    --output_dir models/llama-3.1-8b_frozen_head \
    --num_epochs 3 \
    --learning_rate 5e-4 \
    --batch_size 64 \
    --val_ratio 0.1
```

**Supported models:**
- `llama-3.1-8b`
- `llama-3.3-70b-instruct`
- `mistral-7b`
- `qwen-2.5-7b`

**Key parameters:**
- `--train_ratio`: Subsample training data (default: 1.0)
- `--val_ratio`: Validation split from training data (default: 0.05)
- `--max_length`: Tokenizer max length (default: 256)
- `--grad_accum_steps`: Gradient accumulation steps (default: 1)

**Multi-GPU training:**
```bash
torchrun --nproc_per_node 4 train.py \
    --distributed \
    --data_dir data_en_es \
    --model llama-3.1-8b \
    --output_dir models/llama-3.1-8b_frozen_head \
    --num_epochs 3
```

### 3. Inference

Generate predictions with trained models:

```bash
python inference.py \
    --model_dir models/llama-3.1-8b_frozen_head \
    --data_dir data_en_es \
    --split test \
    --output_file predictions/llama_test.pred
```

### 4. Evaluation

Evaluate predictions:

```bash
python starter_code/eval.py \
    --pred predictions/llama_test.pred \
    --key keys/en_es.slam.20190204.test
```

**Metrics reported:**
- AUC (primary ranking metric)
- F1 Score
- Accuracy
- Average Log Loss

## Experimental Results

### Baseline Model

| Track | Paper AUC | Paper F1 | Our AUC | Our F1 | Status |
|-------|-----------|----------|---------|--------|--------|
| **en_es** | 0.774 | 0.190 | **0.774** | **0.190** | ✅ Replicated |
| **es_en** | 0.746 | 0.175 | **0.746** | **0.177** | ✅ Replicated |
| **fr_en** | 0.771 | 0.281 | **0.770** | **0.281** | ✅ Replicated |

Results match the paper's SLAM_baseline (Table 2) with minimal differences (ΔAUC ≤ 0.001, ΔF1 ≤ 0.002).

### LLM Models

LLM fine-tuning experiments are in progress. Results will be updated here once available.

## Project Structure

```
SLAM/
├── data_en_es/              # Dataset files (en_es track)
├── data_es_en/              # Dataset files (es_en track)
├── data_fr_en/              # Dataset files (fr_en track)
├── keys/                     # Answer keys for evaluation
├── starter_code/             # Original baseline code
│   ├── baseline.py          # Baseline logistic regression
│   ├── eval.py              # Evaluation script
│   └── README.md            # Original documentation
├── models/                   # Trained model checkpoints
├── predictions/             # Prediction outputs
├── data_preprocessing.py    # Data loading for LLM
├── train.py                 # LLM training script
├── inference.py             # LLM inference script
├── run_baseline.sh          # Baseline experiment wrapper
└── requirements.txt         # Python dependencies
```

## Citation

```bibtex
@inproceedings{settles2018duolingo,
  title={The Second Language Acquisition Modeling (SLAM) Shared Task},
  author={Settles, Burr and LaFlair, Geoffrey T and Hagiwara, Masato},
  booktitle={Proceedings of the 13th Workshop on Innovative Use of NLP for Building Educational Applications},
  year={2018}
}
```