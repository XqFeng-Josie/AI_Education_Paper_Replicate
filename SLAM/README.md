# SLAM Paper Replication

This repository contains code for replicating the Second Language Acquisition Modeling (SLAM) shared task experiments, including both baseline and LLM-based approaches.

## Overview

The SLAM task predicts token-level learner errors in second-language learning using large-scale Duolingo traces. The corpus contains >7M tokens from ~6.4k beginners of English, Spanish, and French collected over learners' first 30 days.

**Paper**: [Second Language Acquisition Modeling](https://aclanthology.org/W18-0506.pdf)  
**Dataset**: [Duolingo SLAM Dataset](https://doi.org/10.7910/DVN/8SWHNO)

## Dataset

We focus on the `data_en_es` track (English learners from Spanish). The dataset includes:
- Training data: `en_es.slam.20190204.train`
- Development data: `en_es.slam.20190204.dev` (with key file)
- Test data: `en_es.slam.20190204.test` (with key file)

Each instance contains:
- Token-level features: token, part-of-speech, morphological features, dependency labels
- Exercise metadata: user ID, format, session type, etc.
- Binary label: 0 = incorrect, 1 = correct

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

The data should be in the `data_en_es/` directory. If not, download from the dataset link above.

## Experiments

### Step 1: Run Baseline Model

The baseline is a simple logistic regression model using dataset features.

```bash
# Run baseline on dev set
./run_baseline.sh en_es dev

# Run baseline on test set
./run_baseline.sh en_es test

# For other language tracks (es_en, fr_en)
./run_baseline.sh es_en test
./run_baseline.sh fr_en test
```

This will:
1. Train the baseline logistic regression model
2. Generate predictions in `baseline_<track>_<split>.pred` (e.g., `baseline_en_es_test.pred`)
3. Evaluate using AUC and F1 score

### Step 2: LLM Training with Frozen Backbones

The LLM experiment keeps each backbone (Llama/Mistral/Qwen) **frozen** and only trains a lightweight classification head on top of mean-pooled hidden states. This matches the baseline feature space while keeping the comparison fair across models.

Supported model keys:
- `llama-3.1-8b`
- `llama-3.3-70b-instruct`
- `mistral-7b`
- `qwen-2.5-7b`

**Train a classifier head**

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

Key arguments:
- `--train_ratio`: optional subsampling for quick experiments (default `1.0`)
- `--val_ratio`: fraction of train data held out for validation monitoring (default `0.05`)
- `--max_length`: tokenizer length for compact feature text (default `256`)
- `--grad_accum_steps`: gradient accumulation for memory-friendly large batches
- `--dropout`: dropout probability on the classification head (default `0.1`)

Outputs written to `output_dir`:
- `classifier.pt`: weights of the final linear head
- `tokenizer/`: tokenizer snapshot aligned with the backbone
- `model_config.json`: metadata (backbone path, max length, best epoch, etc.)
- `training_history.json`: per-epoch metrics (loss/AUC/F1/accuracy where available)

Because the backbone never updates, training fits comfortably on a single GPU even for large models; only the head parameters receive gradients.

**Quick dry runs**

```bash
python train.py \
    --data_dir data_en_es \
    --model mistral-7b \
    --output_dir models/mistral7b_debug \
    --train_ratio 0.1 \
    --val_ratio 0.2 \
    --num_epochs 1
```

Use this to verify the pipeline before running the full dataset.

**Other handy flags**
- `--gradient_accum_steps`: virtual batch size via gradient accumulation (default `1`)
- `--num_workers`: DataLoader workers (default `4`)
- `--random_seed`: ensures reproducible sampling/validation splits (default `42`)
- `--distributed`: enable native PyTorch DistributedDataParallel (launch with `torchrun`)
- `--dist_backend`: backend for DDP (default `nccl`)

**Multi-GPU launch example**

```bash
torchrun --nproc_per_node 4 train.py \
    --distributed \
    --data_dir data_en_es \
    --model llama-3.1-8b \
    --output_dir models/llama-3.1-8b_frozen_head \
    --num_epochs 3
```

When `--distributed` is set, each rank loads its own copy of the frozen backbone, and only the lightweight classifier head participates in gradient synchronization. Validation, checkpointing, and logging are handled by rank 0 to avoid redundant disk writes.

Make sure you have access to the underlying Hugging Face model repositories (some Llama builds require authentication).

### Step 3: LLM Inference

Run inference with fine-tuned models:

```bash
# Inference on test set
python inference.py \
    --model_dir models/llama-3.1-8b_frozen_head \
    --data_dir data_en_es \
    --split test \
    --output_file predictions/llama_test.pred

# Inference on dev set
python inference.py \
    --model_dir models/llama-3.1-8b_frozen_head \
    --data_dir data_en_es \
    --split dev \
    --output_file predictions/llama_dev.pred
```

**Inference parameters**:
- `--model_dir`: directory containing `classifier.pt` and `model_config.json`
- `--data_dir`: dataset directory (default `data_en_es`)
- `--split`: `dev` or `test` (default `test`)
- `--output_file`: optional path for `.pred` output (auto-generated if omitted)
- `--batch_size`: inference batch size (default `64`)
- `--max_length`: optional override for tokenizer length (defaults to the value saved during training)

### Step 4: Evaluate Predictions

Evaluate any predictions file:

```bash
# Evaluate test set predictions
python starter_code/eval.py \
    --pred predictions/llama_test.pred \
    --key keys/en_es.slam.20190204.test

# Evaluate dev set predictions
python starter_code/eval.py \
    --pred predictions/llama_dev.pred \
    --key keys/en_es.slam.20190204.dev

# Evaluate baseline predictions (run_baseline.sh handles this automatically)
python starter_code/eval.py \
    --pred data_en_es/baseline_en_es_test.pred \
    --key keys/en_es.slam.20190204.test
```

Key files live under the repository-level `keys/` folder (already provided in this workspace). The `run_baseline.sh` script automatically points to the correct key path.

## Data Format

### Input Format

The baseline features used for LLM input are:
- User ID
- Exercise format (reverse_translate, reverse_tap, listen)
- Token (lowercased)
- Part-of-speech tag
- Morphological feature names (keys only)
- Dependency label

Example input text:
```
User:XEinXf5+ Format:reverse_translate Token:i POS:PRON Morph:Case,Number,Person,PronType DepLabel:nsubj
```

### Output Format

Predictions are written in `.pred` format:
```
instance_id probability
```

Example:
```
DRihrVmh0101 0.85
DRihrVmh0102 0.92
```

## Evaluation Metrics

The evaluation script reports:
- **AUC** (Area Under ROC Curve) - primary ranking metric
- **F1 Score** - secondary metric
- **Accuracy** - using 0.5 as cutoff
- **Average Log Loss**

## Model Configuration

### Baseline Model
- Algorithm: L2-regularized logistic regression
- Training: Stochastic gradient descent (SGD)
- Features: User, format, token, POS, morphological features, dependency labels
- Regularization: Sigma = 20.0
- Learning rate: Eta = 0.1
- Iterations: 10 epochs

### LLM Models
- Encoder: Frozen backbone (no gradients on transformer weights)
- Head: Single linear layer trained with BCE + sigmoid
- Epochs: 1-3 (configure with `--num_epochs`)
- Learning rate: default `5e-4` shared across models
- Batch size: 32-128 depending on GPU memory (set via `--batch_size`)
- Max sequence length: 256 (adjust via `--max_length` if needed)
- Training data: Only the official train split; dev/test remain untouched
- Data sampling: Optional `--train_ratio` and `--val_ratio` for quick ablations

## File Structure

```
SLAM/
├── data_en_es/              # Dataset files (en_es track)
├── data_es_en/              # Dataset files (es_en track)
├── data_fr_en/              # Dataset files (fr_en track)
├── starter_code/            # Original baseline code
│   ├── baseline.py          # Baseline logistic regression model
│   ├── eval.py              # Evaluation script
│   └── README.md            # Original baseline documentation
├── data_preprocessing.py    # Data loading and preprocessing for LLM
├── run_baseline.sh          # Baseline experiment script (shell wrapper)
├── train.py                 # Frozen-backbone LLM training script
├── inference.py             # Frozen-head inference script
├── requirements.txt         # Python dependencies
├── TRAINING_GUIDE.md        # Detailed training guide
└── README.md                # This file
```

## Notes

1. **Data Consistency**: The data preprocessing follows the exact same logic as `baseline.py` to ensure consistency.

2. **Feature Extraction**: LLM inputs use only the baseline features (User, format, token, POS, morphological feature keys, dependency label) as specified in the instructions.

3. **Training Data**: Only the training set is used for fine-tuning. The dev set is not used for training to maintain comparability with the baseline.

4. **Model Access**: Some models (especially Llama) may require authentication or special access. Make sure you have proper access to HuggingFace model repositories.

5. **GPU Requirements**: Even though the LLM weights stay frozen, you still need enough memory to load the backbone for forward passes (70B models require multi-GPU or large-memory instances). Use `--train_ratio`/`--val_ratio` for lighter debugging runs if resources are limited.

## 📊 Experiment Results

### Baseline Model Results

#### Comparison with Paper

*Evaluation on test set (as required by Step 8)*

| Language Track | Paper (SLAM_baseline) | Our Results (Test) |  | Difference | Status |
|----------------|----------------------|-------------------|-------|------------|--------|
| | AUC | F1 | AUC | F1 | ΔAUC | ΔF1 | |
| **en_es** | 0.774 | 0.190 | **0.774** | 0.190 | 0.000 | 0.000 | ✅ Replicated |
| **es_en** | 0.746 | 0.175 | **0.746** | 0.177 | 0.000 | +0.002 | ✅ Replicated |
| **fr_en** | 0.771 | 0.281 | **0.770** | 0.281 | -0.001 | 0.000 | ✅ Replicated |

**Note**: Paper results are from Table 2 (SLAM_baseline). Our results match the paper's reported values with minimal differences (ΔAUC ≤ 0.001, ΔF1 ≤ 0.002).

---

### 🎯 Summary

**Replication Status:**
- ✅ **All three language tracks successfully replicated**: Test set results match the paper's SLAM_baseline results from Table 2
- ✅ **High accuracy replication**: AUC differences ≤ 0.001, F1 differences ≤ 0.002

**Key Findings:**
- **en_es track**: AUC = 0.774, F1 = 0.190 (exact match with paper)
- **es_en track**: AUC = 0.746 (exact match), F1 = 0.177 (paper: 0.175, Δ = +0.002)
- **fr_en track**: AUC = 0.770 (paper: 0.771, Δ = -0.001), F1 = 0.281 (exact match), highest F1 score

**Performance Characteristics:**
- Lower F1 scores (0.177-0.281) indicate class imbalance, which is typical for this task
- AUC is the primary ranking metric for the shared task

---

### LLM Model Results

LLM fine-tuning experiments are in progress. Results will be updated here once available.

**Expected improvements**: Based on the paper's findings, advanced models (including neural approaches) achieved AUC > 0.80, suggesting LLM fine-tuning should outperform the baseline.

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{settles2018duolingo,
  title={The Second Language Acquisition Modeling (SLAM) Shared Task},
  author={Settles, Burr and LaFlair, Geoffrey T and Hagiwara, Masato},
  booktitle={Proceedings of the 13th Workshop on Innovative Use of NLP for Building Educational Applications},
  year={2018}
}
```

## License

This code is provided for research purposes. Please refer to the original dataset license for data usage.

