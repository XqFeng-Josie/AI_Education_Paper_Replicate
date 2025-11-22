# SLAM: Second Language Acquisition Modeling

Replication and extension of the SLAM shared task for predicting learner errors in second-language learning.

**Paper**: [Second Language Acquisition Modeling](https://aclanthology.org/W18-0506.pdf)  
**Dataset**: [Duolingo SLAM Dataset](https://doi.org/10.7910/DVN/8SWHNO)

## Task

Predict **token-level learner errors** using Duolingo learning traces:
- Input: Exercise with token features + user learning history
- Output: Probability of correct answer for each token (0-1)
- Dataset: >7M tokens from ~6.4k learners across 3 language tracks

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Run baseline
./run_baseline.sh en_es test

# Run LLM+MLP
./llm_mlp/run_pipeline.sh llama-3.1-8b en_es

# Run Zero-Shot LLM (70B)
./llm_zeroshot/run_zeroshot_pipeline.sh llama-3.3-70b-instruct en_es dev
```

## Approaches

### 1. Baseline (Logistic Regression)

**Features**: Token, POS, morphology, user ID, format  
**Training**: Scikit-learn LogisticRegression

```bash
./run_baseline.sh en_es test
```

**Files**: `starter_code/baseline.py`, `run_baseline.sh`

### 2. LLM+MLP

**Features**: Exercise context + user learning history  
**Architecture**: Frozen LLM (embeddings) + Trainable MLP (2-layer classifier)

```bash
# Full pipeline: data → embeddings → training → inference → eval
./llm_mlp/run_pipeline.sh llama-3.1-8b en_es
```

**Details**: See [`llm_mlp/README.md`](llm_mlp/README.md) for design rationale and instructions.

### 3. Zero-Shot LLM (70B)

**Features**: Exercise context + user history + direct probability prediction  
**Architecture**: 70B LLM with zero-shot prompting (no training required)

```bash
# Full pipeline: prompts → inference → eval
./llm_zeroshot/run_zeroshot_pipeline.sh llama-3.3-70b-instruct en_es dev

# Test on 10 exercises first
./llm_zeroshot/run_zeroshot_pipeline.sh llama-3.3-70b-instruct en_es dev 10
```

**Details**: See [`llm_zeroshot/README.md`](llm_zeroshot/README.md) for prompt design and usage.

**Requirements**: A100 GPU (40GB+) with int8 quantization

## Results

### en_es Track (English from Spanish learners)

| Model | AUC | F1 | Accuracy | Notes |
|-------|-----|-----|----------|-------|
| **Baseline** | | | | |
| Logistic Regression | 0.774 | 0.190 | - | Paper baseline (replicated) |
| **LLM+MLP** | | | | |
| Llama3.1-8B + MLP | TBD | TBD | TBD | Token-level, exercise context |
| Mistral-7B + MLP | TBD | TBD | TBD | Token-level, exercise context |
| Qwen2.5-7B + MLP | TBD | TBD | TBD | Token-level, exercise context |
| **Zero-Shot LLM** | | | | |
| Llama3.3-70B (zero-shot) | TBD | TBD | TBD | Direct probability output, no training |
| Llama3.1-70B (zero-shot) | TBD | TBD | TBD | Direct probability output, no training |

> **Note**: Fill in results after running experiments. Baseline target: AUC~0.774, F1~0.190

### Class Distribution (en_es)
- **Token-level**: 13% correct, 87% incorrect  
- **Exercise-level** (all tokens correct): 3% correct, 97% incorrect

## Data Format

Each instance contains:
```
Token: "table"
POS: NOUN
Morphology: {Number: Sing}
Dependency: ROOT
Exercise metadata: user, format, session, days
Label: 0 (incorrect) or 1 (correct)
```

## Dataset Structure

```
dataset/
├── en_es.slam.20190204.{train,dev,test}
├── es_en.slam.20190204.{train,dev,test}
├── fr_en.slam.20190204.{train,dev,test}
└── keys/
    ├── en_es.slam.20190204.{dev,test}.key
    └── ...
```

**Splits**:
- Train: For baseline training / user history construction
- Dev: For LLM+MLP training
- Test: For final evaluation

## Reproduction

### Baseline
```bash
# Replicate paper results
./run_baseline.sh en_es test
```

### LLM+MLP
```bash
# Step-by-step
cd llm_mlp
python step1_prepare_data.py --track en_es
python step2_extract_embeddings.py --model llama-3.1-8b --split dev --track en_es
# ... (see llm_mlp/README.md)

# Or run full pipeline
./llm_mlp/run_pipeline.sh llama-3.1-8b en_es
```

## Evaluation

```bash
python starter_code/eval.py \
    --pred predictions/model_en_es_test.pred \
    --key dataset/en_es.slam.20190204.test.key
```

Output metrics: **AUC, F1, Accuracy, Avg Log Loss**

## Environment

```bash
# Python 3.8+
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For Llama models
huggingface-cli login
```