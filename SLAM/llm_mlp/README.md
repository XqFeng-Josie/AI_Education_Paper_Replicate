# LLM+MLP for SLAM: Token-Level Prediction with Exercise Context

## Motivation & Design

### Problem with Baseline
- Baseline uses hand-crafted features without user learning context
- Ignores exercise structure (tokens within same translation exercise)
- Limited ability to model learning patterns

### Our Approach: Token-Level + Exercise Context

**Key Insight**: Each token should see the full exercise context (complete sentence) while making independent predictions.

```
Exercise: "A table for two please"
├── Token 1: 'A' → Prediction (with full exercise context)
├── Token 2: 'table' → Prediction (with full exercise context)  
├── Token 3: 'for' → Prediction (with full exercise context)
└── ...
```

### Architecture

1. **Data Preparation**: Group tokens by exercise, create contextualized prompts
2. **Frozen LLM**: Extract embeddings for each token (with exercise context in prompt)
3. **Trainable MLP**: 2-layer classifier trained on token embeddings
4. **Prediction**: Independent prediction per token

### Prompt Design

```
Learner's history: 14.5 days of practice, 1257 attempts, 21.6% correct.
Performance by format: listen=40.8%, reverse_translate=13.6%, reverse_tap=6.1%.
Exercise (all tokens): "A table for two please".
Current token #1/5: 'A' (POS: DET, Format: reverse_translate, ...).
Will the learner answer this token correctly?
```

**Benefits**:
- ✅ Full exercise context for better understanding
- ✅ User learning history for personalized prediction
- ✅ Token-level granularity (same as baseline for fair comparison)

## Quick Start

```bash
# Full pipeline (data → embeddings → training → inference → eval)
./llm_mlp/run_pipeline.sh llama-3.1-8b en_es
```

## Step-by-Step Usage

### 1. Data Preparation
```bash
python llm_mlp/step1_prepare_data.py --track en_es
```
Output: `llm_mlp/data/en_es_{dev,test}_exercise.jsonl` with contextualized prompts

### 2. Extract Token Embeddings
```bash
python llm_mlp/step2_extract_embeddings.py \
    --model llama-3.1-8b \
    --split dev \
    --track en_es \
    --batch_size 32 \
    --resume
```
Output: `llm_mlp/embeddings/en_es_dev_llama-3.1-8b_token_embeddings.pt`

### 3. Train MLP
```bash
python llm_mlp/step3_train_mlp.py \
    --embeddings_path llm_mlp/embeddings/en_es_dev_llama-3.1-8b_token_embeddings.pt \
    --model_name llama-3.1-8b \
    --track en_es
```
Output: `llm_mlp/models/llama-3.1-8b_en_es/mlp_classifier.pt`

### 4. Inference & Evaluation
```bash
# Generate predictions
python llm_mlp/step4_inference.py \
    --model_dir llm_mlp/models/llama-3.1-8b_en_es \
    --embeddings_path llm_mlp/embeddings/en_es_test_llama-3.1-8b_token_embeddings.pt \
    --output_file llm_mlp/predictions/llama-3.1-8b_en_es_test.pred

# Evaluate
python llm_mlp/step5_evaluate.py \
    --pred llm_mlp/predictions/llama-3.1-8b_en_es_test.pred \
    --key dataset/en_es.slam.20190204.test.key
```

## Technical Details

### MLP Architecture
```python
Linear(4096 → 2048) → ReLU() → Dropout(0.1) → Linear(2048 → 1) → Sigmoid()
```

### Supported Models
- `llama-3.1-8b`: Llama-3.1-8B-Instruct
- `llama-3.3-70b-instruct`: Llama-3.3-70B-Instruct  
- `mistral-7b`: Mistral-7B-Instruct-v0.3
- `qwen-2.5-7b`: Qwen2.5-7B-Instruct

Update paths in `llm_mlp/utils.py` if needed.

### Hyperparameters
- LLM: Frozen (FP16)
- MLP: Trained with AdamW (lr=5e-4, weight_decay=0.01)
- Loss: BCEWithLogitsLoss with pos_weight for class imbalance (~13% positive)
- Batch size: 128 (training), 32 (embedding extraction)
- Epochs: 10 with early stopping

## File Structure
```
llm_mlp/
├── step1_prepare_data.py       # Create prompts with exercise context
├── step2_extract_embeddings.py # Extract token embeddings (frozen LLM)
├── step3_train_mlp.py          # Train MLP classifier
├── step4_inference.py          # Generate predictions
├── step5_evaluate.py           # Evaluation wrapper
├── run_pipeline.sh             # End-to-end automation
├── utils.py                    # UserHistory, prompts, MODEL_MAPPING
├── data_loader.py              # Data loading utilities
├── recover_from_checkpoint.py  # Resume from interrupted extraction
└── README.md                   # This file
```

## Troubleshooting

**Resume interrupted extraction**:
```bash
# Automatically resume from last checkpoint
python llm_mlp/step2_extract_embeddings.py --resume ...
```

**CUDA OOM**: Reduce `--batch_size` to 16 or 8

**Quick test**: Use `--max_exercises 100` in step2 for faster testing
